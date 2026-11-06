"""The training action through the experiment's adapter (needs mjlab; builds the env once).

Device: HWC_TEST_DEVICE (default cuda:0 when available, else cpu).
"""

import os

import numpy as np
import pytest
import torch

from fk import TrainingFK
from geometry import displaced, make_T, pose_from_json
from ik_adapter import TrainingIK, held_target, self_check, world_from_base
from plans import make_plan
from settings import load_settings
from common.poses import quaternion_wxyz

DEVICE = os.environ.get("HWC_TEST_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_settings()


@pytest.fixture(scope="module")
def fk(cfg):
    return TrainingFK(cfg["training"]["scene"])


@pytest.fixture(scope="module")
def env(cfg):
    from sim_reference import build_training_env
    return build_training_env(cfg, DEVICE)


def test_training_configuration_is_preserved(env):
    from sim_reference import training_controller_record
    rec = training_controller_record(env)
    assert rec["physics_dt_s"] == pytest.approx(0.002)
    assert rec["decimation"] == 25
    arm = rec["arm_action"]
    assert arm["class"].endswith("ClippedDifferentialIKAction")
    assert (arm["damping"], arm["max_dq"], arm["joint_limit_weight"], arm["posture_weight"]) == (0.05, 0.5, 1.0, 0.0)
    assert (arm["delta_pos_scale"], arm["delta_ori_scale"], arm["frame_name"]) == (0.01, 0.05, "ee_site")
    assert all(v == 1.0 for n, v in rec["robot_body_gravcomp"].items() if not n.endswith("mocap_base"))
    assert "randomize_stiffness_damping" not in env.cfg.events


@pytest.mark.parametrize("action", [(0.4, -0.3, 0.5, 0.2, -0.6, 0.3), (1.7, 0.0, -2.0, 0.0, 0.0, 1.3)])
def test_adapter_matches_the_action_for_identical_state_and_target(env, action):
    check = self_check(TrainingIK(env), action)
    assert check["desired_position_diff_m"] < 1e-6
    assert check["desired_rotation_diff_rad"] < 1e-5
    assert check["joint_target_diff_rad"] < 1e-5


def test_solve_on_measured_state_does_not_step_physics(env, fk):
    ik = TrainingIK(env)
    q = fk.ready_arm + np.array([0.02, -0.03, 0.01, 0.04, -0.02, 0.05])
    ik.set_target(displaced(fk.tcp(q), [0.0, 0.0, 0.01]))
    t_before = float(env.sim.data.time[0])
    q_target, _ = ik.solve(q)
    assert float(env.sim.data.time[0]) == t_before
    assert np.allclose(env.scene["robot"].data.joint_pos[0, ik.ids].cpu().numpy(), q, atol=1e-6)
    # site pose was refreshed at the measured q
    p = env.sim.data.site_xpos[0, ik.term._frame_id].cpu().numpy()
    assert np.allclose(np.linalg.inv(ik.T_world_base) @ np.append(p, 1.0), np.append(fk.tcp(q)[:3, 3], 1.0), atol=1e-5)
    assert not np.allclose(q_target, q)


def test_sim_holds_absolute_targets_between_outer_updates(env, fk, cfg):
    from sim_reference import SimReference
    sim = SimReference(env, fk)
    plan = make_plan(fk, cfg, "scheduled", "rotation", "y", 1)
    seen = []
    sim.substep_hook = lambda i, held: seen.append((i, held))
    result = sim.run(plan, cfg)
    T_wb = world_from_base(env)
    for i, (pos, quat) in seen:
        k = i // sim.decimation
        if k < len(plan["targets"]):
            T = T_wb @ pose_from_json(plan["targets"][k]["pose"])
            assert np.allclose(pos, T[:3, 3], atol=1e-6)
            q = quaternion_wxyz(T[:3, :3])
            assert abs(abs(float(np.dot(q, quat))) - 1.0) < 1e-6
        else:
            assert np.array_equal(pos, seen[i - 1][1][0]) and np.array_equal(quat, seen[i - 1][1][1])
    sent = [o for o in result["outer_updates"] if o["new_target"]]
    assert [o["index"] for o in sent] == list(range(len(plan["targets"])))
    assert result["settle"]["settled"]
    assert max(result["object_displacement_m"].values(), default=0.0) < 1e-4


def test_sim_relative_test_reaches_goal(env, fk, cfg):
    from sim_reference import SimReference
    sim = SimReference(env, fk)
    sim.substep_hook = None
    plan = make_plan(fk, cfg, "relative", "translation", "x", 1)
    result = sim.run(plan, cfg)
    assert result["stop_reason"] == "goal_reached"
    steps = [o["info"]["step_translation_m"][2] for o in result["outer_updates"] if o["new_target"]]
    assert all(s <= 0.01 + 1e-12 for s in steps)
    goal = pose_from_json(plan["relative"]["goal"])
    final = fk.tcp(np.asarray(result["samples"]["arm_q_rad"][-1]))
    assert np.linalg.norm(final[:3, 3] - goal[:3, 3]) < plan["relative"]["position_tolerance_m"] * 1.5
