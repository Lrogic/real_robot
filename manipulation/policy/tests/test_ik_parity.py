"""ik.DifferentialIK reproduces the training ClippedDifferentialIKAction substep on the same joints.

Builds the jenga training env (one env), writes joints into it, runs the action's own
process_actions (random policy action) and apply_actions, and compares the joint position
target with ik.DifferentialIK on the configured robot model: the goal composed by
actions.ee_delta_target on its FK, then one solve(). Joints are sampled across the soft
limits so the joint-limit rows and the max_dq clamp are exercised. Also checks the sim_ik
config against the embodiment and the training model's kinematics, and, with the task's
action.ee_goal_interpolation on, every substep's subgoal: solve(q, alpha = i / N).

Needs the wxai_mjlab env:

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_ik_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy.actions import ee_delta_target, matrix_to_rotvec  # noqa: E402
from manipulation.policy.ik import ARM_JOINTS, DifferentialIK  # noqa: E402
from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLES = 300


@pytest.fixture(scope="module")
def cfg() -> JengaPolicyConfig:
    return JengaPolicyConfig.load()


@pytest.fixture(scope="module")
def ik(cfg) -> DifferentialIK:
    k = cfg.sim_ik
    return DifferentialIK(k.model, k.site, k.params, cfg.ready_joint_pos)


def test_config_matches_embodiment(cfg, ik):
    from tabletop_sim.embodiments import get_embodiment

    emb = get_embodiment(cfg.embodiment)
    p = cfg.sim_ik.params
    assert emb.arm_joint_names == ARM_JOINTS
    assert cfg.sim_ik.site == emb.tcp_site
    for name in ("damping", "max_dq", "position_weight", "orientation_weight", "joint_limit_weight",
                 "posture_weight"):
        assert getattr(p, name) == getattr(emb.ik, name), name
    assert p.soft_joint_pos_limit_factor == emb.soft_joint_pos_limit_factor
    assert cfg.sim_dt == 0.002 and cfg.control_hz == 20.0

    # The configured model has the training model's kinematics and ranges.
    training = DifferentialIK(emb.xml_path, emb.tcp_site, p, cfg.ready_joint_pos)
    assert np.allclose(ik.soft_lower, training.soft_lower) and np.allclose(ik.soft_upper, training.soft_upper)
    assert np.allclose(ik.ctrl_lower, training.ctrl_lower) and np.allclose(ik.ctrl_upper, training.ctrl_upper)
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(ik.ctrl_lower, ik.ctrl_upper)
        assert np.allclose(ik.fk(q), training.fk(q), atol=1e-12)


def test_subgoal_interpolates_from_start_to_goal(cfg, ik):
    q = cfg.ready_joint_pos
    T_start = ik.fk(q)
    T_goal = ee_delta_target(T_start, [1.0, -0.5, 0.25, 0.3, -1.0, 0.6, 1.0], pos_scale=cfg.ee_pos_scale,
                             rot_scale=cfg.ee_rot_scale, gripper_open=0.04, gripper_closed=0.0).T_base_target
    try:
        ik.set_goal(T_goal)
        direct = ik.solve(q)
        for alpha in (0.0, 0.3, 1.0):  # no start pose: every subgoal is the goal
            assert np.array_equal(ik.solve(q, alpha)[0], direct[0])

        ik.set_goal(T_goal, T_start)
        pos, rot = ik.subgoal(0.0)
        assert np.allclose(pos, T_start[:3, 3], atol=1e-12) and np.allclose(rot, T_start[:3, :3], atol=1e-12)
        for alpha in (1.0, 1.5):
            pos, rot = ik.subgoal(alpha)
            assert np.array_equal(pos, T_goal[:3, 3]) and np.array_equal(rot, T_goal[:3, :3])
        full = matrix_to_rotvec(T_goal[:3, :3] @ T_start[:3, :3].T)
        for alpha in (0.04, 0.5, 0.96):
            pos, rot = ik.subgoal(alpha)
            assert np.allclose(pos, T_start[:3, 3] + alpha * (T_goal[:3, 3] - T_start[:3, 3]), atol=1e-12)
            assert np.allclose(matrix_to_rotvec(rot @ T_start[:3, :3].T), alpha * full, atol=1e-9)
        assert np.array_equal(ik.solve(q)[0], direct[0])  # alpha defaults to the goal
        assert np.allclose(ik.solve(q, 0.0)[1], 0.0, atol=1e-9)  # already at the start pose
    finally:
        ik.set_goal(np.eye(4))


def test_interpolated_substeps_match_mjlab_action(cfg, ik, tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg

    task = json.loads(cfg.task_path.read_text())
    task.setdefault("action", {})["ee_goal_interpolation"] = True
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    env = ManagerBasedRlEnv(make_grasp_env_cfg(cfg.scene_path, task_path, None, 1), device=DEVICE)
    try:
        env.reset()
        term = env.action_manager.get_term("arm")
        assert term.cfg.interpolate_goal is True
        robot, ids, substeps = env.scene["robot"], term._joint_ids, env.cfg.decimation
        assert substeps == round(1.0 / (cfg.control_hz * cfg.sim_dt))
        rng = np.random.default_rng(1)
        q_t = torch.zeros(1, len(ids), device=DEVICE)
        worst = 0.0
        for _ in range(40):
            q = np.clip(cfg.ready_joint_pos + rng.normal(scale=0.3, size=6), ik.ctrl_lower, ik.ctrl_upper)
            action = rng.uniform(-1.5, 1.5, 6)
            q_t[0] = torch.as_tensor(q, dtype=torch.float32)
            robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=ids)
            env.sim.forward()
            q32 = robot.data.joint_pos[0, ids].cpu().numpy().astype(np.float64)
            term.process_actions(torch.as_tensor(action[None], dtype=torch.float32, device=DEVICE))
            target = ee_delta_target(ik.fk(q32), np.r_[action, 1.0], pos_scale=cfg.ee_pos_scale,
                                     rot_scale=cfg.ee_rot_scale, gripper_open=0.04, gripper_closed=0.0)
            ik.set_goal(target.T_base_target, target.T_base_start)
            # The joints stay put, so each call is the next substep's solve from the same q.
            for i in range(1, substeps + 1):
                term.apply_actions()
                expected = robot.data.joint_pos_target[0, ids].cpu().numpy().astype(np.float64)
                _, dq = ik.solve(q32, i / substeps)
                worst = max(worst, float(np.max(np.abs(q32 + dq - expected))))
        print(f"max |numpy - mjlab| joint target over interpolated substeps: {worst:.2e} rad")
        assert worst < 2e-5
    finally:
        env.close()
        ik.set_goal(np.eye(4))


def test_matches_mjlab_action(cfg, ik):
    from mjlab.envs import ManagerBasedRlEnv
    from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg

    env = ManagerBasedRlEnv(make_grasp_env_cfg(cfg.scene_path, cfg.task_path, None, 1), device=DEVICE)
    try:
        env.reset()
        term = env.action_manager.get_term("arm")
        assert type(term).__name__ == "ClippedDifferentialIKAction"
        robot = env.scene["robot"]
        ids = term._joint_ids
        assert [robot.joint_names[i] for i in ids.tolist()] == list(ARM_JOINTS)
        assert np.allclose(term._joint_lower[0].cpu().numpy(), ik.soft_lower, atol=1e-6)
        assert np.allclose(term._joint_upper[0].cpu().numpy(), ik.soft_upper, atol=1e-6)
        root = robot.data.root_link_pose_w[0].cpu().numpy()
        assert np.allclose(root[3:], [1, 0, 0, 0])

        rng = np.random.default_rng(0)
        q_t = torch.zeros(1, len(ids), device=DEVICE)
        worst, worst_policy, worst_fk, clamped, limited = 0.0, 0.0, 0.0, 0, 0
        for i in range(SAMPLES):
            # Half near the ready pose, half anywhere a little beyond the soft limits.
            if i % 2:
                q = cfg.ready_joint_pos + rng.normal(scale=0.3, size=6)
            else:
                lo, hi = ik.soft_lower, ik.soft_upper
                q = rng.uniform(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            q = np.clip(q, ik.ctrl_lower, ik.ctrl_upper)
            # 1-in-4 actions are scaled up 40x so the step exceeds max_dq.
            action = rng.uniform(-1.5, 1.5, 6) * (40.0 if i % 4 == 0 else 1.0)
            q_t[0] = torch.as_tensor(q, dtype=torch.float32)
            robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=ids)
            env.sim.forward()
            q32 = robot.data.joint_pos[0, ids].cpu().numpy().astype(np.float64)

            if i % 4 == 0:
                # Far goals: write the absolute goal directly, as process_actions' clamp would cap them.
                T_goal = ee_delta_target(ik.fk(q32), np.r_[np.clip(action, -1, 1), 1.0], pos_scale=0.4,
                                         rot_scale=1.5, gripper_open=0.04, gripper_closed=0.0).T_base_target
                from mjlab.utils.lab_api.math import quat_from_matrix

                pos_w = T_goal[:3, 3] + root[:3]
                term._desired_pos[:] = torch.as_tensor(pos_w, dtype=torch.float32, device=DEVICE)
                term._desired_quat[:] = quat_from_matrix(torch.as_tensor(T_goal[:3, :3], dtype=torch.float32,
                                                                         device=DEVICE))
            else:
                term.process_actions(torch.as_tensor(action[None], dtype=torch.float32, device=DEVICE))
                T_goal = ee_delta_target(ik.fk(q32), np.r_[action, 1.0], pos_scale=cfg.ee_pos_scale,
                                         rot_scale=cfg.ee_rot_scale, gripper_open=0.04,
                                         gripper_closed=0.0).T_base_target
                desired_w = term._desired_pos[0].cpu().numpy()
                worst_fk = max(worst_fk, float(np.max(np.abs(desired_w - root[:3] - T_goal[:3, 3]))))
            term.apply_actions()
            expected = robot.data.joint_pos_target[0, ids].cpu().numpy().astype(np.float64)

            ik.set_goal(T_goal)
            q_target, dq = ik.solve(q32)
            clamped += bool(np.any(np.abs(dq) >= ik.params.max_dq - 1e-9))
            limited += bool(np.any((q32 < ik.soft_lower) | (q32 > ik.soft_upper)))
            err = float(np.max(np.abs(q32 + dq - expected)))
            worst = max(worst, err)
            if i % 4:
                worst_policy = max(worst_policy, err)
            # MuJoCo clamps the actuator ctrl to its ctrlrange; solve() returns it clamped.
            assert np.allclose(q_target, np.clip(expected, ik.ctrl_lower, ik.ctrl_upper), atol=5e-4)
        print(f"max |numpy - mjlab| joint target: policy-sized goals {worst_policy:.2e} rad, all {worst:.2e} rad; "
              f"goal position {worst_fk:.2e} m; {clamped} max_dq-clamped, {limited} outside the soft limits "
              f"of {SAMPLES}")
        assert worst_fk < 1e-6
        # mjlab solves in float32 on the GPU; far goals near singular poses amplify its rounding.
        assert worst_policy < 2e-5
        assert worst < 5e-4
        assert clamped > 10 and limited > 10
    finally:
        env.close()
