"""Target generation, rotation conventions, final-step capping and scheduled-target consistency."""

import json

import numpy as np
import pytest
import torch

from common.poses import rotvec_to_matrix
from geometry import axis_vector, bounded_step, displaced, make_T, pose_error, rotation_angle
from plans import basic_suite, load_plan, make_plan, save_plan, scheduled_multipliers
from settings import load_settings
from targets import TargetSource

RNG = np.random.default_rng(1)


def random_pose():
    return make_T(rotvec_to_matrix(RNG.normal(size=3)), RNG.normal(size=3) * 0.2)


@pytest.fixture(scope="module")
def cfg():
    return load_settings()


@pytest.fixture(scope="module")
def fk(cfg):
    from fk import TrainingFK
    return TrainingFK(cfg["training"]["scene"])


def test_displaced_matches_mjlab_apply_delta_pose():
    from mjlab.utils.lab_api.math import apply_delta_pose, matrix_from_quat, quat_from_matrix
    for _ in range(20):
        T = random_pose()
        delta = RNG.normal(size=6) * np.array([0.01, 0.01, 0.01, 0.3, 0.3, 0.3])
        pos, quat = apply_delta_pose(torch.tensor(T[None, :3, 3]), quat_from_matrix(torch.tensor(T[None, :3, :3])),
                                     torch.tensor(delta[None]))
        expected = displaced(T, delta[:3], delta[3:])
        assert np.allclose(pos[0].numpy(), expected[:3, 3], atol=1e-12)
        assert np.allclose(matrix_from_quat(quat)[0].numpy(), expected[:3, :3], atol=1e-9)


def test_pose_error_matches_mjlab_compute_pose_error():
    from mjlab.utils.lab_api.math import compute_pose_error, quat_from_matrix
    for _ in range(20):
        A, B = random_pose(), displaced(random_pose(), rotvec=RNG.normal(size=3) * 0.5)
        dp, dr = pose_error(A, B)
        p, r = compute_pose_error(torch.tensor(A[None, :3, 3]), quat_from_matrix(torch.tensor(A[None, :3, :3])),
                                  torch.tensor(B[None, :3, 3]), quat_from_matrix(torch.tensor(B[None, :3, :3])))
        assert np.allclose(p[0].numpy(), dp, atol=1e-12)
        assert np.allclose(r[0].numpy(), dr, atol=1e-6)
        assert np.allclose(displaced(A, dp, dr), B, atol=1e-9)


def test_rotation_targets_are_compositions_about_base_axes(fk, cfg):
    plan = make_plan(fk, cfg, "scheduled", "rotation", "x", -1)
    T_ref = fk.tcp(fk.ready_arm)
    theta = cfg["motion"]["rotation_rad"]
    a = axis_vector("x", -1)
    mults = scheduled_multipliers(**{k: cfg["scheduled"][k] for k in ("out_steps", "back_steps", "hold_steps")})
    for target, s in zip(plan["targets"], mults):
        R = np.asarray(target["pose"]["rotation_matrix"])
        assert np.allclose(target["pose"]["position_m"], T_ref[:3, 3])
        assert np.allclose(R, rotvec_to_matrix(a * s * theta) @ T_ref[:3, :3], atol=1e-12)
        assert rotation_angle(T_ref[:3, :3], R) == pytest.approx(abs(s) * theta, abs=1e-9)
    # The reference orientation is far from identity, so adding the step to its rotation
    # vector (or Euler angles) would be a different rotation.
    rv_ref = np.asarray(plan["reference"]["tcp"]["rotvec_rad"])
    wrong = rotvec_to_matrix(rv_ref + a * theta)
    right = np.asarray(plan["targets"][0]["pose"]["rotation_matrix"])
    assert rotation_angle(wrong, right) > 0.01


def test_translation_targets_keep_orientation(fk, cfg):
    plan = make_plan(fk, cfg, "single", "translation", "z", 1)
    T_ref = fk.tcp(fk.ready_arm)
    (target,) = plan["targets"]
    assert np.allclose(np.asarray(target["pose"]["position_m"]) - T_ref[:3, 3], [0, 0, cfg["motion"]["translation_m"]])
    assert np.allclose(target["pose"]["rotation_matrix"], T_ref[:3, :3])


def test_scheduled_sequence_shape():
    assert scheduled_multipliers(4, 4, 3) == [1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0]
    assert scheduled_multipliers(2, 4, 1) == [1, 2, 1, 0, -1, -2, -2]


@pytest.mark.parametrize("step_limit", ["per_axis", "euclidean"])
def test_bounded_steps_never_pass_goal_and_end_exactly(step_limit):
    for _ in range(20):
        start = random_pose()
        goal = displaced(start, RNG.normal(size=3) * 0.03, RNG.normal(size=3) * 0.15)
        T = start
        dist0, ang0 = np.linalg.norm(goal[:3, 3] - T[:3, 3]), rotation_angle(goal[:3, :3], T[:3, :3])
        for _ in range(200):
            step = bounded_step(T, goal, 0.01, 0.05, step_limit)
            if step_limit == "per_axis":
                assert np.all(np.abs(step.translation) <= 0.01 + 1e-12)
                assert np.all(np.abs(step.rotvec) <= 0.05 + 1e-12)
                dp, _ = pose_error(T, goal)
                assert np.all(np.abs(step.translation) <= np.abs(dp) + 1e-12)
                assert np.all(step.translation * dp >= -1e-15)
            else:
                assert np.linalg.norm(step.translation) <= 0.01 + 1e-12
                assert np.linalg.norm(step.rotvec) <= 0.05 + 1e-12
            dist, ang = np.linalg.norm(goal[:3, 3] - step.T_target[:3, 3]), rotation_angle(goal[:3, :3], step.T_target[:3, :3])
            assert dist <= dist0 + 1e-12 and ang <= ang0 + 1e-9
            dist0, ang0, T = dist, ang, step.T_target
            if step.final:
                assert np.array_equal(step.T_target, goal)
                break
        else:
            pytest.fail("never reached the goal")


def test_single_axis_final_step_is_capped():
    start = make_T(np.eye(3), [0.2, 0.0, 0.1])
    goal = displaced(start, [0.035, 0.0, 0.0])
    T, increments = start, []
    while True:
        s = bounded_step(T, goal, 0.01, 0.05, "per_axis")
        increments.append(s.translation[0])
        T = s.T_target
        if s.final:
            break
    assert np.allclose(increments, [0.01, 0.01, 0.01, 0.005])
    assert np.array_equal(T, goal)


def test_combined_axis_conventions_differ():
    start = make_T(np.eye(3), [0.0, 0.0, 0.0])
    goal = displaced(start, [0.02, 0.02, 0.0])
    per_axis = bounded_step(start, goal, 0.01, 0.05, "per_axis")
    euclid = bounded_step(start, goal, 0.01, 0.05, "euclidean")
    assert np.linalg.norm(per_axis.translation) == pytest.approx(0.01 * np.sqrt(2))
    assert np.linalg.norm(euclid.translation) == pytest.approx(0.01)


def test_target_source_single_and_scheduled_ignore_current_pose(fk, cfg):
    for test in ("single", "scheduled"):
        plan = make_plan(fk, cfg, test, "translation", "y", 1)
        src = TargetSource(plan)
        sent = []
        for k in range(len(plan["targets"]) + 5):
            T, _ = src.update(k, k * plan["outer_period_s"], random_pose())
            sent.append(T)
        expected = [np.asarray(t["pose"]["rotation_matrix"]) for t in plan["targets"]]
        got = [T for T in sent if T is not None]
        assert len(got) == len(expected)
        for T, R, t in zip(got, expected, plan["targets"]):
            assert np.allclose(T[:3, :3], R) and np.allclose(T[:3, 3], t["pose"]["position_m"])
        if test == "single":
            assert sent[0] is not None and all(T is None for T in sent[1:])


def test_target_source_relative_steps_from_current_pose(fk, cfg):
    plan = make_plan(fk, cfg, "relative", "translation", "x", 1)
    src = TargetSource(plan)
    goal = np.asarray(plan["relative"]["goal"]["position_m"])
    current = fk.tcp(fk.ready_arm)
    current[:3, 3] += [0.002, 0.001, 0.0]          # measured pose not on the reference
    T, info = src.update(0, 0.0, current)
    assert T[0, 3] - current[0, 3] == pytest.approx(0.01)
    assert T[1, 3] == pytest.approx(goal[1])        # small off-axis error is corrected in one step
    near = make_T(current[:3, :3], goal - [0.004, 0, 0])
    T, info = src.update(1, 0.05, near)
    assert info["final_step"] and np.allclose(T[:3, 3], goal)
    T, info = src.update(2, 0.10, make_T(np.asarray(plan["relative"]["goal"]["rotation_matrix"]), goal + 0.0005))
    assert T is None and info["reached"] and src.reached_at == pytest.approx(0.10)
    assert src.done(0.10 + plan["relative"]["post_goal_observe_s"]) == "goal_reached"


def test_plan_file_roundtrip_and_tamper_detection(fk, cfg, tmp_path):
    plan = make_plan(fk, cfg, "scheduled", "translation", "z", -1)
    path = save_plan(plan, tmp_path)
    a, b = load_plan(path), load_plan(path)
    assert [t["pose"] for t in a["targets"]] == [t["pose"] for t in b["targets"]] == [t["pose"] for t in plan["targets"]]
    data = json.loads(path.read_text())
    data["targets"][2]["pose"]["position_m"][0] += 1e-4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_plan(path)


def test_basic_suite_is_inside_default_safety_limits(fk, cfg):
    from run_hardware import precheck
    for plan in basic_suite(fk, cfg):
        assert precheck(plan, cfg["hardware"]["safety"]) == [], plan["name"]
