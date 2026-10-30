"""actions.ee_delta_target reproduces the sim's ClippedDifferentialIKAction target and binary gripper.

The mjlab comparison needs the wxai_mjlab env; the rest runs in any env with numpy:

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_actions.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy import observations as obs  # noqa: E402
from manipulation.policy.actions import (  # noqa: E402
    driver_pose6, ee_delta_target, in_workspace, matrix_to_rotvec)

SCALES = dict(pos_scale=0.01, rot_scale=0.05, gripper_open=0.04, gripper_closed=0.0)


def _random_T(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return obs.make_T(obs.quat_wxyz_to_matrix(q / np.linalg.norm(q)), rng.uniform(-0.5, 0.5, 3))


def test_matches_mjlab_apply_delta_pose():
    torch = pytest.importorskip("torch")
    math = pytest.importorskip("mjlab.utils.lab_api.math")
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = _random_T(rng)
        a = rng.uniform(-2.0, 2.0, 7)
        ours = ee_delta_target(T, a, **SCALES)
        delta = np.clip(a[:6], -1.0, 1.0) * np.array([0.01] * 3 + [0.05] * 3)
        pos, quat = math.apply_delta_pose(torch.tensor(T[:3, 3][None]), torch.tensor(obs.pose7(T)[3:][None]),
                                          torch.tensor(delta[None]))
        expected = obs.make_T(obs.quat_wxyz_to_matrix(quat[0].numpy()), pos[0].numpy())
        assert np.allclose(ours.T_base_target, expected, atol=1e-9)


def test_clip_scale_and_gripper():
    T = np.eye(4)
    t = ee_delta_target(T, [3.0, -3.0, 0.5, 0, 0, 0, 0.0], **SCALES)
    assert np.allclose(t.clipped[:3], [1.0, -1.0, 0.5])
    assert np.allclose(t.T_base_target[:3, 3], [0.01, -0.01, 0.005])
    assert t.gripper_open and t.gripper == 0.04  # a = 0 opens, as GripperAction binary
    t = ee_delta_target(T, [0, 0, 0, 0, 0, 2.0, -1e-6], **SCALES)
    assert not t.gripper_open and t.gripper == 0.0
    assert np.isclose(np.arctan2(t.T_base_target[1, 0], t.T_base_target[0, 0]), 0.05)
    half = ee_delta_target(T, [1, 0, 0, 0, 0, 0, 1], **SCALES, action_scale=0.5)
    assert np.allclose(half.T_base_target[:3, 3], [0.005, 0, 0])
    with pytest.raises(ValueError):
        ee_delta_target(T, [np.nan, 0, 0, 0, 0, 0, 1], **SCALES)


def test_driver_pose6_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        T = _random_T(rng)
        p = driver_pose6(T)
        assert np.allclose(obs.make_T(obs.rotvec_to_matrix(p[3:]), p[:3]), T, atol=1e-9)
        assert np.linalg.norm(matrix_to_rotvec(T[:3, :3])) <= np.pi + 1e-9


def test_workspace():
    lo, hi = np.array([0.1, -0.3, 0.0]), np.array([0.6, 0.3, 0.4])
    assert in_workspace([0.3, 0.0, 0.1], lo, hi)
    assert not in_workspace([0.3, 0.0, -0.01], lo, hi)
