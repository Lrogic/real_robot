"""Symmetry selection and observation math (numpy only; any env with pytest).

  python -m pytest real_robot/manipulation/policy/tests/test_symmetry.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy import observations as obs  # noqa: E402
from manipulation.policy.symmetry import (  # noqa: E402
    BOX_SYMMETRY_NAMES,
    box_symmetries,
    half_turn,
    rotation_angle,
    select_closest,
)

# Training reset orientation of the jenga block (tabletop_jenga_wxai.json).
R_TRAIN = obs.quat_wxyz_to_matrix([0.5, 0.5, -0.5, 0.5])


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_box_symmetries_are_identity_and_half_turns():
    syms = box_symmetries()
    assert len(syms) == len(BOX_SYMMETRY_NAMES) == 4
    assert np.allclose(syms[0], np.eye(4))
    for k in range(3):
        assert np.allclose(syms[k + 1][:3, :3], half_turn(np.eye(3)[k]))
        assert np.allclose(syms[k + 1][:3, 3], 0.0)
        assert np.allclose(syms[k + 1] @ syms[k + 1], np.eye(4))


def test_off_centre_box_keeps_its_centre_fixed():
    center = np.array([0.01, -0.02, 0.005])
    axes = _rot_z(30.0)
    for S in box_symmetries(center, axes):
        assert np.allclose(S[:3, :3] @ center + S[:3, 3], center)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("yaw_deg", [-20.0, 0.0, 15.0])
def test_flipped_tracking_is_undone(k, yaw_deg):
    """FP reporting any symmetric pose maps back to the one nearest the training pose."""
    syms = box_symmetries()
    R_true = _rot_z(yaw_deg) @ R_TRAIN
    T_true = obs.make_T(R_true, [0.36, 0.02, 0.04])
    T_fp = T_true @ syms[k]
    choice = select_closest(T_fp, syms, R_TRAIN)
    assert choice.index == k  # S_k is its own inverse
    assert np.allclose(choice.T, T_true)
    assert rotation_angle(choice.T[:3, :3], R_TRAIN) == pytest.approx(np.radians(abs(yaw_deg)), abs=1e-9)
    assert sorted(choice.angles_rad)[1] > np.radians(90.0)


def test_quaternion_sign_rule_and_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = obs.quat_wxyz_to_matrix(q)
        out = obs.quat_wxyz_from_matrix(R)
        assert out[0] >= 0.0
        assert np.allclose(out, q if q[0] >= 0 else -q, atol=1e-9)
    assert np.allclose(obs.quat_wxyz_from_matrix(R_TRAIN), [0.5, 0.5, -0.5, 0.5])


def test_rotvec_matches_quaternion():
    axis = np.array([0.2, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    angle = 1.1
    q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    assert np.allclose(obs.rotvec_to_matrix(axis * angle), obs.quat_wxyz_to_matrix(q))


def test_actor_vector_layout():
    T_tcp = obs.make_T(np.eye(3), [0.25, 0.0, 0.17])
    T_obj = obs.make_T(R_TRAIN, [0.365, 0.0, 0.0375])
    goal = T_obj[:3, 3] + [0, 0, 0.03]
    g = obs.carriages_from_gripper(0.01)
    terms = obs.actor_terms(arm_q=np.arange(6) * 0.1, arm_qd=np.zeros(6), carriage_q=g, carriage_qd=np.zeros(2),
                            T_base_tcp=T_tcp, T_base_obj=T_obj, goal_pos=goal,
                            gripper_open=0.04, gripper_closed=0.0)
    vec = obs.build_actor_obs(terms)
    assert vec.shape == (obs.ACTOR_DIM,) == (37,)
    assert np.allclose(vec[6:8], 0.01)
    assert vec[23] == pytest.approx(0.75)
    assert np.allclose(vec[24:31], [0.365, 0.0, 0.0375, 0.5, 0.5, -0.5, 0.5])
    assert np.allclose(vec[31:34], [0.115, 0.0, -0.1325])
    assert np.allclose(vec[34:37], [0.0, 0.0, 0.03])
    del terms["obj_to_goal_pos"]
    assert obs.build_actor_obs(terms) is None
