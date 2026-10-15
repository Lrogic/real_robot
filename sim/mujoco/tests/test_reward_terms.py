"""Reward kernels on hand-built state (CPU, no simulator)."""

from types import SimpleNamespace

import pytest
import torch

from tabletop_sim.tasks.grasp.mdp import rewards, task_state

GRASP = dict(
  sensor_names=("l", "r"),
  finger_cfg=None,
  finger_outward_axes=((0, 1, 0), (0, -1, 0)),
  min_force=0.2,
  max_angle_deg=110.0,
)
HOLD = dict(command_name="grasp", object_cfg=None, lift_threshold=0.01, **GRASP)


def test_reach_increases_below_old_saturation(monkeypatch):
  d = torch.tensor([0.08, 0.05, 0.04, 0.02, 0.01, 0.0])
  monkeypatch.setattr(task_state, "tcp_to_object_dist", lambda *a: d)
  r = rewards.reach(None, None, None, scale=5.0)
  torch.testing.assert_close(r, 1.0 - torch.tanh(5.0 * d))
  assert (r.diff() > 0).all()
  assert r[-1] == 1.0


@pytest.fixture
def hold_state(monkeypatch):
  """Set grasped / target distance / object speeds per env."""
  state = {}
  monkeypatch.setattr(task_state, "is_grasped", lambda env, *a: state["grasped"])
  monkeypatch.setattr(task_state, "object_to_target_dist", lambda env, *a: state["dist"])
  monkeypatch.setattr(task_state, "object_speeds", lambda env, cfg: (state["lin"], state["ang"]))

  def set_state(grasped, dist, lin, ang):
    n = len(grasped)
    state.update(
      grasped=torch.tensor(grasped),
      dist=torch.as_tensor(dist, dtype=torch.float).expand(n),
      lin=torch.as_tensor(lin, dtype=torch.float).expand(n),
      ang=torch.as_tensor(ang, dtype=torch.float).expand(n),
    )
    return rewards.stable_hold(None, **HOLD)

  return set_state


def test_stable_hold_requires_grasp_and_target_tolerance(hold_state):
  # Perfectly still in every case; only grasp / distance vary.
  r = hold_state([False, True, True, True, False], [0.0, 0.0, 0.01, 0.0101, 0.5], 0.0, 0.0)
  assert r.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
  # Not grasped or out of tolerance stays zero at any speed.
  r = hold_state([False, False], [0.0, 0.0], [0.0, 0.3], [0.0, 2.0])
  assert (r == 0).all()


def test_stable_hold_decreases_smoothly_with_speed(hold_state):
  lin = torch.linspace(0.0, 0.5, 101)
  r = hold_state([True] * 101, 0.005, lin, 0.0)
  torch.testing.assert_close(r, 1.0 - torch.tanh(5.0 * lin))
  assert (r.diff() < 0).all()
  assert r.diff().abs().max() < 0.03  # no jumps: no speed threshold
  ang = torch.linspace(0.0, 2.0, 101)
  r = hold_state([True] * 101, 0.005, 0.0, ang)
  torch.testing.assert_close(r, 1.0 - torch.tanh(2.0 * ang))
  assert (r.diff() < 0).all()
  assert r.diff().abs().max() < 0.05
  # Both speeds multiply.
  r = hold_state([True], 0.0, 0.1, 0.5)
  torch.testing.assert_close(
    r, torch.tensor([(1 - torch.tanh(torch.tensor(0.5))) * (1 - torch.tanh(torch.tensor(1.0)))])
  )


def _joint_env(joint_vel):
  data = SimpleNamespace(joint_vel=torch.tensor(joint_vel, dtype=torch.float))
  return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_joint_speed_penalty_zero_at_or_below_threshold():
  env = _joint_env([[0.0, 0.5, -0.5, 0.2], [-0.49, 0.1, 0.5, -0.3]])
  cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2, 3])
  assert (rewards.joint_speed_excess(env, cfg, 0.5) == 0).all()


def test_joint_speed_penalty_quadratic_per_joint_and_summed():
  env = _joint_env([[0.7, -0.9, 0.3], [1.5, 0.0, 0.0], [0.4, 0.4, 0.4]])
  cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2])
  r = rewards.joint_speed_excess(env, cfg, 0.5)
  torch.testing.assert_close(r, torch.tensor([0.2**2 + 0.4**2, 1.0**2, 0.0]))
  # Total speed (1.2) exceeds the threshold, but no single joint does.
  assert r[2] == 0.0
  # Quadratic: doubling the excess quadruples the penalty.
  r1 = rewards.joint_speed_excess(_joint_env([[0.6]]), SimpleNamespace(name="robot", joint_ids=[0]), 0.5)
  r2 = rewards.joint_speed_excess(_joint_env([[0.7]]), SimpleNamespace(name="robot", joint_ids=[0]), 0.5)
  torch.testing.assert_close(r2, 4 * r1)


def test_joint_speed_penalty_selects_joints_and_threshold():
  env = _joint_env([[2.0, 0.6, 0.06]])
  arm = SimpleNamespace(name="robot", joint_ids=[0, 1])
  gripper = SimpleNamespace(name="robot", joint_ids=[2])
  torch.testing.assert_close(rewards.joint_speed_excess(env, arm, 0.5), torch.tensor([1.5**2 + 0.1**2]))
  assert rewards.joint_speed_excess(env, gripper, 0.5) == 0.0
  torch.testing.assert_close(rewards.joint_speed_excess(env, gripper, 0.05), torch.tensor([0.01**2]))
