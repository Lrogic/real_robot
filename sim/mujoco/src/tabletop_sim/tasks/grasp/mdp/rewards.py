"""Reward terms (unweighted). Weights, including the 1/reward_normalization
factor, are applied through ``RewardTermCfg.weight``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tabletop_sim.tasks.grasp.mdp.commands import GraspTracker, quaternion_angle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _tracker(env: ManagerBasedRlEnv, command_name: str) -> GraspTracker:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GraspTracker)
  term.evaluate()
  return term


def reach(
  env: ManagerBasedRlEnv, command_name: str, scale: float, saturation_dist: float
) -> torch.Tensor:
  """1 - tanh(scale * d), with d floored at saturation_dist so the reward stops
  pulling the TCP into the object once it is close enough to grasp."""
  d = _tracker(env, command_name).tcp_to_obj_dist.clamp(min=saturation_dist)
  return 1.0 - torch.tanh(scale * d)


def grasp(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _tracker(env, command_name).is_grasped.float()


def lift_target(env: ManagerBasedRlEnv, command_name: str, scale: float) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.is_grasped.float() * (1.0 - torch.tanh(scale * t.obj_to_target_dist))


def stable_hold(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  static = (1.0 - torch.tanh(5.0 * t.obj_lin_speed)) * (1.0 - torch.tanh(2.0 * t.obj_ang_speed))
  return t.is_grasped.float() * t.is_lifted.float() * static


def success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _tracker(env, command_name).success.float()


def xy_displacement_penalty(
  env: ManagerBasedRlEnv, command_name: str, tolerance: float
) -> torch.Tensor:
  return (_tracker(env, command_name).xy_displacement - tolerance).clamp(min=0.0)


def orientation_penalty(
  env: ManagerBasedRlEnv, command_name: str, tolerance: float
) -> torch.Tensor:
  t = _tracker(env, command_name)
  angle = quaternion_angle(t.initial_object_quat, t.object_quat_w)
  return (angle - tolerance).clamp(min=0.0)
