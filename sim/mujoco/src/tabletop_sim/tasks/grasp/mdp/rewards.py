"""Reward terms (unweighted). Weights, including the 1/reward_normalization
factor, are applied through ``RewardTermCfg.weight``. Grasp parameters
(``sensor_names`` ... ``max_angle_deg``) are forwarded to
:func:`task_state.is_grasped`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

AxisPair = tuple[tuple[float, float, float], tuple[float, float, float]]


def reach(
  env: ManagerBasedRlEnv,
  tcp_cfg: SceneEntityCfg,
  object_cfg: SceneEntityCfg,
  scale: float,
  saturation_dist: float,
) -> torch.Tensor:
  """1 - tanh(scale * d), with d floored at saturation_dist so the reward stops
  pulling the TCP into the object once it is close enough to grasp."""
  d = task_state.tcp_to_object_dist(env, tcp_cfg, object_cfg).clamp(min=saturation_dist)
  return 1.0 - torch.tanh(scale * d)


def grasp(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: AxisPair,
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  return task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  ).float()


def lift_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  scale: float,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: AxisPair,
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  grasped = task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  )
  dist = task_state.object_to_target_dist(env, command_name, object_cfg)
  return grasped.float() * (1.0 - torch.tanh(scale * dist))


def stable_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  lift_threshold: float,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: AxisPair,
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  grasped = task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  )
  lifted = task_state.is_lifted(env, command_name, object_cfg, lift_threshold)
  lin, ang = task_state.object_speeds(env, object_cfg)
  still = (1.0 - torch.tanh(5.0 * lin)) * (1.0 - torch.tanh(2.0 * ang))
  return grasped.float() * lifted.float() * still


def success(env: ManagerBasedRlEnv, termination_name: str) -> torch.Tensor:
  """This step's result of the success termination (computed before rewards)."""
  return env.termination_manager.get_term(termination_name).float()


def xy_displacement_penalty(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, tolerance: float
) -> torch.Tensor:
  disp = task_state.object_xy_displacement(env, command_name, object_cfg)
  return (disp - tolerance).clamp(min=0.0)


def orientation_penalty(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, tolerance: float
) -> torch.Tensor:
  angle = task_state.object_rotation_from_initial(env, command_name, object_cfg)
  return (angle - tolerance).clamp(min=0.0)
