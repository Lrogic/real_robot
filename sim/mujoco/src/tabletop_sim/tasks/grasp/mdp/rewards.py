"""Reward terms (unweighted, stateless). Weights, including the
1/reward_normalization factor and the sign of penalties, are applied through
``RewardTermCfg.weight``. Grasp parameters (``sensor_names`` ...
``max_angle_deg``) are forwarded to :func:`task_state.is_grasped`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

AxisPair = tuple[tuple[float, float, float], tuple[float, float, float]]


def reach(
  env: ManagerBasedRlEnv,
  tcp_cfg: SceneEntityCfg,
  object_cfg: SceneEntityCfg,
  scale: float,
) -> torch.Tensor:
  d = task_state.tcp_to_object_dist(env, tcp_cfg, object_cfg)
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
  """In [0, 1]: grasped, within ``lift_threshold`` of the target, and still.
  Also the per-step value of the ``mean_hold_score`` metric."""
  grasped = task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  )
  within_target_tolerance = task_state.is_lifted(env, command_name, object_cfg, lift_threshold)
  lin, ang = task_state.object_speeds(env, object_cfg)
  still = (1.0 - torch.tanh(5.0 * lin)) * (1.0 - torch.tanh(2.0 * ang))
  return grasped.float() * within_target_tolerance.float() * still


def joint_speed_excess(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, speed_threshold: float
) -> torch.Tensor:
  """Sum over the selected joints of (|qvel| - threshold)^2, counting only the
  part of each joint's speed above the threshold. Use with a nonpositive weight."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - speed_threshold).clamp_min(0.0)
  return (excess**2).sum(dim=-1)
