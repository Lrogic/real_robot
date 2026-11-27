"""Release termination terms: the object dropped, or a relative object toppled."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state as grasp_state
from tabletop_sim.tasks.release.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

AxisPair = tuple[tuple[float, float, float], tuple[float, float, float]]


def object_dropped(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  drop_below_goal_m: float,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: AxisPair,
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  """Not grasped, and the object's centre is ``drop_below_goal_m`` below the goal's."""
  pos, _ = grasp_state.object_pose_w(env, object_cfg)
  low = pos[:, 2] < task_state.goal(env, command_name).goal_pos[:, 2] - drop_below_goal_m
  grasped = grasp_state.is_grasped(env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg)
  return low & ~grasped


def support_toppled(
  env: ManagerBasedRlEnv,
  command_name: str,
  support_names: tuple[str, ...],
  tilt_deg: float,
  displacement_m: float,
) -> torch.Tensor:
  """Some relative object turned more than ``tilt_deg`` or moved more than
  ``displacement_m`` from its reset pose."""
  d, angle = task_state.support_displacement(env, command_name, support_names)
  return ((angle > math.radians(tilt_deg)) | (d > displacement_m)).any(dim=-1)
