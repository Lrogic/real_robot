"""Episode metrics (logged as ``Episode_Metrics/<name>`` by mjlab's metrics
manager, which evaluates them after rewards, on the pre-reset state)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def success(env: ManagerBasedRlEnv, termination_name: str) -> torch.Tensor:
  """Use with ``reduce="max"``: 1 if the episode ended in success."""
  return env.termination_manager.get_term(termination_name).float()


def grasped_fraction(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]],
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  """Use with ``reduce="sum"``: grasped steps / max episode length."""
  grasped = task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  )
  return grasped.float() / env.max_episode_length


def lift_height(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Use with ``reduce="max"``: height gained over the initial pose, floored at 0."""
  obj_pos, _ = task_state.object_pose_w(env, object_cfg)
  initial = task_state.goal(env, command_name).initial_object_pos
  return (obj_pos[:, 2] - initial[:, 2]).clamp(min=0.0)
