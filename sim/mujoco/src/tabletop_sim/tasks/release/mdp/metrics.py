"""Release episode metrics (logged as ``Episode_Metrics/<name>``; see env_cfg
for the reduce of each)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.release.mdp import rewards, task_state, terminations

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def released_fraction(env: ManagerBasedRlEnv, **released_params) -> torch.Tensor:
  """Use with ``reduce="sum"``: released steps (``released`` > 0.5) / max episode length."""
  return (rewards.released(env, **released_params) > 0.5).float() / env.max_episode_length


def placed_fraction(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, pos_tol: float, rot_tol_deg: float
) -> torch.Tensor:
  """Use with ``reduce="sum"``: steps within the placement tolerances / max episode length."""
  return task_state.is_placed(env, command_name, object_cfg, pos_tol, rot_tol_deg).float() / env.max_episode_length


def position_error(env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
  return task_state.placement_error(env, command_name, object_cfg)[0]


def rotation_error_deg(env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
  return torch.rad2deg(task_state.placement_error(env, command_name, object_cfg)[1])


def collided(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """Use with ``reduce="max"``: 1 if a robot part touched the table or a relative object."""
  return (rewards.robot_collision(env, sensor_names) > 0).float()


def as_float(func):
  """A boolean term as a float metric (``reduce="max"``: did it ever happen)."""

  def metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    return func(env, **params).float()

  metric.__name__ = f"{func.__name__}_metric"
  return metric


object_dropped_metric = as_float(terminations.object_dropped)
support_toppled_metric = as_float(terminations.support_toppled)


def moving_release(env: ManagerBasedRlEnv, action_name: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
  """Use with ``reduce="max"``: 1 for an episode in which the gripper was opened
  while the object was clearly moving (``release_moving`` > 0.5)."""
  return (rewards.release_moving(env, action_name, object_cfg) > 0.5).float()
