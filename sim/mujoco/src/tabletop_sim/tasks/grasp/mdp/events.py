"""Reset events. Domain randomization events will be added alongside these."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import resolve_env_ids

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def reset_object_from_pose_list(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str,
  poses: tuple[tuple[float, ...], ...],
) -> None:
  """Place an object at a pose drawn uniformly from ``poses`` (world frame,
  relative to each env origin), with zero velocity."""
  env_ids = resolve_env_ids(env, env_ids)
  entity: Entity = env.scene[entity_name]
  table = torch.tensor(poses, dtype=torch.float, device=env.device)
  choice = torch.randint(0, table.shape[0], (len(env_ids),), device=env.device)
  pose = table[choice].clone()
  pose[:, :3] += env.scene.env_origins[env_ids]

  if entity.is_fixed_base:
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  else:
    state = torch.zeros(len(env_ids), 13, device=env.device)
    state[:, :7] = pose
    entity.write_root_state_to_sim(state, env_ids=env_ids)


def reset_object_xy_in_disk(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str,
  radius: float,
) -> None:
  """Place an object uniformly in a disk of ``radius`` (m) around its default
  xy. z, orientation, and velocity stay at that default pose."""
  env_ids = resolve_env_ids(env, env_ids)
  entity: Entity = env.scene[entity_name]
  state = entity.data.default_root_state[env_ids].clone()
  state[:, 0:3] += env.scene.env_origins[env_ids]
  n = len(env_ids)
  angle = torch.rand(n, device=env.device) * (2.0 * math.pi)
  # Square-root keeps the sample uniform over the disk area.
  radial = radius * torch.sqrt(torch.rand(n, device=env.device))
  state[:, 0] += radial * torch.cos(angle)
  state[:, 1] += radial * torch.sin(angle)
  entity.write_root_state_to_sim(state, env_ids=env_ids)
