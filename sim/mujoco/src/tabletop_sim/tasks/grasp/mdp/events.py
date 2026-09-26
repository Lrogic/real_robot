"""Reset events. Domain randomization events will be added alongside these."""

from __future__ import annotations

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
