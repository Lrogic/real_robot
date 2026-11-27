"""Release reset events: restore a start state from the dataset, and randomize
the friction of the object-support contact pairs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import curriculum, object_size
from tabletop_sim.tasks.grasp.mdp.events import difficulty_scale_bounds
from tabletop_sim.tasks.release.starts import ReleaseStarts

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_ROWS_ATTR = "_release_start_rows"


def current_rows(env: ManagerBasedRlEnv) -> tuple[ReleaseStarts, torch.Tensor]:
  """The loaded dataset and each env's restored row (set by :class:`RestoreReleaseStart`)."""
  value = getattr(env, _ROWS_ATTR, None)
  if value is None:
    raise RuntimeError("no RestoreReleaseStart event in this env")
  return value


class RestoreReleaseStart(ManagerTermBase):
  """Write a random dataset row, recorded at the env's size variant, into the
  robot (every joint's position and velocity), the manipulated object and
  each relative object. Runs after ``reset_scene_to_default``; the goal
  command reads the chosen row through :func:`current_rows`."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    self.starts = ReleaseStarts.load(Path(p["dataset"]), env.device)
    robot: Entity = env.scene["robot"]
    if tuple(robot.joint_names) != self.starts.joint_names:
      raise ValueError(
        f"dataset joints {self.starts.joint_names} differ from the robot's {tuple(robot.joint_names)}"
      )
    if tuple(p["support_names"]) != self.starts.support_names:
      raise ValueError(
        f"dataset relative objects {self.starts.support_names} differ from {tuple(p['support_names'])}"
      )
    scales = object_size.assigned_scale(env, p["object_name"])
    self._first, self._count = self.starts.rows_for_scales(scales)
    self.rows = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, _ROWS_ATTR, (self.starts, self.rows))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    dataset: str,
    object_name: str,
    support_names: tuple[str, ...],
  ) -> None:
    del dataset
    env_ids = resolve_env_ids(env, env_ids).to(torch.long)
    n = len(env_ids)
    pick = (torch.rand(n, device=env.device) * self._count[env_ids]).long()
    rows = self._first[env_ids] + pick.clamp(max=self._count[env_ids] - 1)
    self.rows[env_ids] = rows
    t = self.starts.tensors
    origins = env.scene.env_origins[env_ids]

    robot: Entity = env.scene["robot"]
    robot.write_joint_state_to_sim(t["robot_joint_pos"][rows], t["robot_joint_vel"][rows], env_ids=env_ids)

    state = t["object_root_state"][rows].clone()
    state[:, :3] += origins
    env.scene[object_name].write_root_state_to_sim(state, env_ids=env_ids)
    for k, name in enumerate(support_names):
      state = t["support_root_state"][rows, k].clone()
      state[:, :3] += origins
      env.scene[name].write_root_state_to_sim(state, env_ids=env_ids)


@requires_model_fields("pair_friction")
class RandomizeSupportPairFriction(ManagerTermBase):
  """Curriculum-scaled friction of the object-support contact pairs, drawn the
  way grasp's ``RandomizePairFriction`` draws the gripper and table pairs: per
  curriculum stage, one log-uniform sliding / torsional / rolling multiplier
  per env, shared by every pair of the group, on the compiled values."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    asset_cfg: SceneEntityCfg = cfg.params["support_cfg"]
    entity: Entity = env.scene[asset_cfg.name]
    if len(entity.indexing.pair_ids[asset_cfg.pair_ids]) == 0:
      raise ValueError(f"no object-support contact pairs match {asset_cfg.pair_names}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    support_cfg: SceneEntityCfg,
    sliding_scale_range: tuple[float, float],
    torsional_scale_range: tuple[float, float],
    rolling_scale_range: tuple[float, float],
  ) -> None:
    env_ids = resolve_env_ids(env, env_ids).to(torch.long)
    state = curriculum.get_curriculum_state(env)
    stages = state.episode_stage[env_ids]
    axis_ranges = {0: sliding_scale_range, 2: torsional_scale_range, 3: rolling_scale_range}
    for stage in torch.unique(stages).tolist():
      ids = env_ids[stages == stage]
      d = state.episode_difficulty[ids[0]]
      ranges = {
        axis: tuple(b.item() for b in difficulty_scale_bounds(d, r)) for axis, r in axis_ranges.items()
      }
      dr.pair_friction(
        env,
        ids,
        ranges=ranges,  # type: ignore[arg-type]
        asset_cfg=support_cfg,
        distribution="log_uniform",
        operation="scale",
        axes=[0, 2, 3],
        shared_random=True,
        isotropic=True,
      )
