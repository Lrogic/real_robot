"""Lift goal: where the object started this episode and where it should go."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class GraspGoalCommandCfg(CommandTermCfg):
  object_name: str
  target_lift_height: float
  # The goal is fixed per episode; it is only re-sampled on reset.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> GraspGoalCommand:
    return GraspGoalCommand(self, env)


class GraspGoalCommand(CommandTerm):
  """Per env: the object's initial world position and the
  lift target ``initial_pos + (0, 0, target_lift_height)``.

  Resets only flag an env: the object is placed by reset events before derived
  poses are refreshed, so the pose is captured in the next ``_update_command``,
  which runs after the forward pass and before observations. mjlab calls it
  with the reset ids from ``env.reset(env_ids)`` but with ``None`` after an
  auto-reset inside ``step()``, hence the flag rather than trusting ``env_ids``.
  """

  cfg: GraspGoalCommandCfg

  def __init__(self, cfg: GraspGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.object_name]
    n, dev = self.num_envs, self.device
    self.initial_object_pos = torch.zeros(n, 3, device=dev)
    self.target_pos = torch.zeros(n, 3, device=dev)
    self._needs_capture = torch.ones(n, dtype=torch.bool, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._needs_capture[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pending = self._needs_capture.clone()
    if env_ids is not None:
      scope = torch.zeros_like(pending)
      scope[env_ids] = True
      pending &= scope
    ids = pending.nonzero(as_tuple=False).squeeze(-1)
    if len(ids) == 0:
      return
    pos = self.object.data.root_link_pos_w[ids]
    self.initial_object_pos[ids] = pos
    self.target_pos[ids] = pos
    self.target_pos[ids, 2] += self.cfg.target_lift_height
    self._needs_capture[ids] = False

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for i in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[i].cpu().numpy(),
        radius=0.008,
        color=(0.1, 0.9, 0.2, 0.6),
        label=f"lift_target_{i}",
      )
