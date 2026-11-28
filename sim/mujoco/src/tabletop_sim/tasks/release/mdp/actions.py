"""The release gripper action: grasp's ``GripperAction``, but every episode
starts with the gripper already commanded closed, as it is when release takes
over from transport on the real arm.

Without this, an episode's first command would count as fresh. With a timed
gripper (``goal_time_s`` > 0) that restarts the closing move from the measured
carriage position, so the squeeze on the held object would drop to nearly zero
for the first steps.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from tabletop_sim.tasks.grasp.mdp.actions import GripperAction, GripperActionCfg
from tabletop_sim.tasks.release.mdp.task_state import stillness_from_speeds

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HeldGripperActionCfg(GripperActionCfg):
  stillness_object: str | None = None
  """When set, record that object's stillness on every step whose command switches to
  open (``HeldGripperAction.release_stillness``)."""
  stillness_speed_scale: float = 11.0
  stillness_spin_scale: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> HeldGripperAction:
    if self.mode not in ("binary", "absolute", "delta"):
      raise ValueError(f"Unknown gripper mode {self.mode!r}")
    return HeldGripperAction(self, env)

  @classmethod
  def from_grasp(cls, cfg: GripperActionCfg, **extra) -> HeldGripperActionCfg:
    return cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}, **extra)


class HeldGripperAction(GripperAction):
  """Reset leaves the target at ``closed_pos`` with no move in progress and
  marks the command as already given, so a first "close" keeps squeezing and
  a first "open" counts as a switch.

  With ``stillness_object`` set, ``release_stillness`` holds, per env, the
  object's stillness on the last step whose command switched to open, measured
  when the command is processed (the state the policy acted on). It is 1 at the
  start of an episode and is overwritten by every later opening, so a regrasp
  and a calmer second release replace a hurried first one."""

  cfg: HeldGripperActionCfg

  def __init__(self, cfg: HeldGripperActionCfg, env):
    super().__init__(cfg, env)
    self.release_stillness = torch.ones(self.num_envs, device=self.device)
    self._stillness_object = env.scene[cfg.stillness_object] if cfg.stillness_object else None

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    if self._stillness_object is None:
      return
    middle = 0.5 * (self.cfg.open_pos + self.cfg.closed_pos)
    toward_open = (self._target[:, 0] - middle) * (self.cfg.open_pos - self.cfg.closed_pos) > 0
    opened = self.switched & toward_open
    if opened.any():
      data = self._stillness_object.data
      lin = torch.linalg.norm(data.root_link_lin_vel_w, dim=-1)
      ang = torch.linalg.norm(data.root_link_ang_vel_w, dim=-1)
      s = stillness_from_speeds(lin, ang, self.cfg.stillness_speed_scale, self.cfg.stillness_spin_scale)
      self.release_stillness[:] = torch.where(opened, s, self.release_stillness)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids] = self.cfg.closed_pos
    self._move_start[ids] = self.cfg.closed_pos
    self._move_substeps[ids] = torch.iinfo(torch.long).max // 2
    self._fresh[ids] = False
    if hasattr(self, "release_stillness"):
      self.release_stillness[ids] = 1.0
