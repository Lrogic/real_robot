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

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HeldGripperActionCfg(GripperActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> HeldGripperAction:
    if self.mode not in ("binary", "absolute", "delta"):
      raise ValueError(f"Unknown gripper mode {self.mode!r}")
    return HeldGripperAction(self, env)

  @classmethod
  def from_grasp(cls, cfg: GripperActionCfg) -> HeldGripperActionCfg:
    return cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init})


class HeldGripperAction(GripperAction):
  """Reset leaves the target at ``closed_pos`` with no move in progress and
  marks the command as already given, so a first "close" keeps squeezing and
  a first "open" counts as a switch."""

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids] = self.cfg.closed_pos
    self._move_start[ids] = self.cfg.closed_pos
    self._move_substeps[ids] = torch.iinfo(torch.long).max // 2
    self._fresh[ids] = False
