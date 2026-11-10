"""Control steps on which the policy's action is discarded and the previous
targets stay, as when the real loop skips a policy tick or withholds a command.

The action terms read the mask (see ``actions.py``); this term only draws it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from tabletop_sim.tasks.grasp.mdp import curriculum

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Uniform = Callable[[int], torch.Tensor]


@dataclass(kw_only=True)
class ActionHoldCfg(CommandTermCfg):
  probability: float
  """Per-step hold probability at episode difficulty 1; p = d * probability."""
  # Never resampled by the timer; resets flag envs instead.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ActionHold:
    return ActionHold(self, env)


class ActionHold(CommandTerm):
  """Per env: whether the next control step holds the previous targets.

  A command term for the same reason as :class:`PerceivedObjectPose`: its
  update runs once per control step, after resets and before the next action.
  Each update draws the mask of the envs in scope independently of earlier
  steps. An env's first step after a reset is never held, because its action
  terms have no previous target yet.
  """

  cfg: ActionHoldCfg

  def __init__(self, cfg: ActionHoldCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.hold = torch.zeros(n, dtype=torch.bool, device=dev)
    self._first_step = torch.ones(n, dtype=torch.bool, device=dev)
    self.uniform: Uniform = lambda count: torch.rand(count, device=dev)
    """Source of U[0, 1) draws, one ``(len(ids),)`` call per update; tests may
    replace it."""

  @property
  def command(self) -> torch.Tensor:
    return self.hold.float().unsqueeze(-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._first_step[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    p = self.cfg.probability * curriculum.episode_difficulty(self._env, ids)
    self.hold[ids] = (self.uniform(len(ids)) < p) & ~self._first_step[ids]
    self._first_step[ids] = False

  def _update_metrics(self) -> None:
    pass
