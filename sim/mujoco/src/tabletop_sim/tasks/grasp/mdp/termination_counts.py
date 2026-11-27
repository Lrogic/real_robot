"""Running totals of episode terminations, per termination term, since the env
was built.

mjlab logs ``Episode_Termination/<term>`` as the number of envs that term ended
*in one control step*, and rsl_rl averages that over the steps of an iteration,
so a rare event (one NaN reset in an iteration of 32 steps) shows up as
0.03. This curriculum term (curriculum terms run on every reset, before the
termination flags are cleared, and mjlab logs what they return) keeps counts
that only grow: ``Curriculum/termination_counts/<term>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

TERM_NAME = "termination_counts"


class TerminationCounts(ManagerTermBase):
  """Cumulative count of the resetting envs each termination term ended."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    del cfg
    self._totals: dict[str, int] = {}

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> dict[str, int]:
    manager = env.termination_manager
    ids = slice(None) if env_ids is None else env_ids
    for name in manager.active_terms:
      done = manager.get_term(name)[ids]
      self._totals[name] = self._totals.get(name, 0) + int(torch.count_nonzero(done))
    return dict(self._totals)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """No-op: the totals must survive every reset."""
    del env_ids
