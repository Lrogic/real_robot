from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tabletop_sim.tasks.grasp.mdp.commands import GraspTracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def task_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Grasped, at the lift target and static for ``stable_hold_duration_sec``."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GraspTracker)
  term.evaluate()
  return term.success.clone()
