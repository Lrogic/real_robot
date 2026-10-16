"""Performance-based difficulty curriculum.

One :class:`DifficultyCurriculum` per env holds the global difficulty/stage, the
evaluation window, and each env's *episode* difficulty, captured when its
episode starts and kept until it ends. Randomization, observation-noise and
reward terms read it through :func:`episode_difficulty`::

  from tabletop_sim.tasks.grasp.mdp import curriculum
  d = curriculum.episode_difficulty(env, env_ids)  # (len(env_ids),) in [0, 1]

With the curriculum disabled every episode difficulty is 0.

mjlab's ``_reset_idx`` runs the curriculum manager before ``sim.reset``, the
reset events and ``metrics_manager.reset``. So at each reset batch the term
reads the completed episodes' accumulated metric, evaluates the window, and
writes the new episodes' difficulty before any reset-time randomization runs.

Curriculum state is not saved in training checkpoints: a resumed run restarts
at ``initial_difficulty``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsManager

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

TERM_NAME = "grasp_difficulty"
"""Curriculum term key; logs appear as ``Curriculum/grasp_difficulty/<key>``."""
_STATE_ATTR = "_grasp_difficulty_curriculum"
_SNAP_EPS = 1e-9


class DifficultyCurriculum:
  """Global difficulty/stage, evaluation window and per-env episode records.

  Stage IDs start at 0 and increase by one per promotion; difficulty is a pure
  function of the stage, so repeated increments cannot drift. A completed
  episode is eligible only if it actually started (not the initialization
  reset), took at least one step, and started at the current stage.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    *,
    enabled: bool = True,
    initial_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
    difficulty_increment: float = 0.05,
    performance_threshold: float = 0.70,
    min_completed_episodes: int = 5000,
  ):
    self.enabled = enabled
    self.initial_difficulty = initial_difficulty if enabled else 0.0
    self.max_difficulty = max_difficulty if enabled else 0.0
    self.difficulty_increment = difficulty_increment
    self.performance_threshold = performance_threshold
    self.min_completed_episodes = min_completed_episodes

    self.stage = 0
    self.window_sum = 0.0
    self.window_count = 0
    self.last_eval_mean = 0.0
    self.last_eval_count = 0
    self.promotions = 0

    self.episode_stage = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.episode_difficulty = torch.full(
      (num_envs,), self.difficulty, dtype=torch.float, device=device
    )
    self.episode_started = torch.zeros(num_envs, dtype=torch.bool, device=device)

  @property
  def difficulty(self) -> float:
    """Current global difficulty, assigned to episodes that start now."""
    d = self.initial_difficulty + self.stage * self.difficulty_increment
    if d >= self.max_difficulty - _SNAP_EPS:
      return self.max_difficulty
    return d

  def update(
    self, env_ids: torch.Tensor, scores: torch.Tensor, episode_lengths: torch.Tensor
  ) -> None:
    """Account for the episodes ending at ``env_ids`` and start their next ones.

    ``scores`` and ``episode_lengths`` are the completed episodes' values, read
    before anything is reset.
    """
    if self.enabled and len(env_ids) > 0:
      eligible = (
        self.episode_started[env_ids]
        & (episode_lengths > 0)
        & (self.episode_stage[env_ids] == self.stage)
      )
      count, total = torch.stack(
        (eligible.sum().double(), scores.double()[eligible].sum())
      ).tolist()
      self.window_count += int(count)
      self.window_sum += total
      if self.window_count >= self.min_completed_episodes:
        self._evaluate()
    self.episode_stage[env_ids] = self.stage
    self.episode_difficulty[env_ids] = self.difficulty
    self.episode_started[env_ids] = True

  def _evaluate(self) -> None:
    mean = self.window_sum / self.window_count
    self.last_eval_mean = mean
    self.last_eval_count = self.window_count
    if mean > self.performance_threshold and self.difficulty < self.max_difficulty:
      self.stage += 1
      self.promotions += 1
    self.window_sum = 0.0
    self.window_count = 0

  def log_values(self) -> dict[str, float]:
    return {
      "difficulty": float(self.difficulty),
      "stage": float(self.stage),
      "window_count": float(self.window_count),
      "last_eval_mean": float(self.last_eval_mean),
      "last_eval_count": float(self.last_eval_count),
      "promotions": float(self.promotions),
    }


def get_curriculum_state(env: ManagerBasedRlEnv) -> DifficultyCurriculum:
  """The env's single curriculum state, built on first use from the registered
  ``grasp_difficulty`` term's params (disabled if there is none). Independent of
  manager construction order."""
  state = getattr(env, _STATE_ATTR, None)
  if state is None:
    term_cfg = env.cfg.curriculum.get(TERM_NAME)
    if term_cfg is None:
      state = DifficultyCurriculum(env.num_envs, env.device, enabled=False)
    else:
      params = {k: v for k, v in term_cfg.params.items() if k != "metric_name"}
      state = DifficultyCurriculum(env.num_envs, env.device, **params)
    setattr(env, _STATE_ATTR, state)
  return state


def episode_difficulty(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None = None
) -> torch.Tensor:
  """Difficulty each env's current episode started with.

  With ``env_ids=None`` this is the shared ``(num_envs,)`` tensor itself, so it
  must not be modified; otherwise a copy for ``env_ids``. Zero when disabled.
  """
  d = get_curriculum_state(env).episode_difficulty
  return d if env_ids is None else d[env_ids]


def _completed_episode_scores(
  env: ManagerBasedRlEnv, metric_name: str, env_ids: torch.Tensor
) -> torch.Tensor:
  """Per-env accumulated value of a ``reduce="sum"`` metric for the episodes
  ending at ``env_ids``.

  mjlab 1.6 has no public per-env accessor, so this is the one place that reads
  ``MetricsManager._episode_sums``/``_term_cfgs``. It is only valid before
  ``metrics_manager.reset`` clears them, i.e. from the curriculum term.
  """
  manager = getattr(env, "metrics_manager", None)
  if not isinstance(manager, MetricsManager):
    raise ValueError(
      f"curriculum metric {metric_name!r} needs a metrics manager; env has none"
    )
  if metric_name not in manager.active_terms:
    raise ValueError(
      f"curriculum metric {metric_name!r} is not a registered metric; "
      f"available: {manager.active_terms}"
    )
  reduce = manager._term_cfgs[manager.active_terms.index(metric_name)].reduce
  if reduce != "sum":
    raise ValueError(f"curriculum metric {metric_name!r} must use reduce='sum', got {reduce!r}")
  return manager._episode_sums[metric_name][env_ids].clone()


class GraspDifficultyCurriculum(ManagerTermBase):
  """Curriculum term driving :class:`DifficultyCurriculum` from completed-episode
  scores. Returns scalar logs."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._metric_name: str = cfg.params["metric_name"]
    self._state = get_curriculum_state(env)

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None, **params
  ) -> dict[str, float]:
    del params  # Already folded into the state.
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    scores = _completed_episode_scores(env, self._metric_name, env_ids)
    self._state.update(env_ids, scores, env.episode_length_buf[env_ids])
    return self._state.log_values()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """No-op: the curriculum manager calls this every reset, and the global
    state and evaluation window must survive it."""
    del env_ids
