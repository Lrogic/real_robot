"""Difficulty-curriculum bookkeeping (CPU, no simulation)."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from tabletop_sim.tasks.grasp.mdp import curriculum
from tabletop_sim.tasks.grasp.mdp.curriculum import DifficultyCurriculum


def _cur(num_envs: int, **kwargs) -> DifficultyCurriculum:
  c = DifficultyCurriculum(num_envs, "cpu", **kwargs)
  _init(c)
  return c


def _init(c: DifficultyCurriculum) -> None:
  """The initialization reset: every env, nothing stepped yet."""
  n = c.episode_stage.numel()
  c.update(torch.arange(n), torch.zeros(n), torch.zeros(n, dtype=torch.long))


def _end(c: DifficultyCurriculum, ids, scores, length: int = 300) -> None:
  ids = torch.as_tensor(ids)
  scores = torch.as_tensor(scores, dtype=torch.float).expand(len(ids)).clone()
  c.update(ids, scores, torch.full((len(ids),), length, dtype=torch.long))


def test_initialization_reset_contributes_nothing():
  c = DifficultyCurriculum(4, "cpu", min_completed_episodes=1)
  # Steps taken before any reset assigned a stage are not a real episode.
  c.update(torch.arange(4), torch.ones(4), torch.full((4,), 10))
  assert c.window_count == 0 and c.window_sum == 0.0 and c.last_eval_count == 0
  assert c.episode_started.all()
  # Started but empty (e.g. a repeated env.reset()) also does not count.
  c.update(torch.arange(4), torch.ones(4), torch.zeros(4, dtype=torch.long))
  assert c.window_count == 0 and c.last_eval_count == 0 and c.stage == 0


def test_individual_scores_accumulate():
  c = _cur(4)
  c.update(torch.tensor([0, 2, 3]), torch.tensor([0.1, 0.5, 0.9]), torch.full((3,), 300))
  assert c.window_count == 3
  assert c.window_sum == pytest.approx(1.5, abs=1e-6)
  _end(c, [1], 0.25)
  assert c.window_count == 4
  assert c.window_sum == pytest.approx(1.75, abs=1e-6)


def test_crossing_batch_included_whole():
  c = _cur(6000)  # default min_completed_episodes = 5000
  _end(c, torch.arange(4990), 0.0)
  assert c.window_count == 4990 and c.last_eval_count == 0
  # 4990 + 20 crosses 5000: all 20 count, and only one evaluation happens.
  _end(c, torch.arange(4990, 5010), 1.0)
  assert c.last_eval_count == 5010
  assert c.last_eval_mean == pytest.approx(20 / 5010)
  assert c.window_count == 0 and c.window_sum == 0.0
  assert c.stage == 0


def test_promotion_is_strictly_greater_than_threshold():
  c = _cur(2, performance_threshold=0.5, min_completed_episodes=2)
  _end(c, [0, 1], 0.5)
  assert c.last_eval_mean == 0.5 and c.stage == 0 and c.difficulty == 0.0
  _end(c, [0, 1], 0.5001)
  assert c.stage == 1 and c.promotions == 1
  assert c.difficulty == pytest.approx(0.05)


def test_default_threshold_boundary():
  c = _cur(10, min_completed_episodes=10)
  _end(c, torch.arange(10), 210 / 300)
  assert c.stage == 0
  _end(c, torch.arange(10), 211 / 300)
  assert c.stage == 1


def test_window_cleared_after_pass_and_fail():
  c = _cur(2, min_completed_episodes=2)
  _end(c, [0, 1], 0.1)
  assert (c.stage, c.window_count, c.window_sum, c.last_eval_count) == (0, 0, 0.0, 2)
  _end(c, [0, 1], 0.9)
  assert (c.stage, c.window_count, c.window_sum, c.last_eval_count) == (1, 0, 0.0, 2)
  assert c.last_eval_mean == pytest.approx(0.9)


def test_failed_evaluation_keeps_same_stage_episodes_eligible():
  c = _cur(4, min_completed_episodes=2)
  _end(c, [0, 1], 0.0)  # fails and clears the window
  assert c.stage == 0 and c.last_eval_count == 2
  _end(c, [2, 3], 1.0)  # started at stage 0, still the current stage
  assert c.last_eval_count == 2 and c.last_eval_mean == pytest.approx(1.0)
  assert c.stage == 1


def test_promotion_excludes_old_stage_and_assigns_new_difficulty():
  c = _cur(4, min_completed_episodes=2)
  _end(c, [0, 1], 1.0)
  assert c.stage == 1
  # The batch that promoted starts its next episodes at the new difficulty.
  assert c.episode_stage.tolist() == [1, 1, 0, 0]
  assert c.episode_difficulty.tolist() == pytest.approx([0.05, 0.05, 0.0, 0.0])
  # Ongoing stage-0 episodes keep 0 and do not count toward stage 1...
  _end(c, [2, 3], 1.0)
  assert c.window_count == 0 and c.stage == 1
  # ...but their next episodes get the promoted difficulty.
  assert c.episode_stage.tolist() == [1, 1, 1, 1]
  assert c.episode_difficulty.tolist() == pytest.approx([0.05] * 4)
  _end(c, [0], 0.3)
  assert c.window_count == 1 and c.window_sum == pytest.approx(0.3)


@pytest.mark.parametrize("increment", [0.05, 0.1, 0.3, 0.07])
def test_difficulty_caps_exactly_at_max(increment):
  c = _cur(1, difficulty_increment=increment, min_completed_episodes=1)
  for _ in range(100):
    _end(c, [0], 1.0)
  assert c.difficulty == 1.0
  assert c.episode_difficulty.item() == 1.0
  assert c.stage == c.promotions == math.ceil(1.0 / increment - 1e-9)
  # Further evaluations still run and clear, but never promote past max.
  assert c.last_eval_count == 1 and c.window_count == 0


def test_never_promotes_when_initial_equals_max():
  c = _cur(1, initial_difficulty=0.4, max_difficulty=0.4, min_completed_episodes=1)
  _end(c, [0], 1.0)
  assert c.stage == 0 and c.difficulty == 0.4 and c.last_eval_count == 1


def test_disabled_reports_zero():
  c = _cur(3, enabled=False, initial_difficulty=0.5, min_completed_episodes=1)
  for _ in range(3):
    _end(c, [0, 1, 2], 1.0)
  assert c.difficulty == 0.0 and c.stage == 0
  assert (c.episode_difficulty == 0).all()
  assert c.window_count == 0 and c.last_eval_count == 0


def test_log_values():
  c = _cur(2, min_completed_episodes=2)
  _end(c, [0, 1], 1.0)
  assert c.log_values() == {
    "difficulty": pytest.approx(0.05),
    "stage": 1.0,
    "window_count": 0.0,
    "last_eval_mean": pytest.approx(1.0),
    "last_eval_count": 2.0,
    "promotions": 1.0,
  }


def _fake_env(curriculum_terms: dict, num_envs: int = 3):
  return SimpleNamespace(
    cfg=SimpleNamespace(curriculum=curriculum_terms), num_envs=num_envs, device="cpu"
  )


def test_accessor_disabled_without_term():
  env = _fake_env({})
  state = curriculum.get_curriculum_state(env)
  assert not state.enabled
  assert curriculum.get_curriculum_state(env) is state
  d = curriculum.episode_difficulty(env)
  assert d is state.episode_difficulty and (d == 0).all() and d.shape == (3,)
  assert curriculum.episode_difficulty(env, torch.tensor([2])).shape == (1,)


def test_accessor_builds_from_term_params():
  term = CurriculumTermCfg(
    func=curriculum.GraspDifficultyCurriculum,
    params={
      "metric_name": "grasped_lifted_frac",
      "initial_difficulty": 0.2,
      "max_difficulty": 0.8,
      "difficulty_increment": 0.1,
      "performance_threshold": 0.6,
      "min_completed_episodes": 7,
    },
  )
  env = _fake_env({curriculum.TERM_NAME: term})
  state = curriculum.get_curriculum_state(env)
  assert state.enabled and state.min_completed_episodes == 7
  assert state.difficulty == pytest.approx(0.2)
  state.episode_difficulty[1] = 0.5
  torch.testing.assert_close(
    curriculum.episode_difficulty(env, torch.tensor([0, 1])), torch.tensor([0.2, 0.5])
  )
