"""rsl_rl PPO runner config for the release task: grasp's settings, its own experiment name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from tabletop_sim.tasks.grasp.rl_cfg import grasp_ppo_runner_cfg


def release_ppo_runner_cfg(experiment_name: str = "wxai_release") -> RslRlOnPolicyRunnerCfg:
  return grasp_ppo_runner_cfg(experiment_name=experiment_name)
