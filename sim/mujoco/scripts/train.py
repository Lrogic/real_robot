"""Train PPO (rsl_rl) on the grasp-lift-hold task.

  python scripts/train.py [--scene JSON] [--task JSON] [--action-space NAME]
                          [--num-envs N] [mjlab TrainConfig flags ...]

Any remaining flags go to mjlab's TrainConfig, e.g. ``--agent.max-iterations
2000``, ``--agent.algorithm.gamma 0.8 --agent.algorithm.lam 0.9`` (the old
ManiSkill run's values), ``--agent.logger wandb``, ``--video``.
Logs are written to ``logs/rsl_rl/<experiment_name>/`` under the current dir.
"""

from __future__ import annotations

import argparse
import sys

import mjlab
import tyro
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg
from tabletop_sim.scene import ACTION_SPACES, load_scene_config
from tabletop_sim.tasks.grasp import TASK_IDS, make_grasp_env_cfg
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene JSON")
  parser.add_argument("--task", default=str(DEFAULT_TASK), help="task JSON")
  parser.add_argument(
    "--action-space", choices=ACTION_SPACES, default=None, help="override scene's action space"
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  ours, rest = parser.parse_known_args()
  if "-h" in rest or "--help" in rest:
    parser.print_help()
    print("\nmjlab TrainConfig flags:\n")

  action_space = ours.action_space or load_scene_config(ours.scene).robot.action_space
  task_id = TASK_IDS[action_space]
  env_cfg = make_grasp_env_cfg(ours.scene, ours.task, action_space, ours.num_envs)
  args = tyro.cli(
    TrainConfig,
    args=rest,
    default=TrainConfig(env=env_cfg, agent=load_rl_cfg(task_id)),
    prog=f"{sys.argv[0]}",
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(task_id=task_id, args=args)


if __name__ == "__main__":
  main()
