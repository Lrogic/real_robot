"""Roll out a policy (trained checkpoint, zero or random) in the viewer.

  python scripts/play.py --checkpoint-file logs/rsl_rl/.../model_1000.pt
  python scripts/play.py --agent random --num-envs 4
  python scripts/play.py --scene my_scene.json --task my_task.json --agent zero
  python scripts/play.py --skill release --checkpoint-file logs/rsl_rl/.../model_N.pt

Remaining flags go to mjlab's PlayConfig (``--viewer viser``, ``--video`` ...).
"""

from __future__ import annotations

import argparse
import sys

import mjlab
import tyro
from mjlab.scripts.play import PlayConfig, run_play
from tabletop_sim.scene import ACTION_SPACES
from tabletop_sim.tasks.grasp import register_play_task
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK
from tabletop_sim.tasks.release import env_cfg as release_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--skill", choices=("grasp", "release"), default="grasp")
  parser.add_argument("--scene", default=None, help="scene JSON (default: the skill's)")
  parser.add_argument("--task", default=None, help="task JSON (default: the skill's)")
  parser.add_argument("--action-space", choices=ACTION_SPACES, default=None)
  ours, rest = parser.parse_known_args()
  if "-h" in rest or "--help" in rest:
    parser.print_help()
    print("\nmjlab PlayConfig flags:\n")

  release = ours.skill == "release"
  scene = ours.scene or str(release_env_cfg.DEFAULT_SCENE if release else DEFAULT_SCENE)
  task = ours.task or str(release_env_cfg.DEFAULT_TASK if release else DEFAULT_TASK)
  if release:
    from tabletop_sim.tasks.release import register_release_task

    task_id = register_release_task(scene, task, ours.action_space)
  else:
    task_id = register_play_task(scene, task, ours.action_space)
  args = tyro.cli(PlayConfig, args=rest, prog=sys.argv[0], config=mjlab.TYRO_FLAGS)
  run_play(task_id, args)


if __name__ == "__main__":
  main()
