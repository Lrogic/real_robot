"""Train PPO (rsl_rl) on the grasp-lift-hold task.

  python scripts/train.py [--scene JSON] [--task JSON] [--action-space NAME]
                          [--num-envs N] [mjlab TrainConfig flags ...]

Any remaining flags go to mjlab's TrainConfig, e.g. ``--agent.max-iterations
2000``, ``--agent.algorithm.gamma 0.8 --agent.algorithm.lam 0.9`` (the old
ManiSkill run's values), ``--agent.logger wandb``, ``--video``.
Logs are written to ``logs/rsl_rl/<experiment_name>/`` under the current dir.
Next to mjlab's ``params/env.yaml`` and ``agent.yaml``, each run folder gets
``params/scene.json`` and ``params/task.json`` (copies of the files as they
were at launch) and ``params/source.json`` (their paths, the action space, the
command line and the repo commit).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mjlab
import mjlab.scripts.train as mjlab_train
import tyro
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg
from tabletop_sim.scene import ACTION_SPACES, load_scene_config
from tabletop_sim.tasks.grasp import TASK_IDS, make_grasp_env_cfg
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK


def _git(*args: str) -> str | None:
  try:
    out = subprocess.run(
      ("git", *args), cwd=Path(__file__).parent, capture_output=True, text=True, check=True
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout.strip()


def _save_sources_with_run(scene: str, task: str, action_space: str) -> None:
  """Make every run write its scene and task JSONs into ``<run>/params``.

  mjlab names the run folder inside ``launch_training``, so this wraps the
  ``run_train`` it calls. The files are read here, before the env is built.
  """
  scene_path, task_path = Path(scene).resolve(), Path(task).resolve()
  status = _git("status", "--porcelain")
  files = {
    "scene.json": scene_path.read_bytes(),
    "task.json": task_path.read_bytes(),
    "source.json": json.dumps(
      {
        "scene": str(scene_path),
        "task": str(task_path),
        "action_space": action_space,
        "argv": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
      },
      indent=2,
    ).encode(),
  }
  run_train = mjlab_train.run_train

  def run_train_with_sources(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
      (params / name).write_bytes(content)
    run_train(task_id, cfg, log_dir)

  mjlab_train.run_train = run_train_with_sources


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
  _save_sources_with_run(ours.scene, ours.task, action_space)
  launch_training(task_id=task_id, args=args)


if __name__ == "__main__":
  main()
