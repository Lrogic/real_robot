"""Release task: place a grasped object at a goal defined relative to other
scene objects, open the gripper and back away.

Unlike grasp, nothing is registered at import: the env needs a generated
start-state dataset (``scripts/generate_release_starts.py``), so the scripts
register it from explicit scene and task files with :func:`register_release_task`.
"""

from pathlib import Path

from mjlab.tasks.registry import list_tasks, register_mjlab_task
from tabletop_sim.scene import load_scene_config
from tabletop_sim.tasks.release.env_cfg import DEFAULT_SCENE, DEFAULT_TASK, make_release_env_cfg
from tabletop_sim.tasks.release.rl_cfg import release_ppo_runner_cfg

TASK_IDS = {"ee_delta_pose_binary_gripper": "WXAI-Release-EE"}


def register_release_task(
  scene_path: str | Path = DEFAULT_SCENE,
  task_path: str | Path = DEFAULT_TASK,
  action_space: str | None = None,
  num_envs: int = 1,
) -> str:
  """Register (once per process) the release env built from these files."""
  action_space = action_space or load_scene_config(scene_path).robot.action_space
  if action_space not in TASK_IDS:
    raise ValueError(f"release supports {sorted(TASK_IDS)}, got {action_space!r}")
  task_id = TASK_IDS[action_space]
  if task_id not in list_tasks():
    env_cfg = make_release_env_cfg(scene_path, task_path, action_space, num_envs)
    register_mjlab_task(
      task_id=task_id,
      env_cfg=env_cfg,
      play_env_cfg=env_cfg,
      rl_cfg=release_ppo_runner_cfg(experiment_name="wxai_release"),
    )
  return task_id


__all__ = ["DEFAULT_SCENE", "DEFAULT_TASK", "TASK_IDS", "make_release_env_cfg", "register_release_task"]
