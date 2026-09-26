from pathlib import Path

from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task
from tabletop_sim.scene import load_scene_config
from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg
from tabletop_sim.tasks.grasp.rl_cfg import grasp_ppo_runner_cfg

TASK_IDS = {
  "ee_delta_pose_binary_gripper": "WXAI-Grasp-Lift-EE",
  "delta_joint_abs_gripper": "WXAI-Grasp-Lift-Joint",
}
PLAY_TASK_ID = "WXAI-Grasp-Play"


def register_play_task(
  scene_path: str | Path, task_path: str | Path, action_space: str | None = None
) -> str:
  """Register an env built from arbitrary scene/task files for mjlab's play loop."""
  action_space = action_space or load_scene_config(scene_path).robot.action_space
  if PLAY_TASK_ID not in list_tasks():
    env_cfg = make_grasp_env_cfg(scene_path, task_path, action_space, num_envs=1)
    register_mjlab_task(
      task_id=PLAY_TASK_ID,
      env_cfg=env_cfg,
      play_env_cfg=env_cfg,
      rl_cfg=load_rl_cfg(TASK_IDS[action_space]),
    )
  return PLAY_TASK_ID

for _action_space, _task_id in TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=make_grasp_env_cfg(action_space=_action_space),
    play_env_cfg=make_grasp_env_cfg(action_space=_action_space),
    rl_cfg=grasp_ppo_runner_cfg(experiment_name=_task_id.lower().replace("-", "_")),
  )

__all__ = [
  "PLAY_TASK_ID",
  "TASK_IDS",
  "grasp_ppo_runner_cfg",
  "make_grasp_env_cfg",
  "register_play_task",
]
