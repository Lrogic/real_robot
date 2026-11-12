"""Task files for the tests, derived from the one checked-in task (configs/tasks/grasp_wxai.json).

The checked-in task has every option; the options that model the real loop are off.
Each variant here is that file with a few values changed, written to a temp folder
once per test session.
"""

from __future__ import annotations

import copy
import json
import tempfile
from functools import cache
from pathlib import Path

from tabletop_sim.tasks.grasp.env_cfg import CONFIGS_DIR, DEFAULT_TASK

CUBE_SCENE = CONFIGS_DIR / "scenes" / "tabletop_cube_wxai.json"
JENGA_SCENE = CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json"
MASTER_TASK = DEFAULT_TASK

_LATENCY = {
  "w_table_contact": -1.0,
  "domain_randomization.object_pose_delay_s": 0.01,
  "domain_randomization.action_hold_probability": 0.1,
}
_NO_NOISE_OR_FRICTION = {
  "lift_threshold": 0.01,
  "domain_randomization.friction.enabled": False,
  "domain_randomization.object_pose_noise.persistent_bias.enabled": False,
  "domain_randomization.object_pose_noise.per_step.enabled": False,
}
_NO_VARIANTS = {"domain_randomization.object_size_variants.enabled": False}
_NO_CURRICULUM = {
  "curriculum.enabled": False,
  "domain_randomization.stiffness_damping_enabled": False,
}

# name -> dotted key -> value, applied to the checked-in task.
VARIANTS: dict[str, dict] = {
  # No curriculum, no randomization beyond the reset pose.
  "standard": {**_NO_NOISE_OR_FRICTION, **_NO_VARIANTS, **_NO_CURRICULUM},
  "curriculum": {**_NO_NOISE_OR_FRICTION, **_NO_VARIANTS},
  "size_variants": {**_NO_NOISE_OR_FRICTION, **_NO_CURRICULUM},
  "curriculum_size_variants": {**_NO_NOISE_OR_FRICTION},
  # The checked-in task: curriculum, size variants, contact friction, pose noise.
  "friction": {},
  # Pose delay, held actions and the table-contact penalty.
  "latency": {**_LATENCY},
  "latency_interp": {**_LATENCY, "action.ee_goal_interpolation": True},
  "latency_interp_gripper": {
    **_LATENCY,
    "action.ee_goal_interpolation": True,
    "action.gripper_goal_time_s": 0.5,
    "w_gripper_switch": -0.2,
  },
  # Command delay and joint friction randomization, no goal interpolation.
  "latency_gripper_joints": {
    **_LATENCY,
    "action.gripper_goal_time_s": 0.5,
    "w_gripper_switch": -0.2,
    "action.command_delay_s": 0.006,
    "domain_randomization.joint_friction.enabled": True,
  },
}


def task_dict(name: str = "friction") -> dict:
  """The variant as a JSON dict (a fresh copy)."""
  data = copy.deepcopy(json.loads(MASTER_TASK.read_text()))
  for dotted, value in VARIANTS[name].items():
    node = data
    *parents, leaf = dotted.split(".")
    for key in parents:
      node = node[key]
    assert leaf in node, dotted
    node[leaf] = value
  return data


@cache
def _folder() -> Path:
  return Path(tempfile.mkdtemp(prefix="tabletop_sim_tasks_"))


@cache
def task_file(name: str) -> Path:
  """Path of the variant's task JSON."""
  path = _folder() / f"{name}.json"
  path.write_text(json.dumps(task_dict(name)))
  return path
