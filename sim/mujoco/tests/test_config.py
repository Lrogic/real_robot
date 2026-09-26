import copy
import json
from pathlib import Path

import pytest

from tabletop_sim.scene import ConfigError, load_scene_config, load_task_config
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK


def _write(tmp_path: Path, data: dict, name: str = "cfg.json") -> Path:
  path = tmp_path / name
  path.write_text(json.dumps(data))
  return path


@pytest.fixture
def scene_dict() -> dict:
  return json.loads(DEFAULT_SCENE.read_text())


@pytest.fixture
def task_dict() -> dict:
  return json.loads(DEFAULT_TASK.read_text())


def test_default_configs_load():
  scene = load_scene_config(DEFAULT_SCENE)
  task = load_task_config(DEFAULT_TASK)
  assert scene.manipulated.name == "cube"
  assert scene.robot.embodiment == "menagerie_wxai"
  assert [o.name for o in scene.objects] == ["table", "cube"]
  assert not scene.objects[0].dynamic
  assert task.timing.decimation == 25
  assert task.timing.step_dt == pytest.approx(0.05)
  assert task.reward_normalization == 10.0


def test_trajectory_key_rejected(tmp_path, scene_dict):
  scene_dict["objects"][1]["trajectory"] = [[0, 0, 0, 1, 0, 0, 0]]
  with pytest.raises(ConfigError, match="initial_poses"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_unknown_key_rejected(tmp_path, scene_dict):
  scene_dict["objects"][1]["colour"] = [1, 0, 0, 1]
  with pytest.raises(ConfigError, match="unknown keys"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_non_unit_quaternion_rejected(tmp_path, scene_dict):
  scene_dict["objects"][1]["initial_poses"] = [[0, 0, 0, 2, 0, 0, 0]]
  with pytest.raises(ConfigError, match="unit length"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_static_manipulated_object_rejected(tmp_path, scene_dict):
  scene_dict["manipulated_object"] = "table"
  with pytest.raises(ConfigError, match="dynamic"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_bad_action_space_rejected(tmp_path, scene_dict):
  scene_dict["robot"]["action_space"] = "joint_velocity"
  with pytest.raises(ConfigError, match="action_space"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_multiple_initial_poses(tmp_path, scene_dict):
  poses = [[-0.25, y, 0.015, 1, 0, 0, 0] for y in (-0.05, 0.0, 0.05)]
  scene_dict["objects"][1]["initial_poses"] = poses
  scene = load_scene_config(_write(tmp_path, scene_dict))
  assert len(scene.manipulated.initial_poses) == 3


def test_reserved_dr_and_curriculum_rejected(tmp_path, task_dict):
  for key in ("domain_randomization", "curriculum"):
    data = copy.deepcopy(task_dict)
    data[key] = {"friction": [0.5, 1.5]}
    with pytest.raises(ConfigError, match="not implemented"):
      load_task_config(_write(tmp_path, data))


def test_incompatible_control_rate_rejected(tmp_path, task_dict):
  task_dict["timing"]["control_hz"] = 30.0
  with pytest.raises(ConfigError, match="control_hz"):
    load_task_config(_write(tmp_path, task_dict))


def test_task_overrides(tmp_path, task_dict):
  task_dict["w_xy"] = 2.0
  task_dict["use_minimal_obs"] = True
  task = load_task_config(_write(tmp_path, task_dict))
  assert task.w_xy == 2.0
  assert task.use_minimal_obs is True
