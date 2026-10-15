import copy
import json
import math
from pathlib import Path

import pytest

from mjlab.envs import mdp as mjlab_mdp
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.scene import ConfigError, load_scene_config, load_task_config
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import (
  DEFAULT_SCENE,
  DEFAULT_TASK,
  make_grasp_env_cfg,
  make_grasp_env_cfg_from_configs,
)


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
  task_dict["w_hold"] = 2.0
  task_dict["include_is_grasped_obs"] = False
  task = load_task_config(_write(tmp_path, task_dict))
  assert task.w_hold == 2.0
  assert task.include_is_grasped_obs is False


def test_reward_terms_and_weights():
  cfg = make_grasp_env_cfg()
  assert cfg.scale_rewards_by_dt is False
  assert cfg.decimation == 25
  weights = {name: term.weight for name, term in cfg.rewards.items()}
  assert weights == pytest.approx({
    "reach": 0.05,
    "grasp": 0.1,
    "lift_target": 0.1,
    "stable_hold": 0.1,
    "arm_speed": -0.01,
    "gripper_speed": 0.0,
    "joint_pos_limits": -1.0,
  })
  assert set(cfg.rewards["reach"].params) == {"tcp_cfg", "object_cfg", "scale"}
  assert cfg.rewards["reach"].params["scale"] == 5.0
  assert cfg.rewards["stable_hold"].params["lift_threshold"] == 0.01
  emb = get_embodiment(load_scene_config(DEFAULT_SCENE).robot.embodiment)
  arm, gripper = cfg.rewards["arm_speed"].params, cfg.rewards["gripper_speed"].params
  assert arm["asset_cfg"].joint_names == emb.arm_joint_names
  assert gripper["asset_cfg"].joint_names == emb.gripper_joint_names
  assert arm["speed_threshold"] == 0.5 and gripper["speed_threshold"] == 0.5
  limits = cfg.rewards["joint_pos_limits"]
  assert limits.func is mjlab_mdp.joint_pos_limits
  assert limits.params["asset_cfg"].joint_names == emb.arm_joint_names
  assert 0.0 < cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor < 1.0


def test_joint_pos_limit_weight_scales_with_normalization(tmp_path, task_dict):
  task_dict.update(w_joint_pos_limits=-4.0, reward_normalization=8.0)
  task = load_task_config(_write(tmp_path, task_dict))
  cfg = make_grasp_env_cfg_from_configs(load_scene_config(DEFAULT_SCENE), task)
  assert cfg.rewards["joint_pos_limits"].weight == pytest.approx(-0.5)


def test_no_success_reward_counter_or_termination():
  cfg = make_grasp_env_cfg()
  assert "success" not in cfg.rewards
  assert list(cfg.terminations) == ["time_out"]
  assert cfg.terminations["time_out"].time_out
  assert set(cfg.metrics) == {"mean_hold_score", "grasped_frac", "max_lift"}
  assert not hasattr(mdp, "StableHoldSuccess")
  assert not hasattr(mdp, "terminations")
  assert not hasattr(mdp.rewards, "success") and not hasattr(mdp.metrics, "success")
  assert not hasattr(mdp.task_state, "is_static")


def test_mean_hold_score_reuses_stable_hold_reward():
  cfg = make_grasp_env_cfg()
  metric, reward = cfg.metrics["mean_hold_score"], cfg.rewards["stable_hold"]
  assert metric.func is mdp.rewards.stable_hold is reward.func
  assert metric.reduce == "mean" and not metric.per_substep
  assert metric.params.keys() == reward.params.keys()
  for key, value in metric.params.items():
    if key in ("object_cfg", "finger_cfg"):
      assert value is not reward.params[key]  # fresh cfgs, resolved independently
      assert value.name == reward.params[key].name
    else:
      assert value == reward.params[key]


@pytest.mark.parametrize(
  "key",
  [
    "w_success",
    "reach_saturation_dist",
    "stable_hold_duration_sec",
    "stable_linear_vel_tol",
    "stable_angular_vel_tol",
    "arm_joint_vel_tol",
    "gripper_joint_vel_tol",
  ],
)
def test_removed_keys_rejected(tmp_path, task_dict, key):
  task_dict[key] = 1.0
  with pytest.raises(ConfigError, match="unknown keys"):
    load_task_config(_write(tmp_path, task_dict))


def test_speed_thresholds_independent(tmp_path, task_dict):
  task_dict.update(arm_speed_threshold=1.25, gripper_speed_threshold=0.02, w_gripper_speed=-0.3)
  task = load_task_config(_write(tmp_path, task_dict))
  cfg = make_grasp_env_cfg_from_configs(load_scene_config(DEFAULT_SCENE), task)
  assert cfg.rewards["arm_speed"].params["speed_threshold"] == 1.25
  assert cfg.rewards["gripper_speed"].params["speed_threshold"] == 0.02
  assert cfg.rewards["arm_speed"].weight == pytest.approx(-0.01)
  assert cfg.rewards["gripper_speed"].weight == pytest.approx(-0.03)


@pytest.mark.parametrize("key", ["arm_speed_threshold", "gripper_speed_threshold"])
@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_bad_speed_thresholds_rejected(tmp_path, task_dict, key, value):
  task_dict[key] = value
  with pytest.raises(ConfigError, match="finite and nonnegative"):
    load_task_config(_write(tmp_path, task_dict))


@pytest.mark.parametrize("key", ["w_arm_speed", "w_gripper_speed", "w_joint_pos_limits"])
@pytest.mark.parametrize("value", [0.1, -math.inf, math.nan])
def test_bad_penalty_weights_rejected(tmp_path, task_dict, key, value):
  task_dict[key] = value
  with pytest.raises(ConfigError, match="finite and nonpositive"):
    load_task_config(_write(tmp_path, task_dict))
