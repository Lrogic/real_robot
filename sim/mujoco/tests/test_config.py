import json
import math
import struct
import zlib
from pathlib import Path

import mujoco
import pytest
import torch

from mjlab.envs import mdp as mjlab_mdp
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.scene import ConfigError, load_scene_config, load_task_config, object_spec
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


def _write_png(path: Path) -> None:
  def chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

  ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
  path.write_bytes(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", ihdr)
    + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    + chunk(b"IEND", b"")
  )


def _mesh_scene(tmp_path: Path, **obj_extra) -> dict:
  mesh = tmp_path / "block.obj"
  mesh.write_text(
    "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\n"
    "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
  )
  obj = {
    "name": "block",
    "primitive": "mesh",
    "dynamic": True,
    "asset_path": "block.obj",
    "color": [0.2, 0.4, 0.6, 1.0],
    "initial_poses": [[0, 0, 0, 1, 0, 0, 0]],
  }
  obj.update(obj_extra)
  return {
    "scene_name": "mesh",
    "manipulated_object": "block",
    "objects": [obj],
  }


def test_texture_path_is_optional(tmp_path):
  scene = load_scene_config(_write(tmp_path, _mesh_scene(tmp_path)))
  assert scene.manipulated.texture_path is None
  model = object_spec(scene.manipulated).compile()
  visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_visual")
  assert model.ntex == 0 and model.geom_matid[visual] == -1
  assert model.geom_rgba[visual].tolist() == pytest.approx([0.2, 0.4, 0.6, 1.0])


def test_texture_path_overrides_color(tmp_path):
  _write_png(tmp_path / "block.png")
  scene = load_scene_config(_write(tmp_path, _mesh_scene(tmp_path, texture_path="block.png")))
  assert scene.manipulated.texture_path == (tmp_path / "block.png").resolve()
  model = object_spec(scene.manipulated).compile()
  visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_visual")
  assert model.ntex == 1 and model.geom_matid[visual] >= 0
  assert model.mat_rgba[model.geom_matid[visual]].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_collision_size_uses_box(tmp_path):
  scene = load_scene_config(
    _write(tmp_path, _mesh_scene(tmp_path, collision_size=[0.075, 0.015, 0.025]))
  )
  assert scene.manipulated.collision_size == (0.075, 0.015, 0.025)
  model = object_spec(scene.manipulated).compile()
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_collision")
  assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
  assert model.geom_size[gid].tolist() == pytest.approx([0.0375, 0.0075, 0.0125])


def test_missing_texture_rejected(tmp_path):
  with pytest.raises(ConfigError, match="texture file not found"):
    load_scene_config(_write(tmp_path, _mesh_scene(tmp_path, texture_path="missing.png")))


def test_texture_path_rejected_for_primitives(tmp_path, scene_dict):
  scene_dict["objects"][1]["texture_path"] = "cube.png"
  with pytest.raises(ConfigError, match="only valid for mesh"):
    load_scene_config(_write(tmp_path, scene_dict))


def test_multiple_initial_poses(tmp_path, scene_dict):
  poses = [[-0.25, y, 0.015, 1, 0, 0, 0] for y in (-0.05, 0.0, 0.05)]
  scene_dict["objects"][1]["initial_poses"] = poses
  scene = load_scene_config(_write(tmp_path, scene_dict))
  assert len(scene.manipulated.initial_poses) == 3


CURRICULUM_TASK = DEFAULT_TASK.parent / "grasp_curriculum_wxai.json"


def test_curriculum_disabled_by_default():
  task = load_task_config(DEFAULT_TASK)
  assert not task.curriculum.enabled
  assert make_grasp_env_cfg(task_path=DEFAULT_TASK).curriculum == {}


def test_curriculum_config_loads():
  c = load_task_config(CURRICULUM_TASK).curriculum
  assert c.enabled and c.metric_name == "grasped_lifted_frac"
  assert (c.initial_difficulty, c.max_difficulty, c.difficulty_increment) == (0.0, 1.0, 0.05)
  assert c.performance_threshold == 0.70
  assert c.min_completed_episodes == 5000 and isinstance(c.min_completed_episodes, int)


def test_curriculum_term_registered_when_enabled():
  cfg = make_grasp_env_cfg(task_path=CURRICULUM_TASK)
  assert list(cfg.curriculum) == [mdp.curriculum.TERM_NAME]
  term = cfg.curriculum[mdp.curriculum.TERM_NAME]
  assert term.func is mdp.curriculum.GraspDifficultyCurriculum
  assert term.params["metric_name"] == "grasped_lifted_frac"
  assert term.params["min_completed_episodes"] == 5000
  assert cfg.metrics["grasped_lifted_frac"].reduce == "sum"


@pytest.mark.parametrize(
  ("override", "match"),
  [
    ({"friction": [0.5, 1.5]}, "unknown keys"),
    ({"enabled": 1}, "must be a bool"),
    ({"metric_name": ""}, "non-empty"),
    ({"initial_difficulty": -0.1}, r"in \[0, 1\]"),
    ({"max_difficulty": 1.5}, r"in \[0, 1\]"),
    ({"performance_threshold": 1.01}, r"in \[0, 1\]"),
    ({"initial_difficulty": 0.6, "max_difficulty": 0.5}, "must not exceed"),
    ({"difficulty_increment": 0.0}, "must be positive"),
    ({"difficulty_increment": -0.05}, "must be positive"),
    ({"min_completed_episodes": 0}, "integer >= 1"),
    ({"min_completed_episodes": 10.5}, "integer >= 1"),
    ({"min_completed_episodes": True}, "integer >= 1"),
    ({"initial_difficulty": "0.1"}, "must be a number"),
  ],
)
def test_curriculum_invalid_rejected(tmp_path, override, match):
  data = json.loads(CURRICULUM_TASK.read_text())
  data["curriculum"].update(override)
  with pytest.raises(ConfigError, match=match):
    load_task_config(_write(tmp_path, data))


def test_curriculum_not_an_object_rejected(tmp_path, task_dict):
  task_dict["curriculum"] = [1, 2]
  with pytest.raises(ConfigError, match="must be an object"):
    load_task_config(_write(tmp_path, task_dict))


@pytest.mark.parametrize(
  ("metric", "match"), [("no_such_metric", "is not a metric"), ("max_lift", "reduce='sum'")]
)
def test_curriculum_invalid_metric_rejected(tmp_path, metric, match):
  data = json.loads(CURRICULUM_TASK.read_text())
  data["curriculum"]["metric_name"] = metric
  path = _write(tmp_path, data)
  load_task_config(path)
  with pytest.raises(ConfigError, match=match):
    make_grasp_env_cfg(task_path=path)


def test_domain_randomization_object_reset_radius(tmp_path, task_dict):
  task_dict["domain_randomization"] = {"object_reset_radius": 0.08}
  task = load_task_config(_write(tmp_path, task_dict))
  assert task.domain_randomization.object_reset_radius == 0.08


def test_domain_randomization_unknown_key_rejected(tmp_path, task_dict):
  task_dict["domain_randomization"] = {"friction": [0.5, 1.5]}
  with pytest.raises(ConfigError, match="unknown keys"):
    load_task_config(_write(tmp_path, task_dict))


def test_domain_randomization_negative_radius_rejected(tmp_path, task_dict):
  task_dict["domain_randomization"] = {"object_reset_radius": -0.01}
  with pytest.raises(ConfigError, match="object_reset_radius"):
    load_task_config(_write(tmp_path, task_dict))


def test_domain_randomization_yaw_range_deg(tmp_path, task_dict):
  task_dict["domain_randomization"] = {"object_reset_yaw_range_deg": 15.0}
  task = load_task_config(_write(tmp_path, task_dict))
  assert task.domain_randomization.object_reset_yaw_range_deg == 15.0


def test_domain_randomization_negative_yaw_range_rejected(tmp_path, task_dict):
  task_dict["domain_randomization"] = {"object_reset_yaw_range_deg": -5.0}
  with pytest.raises(ConfigError, match="object_reset_yaw_range_deg"):
    load_task_config(_write(tmp_path, task_dict))


def test_stiffness_damping_dr_defaults(tmp_path, task_dict):
  task_dict["domain_randomization"] = {}
  dr = load_task_config(_write(tmp_path, task_dict)).domain_randomization
  assert dr.stiffness_damping_enabled is False
  assert dr.stiffness_scale_range == (0.8, 1.2) and dr.damping_scale_range == (0.8, 1.2)
  assert not load_task_config(DEFAULT_TASK).domain_randomization.stiffness_damping_enabled
  assert load_task_config(CURRICULUM_TASK).domain_randomization.stiffness_damping_enabled


def test_stiffness_damping_dr_ranges_load(tmp_path, task_dict):
  task_dict["domain_randomization"].update(
    stiffness_damping_enabled=True, stiffness_scale_range=[0.5, 1], damping_scale_range=[1.1, 1.3]
  )
  dr = load_task_config(_write(tmp_path, task_dict)).domain_randomization
  assert dr.stiffness_damping_enabled is True
  assert dr.stiffness_scale_range == (0.5, 1.0) and dr.damping_scale_range == (1.1, 1.3)


@pytest.mark.parametrize(
  ("override", "match"),
  [
    ({"stiffness_damping_enabled": 1}, "must be a bool"),
    ({"stiffness_scale_range": [1.2, 0.8]}, "lower <= upper"),
    ({"damping_scale_range": [0.0, 1.2]}, "0 < lower"),
    ({"damping_scale_range": [-0.1, 1.2]}, "0 < lower"),
    ({"stiffness_scale_range": [0.8]}, "list of 2"),
    ({"stiffness_scale_range": 0.8}, "list of 2"),
    ({"damping_scale_range": [0.8, "1.2"]}, "list of 2"),
    ({"damping_scale_range": [True, 1.2]}, "list of 2"),
  ],
)
def test_stiffness_damping_dr_invalid_rejected(tmp_path, task_dict, override, match):
  task_dict["domain_randomization"].update(override)
  with pytest.raises(ConfigError, match=match):
    load_task_config(_write(tmp_path, task_dict))


def test_stiffness_damping_event_absent_when_disabled():
  cfg = make_grasp_env_cfg(task_path=DEFAULT_TASK)
  assert "randomize_stiffness_damping" not in cfg.events
  assert list(cfg.events) == ["reset_scene_to_default", "reset_cube_xy"]


@pytest.mark.parametrize(
  ("embodiment", "func", "kp", "damping"),
  [
    (
      "menagerie_wxai",
      mdp.RandomizeKpJointDamping,
      (*(f"joint_{i}" for i in range(6)), "right_carriage_joint"),
      (*(f"joint_{i}" for i in range(6)), "left_carriage_joint", "right_carriage_joint"),
    ),
    (
      "trossen_wxai_base",
      mdp.RandomizeKpActuatorKv,
      (*(f"joint_{i}" for i in range(6)), "left_gripper"),
      (*(f"joint_{i}" for i in range(6)), "left_gripper"),
    ),
  ],
)
def test_stiffness_damping_event_per_embodiment(tmp_path, scene_dict, embodiment, func, kp, damping):
  scene_dict["robot"]["embodiment"] = embodiment
  cfg = make_grasp_env_cfg(_write(tmp_path, scene_dict), CURRICULUM_TASK)
  # Reset events run in insertion order; DR comes after the state resets.
  assert list(cfg.events)[-1] == "randomize_stiffness_damping"
  term = cfg.events["randomize_stiffness_damping"]
  assert term.func is func and term.mode == "reset"
  assert term.params == {
    "entity_name": "robot",
    "stiffness_actuators": kp,
    "damping_targets": damping,
    "stiffness_scale_range": (0.8, 1.2),
    "damping_scale_range": (0.8, 1.2),
  }


def test_difficulty_scale_bounds_interpolate():
  d = torch.tensor([0.0, 0.25, 0.5, 1.0])
  lower, upper = mdp.events.difficulty_scale_bounds(d, (0.8, 1.2))
  torch.testing.assert_close(lower, torch.tensor([1.0, 0.95, 0.9, 0.8]))
  torch.testing.assert_close(upper, torch.tensor([1.0, 1.05, 1.1, 1.2]))
  lower, upper = mdp.events.difficulty_scale_bounds(d, (1.1, 1.5))
  torch.testing.assert_close(lower, torch.tensor([1.0, 1.025, 1.05, 1.1]))
  torch.testing.assert_close(upper, torch.tensor([1.0, 1.125, 1.25, 1.5]))


def test_sample_difficulty_scales_respect_bounds():
  torch.manual_seed(0)
  d = torch.tensor([0.0, 0.5, 1.0]).repeat_interleave(2000)
  s = mdp.events.sample_difficulty_scales(d, (0.8, 1.2), 7)
  assert s.shape == (6000, 7)
  assert (s[:2000] == 1.0).all()
  for block, lo, hi in ((s[2000:4000], 0.9, 1.1), (s[4000:], 0.8, 1.2)):
    assert block.min() >= lo and block.max() <= hi
    assert block.min() < lo + 0.01 and block.max() > hi - 0.01
  # Independent per target and per env.
  assert torch.unique(s[4000:]).numel() > 0.99 * s[4000:].numel()


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
  assert set(cfg.metrics) == {
    "mean_hold_score",
    "grasped_frac",
    "max_lift",
    "grasped_lifted_frac",
  }
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
