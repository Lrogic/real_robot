"""Object pose delay, held actions and the robot-table contact penalty."""

import json
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from tabletop_sim.scene import ConfigError, load_scene_config, load_task_config
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import (
  ACTION_HOLD_NAME,
  CONFIGS_DIR,
  DEFAULT_SCENE,
  DEFAULT_TASK,
  PERCEPTION_NAME,
  POSE_HISTORY_SENSOR,
  TABLE_CONTACT_SENSOR,
  make_grasp_env_cfg,
  make_grasp_env_cfg_from_configs,
)
from tabletop_sim.tasks.grasp.mdp import curriculum, task_state
from tabletop_sim.tasks.grasp.mdp.perception import perturb_pose

CURRICULUM_TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_wxai.json"
LATENCY_TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_size_variants_friction_latency_wxai.json"
DEVICE = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DELAY_S, DELAY_SUBSTEPS = 0.01, 5
OBJECT_POSE = slice(24, 31)  # actor and critic columns


def _write(path: Path, data: dict) -> Path:
  path.write_text(json.dumps(data))
  return path


def _task(tmp_path: Path, base: Path = DEFAULT_TASK, dr: dict | None = None, **top) -> Path:
  data = json.loads(base.read_text())
  data.update(top)
  data.setdefault("domain_randomization", {}).update(dr or {})
  if "curriculum" in data:
    # No promotions of its own; tests set the stage directly.
    data["curriculum"]["min_completed_episodes"] = 10**9
  return _write(tmp_path / "task.json", data)


def _scene(tmp_path: Path, support: dict | None = None) -> Path:
  data = json.loads(DEFAULT_SCENE.read_text())
  if support is not None:
    data["support_surface"] = support
  return _write(tmp_path / "scene.json", data)


def _make_env(scene, task, num_envs=4) -> ManagerBasedRlEnv:
  cfg = make_grasp_env_cfg(scene, task, None, num_envs)
  cfg.seed = 0
  return ManagerBasedRlEnv(cfg, device=DEVICE)


def _set_difficulty(env, d):
  """Difficulty the next resetting episodes start with."""
  state = curriculum.get_curriculum_state(env)
  state.stage = round(d / state.difficulty_increment)
  assert state.difficulty == pytest.approx(d)


def _action(env, **dims):
  """Zero action with the named components set: x, y, z, grip."""
  a = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)
  for name, value in dims.items():
    a[:, {"x": 0, "y": 1, "z": 2, "grip": 6}[name]] = value
  return a


# Config.


def test_defaults_add_nothing():
  task = load_task_config(DEFAULT_TASK)
  dr = task.domain_randomization
  assert dr.object_pose_delay_s == 0.0 and dr.action_hold_probability == 0.0
  assert task.w_table_contact == 0.0 and task.object_pose_delay_substeps == 0
  cfg = make_grasp_env_cfg()
  assert {s.name for s in cfg.scene.sensors}.isdisjoint({POSE_HISTORY_SENSOR, TABLE_CONTACT_SENSOR})
  assert ACTION_HOLD_NAME not in cfg.commands and "table_contact" not in cfg.rewards
  assert {"table_contact_frac", "table_collision"}.isdisjoint(cfg.metrics)
  assert cfg.commands[PERCEPTION_NAME].history_sensor_name is None
  assert all(a.hold_command_name is None for a in cfg.actions.values())


def test_latency_task_wiring():
  task = load_task_config(LATENCY_TASK)
  assert task.object_pose_delay_substeps == DELAY_SUBSTEPS
  scene = load_scene_config(CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json")
  if not scene.manipulated.asset_path.exists():
    pytest.skip("jenga mesh missing")
  cfg = make_grasp_env_cfg_from_configs(scene, task)
  sensors = {s.name: s for s in cfg.scene.sensors}
  assert sensors[POSE_HISTORY_SENSOR].delay_substeps == DELAY_SUBSTEPS
  assert sensors[POSE_HISTORY_SENSOR].object_name == "jenga"
  assert cfg.commands[PERCEPTION_NAME].history_sensor_name == POSE_HISTORY_SENSOR
  assert cfg.commands[ACTION_HOLD_NAME].probability == pytest.approx(0.1)
  assert list(cfg.actions) == ["arm", "gripper"]
  assert all(a.hold_command_name == ACTION_HOLD_NAME for a in cfg.actions.values())
  contact = sensors[TABLE_CONTACT_SENSOR]
  # Everything below the first arm joint; the fixed base is not in the subtree.
  assert (contact.primary.mode, contact.primary.pattern) == ("subtree", "link_1")
  assert (contact.secondary.entity, contact.secondary.pattern) == ("table", "table")
  assert contact.history_length == task.timing.decimation
  reward = cfg.rewards["table_contact"]
  assert reward.func is mdp.rewards.table_contact
  assert reward.weight == pytest.approx(task.w_table_contact / task.reward_normalization)
  assert reward.weight < 0
  frac, collision = cfg.metrics["table_contact_frac"], cfg.metrics["table_collision"]
  assert (frac.func, frac.reduce) == (mdp.rewards.table_contact, "mean")
  assert (collision.func, collision.reduce) == (mdp.metrics.table_touched, "max")
  assert frac.params == collision.params == reward.params


@pytest.mark.parametrize(
  ("dr", "top", "match"),
  [
    ({"object_pose_delay_s": -0.002}, {}, "object_pose_delay_s must be finite and nonnegative"),
    ({"object_pose_delay_s": 0.003}, {}, "multiple of timing.sim_dt"),
    ({"action_hold_probability": -0.1}, {}, r"action_hold_probability must be in \[0, 1\]"),
    ({"action_hold_probability": 1.1}, {}, r"action_hold_probability must be in \[0, 1\]"),
    ({}, {"w_table_contact": 0.5}, "finite and nonpositive"),
  ],
)
def test_config_rejects(tmp_path, dr, top, match):
  with pytest.raises(ConfigError, match=match):
    load_task_config(_task(tmp_path, dr=dr, **top))


def test_table_contact_needs_table_object(tmp_path):
  task = load_task_config(_task(tmp_path, w_table_contact=-1.0))
  for support in (None, {"height": 0.0}):
    scene = load_scene_config(_scene(tmp_path, support))
    with pytest.raises(ConfigError, match="w_table_contact needs"):
      make_grasp_env_cfg_from_configs(scene, task)


# Object pose delay.


class _SubstepPoses:
  """True object pose after every physics substep of the latest env.step()."""

  def __init__(self, env, name="cube"):
    self.env, self.obj, self.poses = env, env.scene[name], []
    self._q_adr = self.obj.indexing.free_joint_q_adr.long()
    inner = env.sim.step

    def step():
      inner()
      self.poses.append(env.sim.data.qpos[:, self._q_adr].clone())

    env.sim.step = step

  def step(self, action):
    self.poses.clear()
    return self.env.step(action)

  def delayed(self):
    return self.poses[-1 - DELAY_SUBSTEPS]

  def push(self, speed=2.0):
    """Give the object a horizontal velocity, different per env, large enough
    that table friction has not stopped it by the end of the control step."""
    v_adr = self.obj.indexing.free_joint_v_adr.long()
    scale = torch.linspace(0.5, 1.0, self.env.num_envs, device=DEVICE)
    self.env.sim.data.qvel[:, v_adr[0]] = speed * scale
    self.env.sim.data.qvel[:, v_adr[1]] = -0.5 * speed * scale


def _pose_b(env, pose_w):
  return task_state.root_relative_pose(env, "robot", pose_w[:, :3], pose_w[:, 3:])


@needs_cuda
def test_actor_sees_true_pose_from_delay_ago(tmp_path):
  env = _make_env(_scene(tmp_path), _task(tmp_path, dr={"object_pose_delay_s": DELAY_S}))
  try:
    rec = _SubstepPoses(env)
    obs, _ = env.reset()
    # No history at reset: the current pose, same as the critic.
    assert torch.equal(obs["actor"][:, OBJECT_POSE], obs["critic"][:, OBJECT_POSE])
    moved = 0.0
    for _ in range(4):
      rec.push()
      obs, *_ = rec.step(_action(env))
      assert len(rec.poses) == env.cfg.decimation
      old, now = rec.delayed(), rec.poses[-1]
      assert torch.allclose(obs["actor"][:, OBJECT_POSE], _pose_b(env, old), atol=1e-6)
      # The critic is not delayed.
      assert torch.allclose(obs["critic"][:, OBJECT_POSE], _pose_b(env, now), atol=1e-6)
      moved = max(moved, (now[:, :3] - old[:, :3]).norm(dim=-1).min().item())
    assert moved > 1e-3  # every env's object moved over the delay, so the two differ
  finally:
    env.close()


@needs_cuda
def test_noise_is_applied_to_the_delayed_pose(tmp_path):
  noise = {"persistent_bias": {"enabled": True}, "per_step": {"enabled": True}}
  dr = {"object_pose_delay_s": DELAY_S, "object_pose_noise": noise}
  env = _make_env(_scene(tmp_path), _task(tmp_path, CURRICULUM_TASK, dr=dr))
  try:
    rec = _SubstepPoses(env)
    _set_difficulty(env, 1.0)
    env.reset()
    term = task_state.perception(env, PERCEPTION_NAME)
    base_quat = env.scene["robot"].data.root_link_quat_w
    for _ in range(4):
      rec.push()
      rec.step(_action(env))
      assert (term.noise_pos_b != 0).any(dim=-1).all() and (term.bias_pos_b != 0).any(dim=-1).all()
      errors = (
        (term.bias_pos_b, term.noise_pos_b),
        (term.bias_rotvec_b, term.noise_rotvec_b),
      )
      old, now = rec.delayed(), rec.poses[-1]
      pos, quat = perturb_pose(old[:, :3], old[:, 3:], base_quat, *errors)
      assert torch.allclose(term.pos_w, pos, atol=1e-6)
      assert torch.allclose(term.quat_w, quat, atol=1e-6)
      # Not the same noise on the current pose.
      pos_now, _ = perturb_pose(now[:, :3], now[:, 3:], base_quat, *errors)
      assert ((term.pos_w - pos_now).norm(dim=-1) > 1e-3).all()
  finally:
    env.close()


@needs_cuda
def test_delay_across_partial_reset(tmp_path):
  env = _make_env(_scene(tmp_path), _task(tmp_path, dr={"object_pose_delay_s": DELAY_S}))
  try:
    rec = _SubstepPoses(env)
    env.reset()
    for _ in range(3):
      rec.push()
      rec.step(_action(env))
    before = env.observation_manager.compute()["actor"].clone()
    ids = torch.tensor([1, 2], device=DEVICE)
    obs, _ = env.reset(env_ids=ids)
    # Reset envs see their new resting pose; the others keep their delayed pose.
    assert torch.equal(obs["actor"][ids][:, OBJECT_POSE], obs["critic"][ids][:, OBJECT_POSE])
    assert not torch.allclose(obs["actor"][ids][:, OBJECT_POSE], before[ids][:, OBJECT_POSE], atol=1e-4)
    keep = torch.tensor([0, 3], device=DEVICE)
    assert torch.equal(obs["actor"][keep][:, OBJECT_POSE], before[keep][:, OBJECT_POSE])
    # The next step's delayed pose belongs to the new episode in every env.
    rec.push()
    obs, *_ = rec.step(_action(env))
    assert torch.allclose(obs["actor"][:, OBJECT_POSE], _pose_b(env, rec.delayed()), atol=1e-6)
  finally:
    env.close()


# Action hold.


def _hold_env(tmp_path, probability=1.0):
  env = _make_env(
    _scene(tmp_path), _task(tmp_path, CURRICULUM_TASK, dr={"action_hold_probability": probability})
  )
  term = env.command_manager.get_term(ACTION_HOLD_NAME)
  assert isinstance(term, mdp.ActionHold)
  return env, term


def _tcp_pos(env):
  arm = env.action_manager.get_term("arm")
  return arm._get_frame_pose()[0].clone()


@needs_cuda
def test_hold_keeps_the_absolute_goal(tmp_path):
  env, term = _hold_env(tmp_path)
  try:
    _set_difficulty(env, 1.0)
    env.reset()
    arm, gripper = env.action_manager.get_term("arm"), env.action_manager.get_term("gripper")
    held = torch.tensor([True, True, False, False], device=DEVICE)
    term.uniform = lambda count: torch.where(held, 0.0, 1.0)[:count]
    assert not term.hold.any()  # first step of the episode
    env.step(_action(env, x=1.0, grip=1.0))
    assert torch.equal(term.hold, held)
    goal_pos, goal_quat = arm._desired_pos.clone(), arm._desired_quat.clone()
    grip_target = gripper._target.clone()
    tcp = _tcp_pos(env)
    # The arm is still on its way, so the old goal is not where it is now.
    assert ((goal_pos - tcp).norm(dim=-1) > 1e-3).all()

    env.step(_action(env, y=-1.0, grip=-1.0))
    scale = arm.cfg.delta_pos_scale
    new_goal = tcp + torch.tensor([0.0, -scale, 0.0], device=DEVICE)
    # Held: bit-identical goal and gripper target, not a delta from the current pose.
    assert torch.equal(arm._desired_pos[held], goal_pos[held])
    assert torch.equal(arm._desired_quat[held], goal_quat[held])
    assert torch.equal(gripper._target[held], grip_target[held])
    assert ((arm._desired_pos[held] - new_goal[held]).norm(dim=-1) > 1e-3).all()
    # Nor the previous delta applied again from the current pose.
    repeated = tcp + torch.tensor([scale, 0.0, 0.0], device=DEVICE)
    assert ((arm._desired_pos[held] - repeated[held]).norm(dim=-1) > 1e-3).all()
    # Not held: the new action, applied to the pose at this step.
    assert torch.allclose(arm._desired_pos[~held], new_goal[~held], atol=1e-6)
    assert (gripper._target[~held] == gripper.cfg.closed_pos).all()
    assert (gripper._target[held] == gripper.cfg.open_pos).all()
  finally:
    env.close()


@needs_cuda
def test_hold_never_on_first_step_and_scales_with_difficulty(tmp_path):
  env, term = _hold_env(tmp_path, probability=0.8)
  try:
    draws = {"u": 0.0}
    term.uniform = lambda count: torch.full((count,), draws["u"], device=DEVICE)
    for d, u, expected in ((1.0, 0.79, True), (1.0, 0.81, False), (0.5, 0.39, True), (0.5, 0.41, False), (0.0, 0.0, False)):
      draws["u"] = u
      _set_difficulty(env, d)
      env.reset()
      assert not term.hold.any()
      env.step(_action(env))
      assert (term.hold == expected).all(), (d, u)
    # A partial reset clears the hold of the reset envs only.
    draws["u"] = 0.0
    _set_difficulty(env, 1.0)
    env.reset()
    env.step(_action(env))
    assert term.hold.all()
    env.reset(env_ids=torch.tensor([0], device=DEVICE))
    assert term.hold.tolist() == [False, True, True, True]
  finally:
    env.close()


# Table contact.


@needs_cuda
def test_table_contact_penalty(tmp_path):
  env = _make_env(_scene(tmp_path, {"object": "table"}), _task(tmp_path, w_table_contact=-2.0))
  try:
    env.reset()
    value = lambda: mdp.rewards.table_contact(env, TABLE_CONTACT_SENSOR)  # noqa: E731
    index = env.reward_manager.active_terms.index("table_contact")
    # At rest at the ready pose nothing touches the table (the fixed base sits on it).
    for _ in range(5):
      env.step(_action(env))
      assert (value() == 0).all()
      assert (env.reward_manager._step_reward[:, index] == 0).all()
    metrics = env.metrics_manager
    assert (metrics._episode_sums["table_contact_frac"] == 0).all()
    assert (metrics._episode_max["table_collision"] == 0).all()
    # Drive the gripper down into the table.
    peak = torch.zeros(env.num_envs, device=DEVICE)
    for _ in range(60):
      _, reward, *_ = env.step(_action(env, z=-1.0))
      peak = torch.maximum(peak, value())
      assert ((value() >= 0) & (value() <= 1)).all()
    assert (peak == 1.0).all()  # in contact for every substep of some step
    assert (metrics._episode_max["table_collision"] == 1).all()
    frac = metrics._episode_sums["table_contact_frac"] / metrics._step_count
    assert ((frac > 0) & (frac < 1)).all()
    task = load_task_config(_task(tmp_path, w_table_contact=-2.0))
    weighted = env.reward_manager._step_reward[:, index]
    assert torch.allclose(weighted, value() * -2.0 / task.reward_normalization)
  finally:
    env.close()
