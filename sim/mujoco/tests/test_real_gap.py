"""Object pose delay, held actions, the robot-table contact penalty, goal
interpolation, the gripper goal time and the arm's joint command path."""

import json
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_error_magnitude
from tabletop_sim.scene import ConfigError, load_scene_config, load_task_config
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import (
  ACTION_HOLD_NAME,
  CONFIGS_DIR,
  PERCEPTION_NAME,
  POSE_HISTORY_SENSOR,
  TABLE_CONTACT_SENSOR,
  make_grasp_env_cfg,
  make_grasp_env_cfg_from_configs,
)
from tabletop_sim.tasks.grasp.mdp import curriculum, task_state
from tabletop_sim.tasks.grasp.mdp.perception import perturb_pose
from task_files import CUBE_SCENE, task_file

STANDARD_TASK = task_file("standard")
CURRICULUM_TASK = task_file("curriculum")
LATENCY_TASK = task_file("latency")
GRIPPER_TASK = task_file("latency_interp_gripper")
INTERP_TASK = task_file("latency_interp")
JOINTS_TASK = task_file("latency_gripper_joints")

DEVICE = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DELAY_S, DELAY_SUBSTEPS = 0.01, 5
OBJECT_POSE = slice(24, 31)  # actor and critic columns


def _write(path: Path, data: dict) -> Path:
  path.write_text(json.dumps(data))
  return path


def _task(tmp_path: Path, base: Path = STANDARD_TASK, dr: dict | None = None, **top) -> Path:
  data = json.loads(base.read_text())
  data.update(top)
  data.setdefault("domain_randomization", {}).update(dr or {})
  if "curriculum" in data:
    # No promotions of its own; tests set the stage directly.
    data["curriculum"]["min_completed_episodes"] = 10**9
  return _write(tmp_path / "task.json", data)


def _scene(tmp_path: Path, support: dict | None = None, embodiment: str | None = None) -> Path:
  data = json.loads(CUBE_SCENE.read_text())
  if support is not None:
    data["support_surface"] = support
  if embodiment is not None:
    data["robot"]["embodiment"] = embodiment
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
  task = load_task_config(STANDARD_TASK)
  dr = task.domain_randomization
  assert dr.object_pose_delay_s == 0.0 and dr.action_hold_probability == 0.0
  assert task.w_table_contact == 0.0 and task.object_pose_delay_substeps == 0
  cfg = make_grasp_env_cfg(CUBE_SCENE, STANDARD_TASK)
  assert {s.name for s in cfg.scene.sensors}.isdisjoint({POSE_HISTORY_SENSOR, TABLE_CONTACT_SENSOR})
  assert ACTION_HOLD_NAME not in cfg.commands and "table_contact" not in cfg.rewards
  assert {"table_contact_frac", "table_collision"}.isdisjoint(cfg.metrics)
  assert cfg.commands[PERCEPTION_NAME].history_sensor_name is None
  assert all(a.hold_command_name is None for a in cfg.actions.values())
  assert cfg.actions["arm"].interpolate_goal is False
  assert cfg.actions["gripper"].goal_time_s == 0.0
  assert "gripper_switch" not in cfg.rewards and "gripper_switches" not in cfg.metrics
  assert cfg.actions["arm"].command_delay_substeps == 0
  assert "randomize_joint_friction" not in cfg.events


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


def test_interpolation_task_wiring(tmp_path):
  assert load_task_config(LATENCY_TASK).action.ee_goal_interpolation is False
  assert load_task_config(INTERP_TASK).action.ee_goal_interpolation is True
  task = _task(tmp_path, action={"ee_goal_interpolation": True})
  assert make_grasp_env_cfg(_scene(tmp_path), task).actions["arm"].interpolate_goal is True
  with pytest.raises(ConfigError, match="needs an end-effector action space"):
    make_grasp_env_cfg(_scene(tmp_path), task, "delta_joint_abs_gripper")


def test_gripper_task_wiring(tmp_path):
  gripper = load_task_config(GRIPPER_TASK)
  assert gripper.action.gripper_goal_time_s == 0.5 and gripper.w_gripper_switch < 0
  task_path = _task(tmp_path, action={"gripper_goal_time_s": 0.5}, w_gripper_switch=-2.0)
  task = load_task_config(task_path)
  cfg = make_grasp_env_cfg(_scene(tmp_path), task_path)
  assert cfg.actions["gripper"].goal_time_s == 0.5
  reward, metric = cfg.rewards["gripper_switch"], cfg.metrics["gripper_switches"]
  assert reward.func is mdp.rewards.gripper_switch
  assert reward.weight == pytest.approx(-2.0 / task.reward_normalization)
  assert (metric.func, metric.reduce, metric.params) == (reward.func, "sum", reward.params)
  for top in ({"action": {"gripper_goal_time_s": 0.5}}, {"w_gripper_switch": -2.0}):
    with pytest.raises(ConfigError, match="needs a binary gripper action space"):
      make_grasp_env_cfg(_scene(tmp_path), _task(tmp_path, **top), "ee_delta_pose_delta_gripper")


@pytest.mark.parametrize(
  ("dr", "top", "match"),
  [
    ({"object_pose_delay_s": -0.002}, {}, "object_pose_delay_s must be finite and nonnegative"),
    ({"object_pose_delay_s": 0.003}, {}, "multiple of timing.sim_dt"),
    ({"action_hold_probability": -0.1}, {}, r"action_hold_probability must be in \[0, 1\]"),
    ({"action_hold_probability": 1.1}, {}, r"action_hold_probability must be in \[0, 1\]"),
    ({}, {"w_table_contact": 0.5}, "finite and nonpositive"),
    ({}, {"action": {"ee_goal_interpolation": 1}}, "ee_goal_interpolation must be a bool"),
    ({}, {"action": {"gripper_goal_time_s": -0.1}}, "gripper_goal_time_s must be finite and non"),
    ({}, {"w_gripper_switch": 0.1}, "finite and nonpositive"),
    ({}, {"action": {"command_delay_s": 0.003}}, "command_delay_s=0.003 must be a multiple"),
    ({"joint_friction": {"enabled": 1}}, {}, "joint_friction.enabled must be a bool"),
    ({"joint_friction": {"frictionloss": [1.0]}}, {}, "unknown keys"),
    ({"joint_friction": {"scale_range": [1.2, 0.8]}}, {}, "scale_range must be finite with"),
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


# Goal interpolation.


class _SubstepGoals:
  """The arm term's IK goal and joint target change on every physics substep
  of the latest env.step()."""

  def __init__(self, env):
    self.env, self.arm = env, env.action_manager.get_term("arm")
    self.pos, self.quat, self.dq = [], [], []
    inner = self.arm.apply_actions

    def apply_actions():
      inner()
      self.pos.append(self.arm._desired_pos.clone())
      self.quat.append(self.arm._desired_quat.clone())
      self.dq.append(self.arm.compute_dq().abs().amax(dim=-1))

    self.arm.apply_actions = apply_actions

  def step(self, action):
    self.pos.clear(), self.quat.clear(), self.dq.clear()
    pos, quat = self.arm._get_frame_pose()
    start = pos.clone(), quat.clone()
    self.env.step(action)
    return start


def _interp_env(tmp_path, base=STANDARD_TASK, dr=None):
  task = _task(tmp_path, base, dr=dr, action={"ee_goal_interpolation": True})
  return _make_env(_scene(tmp_path), task)


@needs_cuda
def test_interpolated_goal_moves_linearly_to_the_step_goal(tmp_path):
  env = _interp_env(tmp_path)
  try:
    env.reset()
    rec = _SubstepGoals(env)
    arm, n = rec.arm, env.cfg.decimation
    action = _action(env)
    action[:, :6] = torch.tensor([1.0, -0.5, 0.25, 0.0, 0.0, 2.0], device=DEVICE)  # yaw clips to 1
    d_pos = torch.tensor([1.0, -0.5, 0.25], device=DEVICE) * arm.cfg.delta_pos_scale
    start_pos, start_quat = rec.step(action)
    assert len(rec.pos) == n
    for i, (pos, quat) in enumerate(zip(rec.pos, rec.quat, strict=True), start=1):
      assert torch.allclose(pos, start_pos + d_pos * (i / n), atol=1e-6)
      angle = quat_error_magnitude(quat, start_quat)
      assert torch.allclose(angle, torch.full_like(angle, arm.cfg.delta_ori_scale * i / n), atol=1e-5)
    # The last substep solves toward the goal process_actions computed.
    assert torch.equal(rec.pos[-1], arm._goal_pos)
    assert torch.equal(rec.quat[-1], arm._goal_quat)
    # A second step starts again from the pose the arm has then.
    start_pos, _ = rec.step(_action(env, z=-1.0))
    first = start_pos + torch.tensor([0.0, 0.0, -arm.cfg.delta_pos_scale / n], device=DEVICE)
    assert torch.allclose(rec.pos[0], first, atol=1e-6)
  finally:
    env.close()


@needs_cuda
def test_interpolation_replaces_the_first_substep_jump(tmp_path):
  env = _interp_env(tmp_path)
  try:
    rec = _SubstepGoals(env)
    arm = rec.arm
    env.reset()
    rec.step(_action(env, x=1.0))
    ramp_first, ramp_goals = rec.dq[0].clone(), len({tuple(p[0].tolist()) for p in rec.pos})
    arm.cfg.interpolate_goal = False
    env.reset()
    start_pos, _ = rec.step(_action(env, x=1.0))
    goal = start_pos + torch.tensor([arm.cfg.delta_pos_scale, 0.0, 0.0], device=DEVICE)
    # Off: every substep solves toward the same goal.
    assert all(torch.equal(p, rec.pos[0]) for p in rec.pos)
    assert torch.allclose(rec.pos[0], goal, atol=1e-6)
    assert ramp_goals == env.cfg.decimation
    assert (ramp_first < 0.2 * rec.dq[0]).all()
  finally:
    env.close()


@needs_cuda
def test_interpolation_keeps_a_held_goal(tmp_path):
  env = _interp_env(tmp_path, CURRICULUM_TASK, dr={"action_hold_probability": 1.0})
  try:
    term = env.command_manager.get_term(ACTION_HOLD_NAME)
    _set_difficulty(env, 1.0)
    env.reset()
    rec = _SubstepGoals(env)
    arm = rec.arm
    held = torch.tensor([True, True, False, False], device=DEVICE)
    term.uniform = lambda count: torch.where(held, 0.0, 1.0)[:count]
    rec.step(_action(env, x=1.0))
    goal_pos, goal_quat = arm._desired_pos.clone(), arm._desired_quat.clone()
    start_pos, _ = rec.step(_action(env, y=-1.0))
    scale, n = arm.cfg.delta_pos_scale, env.cfg.decimation
    for i, (pos, quat) in enumerate(zip(rec.pos, rec.quat, strict=True), start=1):
      assert torch.equal(pos[held], goal_pos[held])
      assert torch.equal(quat[held], goal_quat[held])
      expected = start_pos + torch.tensor([0.0, -scale * i / n, 0.0], device=DEVICE)
      assert torch.allclose(pos[~held], expected[~held], atol=1e-6)
  finally:
    env.close()


# Gripper goal time and switch penalty.


class _SubstepGripper:
  """The gripper term's carriage position target on every physics substep
  since the recorder was made."""

  def __init__(self, env):
    self.env, self.term, self.targets = env, env.action_manager.get_term("gripper"), []
    inner = self.term._position_target

    def position_target():
      target = inner()
      self.targets.append(target.clone())
      return target

    self.term._position_target = position_target

  def q(self):
    return self.term._entity.data.joint_pos[:, self.term._joint_ids].clone()


def _quintic(tau):
  return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


@needs_cuda
def test_gripper_moves_over_its_goal_time(tmp_path):
  goal_time = 0.5
  env = _make_env(_scene(tmp_path), _task(tmp_path, action={"gripper_goal_time_s": goal_time}))
  try:
    env.reset()
    rec = _SubstepGripper(env)
    term, n = rec.term, env.cfg.decimation
    closed, opened = term.cfg.closed_pos, term.cfg.open_pos
    per_move = round(goal_time / env.physics_dt)
    start = rec.q()
    assert ((start - closed).abs() > 0.01).all()  # the ready pose is not closed
    # The first command of the episode starts a move from the measured carriage position.
    for _ in range(4):
      env.step(_action(env, grip=-1.0))
      assert not term.switched.any()
    assert len(rec.targets) == 4 * n
    for k, target in enumerate(rec.targets, start=1):
      assert torch.allclose(target, start + (closed - start) * _quintic(k / per_move), atol=1e-7)
    assert ((rec.targets[-1] - closed).abs() > 1e-3).all()  # 0.2 s in: still on its way

    # A switch mid-move starts a new move from where the carriage is then.
    start, done = rec.q(), len(rec.targets)
    env.step(_action(env, grip=1.0))
    assert term.switched.all()
    for _ in range(per_move // n):
      env.step(_action(env, grip=1.0))
      assert not term.switched.any()  # the same command again is not a switch
    for k, target in enumerate(rec.targets[done:], start=1):
      expected = start + (opened - start) * _quintic(min(k / per_move, 1.0))
      assert torch.allclose(target, expected, atol=1e-7)
    # From the goal time on the target is the commanded position itself.
    assert all((t == opened).all() for t in rec.targets[done + per_move - 1 :])
    assert (rec.q() - opened).abs().max() < 0.01  # there, still ringing on the actuator

    # A reset makes the next command a first command again.
    env.step(_action(env, grip=-1.0))
    assert term.switched.all()
    env.reset(env_ids=torch.tensor([0], device=DEVICE))
    assert term.switched.tolist() == [False, True, True, True]
    env.step(_action(env, grip=1.0))
    assert term.switched.tolist() == [False, True, True, True]
  finally:
    env.close()


@needs_cuda
def test_gripper_goal_time_rule_and_default(tmp_path):
  # Up to 0.2 s the move is linear; without a goal time the target jumps.
  for goal_time in (0.1, 0.0):
    env = _make_env(_scene(tmp_path), _task(tmp_path, action={"gripper_goal_time_s": goal_time}))
    try:
      env.reset()
      rec = _SubstepGripper(env)
      closed, start = rec.term.cfg.closed_pos, rec.q()
      env.step(_action(env, grip=-1.0))
      if goal_time == 0.0:
        assert all((t == closed).all() for t in rec.targets)
        continue
      per_move = round(goal_time / env.physics_dt)
      for k, target in enumerate(rec.targets, start=1):
        assert torch.allclose(target, start + (closed - start) * (k / per_move), atol=1e-7)
    finally:
      env.close()


@needs_cuda
def test_gripper_switch_penalty(tmp_path):
  task_path = _task(
    tmp_path, CURRICULUM_TASK, dr={"action_hold_probability": 1.0}, w_gripper_switch=-2.0
  )
  env = _make_env(_scene(tmp_path), task_path)
  try:
    hold = env.command_manager.get_term(ACTION_HOLD_NAME)
    hold.uniform = lambda count: torch.ones(count, device=DEVICE)  # no holds yet
    _set_difficulty(env, 1.0)
    env.reset()
    index = env.reward_manager.active_terms.index("gripper_switch")
    weight = -2.0 / load_task_config(task_path).reward_normalization
    value = lambda: mdp.rewards.gripper_switch(env, "gripper")  # noqa: E731
    count = 0
    for grip, switched in ((1.0, False), (1.0, False), (-1.0, True), (-1.0, False), (1.0, True)):
      env.step(_action(env, grip=grip))
      assert (value() == float(switched)).all()
      assert torch.allclose(
        env.reward_manager._step_reward[:, index], torch.full_like(value(), weight * switched)
      )
      count += switched
    assert (env.metrics_manager._episode_sums["gripper_switches"] == count).all()
    # A held step keeps the old command, so the discarded switch is not penalized.
    held = torch.tensor([True, True, False, False], device=DEVICE)
    hold.uniform = lambda count: torch.where(held, 0.0, 1.0)[:count]
    env.step(_action(env, grip=1.0))  # draws the hold mask for the next step
    env.step(_action(env, grip=-1.0))
    assert torch.equal(value().bool(), ~held)
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


# Joint command path: friction and command delay.

FRICTION = [0.64, 1.85, 1.92, 0.10, 0.10, 0.16]


def test_joints_task_wiring(tmp_path):
  joints = load_task_config(JOINTS_TASK)
  assert joints.action.command_delay_s == 0.006 and joints.command_delay_substeps == 3
  assert joints.domain_randomization.joint_friction.enabled

  task = _task(
    tmp_path,
    dr={"joint_friction": {"enabled": True}},
    action={"command_delay_s": 0.006},
  )
  cfg = make_grasp_env_cfg(_scene(tmp_path), task)
  assert cfg.actions["arm"].command_delay_substeps == 3
  event = cfg.events["randomize_joint_friction"]
  assert event.func is mdp.RandomizeJointFriction and event.mode == "reset"
  assert event.params["joint_names"] == tuple(f"joint_{i}" for i in range(6))
  assert event.params["scale_range"] == (0.7, 1.3)
  with pytest.raises(ConfigError, match="needs an end-effector action space"):
    make_grasp_env_cfg(
      _scene(tmp_path), _task(tmp_path, action={"command_delay_s": 0.006}), "delta_joint_abs_gripper"
    )
  friction_scene = json.loads((CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai_friction.json").read_text())
  jenga_scene = json.loads((CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json").read_text())
  assert friction_scene["robot"].pop("embodiment") == "trossen_wxai_base_real_friction"
  assert jenga_scene["robot"].pop("embodiment") == "trossen_wxai_base_real"
  assert friction_scene == jenga_scene


@needs_cuda
def test_joint_friction_is_set_and_scaled_with_difficulty(tmp_path):
  task = _task(
    tmp_path,
    CURRICULUM_TASK,
    dr={"joint_friction": {"enabled": True, "scale_range": [0.5, 1.5]}},
  )
  scene = _scene(tmp_path, embodiment="trossen_wxai_base_real_friction")
  env = _make_env(scene, task, num_envs=64)
  try:
    robot = env.scene["robot"]
    arm = env.action_manager.get_term("arm")
    dof = robot.indexing.joint_v_adr[arm._joint_ids]
    nominal = torch.tensor(FRICTION, device=DEVICE)  # the friction model's values
    env.reset()
    assert torch.allclose(env.sim.model.dof_frictionloss[:, dof], nominal.expand(64, -1))
    _set_difficulty(env, 1.0)
    env.reset()
    scale = env.sim.model.dof_frictionloss[:, dof] / nominal
    assert bool(((scale >= 0.5) & (scale <= 1.5)).all())
    assert scale.std() > 0.1 and scale.min() < 0.7 and scale.max() > 1.3
  finally:
    env.close()


class _SubstepCommands:
  """Per physics substep of the latest env.step(): the arm joints and IK step the
  term read, and the position target it set."""

  def __init__(self, env):
    self.env, self.arm = env, env.action_manager.get_term("arm")
    self.robot = env.scene["robot"]
    self.q, self.dq, self.target = [], [], []
    compute_dq, apply_actions = self.arm.compute_dq, self.arm.apply_actions

    def recorded_dq():
      dq = compute_dq()
      self.q.append(self.robot.data.joint_pos[:, self.arm._joint_ids].clone())
      self.dq.append(dq.clone())
      return dq

    def recorded_apply():
      apply_actions()
      self.target.append(self.robot.data.joint_pos_target[:, self.arm._joint_ids].clone())

    self.arm.compute_dq, self.arm.apply_actions = recorded_dq, recorded_apply

  def step(self, action):
    self.q.clear(), self.dq.clear(), self.target.clear()
    self.env.step(action)


def _path_env(tmp_path, **action):
  return _make_env(_scene(tmp_path, embodiment="trossen_wxai_base_real"), _task(tmp_path, action=action))


@needs_cuda
def test_command_delay_shifts_the_joint_targets(tmp_path):
  env = _path_env(tmp_path, command_delay_s=0.006)
  try:
    env.reset()
    rec = _SubstepCommands(env)
    rec.step(_action(env, x=1.0, z=-1.0))
    n = env.cfg.decimation
    assert len(rec.target) == n
    commands = [q + dq for q, dq in zip(rec.q, rec.dq, strict=True)]
    for i in range(n):
      # An episode's first three targets are its first command.
      assert torch.equal(rec.target[i], commands[max(i - 3, 0)])
    last = commands[-3:]
    rec.step(_action(env, y=1.0))
    assert all(torch.equal(rec.target[i], last[i]) for i in range(3))
    assert torch.equal(rec.target[3], rec.q[0] + rec.dq[0])
  finally:
    env.close()


@needs_cuda
def test_command_delay_restarts_with_the_episode(tmp_path):
  env = _path_env(tmp_path, command_delay_s=0.006)
  try:
    env.reset()
    rec = _SubstepCommands(env)
    for _ in range(3):
      rec.step(_action(env, x=1.0, z=-1.0))
    env.reset()
    rec.step(_action(env))
    # The buffer holds the new episode's first command, not the old episode's last ones.
    first = rec.q[0] + rec.dq[0]
    assert all(torch.equal(rec.target[i], first) for i in range(3))
  finally:
    env.close()
