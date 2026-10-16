"""Curriculum-scaled object pose observation noise (persistent bias + per-step
noise) and the actor goal built from it."""

import json
import math
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_angle_axis,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
)
from tabletop_sim.scene import ConfigError, ObjectPoseNoiseConfig, load_scene_config, load_task_config
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import (
  CONFIGS_DIR,
  DEFAULT_SCENE,
  DEFAULT_TASK,
  PERCEPTION_NAME,
  make_grasp_env_cfg,
)
from tabletop_sim.tasks.grasp.mdp import curriculum, object_size, task_state
from tabletop_sim.tasks.grasp.mdp.perception import (
  clipped_gaussian,
  perturb_pose,
  quat_from_rotvec,
)

FRICTION_TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_size_variants_friction_wxai.json"
CURRICULUM_TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_wxai.json"
JENGA_SCENE = CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json"
DEVICE = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_jenga = pytest.mark.skipif(
  not load_scene_config(JENGA_SCENE).manipulated.asset_path.exists(), reason="jenga mesh missing"
)

# Actor/critic columns: object_pose, tcp_to_obj_pos, obj_to_goal_pos.
EE = slice(16, 19)
OBJ_POS, OBJ_QUAT, TCP_TO_OBJ, OBJ_TO_GOAL = slice(24, 27), slice(27, 31), slice(31, 34), slice(34, 37)
OBJECT_COLS = slice(24, 37)
LIFT = 0.03
STEP_T, STEP_R = 0.01, math.radians(5.0)
BIAS_T, BIAS_R = 0.0005, math.radians(0.1)


def _write(path: Path, data: dict) -> Path:
  path.write_text(json.dumps(data))
  return path


def _noise_block(bias=True, step=True, **overrides) -> dict:
  block = {"persistent_bias": {"enabled": bias}, "per_step": {"enabled": step}}
  for key, value in overrides.items():
    source, name = key.split("__")
    block[source][name] = value
  return block


def _task(tmp_path: Path, base: Path = CURRICULUM_TASK, **noise) -> Path:
  data = json.loads(base.read_text())
  data["domain_randomization"]["object_pose_noise"] = _noise_block(**noise)
  # No promotions of its own; tests set the stage directly.
  data["curriculum"]["min_completed_episodes"] = 10**9
  return _write(tmp_path / "task.json", data)


def _cube_scene(tmp_path: Path, yaw_deg: float = 0.0) -> Path:
  """The default cube scene, with the robot base optionally yawed (and tilted
  slightly) so world and base axes differ."""
  data = json.loads(DEFAULT_SCENE.read_text())
  if yaw_deg:
    q = quat_from_euler_xyz(
      torch.tensor([math.radians(4.0)]), torch.tensor([0.0]), torch.tensor([math.radians(yaw_deg)])
    )[0]
    data["robot"]["base_pose"] = [-0.615, 0.05, 0.0, *q.tolist()]
  return _write(tmp_path / "scene.json", data)


def _jenga_scene(tmp_path: Path) -> Path:
  data = json.loads(JENGA_SCENE.read_text())
  for obj in data["objects"]:
    for key in ("asset_path", "texture_path"):
      if key in obj:
        obj[key] = str((JENGA_SCENE.parent / obj[key]).resolve())
  return _write(tmp_path / "scene.json", data)


def _make_env(scene, task, num_envs=4, seed=0) -> ManagerBasedRlEnv:
  cfg = make_grasp_env_cfg(scene, task, None, num_envs)
  cfg.seed = seed
  return ManagerBasedRlEnv(cfg, device=DEVICE)


def _set_difficulty(env, d):
  """Difficulty the next resetting episodes start with."""
  state = curriculum.get_curriculum_state(env)
  state.stage = round(d / state.difficulty_increment)
  assert state.difficulty == pytest.approx(d)


def _zero_action(env):
  return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)


def _term(env) -> mdp.PerceivedObjectPose:
  return task_state.perception(env, PERCEPTION_NAME)


def _true_pose_b(env):
  pos_w, quat_w = task_state.object_pose_w(env, SceneEntityCfg("cube"))
  pose = task_state.root_relative_pose(env, "robot", pos_w, quat_w)
  return pose[:, :3], pose[:, 3:]


def _base(env):
  root = env.scene["robot"].data.root_link_pose_w
  return root[:, :3], root[:, 3:7]


def _to_base_vec(env, vec_w):
  return task_state.root_relative_vec(env, "robot", vec_w)


def _same_rotation(q1, q2, atol=1e-5):
  torch.testing.assert_close(matrix_from_quat(q1), matrix_from_quat(q2), atol=atol, rtol=0)


class _Injector:
  """Replaces the term's N(0, 1) source: each call pops the next queued row
  (broadcast to every env in the call); zeros once the queue is empty."""

  def __init__(self, term):
    self.queue: list[list[float]] = []
    self.calls: list[tuple[int, ...]] = []
    term.standard_normal = self

  def __call__(self, shape):
    self.calls.append(shape)
    row = self.queue.pop(0) if self.queue else [0.0] * 6
    return torch.tensor(row, device=DEVICE).expand(shape).clone()


# --- Math ----------------------------------------------------------------------


def test_quat_from_rotvec():
  assert torch.equal(quat_from_rotvec(torch.zeros(2, 3)), torch.tensor([[1.0, 0, 0, 0]] * 2))
  v = torch.randn(64, 3) * 0.5
  v[0] = torch.tensor([1e-8, 0.0, 0.0])
  q = quat_from_rotvec(v)
  torch.testing.assert_close(torch.linalg.norm(q, dim=-1), torch.ones(64))
  angle = torch.linalg.norm(v, dim=-1)
  ref = quat_from_angle_axis(angle[1:], v[1:] / angle[1:, None])
  torch.testing.assert_close(q[1:], ref, atol=1e-6, rtol=0)


def test_clipped_gaussian():
  z = torch.tensor([[-3.0, -1.0, 0.5, 2.0, 5.0]])
  std = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]])
  torch.testing.assert_close(
    clipped_gaussian(z, std, 2.0), torch.tensor([[-0.2, -0.1, 0.05, 0.2, 0.2]])
  )
  assert torch.equal(clipped_gaussian(z, torch.zeros_like(std), 2.0).abs(), torch.zeros_like(z))


def test_perturb_pose_base_frame_and_order():
  torch.manual_seed(0)
  n = 16
  base_q = quat_from_euler_xyz(torch.full((n,), 0.3), torch.full((n,), -0.2), torch.linspace(-3, 3, n))
  base_p = torch.randn(n, 3)
  obj_q = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
  obj_p = torch.randn(n, 3)
  b_p, e_p = torch.randn(n, 3) * 0.01, torch.randn(n, 3) * 0.01
  b_r, e_r = torch.randn(n, 3) * 0.3, torch.randn(n, 3) * 0.3  # Large: order matters.

  pos_w, quat_w = perturb_pose(obj_p, obj_q, base_q, (b_p, e_p), (b_r, e_r))
  base_inv = quat_inv(base_q)
  pos_b = matrix_from_quat(base_inv) @ (obj_p - base_p)[..., None]
  got_pos_b = matrix_from_quat(base_inv) @ (pos_w - base_p)[..., None]
  torch.testing.assert_close(got_pos_b[..., 0], pos_b[..., 0] + b_p + e_p, atol=1e-5, rtol=0)

  def exp(v):
    return matrix_from_quat(quat_from_rotvec(v))

  rot_b = matrix_from_quat(quat_mul(base_inv, obj_q))
  got_rot_b = matrix_from_quat(quat_mul(base_inv, quat_w))
  torch.testing.assert_close(got_rot_b, exp(e_r) @ exp(b_r) @ rot_b, atol=1e-5, rtol=0)
  assert (got_rot_b - exp(b_r) @ exp(e_r) @ rot_b).abs().max() > 1e-2
  torch.testing.assert_close(torch.linalg.norm(quat_w, dim=-1), torch.ones(n))

  # Rotation only: position untouched; no rotation: quaternion untouched.
  zero = torch.zeros(n, 3)
  pos_w, quat_w = perturb_pose(obj_p, obj_q, base_q, (zero, zero), (b_r, e_r))
  assert torch.equal(pos_w, obj_p)
  pos_w, quat_w = perturb_pose(obj_p, obj_q, base_q, (b_p, zero), (zero, zero))
  assert torch.equal(quat_w, obj_q)


# --- Configuration ---------------------------------------------------------------


def test_config_defaults_and_units():
  c = ObjectPoseNoiseConfig()
  assert not c.persistent_bias.enabled and not c.per_step.enabled
  assert c.per_step.translation_std_m == (0.01,) * 3 and c.per_step.rotation_std_deg == (5.0,) * 3
  assert c.persistent_bias.translation_std_m == (0.0005,) * 3
  assert c.persistent_bias.rotation_std_deg == (0.1,) * 3
  for src in (c.persistent_bias, c.per_step):
    assert src.translation_clip_sigma == src.rotation_clip_sigma == 2.0
  for task in ("grasp_standard_wxai", "grasp_curriculum_wxai", "grasp_curriculum_size_variants_wxai"):
    t = load_task_config(CONFIGS_DIR / "tasks" / f"{task}.json")
    assert t.domain_randomization.object_pose_noise == c

  cfg = make_grasp_env_cfg(JENGA_SCENE, FRICTION_TASK)
  p = cfg.commands[PERCEPTION_NAME]
  assert p.persistent_bias.enabled and p.per_step.enabled
  assert p.per_step.translation_std == (0.01,) * 3
  assert p.per_step.rotation_std == pytest.approx((STEP_R,) * 3)
  assert p.persistent_bias.rotation_std == pytest.approx((BIAS_R,) * 3)
  assert p.target_lift_height == LIFT


def test_config_partial_override_keeps_source_defaults(tmp_path):
  path = _task(tmp_path, per_step__rotation_std_deg=[1, 2, 3], persistent_bias__rotation_clip_sigma=3)
  c = load_task_config(path).domain_randomization.object_pose_noise
  assert c.per_step.rotation_std_deg == (1.0, 2.0, 3.0)
  assert c.per_step.translation_std_m == (0.01,) * 3
  assert c.persistent_bias.rotation_clip_sigma == 3.0
  assert c.persistent_bias.translation_std_m == (0.0005,) * 3


@pytest.mark.parametrize(
  ("override", "match"),
  [
    ({"per_step__enabled": 1}, "must be a bool"),
    ({"per_step__translation_std_m": [0.01, -0.01, 0.01]}, "nonnegative"),
    ({"persistent_bias__rotation_std_deg": [0.1, 0.1]}, "list of 3"),
    ({"per_step__translation_clip_sigma": 0}, "positive"),
    ({"per_step__rotation_clip_sigma": True}, "must be a number"),
    ({"per_step__unknown": 1}, "unknown keys"),
  ],
)
def test_config_rejects(tmp_path, override, match):
  with pytest.raises(ConfigError, match=match):
    load_task_config(_task(tmp_path, **override))


def test_observation_wiring_and_lift_tolerance():
  cfg = make_grasp_env_cfg(JENGA_SCENE, FRICTION_TASK)
  obs = mdp.observations
  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert actor.terms["object_pose"].func is obs.perceived_object_pose
  assert actor.terms["tcp_to_obj_pos"].func is obs.perceived_tcp_to_obj_pos
  assert actor.terms["obj_to_goal_pos"].func is obs.perceived_obj_to_goal_pos
  assert critic.terms["object_pose"].func is obs.object_pose
  assert critic.terms["tcp_to_obj_pos"].func is obs.tcp_to_obj_pos
  assert critic.terms["obj_to_goal_pos"].func is obs.obj_to_goal_pos
  assert list(actor.terms) == [t for t in critic.terms if t != "is_grasped"]
  for group in (actor, critic):
    assert not group.enable_corruption and not group.history_length
    assert all(t.noise is None and t.history_length == 0 for t in group.terms.values())

  assert load_task_config(FRICTION_TASK).lift_threshold == 0.02
  assert cfg.rewards["stable_hold"].params["lift_threshold"] == 0.02
  assert cfg.metrics["mean_hold_score"].params["lift_threshold"] == 0.02
  assert cfg.metrics["grasped_lifted_frac"].params["lift_threshold"] == 0.02
  assert cfg.curriculum[curriculum.TERM_NAME].params["metric_name"] == "grasped_lifted_frac"
  assert load_task_config(DEFAULT_TASK).lift_threshold == 0.01


# --- Simulation --------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize(("noise", "d"), [({"bias": False, "step": False}, 1.0), ({}, 0.0)])
def test_clean_when_disabled_or_zero_difficulty(tmp_path, noise, d):
  env = _make_env(_cube_scene(tmp_path, yaw_deg=30.0), _task(tmp_path, **noise))
  try:
    _set_difficulty(env, d)
    obs, _ = env.reset()
    for _ in range(4):
      assert torch.equal(obs["actor"], obs["critic"][:, :37])
      term, goal = _term(env), task_state.goal(env, "grasp")
      assert torch.equal(term.actor_target_pos, goal.target_pos)
      assert torch.equal(term.pos_w, env.scene["cube"].data.root_link_pos_w)
      obs, *_ = env.step(_zero_action(env))
  finally:
    env.close()


@needs_cuda
def test_injected_samples_frames_and_actor_goal(tmp_path):
  """Deterministic samples on a yawed/tilted base: base-frame composition, the
  first observation, the fixed actor goal and per-step resampling."""
  env = _make_env(_cube_scene(tmp_path, yaw_deg=60.0), _task(tmp_path), num_envs=2)
  try:
    d = 0.5
    _set_difficulty(env, d)
    inject = _Injector(_term(env))
    z_bias = [1.0, -2.5, 0.5, 3.0, -1.0, 0.25]
    z_step = [-0.5, 1.5, 2.0, 0.5, -3.0, 1.0]
    inject.queue = [z_bias, z_step]
    obs, _ = env.reset()
    assert inject.calls == [(2, 6), (2, 6)]  # bias, then first per-step noise

    def expected(z, t_std, r_std):
      z = torch.tensor(z, device=DEVICE)
      return (
        torch.clamp(z[:3] * d * t_std, -2 * d * t_std, 2 * d * t_std),
        torch.clamp(z[3:] * d * r_std, -2 * d * r_std, 2 * d * r_std),
      )

    b_p, b_r = expected(z_bias, BIAS_T, BIAS_R)
    e_p, e_r = expected(z_step, STEP_T, STEP_R)
    term = _term(env)
    torch.testing.assert_close(term.bias_pos_b, b_p.expand(2, 3))
    torch.testing.assert_close(term.bias_rotvec_b, b_r.expand(2, 3))
    torch.testing.assert_close(term.noise_pos_b, e_p.expand(2, 3))
    torch.testing.assert_close(term.noise_rotvec_b, e_r.expand(2, 3))

    # The first observation already carries both sources, in base axes.
    true_p, true_q = _true_pose_b(env)
    actor = obs["actor"]
    torch.testing.assert_close(actor[:, OBJ_POS] - true_p, (b_p + e_p).expand(2, 3), atol=1e-6, rtol=0)
    rot = quat_mul(quat_from_rotvec(e_r), quat_from_rotvec(b_r)).expand(2, 4)
    _same_rotation(actor[:, OBJ_QUAT], quat_mul(rot, true_q))
    torch.testing.assert_close(torch.linalg.norm(actor[:, OBJ_QUAT], dim=-1), torch.ones(2, device=DEVICE))

    # Actor goal from that same corrupted pose; clean goal from the true pose.
    goal = task_state.goal(env, "grasp")
    lift = torch.tensor([0.0, 0.0, LIFT], device=DEVICE)
    actor_goal = term.actor_target_pos.clone()
    torch.testing.assert_close(actor_goal, term.pos_w + lift)
    torch.testing.assert_close(goal.target_pos, env.scene["cube"].data.root_link_pos_w + lift)
    root_p, _ = _base(env)
    torch.testing.assert_close(
      _to_base_vec(env, actor_goal - root_p) - _to_base_vec(env, goal.target_pos - root_p),
      (b_p + e_p).expand(2, 3), atol=1e-6, rtol=0,
    )

    # Next step: only per-step noise is redrawn; bias and actor goal stay.
    z_step2 = [2.0, 0.0, -1.0, -0.5, 0.0, 2.0]
    inject.queue = [z_step2]
    obs, *_ = env.step(_zero_action(env))
    assert inject.calls[2:] == [(2, 6)]
    e2_p, e2_r = expected(z_step2, STEP_T, STEP_R)
    torch.testing.assert_close(term.bias_pos_b, b_p.expand(2, 3))
    torch.testing.assert_close(term.noise_pos_b, e2_p.expand(2, 3))
    assert torch.equal(term.actor_target_pos, actor_goal)
    true_p, true_q = _true_pose_b(env)
    actor = obs["actor"]
    torch.testing.assert_close(actor[:, OBJ_POS] - true_p, (b_p + e2_p).expand(2, 3), atol=1e-6, rtol=0)
    rot = quat_mul(quat_from_rotvec(e2_r), quat_from_rotvec(b_r)).expand(2, 4)
    _same_rotation(actor[:, OBJ_QUAT], quat_mul(rot, true_q))
    goal_b = _to_base_vec(env, actor_goal - root_p)
    torch.testing.assert_close(actor[:, OBJ_POS] + actor[:, OBJ_TO_GOAL], goal_b, atol=1e-6, rtol=0)
  finally:
    env.close()


@needs_cuda
@pytest.mark.parametrize("d", [1.0, 0.4])
def test_clipping_and_difficulty_scaling(tmp_path, d):
  n = 512
  env = _make_env(_cube_scene(tmp_path), _task(tmp_path), num_envs=n)
  try:
    _set_difficulty(env, d)
    biases, noises = [], []
    for _ in range(4):
      env.reset()
      t = _term(env)
      biases.append(torch.cat((t.bias_pos_b, t.bias_rotvec_b), dim=1))
      for _ in range(3):
        noises.append(torch.cat((t.noise_pos_b, t.noise_rotvec_b), dim=1))
        env.step(_zero_action(env))
    for samples, t_std, r_std in ((biases, BIAS_T, BIAS_R), (noises, STEP_T, STEP_R)):
      s = torch.cat(samples)
      std = torch.tensor([t_std] * 3 + [r_std] * 3, device=DEVICE) * d
      ratio = s / std
      assert ratio.abs().max() <= 2.0 + 1e-5
      at_bound = (ratio.abs() > 2.0 - 1e-5).float().mean(dim=0)
      # P(|z| > 2) = 4.55% per component.
      assert ((at_bound > 0.03) & (at_bound < 0.065)).all(), at_bound
      # Std of a standard normal clamped to +-2 is 0.9594.
      torch.testing.assert_close(ratio.std(dim=0), torch.full((6,), 0.9594, device=DEVICE), atol=0.05, rtol=0)
      assert ratio.mean(dim=0).abs().max() < 0.08
      corr = torch.corrcoef(ratio.T)
      assert (corr - torch.eye(6, device=DEVICE)).abs().max() < 0.1
  finally:
    env.close()


@needs_cuda
def test_bias_fixed_noise_changes_and_reads_consistent(tmp_path):
  env = _make_env(_cube_scene(tmp_path, yaw_deg=30.0), _task(tmp_path), num_envs=8)
  try:
    _set_difficulty(env, 1.0)
    obs, _ = env.reset()
    t = _term(env)
    bias = (t.bias_pos_b.clone(), t.bias_rotvec_b.clone())
    goal = t.actor_target_pos.clone()
    root_p, _ = _base(env)
    goal_b = _to_base_vec(env, goal - root_p)
    previous = None
    for _ in range(5):
      # Repeated reads and every object term agree on one cached pose.
      again = env.get_observations()
      assert torch.equal(again["actor"], obs["actor"])
      assert torch.equal(env.get_observations()["actor"], obs["actor"])
      a = obs["actor"]
      torch.testing.assert_close(a[:, OBJ_POS] - a[:, EE], a[:, TCP_TO_OBJ], atol=1e-6, rtol=0)
      torch.testing.assert_close(a[:, OBJ_POS] + a[:, OBJ_TO_GOAL], goal_b, atol=1e-6, rtol=0)
      assert torch.equal(t.bias_pos_b, bias[0]) and torch.equal(t.bias_rotvec_b, bias[1])
      assert torch.equal(t.actor_target_pos, goal)
      noise = torch.cat((t.noise_pos_b, t.noise_rotvec_b), dim=1).clone()
      if previous is not None:
        assert (noise != previous).all()
      previous = noise
      obs, *_ = env.step(_zero_action(env))
  finally:
    env.close()


@needs_cuda
def test_critic_rewards_metrics_and_sim_stay_clean(tmp_path):
  env = _make_env(_cube_scene(tmp_path, yaw_deg=30.0), _task(tmp_path), num_envs=8)
  try:
    _set_difficulty(env, 1.0)
    env.reset()
    obj = env.scene["cube"]
    cube = SceneEntityCfg("cube")
    for _ in range(3):
      obs, *_ = env.step(_zero_action(env))
      critic, actor = obs["critic"], obs["actor"]
      true_p, true_q = _true_pose_b(env)
      assert torch.equal(critic[:, OBJ_POS], true_p) and torch.equal(critic[:, OBJ_QUAT], true_q)
      assert (actor[:, OBJECT_COLS] != critic[:, OBJECT_COLS]).any(dim=1).all()
      assert torch.equal(actor[:, :24], critic[:, :24])  # joints, gripper, FK stay clean
      goal = task_state.goal(env, "grasp")
      torch.testing.assert_close(
        goal.target_pos - goal.initial_object_pos,
        torch.tensor([0.0, 0.0, LIFT], device=DEVICE).expand(8, 3),
      )
      assert not torch.equal(goal.target_pos, _term(env).actor_target_pos)
      assert torch.equal(goal.command, goal.target_pos)
      # Reward terms recomputed from the clean state match what was paid.
      rm = env.reward_manager
      reach = rm.get_term_cfg("reach")
      i = rm.active_terms.index("reach")
      torch.testing.assert_close(
        rm._step_reward[:, i] / reach.weight, mdp.rewards.reach(env, **reach.params)
      )
      assert torch.equal(
        task_state.object_to_target_dist(env, "grasp", cube),
        torch.linalg.norm(obj.data.root_link_pos_w - goal.target_pos, dim=-1),
      )

    # Updating the perceived pose writes nothing to the simulation.
    qpos = env.sim.data.qpos.clone()
    pose = obj.data.root_link_pose_w.clone()
    for _ in range(3):
      _term(env).compute(dt=0.0)
    assert torch.equal(env.sim.data.qpos, qpos) and torch.equal(obj.data.root_link_pose_w, pose)
  finally:
    env.close()


@needs_cuda
def test_orientation_only_bias_keeps_position(tmp_path):
  env = _make_env(
    _cube_scene(tmp_path, yaw_deg=45.0),
    _task(
      tmp_path,
      step=False,
      persistent_bias__translation_std_m=[0, 0, 0],
      persistent_bias__rotation_std_deg=[20, 20, 20],
    ),
  )
  try:
    _set_difficulty(env, 1.0)
    obs, _ = env.reset()
    for _ in range(3):
      a, c = obs["actor"], obs["critic"]
      assert torch.equal(a[:, OBJ_POS], c[:, OBJ_POS])
      assert torch.equal(a[:, TCP_TO_OBJ], c[:, TCP_TO_OBJ])
      assert torch.equal(a[:, OBJ_TO_GOAL], c[:, OBJ_TO_GOAL])
      assert (a[:, OBJ_QUAT] != c[:, OBJ_QUAT]).any(dim=1).all()
      obs, *_ = env.step(_zero_action(env))
  finally:
    env.close()


@needs_cuda
def test_subset_and_auto_resets(tmp_path):
  env = _make_env(_cube_scene(tmp_path), _task(tmp_path), num_envs=4)
  try:
    _set_difficulty(env, 1.0)
    env.reset()
    env.step(_zero_action(env))
    t = _term(env)

    def snapshot():
      return {
        k: getattr(t, k).clone()
        for k in ("bias_pos_b", "bias_rotvec_b", "noise_pos_b", "noise_rotvec_b", "pos_w", "quat_w", "actor_target_pos")
      }

    # Explicit subset reset at the same global step as the last observation.
    before = snapshot()
    step_count = env.common_step_counter
    obs, _ = env.reset(env_ids=torch.tensor([1], device=DEVICE))
    assert env.common_step_counter == step_count
    after = snapshot()
    for k in before:
      assert torch.equal(after[k][[0, 2, 3]], before[k][[0, 2, 3]]), k
      assert not torch.equal(after[k][1], before[k][1]), k
    lift = torch.tensor([0.0, 0.0, LIFT], device=DEVICE)
    torch.testing.assert_close(t.actor_target_pos[1], t.pos_w[1] + lift)
    torch.testing.assert_close(
      task_state.goal(env, "grasp").target_pos[1], env.scene["cube"].data.root_link_pos_w[1] + lift
    )
    assert torch.equal(obs["actor"][1, OBJ_POS], _actor_obj_pos_b(env)[1])

    # Auto-reset inside step(): env 2 times out; its new episode's bias and
    # goal are captured before this step's observation.
    env.episode_length_buf[2] = env.max_episode_length - 1
    before = snapshot()
    obs, *_ = env.step(_zero_action(env))
    after = snapshot()
    for k in ("bias_pos_b", "bias_rotvec_b", "actor_target_pos"):
      assert torch.equal(after[k][[0, 1, 3]], before[k][[0, 1, 3]]), k
      assert not torch.equal(after[k][2], before[k][2]), k
    assert (after["noise_pos_b"] != before["noise_pos_b"]).all()
    torch.testing.assert_close(t.actor_target_pos[2], t.pos_w[2] + lift)
    assert torch.equal(obs["actor"][:, OBJ_POS], _actor_obj_pos_b(env))
    assert env.episode_length_buf[2] == 0
  finally:
    env.close()


def _actor_obj_pos_b(env):
  return mdp.observations.perceived_object_pose(env, PERCEPTION_NAME)[:, :3]


@needs_cuda
@needs_jenga
def test_goals_captured_after_height_correction(tmp_path):
  task = _task(tmp_path, base=FRICTION_TASK)
  env = _make_env(_jenga_scene(tmp_path), task, num_envs=4)
  try:
    _set_difficulty(env, 1.0)
    inject = _Injector(_term(env))
    z_bias = [2.0, -2.0, 1.0, 1.0, 0.0, -1.0]
    z_step = [1.0, 0.5, -1.5, 0.0, 1.0, 0.5]
    inject.queue = [z_bias, z_step]
    env.reset()
    obj = env.scene["jenga"]
    true_pos = obj.data.root_link_pos_w
    # Captured from the variant-seated pose: lowest collision point on the support.
    seat = env.event_manager.get_term_cfg("reset_jenga_xy")
    lowest = object_size.lowest_point_z(
      seat.func._points[seat.func._variant], obj.data.root_link_quat_w
    )
    torch.testing.assert_close(
      true_pos[:, 2] + lowest - env.scene.env_origins[:, 2],
      torch.full((4,), seat.params["support_height"] + seat.params["clearance"], device=DEVICE),
      atol=1e-5, rtol=0,
    )
    assert torch.unique(object_size.assigned_scale(env, "jenga")).numel() > 1
    goal, term = task_state.goal(env, "grasp"), _term(env)
    lift = torch.tensor([0.0, 0.0, LIFT], device=DEVICE)
    assert torch.equal(goal.initial_object_pos, true_pos)
    torch.testing.assert_close(goal.target_pos, true_pos + lift)
    b_p = torch.tensor(z_bias[:3], device=DEVICE) * BIAS_T
    e_p = torch.tensor(z_step[:3], device=DEVICE) * STEP_T
    root_p, _ = _base(env)
    torch.testing.assert_close(
      _to_base_vec(env, term.actor_target_pos - lift - root_p) - _to_base_vec(env, true_pos - root_p),
      (b_p + e_p).expand(4, 3), atol=1e-6, rtol=0,
    )
  finally:
    env.close()
