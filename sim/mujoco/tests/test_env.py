"""Simulation tests (need a CUDA device for MuJoCo Warp)."""

import json

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK, make_grasp_env_cfg
from tabletop_sim.tasks.grasp.mdp import GraspGoalCommand, task_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = "cuda:0"
NUM_ENVS = 4
CUBE_POSES = [[-0.25, y, 0.015, 1.0, 0.0, 0.0, 0.0] for y in (-0.06, 0.0, 0.06)]


def _scene_file(tmp_path_factory, **robot_overrides) -> str:
  data = json.loads(DEFAULT_SCENE.read_text())
  data["robot"].update(robot_overrides)
  data["objects"][1]["initial_poses"] = CUBE_POSES
  path = tmp_path_factory.mktemp("scene") / "scene.json"
  path.write_text(json.dumps(data))
  return str(path)


def _make_env(scene, task=DEFAULT_TASK, action_space=None, num_envs=NUM_ENVS):
  cfg = make_grasp_env_cfg(scene, task, action_space, num_envs)
  cfg.seed = 0
  return ManagerBasedRlEnv(cfg, device=DEVICE)


class _TaskView:
  """Reads task quantities through the env's own resolved term parameters."""

  def __init__(self, env):
    self.env = env
    p = env.reward_manager.get_term_cfg("stable_hold").params
    self.object_cfg = p["object_cfg"]
    self.tcp_cfg = env.reward_manager.get_term_cfg("reach").params["tcp_cfg"]
    self.arm_cfg = env.reward_manager.get_term_cfg("arm_speed").params["asset_cfg"]
    self.gripper_cfg = env.reward_manager.get_term_cfg("gripper_speed").params["asset_cfg"]
    self.limit_cfg = env.reward_manager.get_term_cfg("joint_pos_limits").params["asset_cfg"]
    self.grasp_params = {
      k: p[k]
      for k in ("sensor_names", "finger_cfg", "finger_outward_axes", "min_force", "max_angle_deg")
    }
    self.lift_threshold = p["lift_threshold"]
    self.hold_idx = env.reward_manager.active_terms.index("stable_hold")
    self.hold_weight = env.reward_manager.get_term_cfg("stable_hold").weight

  @property
  def goal(self) -> GraspGoalCommand:
    return task_state.goal(self.env, "grasp")

  @property
  def hold_reward(self):
    """This step's unweighted stable_hold reward, as the reward manager saw it."""
    return self.env.reward_manager._step_reward[:, self.hold_idx] / self.hold_weight

  def joint_vel(self, cfg):
    return self.env.scene["robot"].data.joint_vel[:, cfg.joint_ids]

  @property
  def object_pos_w(self):
    return task_state.object_pose_w(self.env, self.object_cfg)[0]

  @property
  def tcp_pos_w(self):
    return task_state.tcp_pose_w(self.env, self.tcp_cfg)[0]

  @property
  def target_pos(self):
    return self.goal.target_pos

  @property
  def initial_object_pos(self):
    return self.goal.initial_object_pos

  @property
  def is_grasped(self):
    return task_state.is_grasped(self.env, **self.grasp_params)

  @property
  def is_lifted(self):
    return task_state.is_lifted(self.env, "grasp", self.object_cfg, self.lift_threshold)

  def object_speeds(self):
    return task_state.object_speeds(self.env, self.object_cfg)


def _tracker(env) -> _TaskView:
  return _TaskView(env)


@pytest.fixture(scope="module")
def ee_env(tmp_path_factory):
  env = _make_env(_scene_file(tmp_path_factory))
  yield env
  env.close()


def test_observation_layout(ee_env):
  obs, _ = ee_env.reset()
  assert obs["actor"].shape == (NUM_ENVS, 37)
  assert obs["critic"].shape == (NUM_ENVS, 38)
  assert ee_env.action_manager.total_action_dim == 7
  assert ee_env.max_episode_length == 300
  t = _tracker(ee_env)
  actor = obs["actor"]
  # object_pose z in root frame (base at z=0) and obj_to_goal = +lift height.
  torch.testing.assert_close(actor[:, 26], t.object_pos_w[:, 2], atol=1e-4, rtol=0)
  torch.testing.assert_close(
    actor[:, 34:37], torch.tensor([0.0, 0.0, 0.03], device=DEVICE).expand(NUM_ENVS, 3),
    atol=1e-4, rtol=0,
  )
  # Closure at the ready pose (carriages at 0.026 of 0.044); nothing grasped.
  torch.testing.assert_close(
    actor[:, 23], torch.full((NUM_ENVS,), 1 - 0.026 / 0.044, device=DEVICE), atol=1e-3, rtol=0
  )
  torch.testing.assert_close(obs["critic"][:, :37], actor)
  assert (obs["critic"][:, 37] == 0).all()


def test_resets_sample_initial_poses(ee_env):
  candidates = torch.tensor(CUBE_POSES, device=DEVICE)[:, :3]
  seen = set()
  for _ in range(6):
    ee_env.reset()
    pos = _tracker(ee_env).object_pos_w
    dist = torch.cdist(pos, candidates)
    assert (dist.min(dim=1).values < 1e-3).all()
    seen.update(dist.argmin(dim=1).tolist())
    torch.testing.assert_close(_tracker(ee_env).initial_object_pos, pos, atol=1e-4, rtol=0)
  assert len(seen) > 1


def test_reward_matches_formula(ee_env):
  ee_env.reset()
  t = _tracker(ee_env)
  for _ in range(10):
    action = torch.rand(NUM_ENVS, 7, device=DEVICE) * 2 - 1
    _, reward, terminated, truncated, _ = ee_env.step(action)
    live = ~(terminated | truncated)
    d_tcp = torch.linalg.norm(t.object_pos_w - t.tcp_pos_w, dim=-1)
    d_goal = torch.linalg.norm(t.object_pos_w - t.target_pos, dim=-1)
    grasped = t.is_grasped.float()
    lifted = (d_goal <= 0.01).float()
    lin, ang = t.object_speeds()
    arm_excess = ((t.joint_vel(t.arm_cfg).abs() - 0.5).clamp(min=0) ** 2).sum(-1)
    q = t.env.scene["robot"].data.joint_pos[:, t.limit_cfg.joint_ids]
    soft = t.env.scene["robot"].data.soft_joint_pos_limits[:, t.limit_cfg.joint_ids]
    outside = ((soft[..., 0] - q).clamp(min=0) + (q - soft[..., 1]).clamp(min=0)).sum(-1)
    expected = (
      0.5 * (1 - torch.tanh(5 * d_tcp))
      + 1.0 * grasped
      + 1.0 * grasped * (1 - torch.tanh(30 * d_goal))
      + 1.0 * grasped * lifted * (1 - torch.tanh(5 * lin)) * (1 - torch.tanh(2 * ang))
      - 0.1 * arm_excess
      - 10.0 * outside
    ) / 10.0
    assert torch.isfinite(reward).all()
    # mjlab evaluates rewards before its post-step forward pass, so body poses
    # lag the state read here by one physics substep.
    torch.testing.assert_close(reward[live], expected[live], atol=5e-4, rtol=0)


def _scripted_action(t, i, release=False):
  obj = t.object_pos_w
  if i < 30:
    goal, grip = obj + torch.tensor([0.0, 0.0, 0.06], device=DEVICE), 1.0
  elif i < 50:
    goal, grip = obj.clone(), 1.0
  elif i < 65:
    goal, grip = obj.clone(), -1.0
  else:
    goal, grip = t.target_pos.clone(), -1.0
  action = torch.zeros(t.env.num_envs, 7, device=DEVICE)
  action[:, :3] = ((goal - t.tcp_pos_w) / 0.01).clamp(-1, 1)
  action[:, 6] = 1.0 if release else grip
  return action


class _HoldMeans:
  """Independent per-env episode sum / step count of the unweighted stable_hold
  reward, to check mjlab's ``mean_hold_score`` against."""

  def __init__(self, t):
    self.t = t
    self.sum = torch.zeros(t.env.num_envs, device=DEVICE)
    self.count = torch.zeros(t.env.num_envs, device=DEVICE)

  def record_step(self):
    self.sum += self.t.hold_reward
    self.count += 1

  def mean(self, ids):
    return self.sum[ids] / self.count[ids]

  def reset(self, ids):
    self.sum[ids] = 0.0
    self.count[ids] = 0.0


def _metric_state(env, name="mean_hold_score"):
  m = env.metrics_manager
  return m._episode_sums[name].clone(), m._step_count.clone()


def _scripted_grasp(env, steps=110):
  """Hover, descend, close, lift to target, hold. Returns per-step logs and the
  independent hold-score tracker."""
  t = _tracker(env)
  env.reset()
  means = _HoldMeans(t)
  logs = []
  for i in range(steps):
    sums_before, counts_before = _metric_state(env)
    _, reward, terminated, truncated, extras = env.step(_scripted_action(t, i))
    means.record_step()
    logs.append(
      dict(step=i, grasped=t.is_grasped.clone(), lifted=t.is_lifted.clone(),
           terminated=terminated.clone(), truncated=truncated.clone(),
           reward=reward.clone(), hold=t.hold_reward.clone(),
           running_mean=means.mean(slice(None)).clone(),
           metric_sum_before=sums_before, metric_count_before=counts_before,
           log=dict(extras["log"]))
    )
  return logs, means


def test_no_success_machinery(ee_env):
  assert "success" not in ee_env.reward_manager.active_terms
  assert ee_env.termination_manager.active_terms == ["time_out"]
  assert "success" not in ee_env.metrics_manager.active_terms


def test_scripted_grasp_holds_until_time_limit(ee_env):
  logs, means = _scripted_grasp(ee_env, steps=ee_env.max_episode_length)
  assert len(logs) == 300
  # Contact-force grasp detection: not grasped while open, grasped once closed.
  assert not any(log["grasped"].any() for log in logs[:45])
  assert logs[62]["grasped"].all()
  assert any(log["lifted"].all() for log in logs[65:85])
  # Holding at the target never ends the episode; only the time limit does.
  held = [log for log in logs[:-1] if (log["grasped"] & log["lifted"]).all()]
  assert len(held) > 150
  assert all((log["hold"] > 0.8).all() for log in held[20:])
  assert not any((log["terminated"] | log["truncated"]).any() for log in logs[:-1])
  assert logs[-1]["truncated"].all() and not logs[-1]["terminated"].any()
  # Hold reward is still paid every held step, not just once.
  assert (logs[-2]["hold"] > 0.8).all() and (logs[-2]["reward"] > 0.3).all()

  # mean_hold_score is the per-env mean of the unweighted stable_hold reward
  # over all 300 steps, including the final pre-reset one.
  final = logs[-1]
  assert (final["metric_count_before"] == 299).all()
  torch.testing.assert_close(final["metric_sum_before"], means.sum - final["hold"])
  per_env = means.mean(slice(None))
  logged = final["log"]["Episode_Metrics/mean_hold_score"]
  torch.testing.assert_close(logged, per_env.mean(), atol=1e-5, rtol=0)
  without_final = (means.sum - final["hold"]) / 299
  assert (logged - without_final.mean()).abs() > 1e-4
  assert 0.0 < logged < 1.0
  assert "Episode_Metrics/grasped_frac" in final["log"]
  assert "Episode_Metrics/max_lift" in final["log"]
  # The auto-reset zeroed every env's accumulator.
  sums, counts = _metric_state(ee_env)
  assert (sums == 0).all() and (counts == 0).all()


def test_partial_reset_leaves_other_envs(ee_env):
  logs, means = _scripted_grasp(ee_env, steps=100)
  t = _tracker(ee_env)
  keep = torch.tensor([0, 2, 3], device=DEVICE)
  target_before = t.target_pos[keep].clone()
  initial_before = t.initial_object_pos[keep].clone()
  sums_before, counts_before = _metric_state(ee_env)
  assert (logs[-1]["hold"] > 0).all()

  _, extras = ee_env.reset(env_ids=torch.tensor([1], device=DEVICE))
  # The logged score is env 1's own episode mean, unmixed with the others.
  torch.testing.assert_close(
    extras["log"]["Episode_Metrics/mean_hold_score"], means.mean(1), atol=1e-6, rtol=0
  )
  sums, counts = _metric_state(ee_env)
  torch.testing.assert_close(sums[keep], sums_before[keep])
  assert (counts[keep] == 100).all() and counts[1] == 0 and sums[1] == 0
  means.reset(1)
  torch.testing.assert_close(t.target_pos[keep], target_before)
  torch.testing.assert_close(t.initial_object_pos[keep], initial_before)
  torch.testing.assert_close(t.initial_object_pos[1], t.object_pos_w[1], atol=1e-4, rtol=0)
  torch.testing.assert_close(
    t.target_pos[1] - t.initial_object_pos[1],
    torch.tensor([0.0, 0.0, 0.03], device=DEVICE), atol=1e-6, rtol=0,
  )

  # Envs 0, 2, 3 time out together at step 300; env 1 is 100 steps behind.
  for i in range(100, 300):
    _, _, terminated, truncated, extras = ee_env.step(_scripted_action(t, i))
    means.record_step()
    if i < 299:
      assert not (terminated | truncated).any()
  assert truncated.tolist() == [True, False, True, True]
  torch.testing.assert_close(
    extras["log"]["Episode_Metrics/mean_hold_score"], means.mean(keep).mean(),
    atol=1e-5, rtol=0,
  )
  sums, counts = _metric_state(ee_env)
  assert (counts[keep] == 0).all() and counts[1] == 200
  torch.testing.assert_close(sums[1], means.sum[1], atol=1e-4, rtol=0)


def _limit_joints(env):
  cfg = env.reward_manager.get_term_cfg("joint_pos_limits").params["asset_cfg"]
  robot = env.scene["robot"]
  model_range = torch.tensor(
    env.sim.mj_model.jnt_range[[robot.indexing.joint_ids[i].item() for i in cfg.joint_ids]],
    device=DEVICE, dtype=torch.float,
  )
  return cfg, robot, model_range


def test_soft_limits_are_central_90_percent(ee_env):
  cfg, robot, model_range = _limit_joints(ee_env)
  assert len(cfg.joint_ids) == 8  # 6 arm + 2 carriages
  soft = robot.data.soft_joint_pos_limits[:, cfg.joint_ids]
  mid = model_range.mean(-1)
  half = 0.5 * (model_range[:, 1] - model_range[:, 0])
  expected = torch.stack((mid - 0.9 * half, mid + 0.9 * half), dim=-1)
  torch.testing.assert_close(soft, expected.expand_as(soft), atol=1e-6, rtol=0)
  torch.testing.assert_close(
    robot.data.joint_pos_limits[:, cfg.joint_ids], model_range.expand_as(soft), atol=1e-6, rtol=0
  )


def test_joint_pos_limit_penalty_sign_and_normalization(ee_env):
  ee_env.reset()
  cfg, robot, model_range = _limit_joints(ee_env)
  ids = torch.as_tensor(cfg.joint_ids, device=DEVICE)
  soft = robot.data.soft_joint_pos_limits[0, ids]
  idx = ee_env.reward_manager.active_terms.index("joint_pos_limits")
  weight = ee_env.reward_manager.get_term_cfg("joint_pos_limits").weight
  assert weight == pytest.approx(-1.0)

  def penalty_at(q_sel):
    q = robot.data.joint_pos.clone()
    q[:, ids] = q_sel
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    ee_env.sim.forward()
    ee_env.reward_manager.compute(dt=ee_env.step_dt)
    return ee_env.reward_manager._step_reward[:, idx].clone()

  # Exactly at either soft bound, and anywhere inside: zero.
  for q_sel in (soft[:, 0], soft[:, 1], soft.mean(-1)):
    assert (penalty_at(q_sel.expand(NUM_ENVS, -1)) == 0).all()
  # Per-joint distance outside the soft range, summed (not squared), times -1.
  q_sel = soft.mean(-1).clone()
  q_sel[0] = soft[0, 1] + 0.1     # arm joint above
  q_sel[1] = soft[1, 0] - 0.05    # arm joint below
  q_sel[6:] = model_range[6:, 1]  # both carriages at the hard open limit
  outside = 0.1 + 0.05 + (model_range[6:, 1] - soft[6:, 1]).sum()
  torch.testing.assert_close(
    penalty_at(q_sel.expand(NUM_ENVS, -1)),
    torch.full((NUM_ENVS,), -1.0 * outside.item(), device=DEVICE), atol=1e-5, rtol=0,
  )
  ee_env.reset()


def test_delta_joint_action(tmp_path_factory):
  env = _make_env(_scene_file(tmp_path_factory), action_space="delta_joint_abs_gripper")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (NUM_ENVS, 37)
    arm = env.action_manager.get_term("arm")
    q0 = obs["actor"][:, :6].clone()
    action = torch.zeros(NUM_ENVS, 7, device=DEVICE)
    action[:, 0] = 1.0
    action[:, 1] = -5.0  # clipped to -1
    env.step(action)
    limits = arm._entity.data.soft_joint_pos_limits[:, arm._joint_ids]
    expected = (q0 + action[:, :6].clamp(-1, 1) * 0.1).clamp(limits[..., 0], limits[..., 1])
    torch.testing.assert_close(arm._target, expected, atol=1e-5, rtol=0)
    # The (heavily damped, system-identified) base joint starts moving toward it.
    q1 = env.observation_manager.compute()["actor"][:, 0]
    assert ((q1 - q0[:, 0]) > 0.003).all()
    # Driving a joint at its lower bound: the target stops at the 90% soft
    # limit, not the model limit.
    hard = arm._entity.data.joint_pos_limits[:, arm._joint_ids]
    action.zero_()
    action[:, 3] = -1.0
    for _ in range(20):
      env.step(action)
    torch.testing.assert_close(arm._target[:, 3], limits[:, 3, 0], atol=1e-6, rtol=0)
    assert (limits[:, 3, 0] > hard[:, 3, 0] + 0.1).all()
  finally:
    env.close()


@pytest.mark.parametrize("action_space", ["delta_joint_delta_gripper", "ee_delta_pose_delta_gripper"])
def test_delta_gripper_action(tmp_path_factory, action_space):
  env = _make_env(_scene_file(tmp_path_factory), action_space=action_space)
  try:
    env.reset()
    gripper = env.action_manager.get_term("gripper")
    assert env.action_manager.total_action_dim == 7

    def carriage():
      return gripper._entity.data.joint_pos[:, gripper._joint_ids].clone()

    action = torch.zeros(NUM_ENVS, 7, device=DEVICE)
    action[:, 6] = -0.5
    q0 = carriage()
    env.step(action)
    torch.testing.assert_close(gripper._target, q0 - 0.005, atol=1e-6, rtol=0)
    # Target follows the measured position, and is clamped to the open limit.
    action[:, 6] = 5.0
    for _ in range(8):
      q = carriage()
      env.step(action)
      torch.testing.assert_close(gripper._target, (q + 0.01).clamp(max=0.044), atol=1e-6, rtol=0)
    assert (gripper._target == 0.044).all()
    assert (carriage() > 0.035).all()
  finally:
    env.close()


def test_trossen_embodiment(tmp_path_factory):
  scene = _scene_file(tmp_path_factory, embodiment="trossen_wxai_base")
  env = _make_env(scene, num_envs=2)
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 37)
    for _ in range(5):
      obs, reward, *_ = env.step(torch.zeros(2, 7, device=DEVICE))
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
  finally:
    env.close()

