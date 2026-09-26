"""Simulation tests (need a CUDA device for MuJoCo Warp)."""

import json
import math

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK, make_grasp_env_cfg
from tabletop_sim.tasks.grasp.mdp import GraspTracker

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


def _tracker(env) -> GraspTracker:
  term = env.command_manager.get_term("grasp")
  assert isinstance(term, GraspTracker)
  return term


@pytest.fixture(scope="module")
def ee_env(tmp_path_factory):
  env = _make_env(_scene_file(tmp_path_factory))
  yield env
  env.close()


def test_observation_layout(ee_env):
  obs, _ = ee_env.reset()
  assert obs["actor"].shape == (NUM_ENVS, 38)
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
  assert (actor[:, 37] == 0).all()


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
    lin = torch.linalg.norm(t.object.data.root_link_lin_vel_w, dim=-1)
    ang = torch.linalg.norm(t.object.data.root_link_ang_vel_w, dim=-1)
    expected = (
      0.5 * (1 - torch.tanh(5 * d_tcp.clamp(min=0.05)))
      + 1.0 * grasped
      + 1.0 * grasped * (1 - torch.tanh(30 * d_goal))
      + 1.0 * grasped * lifted * (1 - torch.tanh(5 * lin)) * (1 - torch.tanh(2 * ang))
      + 5.0 * t.success.float()
    ) / 10.0
    assert torch.isfinite(reward).all()
    # mjlab evaluates rewards before its post-step forward pass, so body poses
    # lag the state read here by one physics substep.
    torch.testing.assert_close(reward[live], expected[live], atol=5e-4, rtol=0)


def _scripted_grasp(env, steps=110):
  """Hover, descend, close, lift to target, hold. Returns per-step logs."""
  t = _tracker(env)
  env.reset()
  logs = []
  for i in range(steps):
    obj = t.object_pos_w
    if i < 30:
      goal, grip = obj + torch.tensor([0.0, 0.0, 0.06], device=DEVICE), 1.0
    elif i < 50:
      goal, grip = obj.clone(), 1.0
    elif i < 65:
      goal, grip = obj.clone(), -1.0
    else:
      goal, grip = t.target_pos.clone(), -1.0
    action = torch.zeros(env.num_envs, 7, device=DEVICE)
    action[:, :3] = ((goal - t.tcp_pos_w) / 0.01).clamp(-1, 1)
    action[:, 6] = grip
    _, reward, terminated, _, _ = env.step(action)
    logs.append(
      dict(step=i, grasped=t.is_grasped.clone(), lifted=t.is_lifted.clone(),
           terminated=terminated.clone(), reward=reward.clone())
    )
  return logs


def test_scripted_grasp_succeeds(ee_env):
  logs = _scripted_grasp(ee_env)
  # Contact-force grasp detection: not grasped while open, grasped once closed.
  assert not any(log["grasped"].any() for log in logs[:45])
  assert logs[62]["grasped"].all()
  assert any(log["lifted"].all() for log in logs[65:85])
  # Success = grasped + lifted + static for ceil(1 s * 20 Hz) = 20 steps; the env
  # terminates on success and the success step carries the +w_success bonus.
  term_steps = [log["step"] for log in logs if log["terminated"].all()]
  assert term_steps, "scripted grasp never reached success"
  success_log = logs[term_steps[0]]
  assert (success_log["reward"] > 0.5).all()


def test_delta_joint_action(tmp_path_factory):
  env = _make_env(_scene_file(tmp_path_factory), action_space="delta_joint_abs_gripper")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (NUM_ENVS, 38)
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
  finally:
    env.close()


def test_minimal_obs_and_trossen_embodiment(tmp_path_factory):
  task = json.loads(DEFAULT_TASK.read_text())
  task["use_minimal_obs"] = True
  task_path = tmp_path_factory.mktemp("task") / "task.json"
  task_path.write_text(json.dumps(task))
  scene = _scene_file(tmp_path_factory, embodiment="trossen_wxai_base")
  env = _make_env(scene, task=str(task_path), num_envs=2)
  try:
    obs, _ = env.reset()
    # qpos 8 + qvel 8 + ee_pose 7 + closure 1 + tcp_to_goal 3 + obj_initial_pose 7.
    assert obs["actor"].shape == (2, 34)
    for _ in range(5):
      obs, reward, *_ = env.step(torch.zeros(2, 7, device=DEVICE))
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
  finally:
    env.close()


def test_hold_steps_rounding():
  cfg = make_grasp_env_cfg()
  assert cfg.commands["grasp"].hold_steps == math.ceil(1.0 * 20)
  assert cfg.scale_rewards_by_dt is False
  assert cfg.decimation == 25
