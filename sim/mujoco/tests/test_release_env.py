"""Release env on a small start-state dataset (200 rows, enough to cover
every size variant) made by the real generator (grasp policy + scripted
transport) on the tower scene. Needs CUDA and the upright-block grasp
checkpoint (UPRIGHT_GRASP_CHECKPOINT, the policy the tower scene was built
for); skips otherwise."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from tabletop_sim import CONFIGS_DIR, PACKAGE_ROOT
from tabletop_sim.scene import load_scene_config
from tabletop_sim.scene.release_config import load_release_task_config
from tabletop_sim.tasks.grasp.mdp import task_state
from tabletop_sim.tasks.release import env_cfg as rc
from tabletop_sim.tasks.release import starts
from tabletop_sim.tasks.release.mdp import current_rows, placement, rewards, terminations

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = "cuda:0"
SCENE = CONFIGS_DIR / "release" / "scenes" / "tabletop_jenga_tower_wxai.json"
TASK = CONFIGS_DIR / "release" / "tasks" / "release_tower_wxai.json"
UPRIGHT_GRASP_CHECKPOINT = (
  PACKAGE_ROOT / "logs/rsl_rl/joint_friction-no_slip_interpolation/2026-11-19_14-33-28/model_999.pt"
)
NUM_ENVS = 32


def _grasp_checkpoint() -> Path:
  return UPRIGHT_GRASP_CHECKPOINT


def _task_file(tmp: Path, dataset: Path, **edits) -> Path:
  data = json.loads(TASK.read_text())
  data["start"]["dataset"] = str(dataset)
  for key, value in edits.items():
    node = data
    *parents, leaf = key.split(".")
    for p in parents:
      node = node[p]
    node[leaf] = value
  path = tmp / f"task_{abs(hash(tuple(sorted(edits.items())))) % 10**8}.json"
  path.write_text(json.dumps(data))
  return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
  ckpt = _grasp_checkpoint()
  if not ckpt.is_file():
    pytest.skip(f"grasp checkpoint {ckpt} not found")
  spec = importlib.util.spec_from_file_location(
    "generate_release_starts", PACKAGE_ROOT / "scripts" / "generate_release_starts.py"
  )
  gen = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(gen)  # type: ignore[union-attr]
  out = tmp_path_factory.mktemp("starts") / "starts.pt"
  args = Namespace(
    scene=str(SCENE), task=str(TASK), grasp_checkpoint=str(ckpt),
    grasp_task=str(ckpt.parent / "params" / "task.json"), out=str(out), rows=200, num_envs=256,
    difficulty=1.0, hold_steps=10, grasp_timeout=250, phase_timeout=120, seed=0, device=DEVICE,
  )
  torch.manual_seed(0)
  gen.Generator(args).run()
  return out


def _env(tmp: Path, dataset: Path, num_envs: int = NUM_ENVS, **edits) -> ManagerBasedRlEnv:
  task = load_release_task_config(_task_file(tmp, dataset, **edits))
  cfg = rc.make_release_env_cfg_from_configs(load_scene_config(SCENE), task, None, num_envs)
  cfg.seed = 0
  return ManagerBasedRlEnv(cfg, device=DEVICE)


def _grasp_params(env) -> dict:
  p = env.termination_manager.get_term_cfg("object_dropped").params
  return {k: p[k] for k in ("sensor_names", "finger_cfg", "finger_outward_axes", "min_force", "max_angle_deg")}


def test_dataset_rows_are_held_and_inside_the_ball(dataset):
  data = torch.load(dataset, weights_only=False)
  meta = json.loads(starts.meta_path(dataset).read_text())
  task = load_release_task_config(TASK)
  assert data["variant_scale"].shape[0] == 200 and meta["rows"] == 200
  assert meta["kept"] >= 200 and meta["attempts"] >= meta["kept"]
  delta = data["object_root_state"][:, :3] - data["goal_pose"][:, :3]
  assert float(torch.linalg.norm(delta, dim=-1).max()) <= task.start.ball_radius + 1e-6
  assert float(delta[:, 2].min()) >= task.start.min_height_above_goal - 0.002
  assert data["support_names"] == ["base_block"]


def test_reset_restores_rows_and_keeps_the_grasp(tmp_path, dataset):
  env = _env(tmp_path, dataset)
  env.reset()
  st, rows = current_rows(env)
  obj, robot = env.scene["jenga"], env.scene["robot"]
  want = st.tensors["object_root_state"][rows, :3] + env.scene.env_origins
  assert torch.allclose(obj.data.root_link_pos_w, want, atol=1e-5)
  assert torch.allclose(robot.data.joint_pos, st.tensors["robot_joint_pos"][rows], atol=1e-5)
  gp = _grasp_params(env)
  assert float(task_state.is_grasped(env, **gp).float().mean()) == 1.0

  # Hold still with the gripper closed: the block stays in the hand (the held
  # gripper action keeps squeezing) and nothing touches the table or the base block.
  act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)
  act[:, -1] = -1.0
  collision = rewards.robot_collision
  names = env.reward_manager.get_term_cfg("robot_collision").params["sensor_names"]
  touched = torch.zeros(env.num_envs, dtype=torch.bool, device=DEVICE)
  for _ in range(10):
    _, _, terminated, _, _ = env.step(act)
    touched |= collision(env, names) > 0
    assert not terminated.any()
  assert float(task_state.is_grasped(env, **gp).float().mean()) >= 0.95
  assert float(touched.float().mean()) <= 0.05
  env.close()


def test_actor_and_critic_goal_obs_agree_without_noise(tmp_path, dataset):
  env = _env(
    tmp_path,
    dataset,
    **{
      "goal_pose_noise.enabled": False,
      "domain_randomization.object_pose_noise.persistent_bias.enabled": False,
      "domain_randomization.object_pose_noise.per_step.enabled": False,
    },
  )
  obs, _ = env.reset()
  om = env.observation_manager
  def term(group, name):
    names = om.active_terms[group]
    dims = [d[0] for d in om.group_obs_term_dim[group]]
    start = sum(dims[: names.index(name)])
    return obs[group][:, start : start + dims[names.index(name)]]
  for name in ("object_pose", "tcp_to_obj_pose", "obj_to_goal_pose"):
    assert torch.allclose(term("actor", name), term("critic", name), atol=1e-5), name
  # The goal observation uses the symmetric goal nearest the object, so at the
  # start its rotation part is a small turn (TCP noise + slip only).
  q = term("critic", "obj_to_goal_pose")[:, 3:]
  angle = 2 * torch.acos(q[:, 0].clamp(max=1.0))
  assert float(torch.rad2deg(angle).max()) < 25.0
  env.close()


def test_place_reward_and_topple_termination(tmp_path, dataset):
  env = _env(tmp_path, dataset, num_envs=4)
  env.reset()
  goal = env.command_manager.get_term(rc.COMMAND_NAME)
  obj = env.scene["jenga"]
  ocfg = env.reward_manager.get_term_cfg("place").params["object_cfg"]
  # Put the object exactly at a symmetric copy of the goal: place = 1.
  pose = torch.cat((goal.goal_pos, placement.closest_symmetric_quat(goal.goal_quat, obj.data.root_link_quat_w, "box")), -1)
  obj.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  r = rewards.place(env, rc.COMMAND_NAME, ocfg, 30.0, 2.0)
  assert torch.allclose(r, torch.ones_like(r), atol=1e-4)
  # Push the base block 5 cm: toppled (displacement > 3 cm) and disturbed.
  base = env.scene["base_block"]
  moved = torch.cat((base.data.root_link_pos_w + torch.tensor([0.05, 0, 0], device=DEVICE), base.data.root_link_quat_w), -1)
  base.write_root_link_pose_to_sim(moved)
  env.sim.forward()
  assert bool(terminations.support_toppled(env, rc.COMMAND_NAME, ("base_block",), 30.0, 0.03).all())
  assert float(rewards.support_disturb(env, rc.COMMAND_NAME, ("base_block",), 50.0, 5.0).min()) > 0.9
  env.close()


def test_termination_counts_are_running_totals(tmp_path, dataset):
  """Curriculum/termination_counts/<term> is the total number of episodes each
  term ended since the env was built (not a per-step average)."""
  env = _env(tmp_path, dataset)
  env.reset()
  act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=DEVICE)
  act[:, -1] = 1.0  # open at once: the block falls off and episodes end as object_dropped
  dropped = 0
  previous = 0
  for _ in range(40):
    env.step(act)
    dropped += int(env.termination_manager.get_term("object_dropped").sum())
    totals = env.curriculum_manager._curriculum_state["termination_counts"]
    assert totals["object_dropped"] >= previous
    previous = totals["object_dropped"]
  assert dropped > 0
  assert totals["object_dropped"] == dropped
  env.close()


def test_release_moving_penalizes_opening_on_a_moving_object(tmp_path, dataset):
  """release_moving fires only on the step the gripper is commanded open, and is
  about 0 when the held object is still and large when it moves."""
  env = _env(tmp_path, dataset, num_envs=16)
  ocfg = env.reward_manager.get_term_cfg("release_moving").params["object_cfg"]
  dim = env.action_manager.total_action_dim

  def penalty_on_open(arm_action):
    env.reset()
    hold = torch.zeros(env.num_envs, dim, device=DEVICE)
    hold[:, -1] = -1.0
    for _ in range(15):  # settle in the hand, gripper closed
      env.step(hold)
    move = hold.clone()
    move[:, :6] = torch.tensor(arm_action, device=DEVICE)
    for _ in range(3):
      env.step(move)
    assert float(rewards.release_moving(env, "gripper", ocfg).abs().max()) == 0.0  # no switch yet
    opened = move.clone()
    opened[:, -1] = 1.0
    env.step(opened)
    at_open = rewards.release_moving(env, "gripper", ocfg).clone()
    env.step(opened)
    after = rewards.release_moving(env, "gripper", ocfg)
    assert float(after.abs().max()) == 0.0  # only the switching step counts
    return at_open

  still = penalty_on_open([0.0] * 6)
  moving = penalty_on_open([0.0, 0.0, -1.0, 0.0, 0.0, 1.0])  # descend and turn at full speed
  assert float(still.median()) < 0.1
  assert float(moving.median()) > 0.3
  env.close()
