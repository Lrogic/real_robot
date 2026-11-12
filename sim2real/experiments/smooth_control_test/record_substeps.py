"""Record the arm's motion at every 2 ms physics substep while the trained policy runs in sim.

Answers whether the sim arm moves continuously or stops and goes within each 50 ms control
period. Same env as scripts/play_with_stats.py (the play task built from the scene and task
JSON, with its randomization); sim.step is wrapped to log TCP position, arm joint velocities
and actuator targets after every substep.

Also runs two open-loop baselines from the same reset: a constant +x TCP action and a
constant +z action, at full (1.0) and half (0.5) magnitude.

Launch from the wxai_mjlab environment, in this folder:

  python record_substeps.py
  python record_substeps.py --episodes 3 --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parents[2] / "sim" / "mujoco"
DEFAULT_CKPT = (
  SIM_ROOT / "logs/rsl_rl/grasp_jenga_DR_adjusted_gripper_width_with_grav_comp_real_robot_parameters"
  "_higher_gain_adjusted_quat/2026-10-28_14-55-03/model_999.pt"
)
DEFAULT_SCENE = SIM_ROOT / "configs/scenes/tabletop_jenga_wxai.json"
DEFAULT_TASK = SIM_ROOT / "configs/tasks/grasp_wxai.json"


class SubstepRecorder:
  def __init__(self, env):
    self.env = env
    robot = env.scene["robot"]
    term = env.action_manager.get_term("arm")
    self.site_id = term._frame_id
    self.dof_ids = term._joint_dof_ids
    self.q_adr = robot.indexing.joint_q_adr[term._joint_ids]
    self.rows: list[np.ndarray] = []
    self.enabled = False
    sim = env.sim
    orig = sim.step

    def step(*a, **k):
      out = orig(*a, **k)
      if self.enabled:
        self.record()
      return out

    sim.step = step

  def record(self) -> None:
    d = self.env.sim.data
    tcp = d.site_xpos[0, self.site_id]
    qd = d.qvel[0, self.dof_ids]
    q = d.qpos[0, self.q_adr]
    # The arm's six position actuators come first in the model.
    ctrl = d.ctrl[0, :6]
    self.rows.append(torch.cat([tcp, q, qd, ctrl]).float().cpu().numpy())

  def take(self) -> np.ndarray:
    out = np.stack(self.rows) if self.rows else np.zeros((0, 21))
    self.rows = []
    return out


def build(args):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from tabletop_sim.tasks.grasp import register_play_task

  task_id = register_play_task(args.scene, args.task)
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(task_id) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  return env, wrapped, policy


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
  p.add_argument("--scene", default=str(DEFAULT_SCENE))
  p.add_argument("--task", default=str(DEFAULT_TASK))
  p.add_argument("--episodes", type=int, default=3)
  p.add_argument("--steps", type=int, default=300, help="policy steps per episode")
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  p.add_argument("--out", type=Path, default=HERE / "results" / "substeps.npz")
  args = p.parse_args()

  env, wrapped, policy = build(args)
  rec = SubstepRecorder(env)
  decimation = env.cfg.decimation
  sim_dt = env.cfg.sim.mujoco.timestep
  print(f"decimation={decimation} sim_dt={sim_dt} action_dim={env.action_manager.total_action_dim}")

  saved: dict[str, np.ndarray] = {}
  with torch.inference_mode():
    for ep in range(args.episodes):
      obs, _ = wrapped.reset()
      rec.enabled = True
      rec.take()
      actions = []
      for _ in range(args.steps):
        a = policy(obs)
        actions.append(a[0].cpu().numpy())
        obs, *_ = wrapped.step(a)
      saved[f"policy_{ep}"] = rec.take()
      saved[f"policy_{ep}_actions"] = np.stack(actions)
      rec.enabled = False
      print(f"episode {ep}: {saved[f'policy_{ep}'].shape[0]} substeps")

    dim = env.action_manager.total_action_dim
    for name, axis, mag in (("const_x1", 0, 1.0), ("const_x05", 0, 0.5), ("const_z1", 2, 1.0)):
      wrapped.reset()
      rec.enabled = True
      rec.take()
      a = torch.zeros(1, dim, device=env.device)
      a[0, axis] = mag
      a[0, -1] = 1.0
      for _ in range(40):
        wrapped.step(a)
      saved[name] = rec.take()
      rec.enabled = False

  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, decimation=decimation, sim_dt=sim_dt, **saved)
  print(f"saved {args.out}")


if __name__ == "__main__":
  main()
