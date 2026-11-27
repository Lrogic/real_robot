"""Scripted grasp, hold and push in the training env (MuJoCo Warp) for several robot models.

Compares how a gripped jenga block behaves with the original model, the joint-friction
model, and the friction model with the gripper matched to the real one
(trossen_wxai_base_real_friction_grip). Each model runs the jenga scene and
configs/grasp/tasks/grasp_wxai.json with two edits: the block's reset yaw is 0, so the scripted
grasp needs no rotation, and action.gripper_goal_time_s is --gripper-goal-time-s (the real
loop's 0.5 s by default). Block size variants stay on.

Every env runs the same script at the 20 Hz control rate, with TCP goals clipped to the
policy's 10 mm per tick and no rotation command:

  0-1.5 s   above the block (6 cm), gripper open
  1.5-2.5 s down to 8 mm above the block centre
  2.5-3.5 s close (binary gripper action)
  then, by env index mod 4:
    hold    lift 50 mm and hold 4 s
    push    every tick, a TCP goal dz below the measured TCP, for 1.5 s; dz = 2, 5, 10 mm

Reported: the squeeze (gripper actuator force) and the finger offset (left minus right
carriage) after the close; for hold, how far the block slid out along the fingers during the
lift and over the 4 s hold, and how high it was lifted; for push, how far the fingers slid
down over the block, how far the gripper rotated (the script holds no orientation), and how
far the block rotated inside the grip. Slides are measured along the TCP x axis (along the
fingers), so a rotating gripper does not count as a slide. Then env.step throughput at --throughput-envs random-action
envs.

Needs CUDA. Launch from this directory with the wxai_mjlab environment:

  /opt/conda-envs/wxai_mjlab/bin/python run_sim.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[3] / "sim" / "mujoco"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "cuda:0"
SCENES = {
  "trossen_wxai_base_real": "tabletop_jenga_wxai.json",
  "trossen_wxai_base_real_friction": "tabletop_jenga_wxai_friction.json",
  "trossen_wxai_base_real_friction_grip": "tabletop_jenga_wxai_friction_grip.json",
}
PUSH_DZ_M = (0.002, 0.005, 0.010)
LIFT_M = 0.05
STEPS = dict(above=30, descend=50, close=70, lift_end=90, push_end=100, hold_end=170)


def _task_file(tmp: Path, goal_time_s: float) -> Path:
  data = json.loads((SIM / "configs" / "grasp" / "tasks" / "grasp_wxai.json").read_text())
  data["action"]["gripper_goal_time_s"] = goal_time_s
  data["domain_randomization"]["object_reset_yaw_range_deg"] = 0.0
  out = tmp / "task.json"
  out.write_text(json.dumps(data))
  return out


def _make_env(embodiment: str, task: Path, num_envs: int):
  from mjlab.envs import ManagerBasedRlEnv
  from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg

  cfg = make_grasp_env_cfg(SIM / "configs" / "grasp" / "scenes" / SCENES[embodiment], task, None, num_envs)
  return ManagerBasedRlEnv(cfg, device=DEVICE)


def _quat_conj(q):
  return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _quat_mul(a, b):
  w1, x1, y1, z1 = a.unbind(-1)
  w2, x2, y2, z2 = b.unbind(-1)
  return torch.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], -1)


def _angle_deg(q):
  return torch.rad2deg(2 * torch.arccos(q[..., 0].abs().clamp(max=1.0)))


def _along_fingers(tcp_quat, d):
  """Component of d (block centre minus TCP, world) along the TCP x axis, which points
  out of the gripper along the fingers. It grows when the block slides out toward the
  fingertips and shrinks when the fingers slide down over the block."""
  w, x, y, z = tcp_quat.unbind(-1)
  axis = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
  return (axis * d).sum(-1).clone()


def _summary(x: torch.Tensor) -> dict:
  x = x.float()
  return {"median": float(x.median()), "min": float(x.min()), "max": float(x.max())}


def scripted_grasp(env) -> dict:
  robot, obj = env.scene["robot"], env.scene["jenga"]
  site = robot.find_sites(("ee_site",))[0][0]
  carriages, _ = robot.find_joints(("left_carriage_joint", "right_carriage_joint"), preserve_order=True)
  gripper = robot.find_actuators(("left_gripper",))[0][0]
  n = env.num_envs
  group = torch.arange(n, device=DEVICE) % 4
  dz = torch.tensor((0.0, *PUSH_DZ_M), device=DEVICE)[group]
  push = group > 0
  env.reset()
  snap, finite = {}, True
  lift_goal = push_xy = None
  for i in range(STEPS["hold_end"]):
    tcp = robot.data.site_pos_w[:, site]
    o = obj.data.root_link_pos_w
    if i < STEPS["above"]:
      goal, grip = o + torch.tensor([0.0, 0.0, 0.06], device=DEVICE), 1.0
    elif i < STEPS["close"]:
      goal = o + torch.tensor([0.0, 0.0, 0.008], device=DEVICE)
      grip = 1.0 if i < STEPS["descend"] else -1.0
    else:
      if lift_goal is None:
        lift_goal = tcp + torch.tensor([0.0, 0.0, LIFT_M], device=DEVICE)
        push_xy = tcp[:, :2].clone()
      goal, grip = lift_goal, -1.0
    action = torch.zeros(n, 7, device=DEVICE)
    action[:, :3] = ((goal - tcp) / 0.01).clamp(-1, 1)
    if i >= STEPS["close"]:
      action[push, :2] = ((push_xy[push] - tcp[push, :2]) / 0.01).clamp(-1, 1)
      action[push, 2] = -dz[push] / 0.01 if i < STEPS["push_end"] else 0.0
      if i >= STEPS["push_end"]:
        action[push, :3] = 0.0
    action[:, 6] = grip
    if i in (STEPS["close"] - 1, STEPS["lift_end"] - 1, STEPS["push_end"] - 1, STEPS["hold_end"] - 1):
      snap[i + 1] = {
        "along_fingers": _along_fingers(robot.data.site_quat_w[:, site], o - tcp),
        "obj_z": o[:, 2].clone(),
        "obj_quat": obj.data.root_link_quat_w.clone(),
        "tcp_quat": robot.data.site_quat_w[:, site].clone(),
        "squeeze": robot.data.actuator_force[:, gripper].clone(),
        "carriages": robot.data.joint_pos[:, carriages].clone(),
      }
    env.step(action)
    finite &= bool(torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(obj.data.root_link_pos_w).all())

  closed, lifted, pushed, held = (snap[STEPS[k]] for k in ("close", "lift_end", "push_end", "hold_end"))
  hold = group == 0
  out = {
    "finite": finite,
    "after_close": {
      "squeeze_n": _summary(-closed["squeeze"]),
      "finger_offset_mm": _summary(1e3 * (closed["carriages"][:, 0] - closed["carriages"][:, 1])),
    },
    "hold": {
      "envs": int(hold.sum()),
      "slip_during_lift_mm": _summary(1e3 * (lifted["along_fingers"] - closed["along_fingers"])[hold]),
      "slip_over_4s_mm": _summary(1e3 * (held["along_fingers"] - lifted["along_fingers"])[hold]),
      "lifted_mm": _summary(1e3 * (held["obj_z"] - closed["obj_z"])[hold]),
    },
    "push": {},
  }

  def in_grip(s):
    return _quat_mul(_quat_conj(s["tcp_quat"]), s["obj_quat"])

  for k, d in enumerate(PUSH_DZ_M, start=1):
    m = group == k
    out["push"][f"{round(d * 1e3)}mm_per_tick"] = {
      "envs": int(m.sum()),
      "fingers_slid_mm": _summary(1e3 * (closed["along_fingers"] - pushed["along_fingers"])[m]),
      "gripper_rotated_deg": _summary(_angle_deg(_quat_mul(_quat_conj(closed["tcp_quat"]), pushed["tcp_quat"]))[m]),
      "block_rotated_in_grip_deg": _summary(_angle_deg(_quat_mul(_quat_conj(in_grip(closed)), in_grip(pushed)))[m]),
    }
  return out


def throughput(embodiment: str, task: Path, num_envs: int, steps: int = 100) -> float:
  env = _make_env(embodiment, task, num_envs)
  try:
    env.reset()
    for _ in range(10):
      env.step(torch.rand(num_envs, 7, device=DEVICE) * 2 - 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
      env.step(torch.rand(num_envs, 7, device=DEVICE) * 2 - 1)
    torch.cuda.synchronize()
    return num_envs * steps / (time.perf_counter() - t0)
  finally:
    env.close()


def _print(name: str, r: dict) -> None:
  def f(s):
    return f"{s['median']:6.2f} [{s['min']:6.2f}, {s['max']:6.2f}]"

  print(f"\n== {name} (finite throughout: {r['finite']}); median [min, max]")
  print(f"  after close: squeeze {f(r['after_close']['squeeze_n'])} N, finger offset {f(r['after_close']['finger_offset_mm'])} mm")
  h = r["hold"]
  print(f"  hold ({h['envs']} envs): slip during lift {f(h['slip_during_lift_mm'])} mm, over 4 s {f(h['slip_over_4s_mm'])} mm, "
        f"lifted {f(h['lifted_mm'])} mm")
  for key, p in r["push"].items():
    print(f"  push {key} ({p['envs']} envs): fingers slid {f(p['fingers_slid_mm'])} mm, gripper rotated "
          f"{f(p['gripper_rotated_deg'])} deg, block rotated in grip {f(p['block_rotated_in_grip_deg'])} deg")
  if "env_steps_per_s" in r:
    print(f"  throughput: {r['env_steps_per_s']:,.0f} env-steps/s")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--embodiments", nargs="+", default=list(SCENES), choices=list(SCENES))
  parser.add_argument("--gripper-goal-time-s", type=float, default=0.5)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--throughput-envs", type=int, default=4096, help="0 skips the throughput timing")
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise SystemExit("needs CUDA (MuJoCo Warp)")
  output = args.output or RESULTS_DIR / f"gripper_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
  payload = {"experiment": "gripper_hold_push_sim", "gripper_goal_time_s": args.gripper_goal_time_s,
             "num_envs": args.num_envs, "push_dz_m": PUSH_DZ_M, "lift_m": LIFT_M, "results": {}}
  with tempfile.TemporaryDirectory() as tmp:
    task = _task_file(Path(tmp), args.gripper_goal_time_s)
    for name in args.embodiments:
      env = _make_env(name, task, args.num_envs)
      try:
        result = scripted_grasp(env)
      finally:
        env.close()
      if args.throughput_envs:
        result["env_steps_per_s"] = throughput(name, task, args.throughput_envs)
      payload["results"][name] = result
      _print(name, result)
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2) + "\n")
  print(f"\nSaved {output}")


if __name__ == "__main__":
  main()
