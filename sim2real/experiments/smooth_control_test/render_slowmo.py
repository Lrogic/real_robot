"""Slow-motion video of the recorded sim arm motion: every 2 ms substep vs. one frame per policy step.

Replays the arm joints from results/substeps.npz (record_substeps.py) on the training robot
model. Left: every substep (what the sim arm does). Right: the pose held for each 50 ms
policy step, i.e. what a play/training video at render_fps = control_hz shows. Both play
10x slower than real time. Also prints joint acceleration stats.

  MUJOCO_GL=egl python render_slowmo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODEL = HERE.parents[2] / "sim/mujoco/assets/robots/trossen_wxai_base/wxai/wxai_base_real_robot.xml"
Q, QD = slice(3, 9), slice(9, 15)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--episode", default="policy_0")
  p.add_argument("--seconds", type=float, default=1.0)
  p.add_argument("--slowdown", type=float, default=10.0)
  args = p.parse_args()

  data = np.load(RES / "substeps.npz")
  n, dt = int(data["decimation"]), float(data["sim_dt"])
  rows = data[args.episode]

  qd = rows[:, QD]
  acc = np.diff(qd, axis=0) / dt
  print("per joint: p99 |qdd| [rad/s^2], p99 |qd| [rad/s]")
  for j in range(6):
    print(f"  joint_{j}: {np.percentile(np.abs(acc[:, j]), 99):7.1f}  {np.percentile(np.abs(qd[:, j]), 99):.2f}")

  tcp = rows[:, :3]
  speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / dt
  win = int(args.seconds / dt)
  i0 = int(np.argmax(np.convolve(speed, np.ones(win), "valid")))
  i0 -= i0 % n
  q = rows[i0 : i0 + win, Q]

  m = mujoco.MjModel.from_xml_path(str(MODEL))
  m.vis.headlight.ambient[:] = 0.5
  m.vis.headlight.diffuse[:] = 0.7
  # The robot meshes are near-black; flat light gray reads better on the black background.
  m.geom_matid[:] = -1
  m.geom_rgba[:] = [0.75, 0.75, 0.8, 1.0]
  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, 480, 480)
  d.qpos[:6] = q[0]
  mujoco.mj_forward(m, d)
  ee = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")]
  cam = mujoco.MjvCamera()
  cam.lookat[:] = 0.5 * ee + [0.0, 0.0, 0.1]
  cam.distance, cam.azimuth, cam.elevation = 1.0, 120.0, -15.0
  fps = 1.0 / dt / args.slowdown

  frames = []
  cache: dict[int, np.ndarray] = {}
  for i in range(len(q)):
    held = i - i % n
    pair = []
    for k in (i, held):
      if k not in cache:
        d.qpos[:6] = q[k]
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        cache[k] = r.render().copy()
      pair.append(cache[k])
    frames.append(np.concatenate(pair, axis=1))
    cache = {k: v for k, v in cache.items() if k >= held}
  out = RES / "slowmo_substep_vs_policy_rate.mp4"
  media.write_video(str(out), frames, fps=fps)
  print(f"saved {out} ({len(frames)} frames, {args.slowdown:g}x slow motion, window starts {i0 * dt:.2f} s)")


if __name__ == "__main__":
  main()
