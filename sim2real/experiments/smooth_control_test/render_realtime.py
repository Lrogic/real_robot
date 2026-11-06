"""Real-time video of the recorded sim arm motion at a high frame rate, with a scrolling speed trace.

Replays the arm joints from results/substeps.npz (record_substeps.py) on the training robot
model, sampling the 500 Hz substeps at --fps. Left: the actual motion. Right: the pose held
for each 50 ms policy step (what a 20 fps play video shows). Below: TCP speed and wrist
(joint_3..5) velocities over a 1 s window scrolling with the video; the red line is now and
the dotted lines are new policy actions.

  MUJOCO_GL=egl python render_realtime.py --fps 60
  MUJOCO_GL=egl python render_realtime.py --fps 120 --episode policy_1 --seconds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODEL = HERE.parents[2] / "sim/mujoco/assets/robots/trossen_wxai_base/wxai/wxai_base_real_robot.xml"
TCP, Q, QD = slice(0, 3), slice(3, 9), slice(9, 15)
W, H, TRACE_H = 480, 480, 240
# The trace window under the two 480 px views spans exactly 1 s.
PX_PER_S = 2 * W
RESET_JUMP_M = 0.01


def trace_strip(t, speed, qd, n, dt) -> np.ndarray:
  """One wide image of the whole recording, PX_PER_S pixels per second."""
  dur = t[-1] + dt
  fig = plt.figure(figsize=(dur * PX_PER_S / 100, TRACE_H / 100), dpi=100)
  ax = fig.add_axes([0, 0.08, 1, 0.92])
  ax.plot(t, speed, "k", lw=1.5, label="TCP speed [m/s]")
  for j, c in zip(range(3, 6), ("tab:red", "tab:purple", "tab:brown")):
    ax.plot(t, qd[:, j] / 5, c, lw=1, label=f"joint_{j} vel / 5 [rad/s]")
  for b in np.arange(0, dur, n * dt):
    ax.axvline(b, color="gray", lw=0.5, ls=":")
  ax.axhline(0, color="gray", lw=0.5)
  ax.set_xlim(0, dur)
  ax.set_ylim(-0.9, 0.9)
  ax.tick_params(axis="x", direction="in", pad=-12, labelsize=7)
  ax.set_yticks([])
  fig.canvas.draw()
  img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return img


def legend_img() -> np.ndarray:
  fig = plt.figure(figsize=(2 * W / 100, 0.3), dpi=100)
  items = [("k", "TCP speed [m/s]"), ("tab:red", "joint_3 / 5"), ("tab:purple", "joint_4 / 5"),
           ("tab:brown", "joint_5 / 5 [rad/s]"), ("gray", ": = new action")]
  for i, (c, s) in enumerate(items):
    fig.text(0.01 + i * 0.2, 0.3, "— " + s, color=c, fontsize=8)
  fig.canvas.draw()
  img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return img


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--episode", default="policy_0")
  p.add_argument("--fps", type=float, default=60.0)
  p.add_argument("--start", type=float, default=0.0, help="seconds into the episode")
  p.add_argument("--seconds", type=float, default=None, help="default: until the first reset")
  args = p.parse_args()

  data = np.load(RES / "substeps.npz")
  n, dt = int(data["decimation"]), float(data["sim_dt"])
  rows = data[args.episode]
  step = np.linalg.norm(np.diff(rows[:, TCP], axis=0, prepend=rows[:1, TCP]), axis=1)
  resets = np.flatnonzero(step > RESET_JUMP_M)
  end = int(resets[0]) if len(resets) else len(rows)
  i0 = int(args.start / dt)
  i1 = end if args.seconds is None else min(end, i0 + int(args.seconds / dt))
  rows = rows[i0:i1]
  speed = step[i0:i1] / dt
  t = np.arange(len(rows)) * dt
  strip = trace_strip(t, speed, rows[:, QD], n, dt)
  legend = legend_img()

  m = mujoco.MjModel.from_xml_path(str(MODEL))
  m.vis.headlight.ambient[:] = 0.5
  m.vis.headlight.diffuse[:] = 0.7
  # The robot meshes are near-black; flat light gray reads better on the black background.
  m.geom_matid[:] = -1
  m.geom_rgba[:] = [0.75, 0.75, 0.8, 1.0]
  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, H, W)
  d.qpos[:6] = rows[0, Q]
  mujoco.mj_forward(m, d)
  ee = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")].copy()
  cam = mujoco.MjvCamera()
  cam.lookat[:] = 0.5 * ee + [0.0, 0.0, 0.1]
  cam.distance, cam.azimuth, cam.elevation = 1.0, 120.0, -15.0

  def render(k: int) -> np.ndarray:
    d.qpos[:6] = rows[k, Q]
    mujoco.mj_forward(m, d)
    r.update_scene(d, cam)
    return r.render().copy()

  half = PX_PER_S // 2
  pad = np.full((strip.shape[0], half, 3), 255, np.uint8)
  strip = np.concatenate([pad, strip, pad], axis=1)
  frames = []
  for f in range(int(t[-1] * args.fps)):
    k = min(int(round(f / args.fps / dt)), len(rows) - 1)
    held = k - k % n
    top = np.concatenate([render(k), render(held)], axis=1)
    x = int(k * dt * PX_PER_S)
    win = strip[:, x : x + PX_PER_S].copy()
    win[:, W - 1 : W + 1] = [220, 0, 0]
    frames.append(np.concatenate([top, legend, win], axis=0))
  out = RES / f"realtime_{args.episode}_{int(args.fps)}fps.mp4"
  media.write_video(str(out), frames, fps=args.fps)
  print(f"saved {out} ({len(frames)} frames, {t[-1]:.2f} s real time at {args.fps:g} fps)")


if __name__ == "__main__":
  main()
