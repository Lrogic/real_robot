"""Render side-by-side videos of a gripped jenga block: the joint-friction model (before the
gripper change) next to the grip model (trossen_wxai_base_real_friction_grip).

Two videos, each the same scripted sequence on both models in lockstep:

  hold.mp4   grasp the upright block, lift it 50 mm, hold it to the end (--duration-s, 20 s)
  push.mp4   grasp the block on the table, then every 50 ms set the TCP goal --push-mm below
             the measured TCP (the policy's relative-action pattern) for --push-s (8 s), then
             hold the TCP where it is to the end

The grasp: open gripper above the block, down to 11 mm above its centre, close with the real
loop's 0.5 s gripper goal time (the driver's quintic, as action.gripper_goal_time_s models it).
The arm follows TCP goals with a damped least-squares IK solve every 2 ms physics step that
also holds the starting gripper orientation. The overlay shows how far the block has moved
along the fingers since the close, and the squeeze.

CPU MuJoCo on the training models (scene JSON compiled as training compiles it, with the
embodiment's solver settings). Launch from this directory with the wxai_mjlab environment:

  MUJOCO_GL=egl /opt/conda-envs/wxai_mjlab/bin/python render_sim.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from common.sim import apply_training_solver  # noqa: E402

SIM = Path(__file__).resolve().parents[3] / "sim" / "mujoco"
OUT_DIR = Path(__file__).resolve().parent / "videos"
MODELS = (
  ("Before: trossen_wxai_base_real_friction", "tabletop_jenga_wxai_friction.json"),
  ("After: trossen_wxai_base_real_friction_grip", "tabletop_jenga_wxai_friction_grip.json"),
)
BLOCK_XY = (-0.30, 0.0)
GRASP_ABOVE_CENTRE_M = 0.011
CLOSE_GOAL_TIME_S = 0.5
FPS = 30
WIDTH, HEIGHT = 640, 540


def compile_scene(scene_name: str):
  from mjlab.scene import Scene
  from tabletop_sim.embodiments import get_embodiment
  from tabletop_sim.scene import load_scene_config
  from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg

  scene_path = SIM / "configs" / "grasp" / "scenes" / scene_name
  cfg = make_grasp_env_cfg(scene_path, SIM / "configs" / "grasp" / "tasks" / "grasp_wxai.json", None, 1)
  model = Scene(cfg.scene, "cpu").compile()
  emb = get_embodiment(load_scene_config(scene_path).robot.embodiment)
  apply_training_solver(model, emb, 0.002)
  return model, emb


class Rig:
  """One model: the arm on per-substep IK toward a TCP goal, the gripper on a position target."""

  def __init__(self, model, emb):
    self.m, self.d, self.emb = model, mujoco.MjData(model), emb
    m, d = self.m, self.d
    joint = lambda n: m.joint(f"robot/{n}")  # noqa: E731
    self.arm_q = [m.jnt_qposadr[joint(f"joint_{i}").id] for i in range(6)]
    self.arm_v = [m.jnt_dofadr[joint(f"joint_{i}").id] for i in range(6)]
    self.arm_u = [m.actuator(f"robot/joint_{i}").id for i in range(6)]
    self.grip_u = m.actuator("robot/left_gripper").id
    self.site = m.site(f"robot/{emb.tcp_site}").id
    self.block = m.body("jenga/jenga").id
    for name, value in emb.ready_joint_pos.items():
      d.qpos[m.jnt_qposadr[joint(name).id]] = value
    for name in emb.gripper_joint_names:
      d.qpos[m.jnt_qposadr[joint(name).id]] = emb.gripper_open
    block_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == self.block and m.geom_contype[g] == 1)
    adr = m.jnt_qposadr[m.joint("jenga/jenga_joint").id]
    half_height = float(m.geom_size[block_geom].max())  # the active size variant, standing on end
    d.qpos[adr:adr + 7] = [*BLOCK_XY, half_height, 0.5, 0.5, -0.5, 0.5]
    d.ctrl[self.arm_u] = d.qpos[self.arm_q]
    d.ctrl[self.grip_u] = emb.gripper_open
    mujoco.mj_forward(m, d)
    self.goal = d.site_xpos[self.site].copy()
    self.R_goal = d.site_xmat[self.site].reshape(3, 3).copy()
    self.grip_from = self.grip_to = emb.gripper_open
    self.grip_t = 0.0
    self.half_height = half_height

  def tcp(self):
    return self.d.site_xpos[self.site].copy()

  def close_gripper(self):
    self.grip_from = float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"robot/{self.emb.gripper_actuator_joint}").id]])
    self.grip_to, self.grip_t = self.emb.gripper_closed, 0.0

  def along_fingers(self):
    """Block centre minus TCP along the TCP x axis (pointing out along the fingers)."""
    axis = self.d.site_xmat[self.site].reshape(3, 3)[:, 0]
    return float(axis @ (self.d.xpos[self.block] - self.d.site_xpos[self.site]))

  def squeeze(self):
    return -float(self.d.actuator_force[self.grip_u])

  def step(self):
    m, d = self.m, self.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, self.site)
    J = np.vstack([jp, jr])[:, self.arm_v]
    R = d.site_xmat[self.site].reshape(3, 3)
    q_cur, q_goal, rv = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q_cur, R.flatten())
    mujoco.mju_mat2Quat(q_goal, self.R_goal.flatten())
    mujoco.mju_subQuat(rv, q_goal, q_cur)
    e = np.concatenate([self.goal - d.site_xpos[self.site], R @ rv])
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
    d.ctrl[self.arm_u] = d.qpos[self.arm_q] + dq
    self.grip_t += m.opt.timestep
    tau = min(self.grip_t / CLOSE_GOAL_TIME_S, 1.0)
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    d.ctrl[self.grip_u] = self.grip_from + (self.grip_to - self.grip_from) * s
    mujoco.mj_step(m, d)


class Movie:
  def __init__(self, rigs, labels, path: Path, lookat):
    self.rigs, self.labels, self.path = rigs, labels, path
    self.renderers = [mujoco.Renderer(r.m, HEIGHT, WIDTH) for r in rigs]
    self.cam = mujoco.MjvCamera()
    self.cam.lookat[:] = lookat
    self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.24, 180.0, -6.0
    self.opt = mujoco.MjvOption()
    self.frames, self.t, self.next_frame = [], 0.0, 0.0
    self.phase, self.ref = "", [None] * len(rigs)
    try:
      self.font = ImageFont.truetype("DejaVuSans.ttf", 17)
      self.big = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
    except OSError:
      self.font = self.big = ImageFont.load_default()

  def mark_reference(self):
    self.ref = [r.along_fingers() for r in self.rigs]

  def run(self, seconds, before_tick=None):
    dt = self.rigs[0].m.opt.timestep
    tick_steps = round(0.05 / dt)
    for k in range(round(seconds / dt)):
      if before_tick is not None and k % tick_steps == 0:
        for r in self.rigs:
          before_tick(r)
      for r in self.rigs:
        r.step()
      self.t += dt
      if self.t >= self.next_frame:
        self.frames.append(self.frame())
        self.next_frame += 1.0 / FPS

  def move(self, delta, seconds):
    starts = [r.goal.copy() for r in self.rigs]
    dt = self.rigs[0].m.opt.timestep
    n = round(seconds / dt)
    for k in range(n):
      for r, s in zip(self.rigs, starts):
        r.goal = s + np.asarray(delta) * (k + 1) / n
        r.step()
      self.t += dt
      if self.t >= self.next_frame:
        self.frames.append(self.frame())
        self.next_frame += 1.0 / FPS

  def frame(self):
    panels = []
    for r, renderer, label, ref in zip(self.rigs, self.renderers, self.labels, self.ref):
      renderer.update_scene(r.d, self.cam, self.opt)
      img = Image.fromarray(renderer.render())
      draw = ImageDraw.Draw(img)
      draw.rectangle([0, 0, WIDTH, 92], fill=(0, 0, 0))
      draw.text((10, 6), label, font=self.big, fill=(255, 255, 255))
      draw.text((10, 34), f"t {self.t:4.1f} s   {self.phase}", font=self.font, fill=(220, 220, 220))
      if ref is None:
        draw.text((10, 60), "squeeze  --", font=self.font, fill=(220, 220, 220))
      else:
        slip = 1e3 * (r.along_fingers() - ref)
        colour = (255, 120, 120) if abs(slip) > 2 else (140, 255, 140)
        draw.text((10, 60), f"squeeze {r.squeeze():5.1f} N   block moved along fingers {slip:+6.1f} mm",
                  font=self.font, fill=colour)
      panels.append(np.asarray(img))
    return np.concatenate(panels, axis=1)

  def save(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(self.path, self.frames, fps=FPS, quality=8, macro_block_size=8)
    print(f"Saved {self.path} ({len(self.frames)} frames)")


def grasp(movie: Movie):
  movie.phase = "settle"
  movie.run(0.3)
  movie.phase = "move above the block"
  for r in movie.rigs:
    r.goal_target = np.array([*BLOCK_XY, r.half_height + 0.06])
  movie.move(movie.rigs[0].goal_target - movie.rigs[0].goal, 0.8)
  movie.phase = "down to the block"
  movie.move([0.0, 0.0, -(0.06 - GRASP_ABOVE_CENTRE_M)], 0.8)
  movie.run(0.2)
  movie.phase = "close (0.5 s goal time)"
  for r in movie.rigs:
    r.close_gripper()
  movie.run(0.8)
  movie.mark_reference()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--push-mm", type=float, default=5.0, help="TCP goal below the measured TCP each 50 ms tick")
  parser.add_argument("--push-s", type=float, default=8.0, help="how long the push lasts")
  parser.add_argument("--duration-s", type=float, default=20.0,
                      help="video length; the hold, and the rest after the push, fill it")
  parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
  parser.add_argument("--only", choices=("hold", "push"), default=None)
  args = parser.parse_args()
  compiled = [compile_scene(scene) for _, scene in MODELS]
  labels = [label for label, _ in MODELS]

  def fresh():
    return [Rig(m, emb) for m, emb in compiled]

  if args.only in (None, "hold"):
    rigs = fresh()
    lookat = [BLOCK_XY[0], BLOCK_XY[1], rigs[0].half_height + 0.045]
    movie = Movie(rigs, labels, args.out_dir / "hold.mp4", lookat)
    grasp(movie)
    movie.phase = "lift 50 mm"
    movie.move([0.0, 0.0, 0.05], 0.5)
    hold_s = args.duration_s - movie.t
    movie.phase = f"hold {hold_s:.1f} s"
    movie.run(hold_s)
    movie.save()

  if args.only in (None, "push"):
    rigs = fresh()
    lookat = [BLOCK_XY[0], BLOCK_XY[1], rigs[0].half_height + 0.02]
    movie = Movie(rigs, labels, args.out_dir / "push.mp4", lookat)
    grasp(movie)
    dz = args.push_mm / 1000.0

    def push(r):
      r.goal = r.tcp() - np.array([0.0, 0.0, dz])

    movie.phase = f"push down: goal {args.push_mm:g} mm below the TCP every 50 ms"
    movie.run(args.push_s, before_tick=push)
    for r in movie.rigs:
      r.goal = r.tcp()
    movie.phase = "stop pushing"
    movie.run(max(args.duration_s - movie.t, 0.0))
    movie.save()


if __name__ == "__main__":
  main()
