"""Generate the release task's start states: run the trained grasp policy, then
a scripted IK transport to just above the goal, and keep the states where the
object is still held and nothing collided (env ``wxai_mjlab``, needs CUDA).

  python scripts/generate_release_starts.py --grasp-checkpoint logs/rsl_rl/<exp>/<run>/model_N.pt \
      [--scene JSON] [--task JSON] [--grasp-task JSON] [--rows N] [--num-envs N] [--out PT]

Per env, repeated until ``--rows`` states are kept:
  1. Grasp: the grasp policy acts (relative objects parked off the table) until
     ``stable_hold`` > 0.5 for ``--hold-steps`` steps.
  2. Lift: measure the object pose in the TCP frame, sample the anchor, the
     goal (symmetric copy closest to the object's orientation) and the arrival
     offset (ball cap above the goal, world axes); drive the TCP straight up to
     z_safe = top of the relative objects + transport_clearance + how far the
     fingers and object reach below the TCP, turning it to the goal orientation
     on the way (the relative objects are still parked; turning first keeps the
     arm inside its joint limits at that height).
  3. Place the relative objects at the anchor. A contact with the robot or the
     object on the next step rejects the attempt.
  4. Across at z_safe to above the target.
  5. Down to the TCP target (puts the object at goal + offset).
  6. Turn the TCP by roll / pitch / yaw ~ N(0, tcp_rotation_noise_std_deg)
     about its own axes (clipped at 2 std).
  7. Settle for a random number of steps, then check and record.

Scripted actions are clip(error / action scale, -1, 1) through the env's own
IK action, so transport moves no faster than the policy's action limits.
Writes the ``.pt`` and ``.meta.json`` (rejection counts included).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import joint_pos_limits
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  combine_frame_transforms,
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.scene import load_scene_config, load_task_config
from tabletop_sim.scene.release_config import load_release_task_config
from tabletop_sim.tasks.grasp import env_cfg as grasp_cfg
from tabletop_sim.tasks.grasp.mdp import object_size, rewards, task_state
from tabletop_sim.tasks.grasp.rl_cfg import grasp_ppo_runner_cfg
from tabletop_sim.tasks.release import starts as starts_io
from tabletop_sim.tasks.release.env_cfg import DEFAULT_SCENE, DEFAULT_TASK, robot_contact_sensor
from tabletop_sim.tasks.release.mdp import placement

GRASP, LIFT, PLACED, ACROSS, DOWN, NOISE, SETTLE = range(7)
PHASE_NAMES = ("grasp", "lift", "placed", "across", "down", "noise", "settle")
# Via points (lift, across) only need to be passed near; the descent ends at the target.
VIA_POS_TOL, VIA_ROT_TOL = 0.015, math.radians(5.0)
POS_TOL, ROT_TOL = 0.005, math.radians(3.0)
FINE_POS_TOL, FINE_ROT_TOL = 0.003, math.radians(0.5)


def _git_commit() -> str | None:
  try:
    out = subprocess.run(
      ("git", "rev-parse", "HEAD"), cwd=Path(__file__).parent, capture_output=True, text=True, check=True
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout.strip()


def object_contact_sensor(obj: str, target: str, decimation: int) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=f"{obj}_{target}_contact",
    primary=ContactMatch(mode="body", pattern=obj, entity=obj),
    secondary=ContactMatch(mode="body", pattern=target, entity=target),
    fields=("found", "force"),
    reduce="netforce",
    history_length=decimation,
  )


def any_contact(env: ManagerBasedRlEnv, names: list[str]) -> torch.Tensor:
  hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in names:
    history = env.scene[name].data.force_history
    hit |= (torch.linalg.norm(history, dim=-1) > 0).reshape(env.num_envs, -1).any(dim=-1)
  return hit


class Generator:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.scene = load_scene_config(args.scene)
    self.task = load_release_task_config(args.task)
    self.task.validate_scene(self.scene)
    grasp_task = load_task_config(args.grasp_task)
    rel_sizes = self.task.domain_randomization.object_size_variants
    g_sizes = grasp_task.domain_randomization.object_size_variants
    if (rel_sizes.enabled, rel_sizes.scales) != (g_sizes.enabled, g_sizes.scales):
      raise SystemExit("the release and grasp tasks must use the same object size variants")
    # Fixed difficulty: the grasp phase sees the DR and pose noise at that level.
    d = args.difficulty
    grasp_task = replace(
      grasp_task,
      curriculum=replace(grasp_task.curriculum, enabled=True, initial_difficulty=d, max_difficulty=d),
    )
    self.grasp_task = grasp_task
    self.emb = get_embodiment(self.scene.robot.embodiment)
    self.obj = self.scene.manipulated_object
    self.supports = self.task.relative_placement.object_names

    cfg = grasp_cfg.make_grasp_env_cfg_from_configs(self.scene, grasp_task, None, args.num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = 1.0e6  # The script decides when an attempt ends.
    dec = grasp_task.timing.decimation
    surface = self.scene.support_surface.object  # type: ignore[union-attr]
    self.robot_sensors = [f"robot_{t}_contact" for t in (surface, *self.supports)]
    self.object_sensors = [f"{self.obj}_{t}_contact" for t in (surface, *self.supports)]
    # A grasp task with w_table_contact already has the robot-table sensor, under the same name
    # and with the same definition (moving robot links vs the table, one sample per substep).
    have = {s.name for s in cfg.scene.sensors}
    cfg.scene.sensors = (
      *cfg.scene.sensors,
      *(s for s in (robot_contact_sensor(self.emb, t, dec) for t in (surface, *self.supports))
        if s.name not in have),
      *(object_contact_sensor(self.obj, t, dec) for t in (surface, *self.supports)),
    )
    self.env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    self.wrapped = RslRlVecEnvWrapper(self.env)
    runner = MjlabOnPolicyRunner(self.wrapped, asdict(grasp_ppo_runner_cfg()), device=args.device)
    runner.load(args.grasp_checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    self.policy = runner.get_inference_policy(device=args.device)
    self._setup_handles()

  def _setup_handles(self) -> None:
    env, emb = self.env, self.emb
    ent = grasp_cfg._Entities(emb, self.obj)
    self.hold_params = ent.hold_params(self.grasp_task)
    self.grasp_params = ent.grasp_params(self.grasp_task)
    self.arm_cfg = ent.arm()
    self.tcp_cfg = ent.tcp()
    for c in (self.hold_params["object_cfg"], self.hold_params["finger_cfg"], self.grasp_params["finger_cfg"],
              self.arm_cfg, self.tcp_cfg):
      c.resolve(env.scene)
    self.robot = env.scene["robot"]
    self.object = env.scene[self.obj]
    m = env.sim.mj_model
    finger_bodies = {m.body(f"robot/{b}").id for b in emb.finger_bodies}
    self.finger_geoms = [
      g for g in range(m.ngeom)
      if m.geom_bodyid[g] in finger_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    signs = torch.tensor([[1 if (i >> k) & 1 else -1 for k in range(3)] for i in range(8)], dtype=torch.float)
    half = torch.tensor(m.geom_size[self.finger_geoms], dtype=torch.float)  # boxes (or bounds)
    for g in self.finger_geoms:
      if m.geom_type[g] not in (6, 7):  # box, mesh: geom_size of a mesh is its bounding half-size
        raise SystemExit(f"finger geom {m.geom(g).name} has unsupported type {m.geom_type[g]}")
    self.finger_corners = (signs[None] * half[:, None]).to(env.device)  # (G, 8, 3) geom frame
    self.obj_points, self.obj_variant = object_size.collision_points_per_variant(env, self.obj)
    self.obj_scale = object_size.assigned_scale(env, self.obj)
    self.support_points = [object_size.collision_points_per_variant(env, s)[0][0] for s in self.supports]
    self.support_height = self.scene.support_height()
    act = self.grasp_task.action
    self.pos_scale, self.rot_scale = act.ee_pos_scale, act.ee_rot_scale

    n, dev = env.num_envs, env.device
    self.phase = torch.zeros(n, dtype=torch.long, device=dev)
    self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.held_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.settle_left = torch.zeros(n, dtype=torch.long, device=dev)
    self.wp_pos = torch.zeros(n, 3, device=dev)
    self.wp_quat = torch.zeros(n, 4, device=dev)
    self.wp_quat[:, 0] = 1
    self.final_pos = torch.zeros(n, 3, device=dev)
    self.final_quat = self.wp_quat.clone()
    self.z_safe = torch.zeros(n, device=dev)
    self.anchor = torch.zeros(n, 7, device=dev)
    self.goal = torch.zeros(n, 7, device=dev)
    self.support_state = torch.zeros(n, len(self.supports), 13, device=dev)
    self.tcp_obj = torch.zeros(n, 7, device=dev)

  # Geometry.

  def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    return task_state.tcp_pose_w(self.env, self.tcp_cfg)

  def hang_below_tcp(self, ids: torch.Tensor, tcp_quat_target: torch.Tensor) -> torch.Tensor:
    """How far (m) the fingers and the object reach below the TCP if the TCP
    had orientation ``tcp_quat_target``; the hand's shape is measured now."""
    env = self.env
    tcp_pos, tcp_quat = self.tcp_pose()
    tp, tq = tcp_pos[ids], tcp_quat[ids]
    # Finger corners in the TCP frame.
    g = self.finger_geoms
    gpos = env.sim.data.geom_xpos[ids][:, g]  # (n, G, 3)
    gmat = env.sim.data.geom_xmat[ids][:, g].reshape(len(ids), len(g), 3, 3)
    corners_w = gpos[:, :, None, :] + torch.einsum("ngij,gkj->ngki", gmat, self.finger_corners)
    corners_w = corners_w.reshape(len(ids), -1, 3)
    # Object collision points in the TCP frame.
    pts = self.obj_points[self.obj_variant[ids]]  # (n, P, 3) object frame
    opos, oquat = self.object.data.root_link_pos_w[ids], self.object.data.root_link_quat_w[ids]
    pts_w = opos[:, None] + quat_apply(oquat[:, None].expand(-1, pts.shape[1], 4), pts)
    allp = torch.cat((corners_w, pts_w), dim=1)
    k = allp.shape[1]
    in_tcp = quat_apply(quat_inv(tq)[:, None].expand(-1, k, 4), allp - tp[:, None])
    rotated = quat_apply(tcp_quat_target[:, None].expand(-1, k, 4), in_tcp)
    return (-rotated[..., 2]).amax(dim=1).clamp(min=0.0)

  def choose_goal_quat(
    self,
    g_quat: torch.Tensor,
    oquat: torch.Tensor,
    q_to: torch.Tensor,
    goal_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
  ) -> torch.Tensor:
    """The symmetric goal orientation the transport aims for: the copies
    within 5 deg of the smallest turn from the object's current orientation,
    and among those the one whose TCP approach axis (TCP x) points most
    directly away from the robot base, horizontally. Ties are common (an
    upright block is 90 deg from both ends of a horizontal goal), and reaching
    forward stays inside the arm's joint limits where reaching back does not."""
    sym = placement.symmetry_quats(self.task.relative_placement.goal_symmetry, g_quat.device)
    n, k = g_quat.shape[0], sym.shape[0]
    cand = quat_mul(g_quat[:, None].expand(n, k, 4).reshape(-1, 4), sym[None].expand(n, k, 4).reshape(-1, 4))
    cand = cand.reshape(n, k, 4)
    angle = placement.rotation_angle(cand, oquat[:, None].expand(n, k, 4))
    near = angle <= angle.min(dim=1, keepdim=True).values + math.radians(5.0)
    tcp_q = quat_mul(cand.reshape(-1, 4), quat_inv(q_to)[:, None].expand(n, k, 4).reshape(-1, 4))
    approach = quat_apply(tcp_q, torch.tensor([1.0, 0.0, 0.0], device=g_quat.device).expand(n * k, 3))
    approach = approach.reshape(n, k, 3)
    radial = (goal_pos_w - base_pos_w)[:, :2]
    radial = radial / torch.linalg.norm(radial[:, :2], dim=-1, keepdim=True).clamp_min(1e-9)
    score = (approach[..., :2] * radial[:, None]).sum(-1)
    score = torch.where(near, score, torch.full_like(score, -1e9))
    best = score.argmax(dim=1)
    return cand[torch.arange(n, device=g_quat.device), best]

  # Phases.

  def start_lift(self, ids: torch.Tensor) -> None:
    env, rp, st = self.env, self.task.relative_placement, self.task.start
    n = len(ids)
    origins = env.scene.env_origins[ids]
    tcp_pos, tcp_quat = self.tcp_pose()
    tp, tq = tcp_pos[ids], tcp_quat[ids]
    opos, oquat = self.object.data.root_link_pos_w[ids], self.object.data.root_link_quat_w[ids]
    p_to, q_to = subtract_frame_transforms(tp, tq, opos, oquat)
    self.tcp_obj[ids] = torch.cat((p_to, q_to), dim=-1)

    a_pos, a_quat = placement.sample_anchor(
      n, rp.anchor_center, rp.anchor_radius, rp.anchor_yaw_deg, rp.anchor_yaw_range_deg,
      self.support_height, env.device,
    )
    self.anchor[ids] = torch.cat((a_pos, a_quat), dim=-1)
    g_pos, g_quat = placement.compose(a_pos, a_quat, rp.goal_pose)
    self.goal[ids] = torch.cat((g_pos, g_quat), dim=-1)
    top = torch.full((n,), -1.0e9, device=env.device)
    for k, (name, pose) in enumerate(rp.objects):
      s_pos, s_quat = placement.compose(a_pos, a_quat, pose)
      state = torch.zeros(n, 13, device=env.device)
      state[:, :3], state[:, 3:7] = s_pos, s_quat
      self.support_state[ids, k] = state
      pts = self.support_points[k][None].expand(n, -1, -1)
      z = s_pos[:, 2:3] + quat_apply(s_quat[:, None].expand(-1, pts.shape[1], 4), pts)[..., 2]
      top = torch.maximum(top, z.amax(dim=1))

    goal_q = self.choose_goal_quat(g_quat, oquat, q_to, g_pos + origins, self.robot.data.root_link_pos_w[ids])
    delta = placement.sample_ball_cap(n, st.ball_radius, st.min_height_above_goal, env.device)
    obj_t_pos = g_pos + origins + delta
    tcp_t_quat = quat_mul(goal_q, quat_inv(q_to))
    tcp_t_pos = obj_t_pos - quat_apply(tcp_t_quat, p_to)
    self.final_pos[ids], self.final_quat[ids] = tcp_t_pos, tcp_t_quat

    hang = torch.maximum(self.hang_below_tcp(ids, tcp_t_quat), self.hang_below_tcp(ids, tq))
    z_safe = origins[:, 2] + top + st.transport_clearance + hang
    self.z_safe[ids] = torch.maximum(z_safe, tp[:, 2])
    self.wp_pos[ids] = torch.cat((tp[:, :2], self.z_safe[ids, None]), dim=-1)
    self.wp_quat[ids] = tcp_t_quat
    self.set_phase(ids, LIFT)

  def place_supports(self, ids: torch.Tensor) -> None:
    origins = self.env.scene.env_origins[ids]
    for k, name in enumerate(self.supports):
      state = self.support_state[ids, k].clone()
      state[:, :3] += origins
      self.env.scene[name].write_root_state_to_sim(state, env_ids=ids)
    self.set_phase(ids, PLACED)

  def start_across(self, ids: torch.Tensor) -> None:
    self.wp_pos[ids] = torch.cat((self.final_pos[ids, :2], self.z_safe[ids, None]), dim=-1)
    self.wp_quat[ids] = self.final_quat[ids]
    self.set_phase(ids, ACROSS)

  def start_down(self, ids: torch.Tensor) -> None:
    self.wp_pos[ids], self.wp_quat[ids] = self.final_pos[ids], self.final_quat[ids]
    self.set_phase(ids, DOWN)

  def start_noise(self, ids: torch.Tensor) -> None:
    std = math.radians(self.task.start.tcp_rotation_noise_std_deg)
    n = len(ids)
    rotvec = (torch.randn(n, 3, device=self.env.device) * std).clamp(-2 * std, 2 * std)
    angle = torch.linalg.norm(rotvec, dim=-1)
    axis = rotvec / angle.clamp_min(1e-9)[:, None]
    q = quat_from_angle_axis(angle, axis)
    # Right-multiplied: about the TCP's own axes.
    self.wp_quat[ids] = quat_mul(self.final_quat[ids], q)
    self.wp_pos[ids] = self.final_pos[ids]
    self.set_phase(ids, NOISE)

  def start_settle(self, ids: torch.Tensor) -> None:
    lo, hi = self.task.start.settle_steps_range
    self.settle_left[ids] = torch.randint(lo, hi + 1, (len(ids),), device=self.env.device)
    self.set_phase(ids, SETTLE)

  def set_phase(self, ids: torch.Tensor, phase: int) -> None:
    self.phase[ids] = phase
    self.phase_steps[ids] = 0

  def restart(self, ids: torch.Tensor) -> None:
    if len(ids) == 0:
      return
    self.env.reset(env_ids=ids)
    self.set_phase(ids, GRASP)
    self.held_steps[ids] = 0

  # Control.

  def scripted_action(self) -> torch.Tensor:
    tcp_pos, tcp_quat = self.tcp_pose()
    dp = (self.wp_pos - tcp_pos) / self.pos_scale
    rot = axis_angle_from_quat(quat_mul(self.wp_quat, quat_inv(tcp_quat))) / self.rot_scale
    act = torch.cat((dp.clamp(-1, 1), rot.clamp(-1, 1), -torch.ones(self.env.num_envs, 1, device=self.env.device)), dim=-1)
    return act

  def at_waypoint(self, pos_tol: float, rot_tol: float) -> torch.Tensor:
    tcp_pos, tcp_quat = self.tcp_pose()
    return (torch.linalg.norm(self.wp_pos - tcp_pos, dim=-1) < pos_tol) & (
      placement.rotation_angle(self.wp_quat, tcp_quat) < rot_tol
    )

  def run(self) -> None:
    args, env = self.args, self.env
    reasons: dict[str, int] = {}
    kept: dict[str, list[torch.Tensor]] = {k: [] for k in starts_io.TENSOR_KEYS}
    num_kept = 0
    attempts = 0
    t0 = time.time()
    self.restart(torch.arange(env.num_envs, device=env.device))
    obs = self.wrapped.get_observations()
    step = 0

    def reject(mask: torch.Tensor, reason: str) -> torch.Tensor:
      ids = mask.nonzero(as_tuple=False).squeeze(-1)
      if len(ids):
        reasons[reason] = reasons.get(reason, 0) + len(ids)
      return ids

    while num_kept < args.rows:
      with torch.no_grad():
        policy_act = self.policy(obs)
      act = torch.where((self.phase == GRASP)[:, None], policy_act, self.scripted_action())
      obs, _, dones, _ = self.wrapped.step(act)
      step += 1
      self.phase_steps += 1
      restart = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      if dones.any():  # NaN resets: the env already restarted itself.
        ids = reject(dones.bool(), "physics_reset")
        self.set_phase(ids, GRASP)
        self.held_steps[ids] = 0

      grasped = task_state.is_grasped(env, **self.grasp_params)
      transporting = (self.phase != GRASP) & ~dones.bool()

      # 1. Grasp.
      g = (self.phase == GRASP) & ~dones.bool()
      hold = rewards.stable_hold(env, **self.hold_params) > 0.5
      self.held_steps = torch.where(g & hold, self.held_steps + 1, torch.zeros_like(self.held_steps))
      done_grasp = g & (self.held_steps >= args.hold_steps)
      restart |= g & (self.phase_steps >= args.grasp_timeout) & ~done_grasp
      reject(g & (self.phase_steps >= args.grasp_timeout) & ~done_grasp, "grasp_timeout")

      # Checks for every transport phase.
      lost = transporting & ~grasped
      restart |= lost
      reject(lost, "lost_grasp")
      placed = transporting & (self.phase >= ACROSS)
      newly_placed = transporting & (self.phase == PLACED)
      robot_hit = any_contact(env, self.robot_sensors[1:]) | (any_contact(env, self.robot_sensors[:1]))
      obj_hit = any_contact(env, self.object_sensors)
      bad = (placed | newly_placed) & (robot_hit | obj_hit) & ~restart
      restart |= bad
      reject(bad & newly_placed, "collision_at_placement")
      reject(bad & placed, "collision_in_transport")
      timeout = transporting & (self.phase_steps >= args.phase_timeout) & ~restart
      restart |= timeout
      for ph in range(LIFT, SETTLE + 1):
        reject(timeout & (self.phase == ph), f"timeout_{PHASE_NAMES[ph]}")

      ok = ~restart
      via = self.at_waypoint(VIA_POS_TOL, VIA_ROT_TOL)
      coarse = self.at_waypoint(POS_TOL, ROT_TOL)
      fine = self.at_waypoint(FINE_POS_TOL, FINE_ROT_TOL)
      # Advance phases (each env at most one step per control step).
      settle_done = ok & (self.phase == SETTLE) & (self.settle_left <= 0)
      self.settle_left -= ((self.phase == SETTLE) & ok).long()
      to_settle = ok & (self.phase == NOISE) & fine
      to_noise = ok & (self.phase == DOWN) & coarse
      to_down = ok & (self.phase == ACROSS) & via
      to_across = ok & (self.phase == PLACED)
      to_place = ok & (self.phase == LIFT) & via
      for mask, fn in (
        (to_settle, self.start_settle),
        (to_noise, self.start_noise),
        (to_down, self.start_down),
        (to_across, self.start_across),
        (to_place, self.place_supports),
      ):
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
          fn(ids)
      ids = done_grasp.nonzero(as_tuple=False).squeeze(-1)
      if len(ids):
        self.start_lift(ids)
        attempts += len(ids)

      # Record.
      ids = settle_done.nonzero(as_tuple=False).squeeze(-1)
      if len(ids):
        num_kept += self.record(ids, kept, reasons)
        restart[ids] = True

      self.restart(restart.nonzero(as_tuple=False).squeeze(-1))
      if restart.any():
        obs = self.wrapped.get_observations()
      if step % 50 == 0:
        counts = torch.bincount(self.phase, minlength=7).tolist()
        print(
          f"step {step}: kept {num_kept}/{args.rows} of {attempts} attempts, "
          f"phases {dict(zip(PHASE_NAMES, counts))}, rejected {reasons}, {time.time() - t0:.0f}s",
          flush=True,
        )
    self.save(kept, reasons, attempts, num_kept, time.time() - t0)

  def record(self, ids: torch.Tensor, kept: dict, reasons: dict) -> int:
    env, st = self.env, self.task.start
    origins = env.scene.env_origins[ids]
    opos = self.object.data.root_link_pos_w[ids]
    delta = opos - (self.goal[ids, :3] + origins)
    checks = {
      "lost_grasp_at_record": task_state.is_grasped(env, **self.grasp_params)[ids],
      "outside_ball": (torch.linalg.norm(delta, dim=-1) <= st.ball_radius)
      & (delta[:, 2] >= st.min_height_above_goal - 0.002),
      "joint_limits": joint_pos_limits(env, self.arm_cfg)[ids] <= 0.0,
    }
    moved = torch.zeros(len(ids), dtype=torch.bool, device=env.device)
    for k, name in enumerate(self.supports):
      data = env.scene[name].data
      p0 = self.support_state[ids, k, :3] + origins
      moved |= torch.linalg.norm(data.root_link_pos_w[ids] - p0, dim=-1) > 0.002
      moved |= placement.rotation_angle(data.root_link_quat_w[ids], self.support_state[ids, k, 3:7]) > math.radians(2)
    checks["supports_moved"] = ~moved
    good = torch.ones(len(ids), dtype=torch.bool, device=env.device)
    for reason, passed in checks.items():
      failed = good & ~passed
      if failed.any():
        reasons[reason] = reasons.get(reason, 0) + int(failed.sum())
      good &= passed
    ids = ids[good]
    origins = origins[good]
    if len(ids) == 0:
      return 0
    obj_state = torch.cat(
      (self.object.data.root_link_pos_w[ids] - origins, self.object.data.root_link_quat_w[ids],
       self.object.data.root_link_lin_vel_w[ids], self.object.data.root_link_ang_vel_w[ids]), dim=-1,
    )
    sup = []
    for name in self.supports:
      d = env.scene[name].data
      sup.append(torch.cat((d.root_link_pos_w[ids] - origins, d.root_link_quat_w[ids],
                            d.root_link_lin_vel_w[ids], d.root_link_ang_vel_w[ids]), dim=-1))
    kept["robot_joint_pos"].append(self.robot.data.joint_pos[ids].clone())
    kept["robot_joint_vel"].append(self.robot.data.joint_vel[ids].clone())
    kept["object_root_state"].append(obj_state)
    kept["support_root_state"].append(torch.stack(sup, dim=1))
    kept["goal_pose"].append(self.goal[ids].clone())
    kept["anchor_pose"].append(self.anchor[ids].clone())
    kept["variant_scale"].append(self.obj_scale[ids].clone())
    return len(ids)

  def save(self, kept: dict, reasons: dict, attempts: int, num_kept: int, seconds: float) -> None:
    rows = {k: torch.cat(v)[: self.args.rows] for k, v in kept.items()}
    sizes = self.task.domain_randomization.object_size_variants
    meta = {
      **json.loads(json.dumps(starts_io.generation_key(self.task))),
      "scene_name": self.scene.scene_name,
      "variant_scales": sorted(set(round(s, 6) for s in rows["variant_scale"].tolist())),
      "size_variants_enabled": sizes.enabled,
      "scene": str(Path(self.args.scene).resolve()),
      "task": str(Path(self.args.task).resolve()),
      "grasp_task": str(Path(self.args.grasp_task).resolve()),
      "grasp_checkpoint": str(Path(self.args.grasp_checkpoint).resolve()),
      "grasp_checkpoint_sha256": starts_io.file_sha256(Path(self.args.grasp_checkpoint)),
      "difficulty": self.args.difficulty,
      "seed": self.args.seed,
      "attempts": attempts,
      "kept": num_kept,
      "rejected": reasons,
      "seconds": round(seconds, 1),
      "git_commit": _git_commit(),
    }
    out = Path(self.args.out)
    starts_io.save(out, rows, self.robot.joint_names, self.supports, meta)
    print(f"wrote {out} ({rows['variant_scale'].shape[0]} rows) and {starts_io.meta_path(out).name}")
    print(json.dumps({k: meta[k] for k in ("attempts", "kept", "rejected", "seconds")}, indent=2))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="release scene JSON")
  parser.add_argument("--task", default=str(DEFAULT_TASK), help="release task JSON")
  parser.add_argument("--grasp-checkpoint", required=True, help="grasp policy model_N.pt")
  parser.add_argument(
    "--grasp-task", default=None, help="grasp task JSON (default: the checkpoint run's params/task.json)"
  )
  parser.add_argument("--out", default=None, help="output .pt (default: the release task's start.dataset)")
  parser.add_argument("--rows", type=int, default=50_000)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--difficulty", type=float, default=1.0, help="grasp curriculum difficulty to run at")
  parser.add_argument("--hold-steps", type=int, default=10)
  parser.add_argument("--grasp-timeout", type=int, default=250)
  parser.add_argument("--phase-timeout", type=int, default=120)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.grasp_task is None:
    args.grasp_task = str(Path(args.grasp_checkpoint).resolve().parent / "params" / "task.json")
  if args.out is None:
    dataset = load_release_task_config(args.task).start.dataset
    if dataset is None:
      raise SystemExit("the task has no start.dataset; pass --out")
    args.out = str(dataset)
  torch.manual_seed(args.seed)
  Generator(args).run()


if __name__ == "__main__":
  main()
