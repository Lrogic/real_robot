"""Grasp tracker: per-episode task state for the reach-grasp-lift-hold task.

The tracker owns everything that has memory across steps (initial object pose,
lift target, stable-hold counter, success latch) plus the per-step task metrics
that observations, rewards and terminations share.

Update order within one env step (see ``ManagerBasedRlEnv.step``):

1. Terminations and rewards call :meth:`GraspTracker.evaluate`. The first call in
   a step recomputes metrics and advances the hold counter exactly once.
2. After resets and the kinematics refresh, the command manager calls
   ``compute`` which initialises freshly reset envs and refreshes the
   instantaneous metrics (no counter advance) for the observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Absolute rotation angle between two (w, x, y, z) quaternions."""
  q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True).clamp_min(1e-9)
  q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True).clamp_min(1e-9)
  dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0)
  return 2.0 * torch.acos(dot)


def gripper_closure(
  left: torch.Tensor, right: torch.Tensor, open_pos: float, closed_pos: float
) -> torch.Tensor:
  """Unitless jaw closure: fully closed = 1, fully open = 0."""
  opening = ((left + right) * 0.5 - closed_pos) / (open_pos - closed_pos)
  return 1.0 - opening.clamp(0.0, 1.0)


@dataclass(kw_only=True)
class GraspTrackerCfg(CommandTermCfg):
  object_name: str
  tcp_site: str
  arm_joint_names: tuple[str, ...]
  gripper_joint_names: tuple[str, str]
  finger_sensor_names: tuple[str, str]
  finger_bodies: tuple[str, str]
  finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]]
  gripper_open: float
  gripper_closed: float
  hold_steps: int
  lift_threshold: float
  target_lift_height: float
  stable_linear_vel_tol: float
  stable_angular_vel_tol: float
  arm_joint_vel_tol: float
  gripper_joint_vel_tol: float
  grasp_min_force: float = 0.2
  grasp_max_angle_deg: float = 110.0
  robot_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> GraspTracker:
    return GraspTracker(self, env)


class GraspTracker(CommandTerm):
  cfg: GraspTrackerCfg

  def __init__(self, cfg: GraspTrackerCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.robot_name]
    self.object: Entity = env.scene[cfg.object_name]
    self.finger_sensors: tuple[ContactSensor, ContactSensor] = (
      env.scene[cfg.finger_sensor_names[0]],
      env.scene[cfg.finger_sensor_names[1]],
    )

    self._tcp_id = self.robot.find_sites(cfg.tcp_site, preserve_order=True)[0][0]
    self._arm_ids = torch.tensor(
      self.robot.find_joints(cfg.arm_joint_names, preserve_order=True)[0], device=self.device
    )
    self._gripper_ids = torch.tensor(
      self.robot.find_joints(cfg.gripper_joint_names, preserve_order=True)[0],
      device=self.device,
    )
    self._finger_body_ids = self.robot.find_bodies(cfg.finger_bodies, preserve_order=True)[0]
    self._finger_axes = torch.tensor(cfg.finger_outward_axes, device=self.device)
    self._cos_max_angle = math.cos(math.radians(cfg.grasp_max_angle_deg))

    n, dev = self.num_envs, self.device
    self.initial_object_pos = torch.zeros(n, 3, device=dev)
    self.initial_object_quat = torch.zeros(n, 4, device=dev)
    self.initial_object_quat[:, 0] = 1.0
    self.obj_initial_pose_b = torch.zeros(n, 7, device=dev)
    self.target_pos = torch.zeros(n, 3, device=dev)
    self.hold_counter = torch.zeros(n, dtype=torch.long, device=dev)
    self.success = torch.zeros(n, dtype=torch.bool, device=dev)
    self.first_success = torch.zeros(n, dtype=torch.bool, device=dev)
    self.was_success = torch.zeros(n, dtype=torch.bool, device=dev)
    self._needs_init = torch.ones(n, dtype=torch.bool, device=dev)
    self._last_eval_step = -1

    zeros = lambda: torch.zeros(n, device=dev)  # noqa: E731
    bools = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    self.is_grasped, self.is_lifted, self.is_static, self.success_now = (
      bools(), bools(), bools(), bools()
    )
    self.tcp_to_obj_dist, self.obj_to_target_dist, self.xy_displacement = (
      zeros(), zeros(), zeros()
    )
    self.obj_lin_speed, self.obj_ang_speed = zeros(), zeros()
    self.arm_vel_max, self.gripper_vel_max = zeros(), zeros()

    self.metrics["success"] = zeros()
    self.metrics["grasped_frac"] = zeros()
    self.metrics["max_lift"] = zeros()

  # State accessors (world frame unless noted).

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def root_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    pose = self.robot.data.root_link_pose_w
    return pose[:, :3], pose[:, 3:7]

  @property
  def tcp_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._tcp_id]

  @property
  def tcp_quat_w(self) -> torch.Tensor:
    return self.robot.data.site_quat_w[:, self._tcp_id]

  @property
  def object_pos_w(self) -> torch.Tensor:
    return self.object.data.root_link_pos_w

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.object.data.root_link_quat_w

  def to_root_frame(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    root_pos, root_quat = self.root_pose_w
    pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)
    return torch.cat((pos_b, quat_b), dim=-1)

  def arm_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self._arm_ids]

  def gripper_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self._gripper_ids]

  def closure(self) -> torch.Tensor:
    q = self.gripper_joint_pos()
    return gripper_closure(q[:, 0], q[:, 1], self.cfg.gripper_open, self.cfg.gripper_closed)

  # Task evaluation.

  def evaluate(self) -> None:
    """Advance the stable-hold state once per env step (idempotent within a step)."""
    step = self._env.common_step_counter
    if step == self._last_eval_step:
      return
    self._last_eval_step = step
    self._refresh()
    next_counter = torch.where(
      self.success_now, self.hold_counter + 1, torch.zeros_like(self.hold_counter)
    )
    self.hold_counter[:] = next_counter
    self.success[:] = self.hold_counter >= self.cfg.hold_steps
    self.first_success[:] = self.success & ~self.was_success
    self.was_success |= self.success

    self.metrics["success"] = torch.maximum(self.metrics["success"], self.success.float())
    self.metrics["grasped_frac"] += self.is_grasped.float() / self._env.max_episode_length
    lift = self.object_pos_w[:, 2] - self.initial_object_pos[:, 2]
    self.metrics["max_lift"] = torch.maximum(self.metrics["max_lift"], lift)

  def _refresh(self) -> None:
    obj_pos = self.object_pos_w
    self.is_grasped[:] = self._compute_is_grasped()
    self.tcp_to_obj_dist[:] = torch.linalg.norm(obj_pos - self.tcp_pos_w, dim=-1)
    self.obj_to_target_dist[:] = torch.linalg.norm(obj_pos - self.target_pos, dim=-1)
    self.is_lifted[:] = self.obj_to_target_dist <= self.cfg.lift_threshold
    self.xy_displacement[:] = torch.linalg.norm(
      obj_pos[:, :2] - self.initial_object_pos[:, :2], dim=-1
    )
    self.obj_lin_speed[:] = torch.linalg.norm(self.object.data.root_link_lin_vel_w, dim=-1)
    self.obj_ang_speed[:] = torch.linalg.norm(self.object.data.root_link_ang_vel_w, dim=-1)
    qvel = self.robot.data.joint_vel
    self.arm_vel_max[:] = qvel[:, self._arm_ids].abs().amax(dim=-1)
    self.gripper_vel_max[:] = qvel[:, self._gripper_ids].abs().amax(dim=-1)
    # Revolute (rad/s) and prismatic (m/s) velocities are thresholded separately.
    self.is_static[:] = (
      (self.obj_lin_speed < self.cfg.stable_linear_vel_tol)
      & (self.obj_ang_speed < self.cfg.stable_angular_vel_tol)
      & (self.arm_vel_max < self.cfg.arm_joint_vel_tol)
      & (self.gripper_vel_max < self.cfg.gripper_joint_vel_tol)
    )
    self.success_now[:] = self.is_grasped & self.is_lifted & self.is_static

  def finger_contact_forces(self) -> torch.Tensor:
    """Net contact force on each finger from the object, world frame. [N, 2, 3]."""
    forces = [s.data.force for s in self.finger_sensors]
    assert all(f is not None for f in forces)
    # The sensor reports the primary's (finger's) force on the secondary (object).
    return -torch.stack([f[:, 0] for f in forces], dim=1)  # type: ignore[index]

  def _compute_is_grasped(self) -> torch.Tensor:
    """Both fingers push on the object with enough force, roughly along their
    closing direction (same criterion as ManiSkill's ``is_grasping``)."""
    forces = self.finger_contact_forces()
    finger_quat = self.robot.data.body_link_quat_w[:, self._finger_body_ids]
    axes = quat_apply(finger_quat, self._finger_axes.expand(self.num_envs, 2, 3))
    magnitude = torch.linalg.norm(forces, dim=-1)
    cos = torch.sum(forces * axes, dim=-1) / magnitude.clamp_min(1e-9)
    per_finger = (magnitude >= self.cfg.grasp_min_force) & (cos >= self._cos_max_angle)
    return per_finger.all(dim=-1)

  # CommandTerm hooks.

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Called on reset before kinematics are refreshed; defer to _update_metrics.
    self._needs_init[env_ids] = True
    self.hold_counter[env_ids] = 0
    self.success[env_ids] = False
    self.first_success[env_ids] = False
    self.was_success[env_ids] = False

  def _update_metrics(self) -> None:
    init_ids = self._needs_init.nonzero(as_tuple=False).squeeze(-1)
    if len(init_ids) > 0:
      obj_pos = self.object_pos_w[init_ids]
      obj_quat = self.object_quat_w[init_ids]
      self.initial_object_pos[init_ids] = obj_pos
      self.initial_object_quat[init_ids] = obj_quat
      self.target_pos[init_ids] = obj_pos
      self.target_pos[init_ids, 2] += self.cfg.target_lift_height
      self.obj_initial_pose_b[init_ids] = self.to_root_frame(
        self.object_pos_w, self.object_quat_w
      )[init_ids]
      self._needs_init[init_ids] = False
    self._refresh()

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for i in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[i].cpu().numpy(),
        radius=0.008,
        color=(0.1, 0.9, 0.2, 0.6),
        label=f"lift_target_{i}",
      )
