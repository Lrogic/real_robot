"""Stateless task checks shared by rewards, observations, terminations and
metrics. Every function recomputes from the current sim state and the goal
command; none of them keeps memory between calls.

Robot elements are passed as resolved ``SceneEntityCfg``s (``tcp_cfg`` with one
site, ``arm_cfg`` / ``gripper_cfg`` with joints, ``finger_cfg`` with the two
finger bodies in (left, right) order); the object as ``object_cfg``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, subtract_frame_transforms
from tabletop_sim.tasks.grasp.mdp.commands import GraspGoalCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


# Pure math.


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


@functools.lru_cache(maxsize=32)
def _const(values: tuple, device: str) -> torch.Tensor:
  return torch.tensor(values, dtype=torch.float, device=device)


# State accessors.


def goal(env: ManagerBasedRlEnv, command_name: str) -> GraspGoalCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GraspGoalCommand)
  return term


def tcp_pose_w(env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene[tcp_cfg.name]
  # A selection covering every site resolves to slice(None) instead of a list.
  ids = tcp_cfg.site_ids
  site = 0 if isinstance(ids, slice) else ids[0]
  return robot.data.site_pos_w[:, site], robot.data.site_quat_w[:, site]


def object_pose_w(
  env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  obj: Entity = env.scene[object_cfg.name]
  return obj.data.root_link_pos_w, obj.data.root_link_quat_w


def root_relative_pose(
  env: ManagerBasedRlEnv, robot_name: str, pos_w: torch.Tensor, quat_w: torch.Tensor
) -> torch.Tensor:
  """World pose -> [x, y, z, qw, qx, qy, qz] in the robot root frame."""
  root = env.scene[robot_name].data.root_link_pose_w
  pos_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], pos_w, quat_w)
  return torch.cat((pos_b, quat_b), dim=-1)


# Grasp detection.


def finger_contact_forces(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, str]
) -> torch.Tensor:
  """Net contact force the object exerts on each finger, world frame. [N, 2, 3]."""
  forces = []
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    assert sensor.data.force is not None
    forces.append(sensor.data.force[:, 0])
  # The sensor reports the primary's (finger's) force on the secondary (object).
  return -torch.stack(forces, dim=1)


def is_grasped(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]],
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  """Both fingers push on the object with at least ``min_force`` newtons, within
  ``max_angle_deg`` of their outward axis (ManiSkill's ``is_grasping`` test)."""
  forces = finger_contact_forces(env, sensor_names)
  robot: Entity = env.scene[finger_cfg.name]
  finger_quat = robot.data.body_link_quat_w[:, finger_cfg.body_ids]
  axes_local = _const(finger_outward_axes, env.device).expand(env.num_envs, 2, 3)
  axes = quat_apply(finger_quat, axes_local)
  magnitude = torch.linalg.norm(forces, dim=-1)
  cos = torch.sum(forces * axes, dim=-1) / magnitude.clamp_min(1e-9)
  per_finger = (magnitude >= min_force) & (cos >= math.cos(math.radians(max_angle_deg)))
  return per_finger.all(dim=-1)


# Distances.


def tcp_to_object_dist(
  env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  tcp_pos, _ = tcp_pose_w(env, tcp_cfg)
  obj_pos, _ = object_pose_w(env, object_cfg)
  return torch.linalg.norm(obj_pos - tcp_pos, dim=-1)


def object_to_target_dist(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  obj_pos, _ = object_pose_w(env, object_cfg)
  return torch.linalg.norm(obj_pos - goal(env, command_name).target_pos, dim=-1)


def is_lifted(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, lift_threshold: float
) -> torch.Tensor:
  return object_to_target_dist(env, command_name, object_cfg) <= lift_threshold


def object_xy_displacement(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  obj_pos, _ = object_pose_w(env, object_cfg)
  initial = goal(env, command_name).initial_object_pos
  return torch.linalg.norm(obj_pos[:, :2] - initial[:, :2], dim=-1)


def object_rotation_from_initial(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  _, obj_quat = object_pose_w(env, object_cfg)
  return quaternion_angle(goal(env, command_name).initial_object_quat, obj_quat)


# Velocities and stability.


def object_speeds(
  env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """(linear m/s, angular rad/s) speed of the object."""
  data = env.scene[object_cfg.name].data
  return (
    torch.linalg.norm(data.root_link_lin_vel_w, dim=-1),
    torch.linalg.norm(data.root_link_ang_vel_w, dim=-1),
  )


def is_static(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  arm_cfg: SceneEntityCfg,
  gripper_cfg: SceneEntityCfg,
  linear_vel_tol: float,
  angular_vel_tol: float,
  arm_joint_vel_tol: float,
  gripper_joint_vel_tol: float,
) -> torch.Tensor:
  """Object, arm and gripper are all nearly still. Revolute (rad/s) and
  prismatic (m/s) joint speeds are thresholded separately."""
  lin, ang = object_speeds(env, object_cfg)
  qvel = env.scene[arm_cfg.name].data.joint_vel
  arm_max = qvel[:, arm_cfg.joint_ids].abs().amax(dim=-1)
  gripper_max = qvel[:, gripper_cfg.joint_ids].abs().amax(dim=-1)
  return (
    (lin < linear_vel_tol)
    & (ang < angular_vel_tol)
    & (arm_max < arm_joint_vel_tol)
    & (gripper_max < gripper_joint_vel_tol)
  )
