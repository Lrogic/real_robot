"""Observation terms. Poses are relative to the robot root; displacement vectors
are expressed in world axes (same conventions as the original ManiSkill task)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_pos(
  env: ManagerBasedRlEnv, arm_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Arm joints then (left, right) gripper joints."""
  q = env.scene[arm_cfg.name].data.joint_pos
  return torch.cat((q[:, arm_cfg.joint_ids], q[:, gripper_cfg.joint_ids]), dim=-1)


def joint_vel(
  env: ManagerBasedRlEnv, arm_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg
) -> torch.Tensor:
  qd = env.scene[arm_cfg.name].data.joint_vel
  return torch.cat((qd[:, arm_cfg.joint_ids], qd[:, gripper_cfg.joint_ids]), dim=-1)


def ee_pose(env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg) -> torch.Tensor:
  pos, quat = task_state.tcp_pose_w(env, tcp_cfg)
  return task_state.root_relative_pose(env, tcp_cfg.name, pos, quat)


def gripper_width(
  env: ManagerBasedRlEnv, gripper_cfg: SceneEntityCfg, open_pos: float, closed_pos: float
) -> torch.Tensor:
  """Normalised closure (closed = 1, open = 0); name kept from the old task."""
  q = env.scene[gripper_cfg.name].data.joint_pos[:, gripper_cfg.joint_ids]
  return task_state.gripper_closure(q[:, 0], q[:, 1], open_pos, closed_pos).unsqueeze(-1)


def object_pose(
  env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg, robot_name: str = "robot"
) -> torch.Tensor:
  pos, quat = task_state.object_pose_w(env, object_cfg)
  return task_state.root_relative_pose(env, robot_name, pos, quat)


def tcp_to_obj_pos(
  env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  tcp_pos, _ = task_state.tcp_pose_w(env, tcp_cfg)
  obj_pos, _ = task_state.object_pose_w(env, object_cfg)
  return obj_pos - tcp_pos


def obj_to_goal_pos(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  obj_pos, _ = task_state.object_pose_w(env, object_cfg)
  return task_state.goal(env, command_name).target_pos - obj_pos


def is_grasped(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, str],
  finger_cfg: SceneEntityCfg,
  finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]],
  min_force: float,
  max_angle_deg: float,
) -> torch.Tensor:
  grasped = task_state.is_grasped(
    env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
  )
  return grasped.float().unsqueeze(-1)
