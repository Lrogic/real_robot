"""Observation terms. Poses are relative to the robot root; displacement vectors
are expressed in world axes (same conventions as the original ManiSkill task)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tabletop_sim.tasks.grasp.mdp.commands import GraspTracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _tracker(env: ManagerBasedRlEnv, command_name: str) -> GraspTracker:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, GraspTracker)
  return term


def joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Arm joints then (left, right) gripper joints."""
  t = _tracker(env, command_name)
  return torch.cat((t.arm_joint_pos(), t.gripper_joint_pos()), dim=-1)


def joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  qvel = t.robot.data.joint_vel
  return torch.cat((qvel[:, t._arm_ids], qvel[:, t._gripper_ids]), dim=-1)


def ee_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.to_root_frame(t.tcp_pos_w, t.tcp_quat_w)


def gripper_width(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalised closure (closed = 1, open = 0); name kept from the old task."""
  return _tracker(env, command_name).closure().unsqueeze(-1)


def object_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.to_root_frame(t.object_pos_w, t.object_quat_w)


def tcp_to_obj_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.object_pos_w - t.tcp_pos_w


def obj_to_goal_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.target_pos - t.object_pos_w


def tcp_to_goal_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _tracker(env, command_name)
  return t.target_pos - t.tcp_pos_w


def obj_initial_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _tracker(env, command_name).obj_initial_pose_b


def is_grasped(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _tracker(env, command_name).is_grasped.float().unsqueeze(-1)
