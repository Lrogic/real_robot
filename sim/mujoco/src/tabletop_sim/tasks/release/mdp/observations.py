"""Release observation terms, in the robot base frame. ``qpos``, ``qvel``,
``ee_pose``, ``gripper_width``, ``object_pose`` and ``is_grasped`` reuse grasp's
terms. Relative poses are 7 numbers: the position difference in base axes and
the rotation ``q_to * q_from^-1`` in the base frame, with w >= 0."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul, quat_unique
from tabletop_sim.tasks.grasp.mdp import task_state
from tabletop_sim.tasks.release.mdp import placement

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from tabletop_sim.tasks.release.mdp.commands import PerceivedGoal, ReleaseGoalCommand


def relative_pose_b(
  env: ManagerBasedRlEnv,
  robot_name: str,
  pos_from: torch.Tensor,
  quat_from: torch.Tensor,
  pos_to: torch.Tensor,
  quat_to: torch.Tensor,
) -> torch.Tensor:
  """World poses -> [dp (base axes), q_b_to * q_b_from^-1 (w >= 0)]."""
  q_bw = quat_inv(env.scene[robot_name].data.root_link_quat_w)
  dp = quat_apply_inverse(env.scene[robot_name].data.root_link_quat_w, pos_to - pos_from)
  q_to_b, q_from_b = quat_mul(q_bw, quat_to), quat_mul(q_bw, quat_from)
  return torch.cat((dp, quat_unique(placement.relative_rotation(q_to_b, q_from_b))), dim=-1)


def goal(env: ManagerBasedRlEnv, command_name: str) -> ReleaseGoalCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def tcp_to_obj_pose(
  env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
  tcp_pos, tcp_quat = task_state.tcp_pose_w(env, tcp_cfg)
  obj_pos, obj_quat = task_state.object_pose_w(env, object_cfg)
  return relative_pose_b(env, tcp_cfg.name, tcp_pos, tcp_quat, obj_pos, obj_quat)


def obj_to_goal_pose(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, robot_name: str = "robot"
) -> torch.Tensor:
  """Object to the clean goal, using the symmetric goal orientation closest to the object's."""
  obj_pos, obj_quat = task_state.object_pose_w(env, object_cfg)
  g = goal(env, command_name)
  return relative_pose_b(env, robot_name, obj_pos, obj_quat, g.goal_pos, g.closest_goal_quat(obj_quat))


def support_poses(
  env: ManagerBasedRlEnv, support_names: tuple[str, ...], robot_name: str = "robot"
) -> torch.Tensor:
  """Critic only: every relative object's pose in the base frame, concatenated."""
  out = []
  for name in support_names:
    data = env.scene[name].data
    out.append(task_state.root_relative_pose(env, robot_name, data.root_link_pos_w, data.root_link_quat_w))
  return torch.cat(out, dim=-1)


# Actor counterparts: the perceived object pose (grasp's PerceivedObjectPose)
# and the perceived goal (PerceivedGoal).


def perceived_tcp_to_obj_pose(
  env: ManagerBasedRlEnv, tcp_cfg: SceneEntityCfg, perception_name: str
) -> torch.Tensor:
  tcp_pos, tcp_quat = task_state.tcp_pose_w(env, tcp_cfg)
  p = task_state.perception(env, perception_name)
  return relative_pose_b(env, tcp_cfg.name, tcp_pos, tcp_quat, p.pos_w, p.quat_w)


def perceived_obj_to_goal_pose(
  env: ManagerBasedRlEnv,
  perception_name: str,
  goal_perception_name: str,
  symmetry: str,
  robot_name: str = "robot",
) -> torch.Tensor:
  p = task_state.perception(env, perception_name)
  g: PerceivedGoal = env.command_manager.get_term(goal_perception_name)  # type: ignore[assignment]
  goal_quat = placement.closest_symmetric_quat(g.quat_w, p.quat_w, symmetry)
  return relative_pose_b(env, robot_name, p.pos_w, p.quat_w, g.pos_w, goal_quat)
