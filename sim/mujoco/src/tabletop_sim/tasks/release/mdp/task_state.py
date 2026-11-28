"""Stateless release checks shared by rewards, terminations and metrics (the
counterpart of grasp's ``task_state``): placement error to the closest
symmetric goal, finger clearance, gripper opening, stillness, and how far the
relative objects moved."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state as grasp_state
from tabletop_sim.tasks.release.mdp import placement

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from tabletop_sim.tasks.release.mdp.commands import ReleaseGoalCommand


def goal(env: ManagerBasedRlEnv, command_name: str) -> ReleaseGoalCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def placement_error(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """(position error m, rotation error rad to the closest symmetric goal)."""
  pos, quat = grasp_state.object_pose_w(env, object_cfg)
  g = goal(env, command_name)
  d_pos = torch.linalg.norm(pos - g.goal_pos, dim=-1)
  d_rot = placement.symmetric_rotation_distance(g.goal_quat, quat, g.cfg.symmetry)
  return d_pos, d_rot


def is_placed(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, pos_tol: float, rot_tol_deg: float
) -> torch.Tensor:
  d_pos, d_rot = placement_error(env, command_name, object_cfg)
  return (d_pos <= pos_tol) & (d_rot <= math.radians(rot_tol_deg))


def fingers_clear(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """No finger touches the object."""
  touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    assert sensor.data.found is not None
    touching |= (sensor.data.found > 0).reshape(env.num_envs, -1).any(dim=-1)
  return ~touching


def gripper_open(
  env: ManagerBasedRlEnv, gripper_cfg: SceneEntityCfg, open_pos: float, closed_pos: float, max_closure: float
) -> torch.Tensor:
  q = env.scene[gripper_cfg.name].data.joint_pos[:, gripper_cfg.joint_ids]
  return grasp_state.gripper_closure(q[:, 0], q[:, 1], open_pos, closed_pos) <= max_closure


def stillness_from_speeds(
  lin: torch.Tensor, ang: torch.Tensor, speed_scale: float = 5.0, spin_scale: float = 2.0
) -> torch.Tensor:
  """In (0, 1]: (1 - tanh(speed_scale * speed)) * (1 - tanh(spin_scale * spin)); 1 at rest."""
  return (1.0 - torch.tanh(speed_scale * lin)) * (1.0 - torch.tanh(spin_scale * ang))


def stillness(env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
  """In (0, 1]: 1 at rest (grasp's ``stable_hold`` form, scales 5 and 2)."""
  lin, ang = grasp_state.object_speeds(env, object_cfg)
  return stillness_from_speeds(lin, ang)




def support_displacement(
  env: ManagerBasedRlEnv, command_name: str, support_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per env and relative object ``(N, S)``: (distance m, rotation rad) from its reset pose."""
  g = goal(env, command_name)
  pos = torch.stack([env.scene[n].data.root_link_pos_w for n in support_names], dim=1)
  quat = torch.stack([env.scene[n].data.root_link_quat_w for n in support_names], dim=1)
  return torch.linalg.norm(pos - g.support_pos0, dim=-1), placement.rotation_angle(quat, g.support_quat0)
