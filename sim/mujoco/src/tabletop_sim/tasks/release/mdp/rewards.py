"""Release reward terms (unweighted, stateless; weights and signs come from the
term configs). Speed, joint-limit, gripper-switch and contact-fraction terms
reuse grasp's ``rewards`` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import rewards as grasp_rewards
from tabletop_sim.tasks.grasp.mdp import task_state as grasp_state
from tabletop_sim.tasks.release.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def place(
  env: ManagerBasedRlEnv, command_name: str, object_cfg: SceneEntityCfg, pos_scale: float, rot_scale: float
) -> torch.Tensor:
  """In (0, 1]: 1 - tanh(pos_scale * d_pos + rot_scale * d_rot)."""
  d_pos, d_rot = task_state.placement_error(env, command_name, object_cfg)
  return 1.0 - torch.tanh(pos_scale * d_pos + rot_scale * d_rot)


def released(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  pos_tol: float,
  rot_tol_deg: float,
  sensor_names: tuple[str, ...],
  gripper_cfg: SceneEntityCfg,
  open_pos: float,
  closed_pos: float,
  max_closure: float,
) -> torch.Tensor:
  """In [0, 1]: placed within the tolerances, no finger contact, gripper open,
  times the object's stillness."""
  ok = (
    task_state.is_placed(env, command_name, object_cfg, pos_tol, rot_tol_deg)
    & task_state.fingers_clear(env, sensor_names)
    & task_state.gripper_open(env, gripper_cfg, open_pos, closed_pos, max_closure)
  )
  return ok.float() * task_state.stillness(env, object_cfg)


def retreat(
  env: ManagerBasedRlEnv,
  tcp_cfg: SceneEntityCfg,
  retreat_height: float,
  **released_params,
) -> torch.Tensor:
  """``released`` times how far (as a fraction of ``retreat_height``, capped at
  1) the TCP has risen above the object's centre."""
  tcp_pos, _ = grasp_state.tcp_pose_w(env, tcp_cfg)
  obj_pos, _ = grasp_state.object_pose_w(env, released_params["object_cfg"])
  lift = ((tcp_pos[:, 2] - obj_pos[:, 2]) / retreat_height).clamp(0.0, 1.0)
  return released(env, **released_params) * lift


def robot_collision(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """In [0, 1]: the largest of the sensors' contact fractions over the control
  step's substeps (grasp's ``table_contact`` per sensor)."""
  fractions = torch.stack([grasp_rewards.table_contact(env, name) for name in sensor_names])
  return fractions.max(dim=0).values


def support_disturb(
  env: ManagerBasedRlEnv, command_name: str, support_names: tuple[str, ...], pos_scale: float, rot_scale: float
) -> torch.Tensor:
  """In [0, 1): mean over relative objects of tanh(pos_scale * d + rot_scale * angle)."""
  d, angle = task_state.support_displacement(env, command_name, support_names)
  return torch.tanh(pos_scale * d + rot_scale * angle).mean(dim=-1)


def gripper_opened(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """True on the control step whose gripper target switched from closed to open
  (``GripperAction.switched`` toward the open side)."""
  term = env.action_manager.get_term(action_name)
  middle = 0.5 * (term.cfg.open_pos + term.cfg.closed_pos)  # type: ignore[attr-defined]
  toward_open = (term._target[:, 0] - middle) * (term.cfg.open_pos - term.cfg.closed_pos) > 0  # type: ignore[attr-defined]
  return term.switched & toward_open  # type: ignore[attr-defined]


def release_moving(env: ManagerBasedRlEnv, action_name: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
  """In [0, 1): on the step the gripper is commanded open, 1 - stillness of the
  object (about 1 when it moves at ~0.1 m/s or spins at ~1 rad/s, about 0 at
  rest); 0 on every other step. Use with a nonpositive weight."""
  return gripper_opened(env, action_name).float() * (1.0 - task_state.stillness(env, object_cfg))
