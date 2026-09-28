from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from tabletop_sim.tasks.grasp.mdp import task_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg


class StableHoldSuccess(ManagerTermBase):
  """Success once the object has been grasped, at the lift target and still for
  ``hold_steps`` consecutive control steps.

  Owns the per-env hold counter. The termination manager calls this once per
  env step (before rewards), and ``reset`` when an episode ends. Rewards read
  the result via ``env.termination_manager.get_term(name)``.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.hold_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    hold_steps: int,
    object_cfg: SceneEntityCfg,
    lift_threshold: float,
    sensor_names: tuple[str, str],
    finger_cfg: SceneEntityCfg,
    finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]],
    min_force: float,
    max_angle_deg: float,
    arm_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    linear_vel_tol: float,
    angular_vel_tol: float,
    arm_joint_vel_tol: float,
    gripper_joint_vel_tol: float,
  ) -> torch.Tensor:
    holding = (
      task_state.is_grasped(
        env, sensor_names, finger_cfg, finger_outward_axes, min_force, max_angle_deg
      )
      & task_state.is_lifted(env, command_name, object_cfg, lift_threshold)
      & task_state.is_static(
        env,
        object_cfg,
        arm_cfg,
        gripper_cfg,
        linear_vel_tol,
        angular_vel_tol,
        arm_joint_vel_tol,
        gripper_joint_vel_tol,
      )
    )
    self.hold_counter = torch.where(
      holding, self.hold_counter + 1, torch.zeros_like(self.hold_counter)
    )
    return self.hold_counter >= hold_steps

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.hold_counter[env_ids] = 0
