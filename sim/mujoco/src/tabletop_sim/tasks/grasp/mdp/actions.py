"""Action terms. Every policy input is clipped to [-1, 1] before scaling.

Each term can be given the name of an :class:`ActionHold` command. On a control
step that command marks as held, an env's stored target (the absolute TCP goal,
or the joint target) is left exactly as it was: the new action is discarded and
the previous delta is not applied again from the current state.

The end-effector term can also interpolate its goal (``interpolate_goal``): IK
then tracks a subgoal that moves at constant velocity from the TCP pose at the
start of the control step to the goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import apply_delta_pose
from tabletop_sim.tasks.grasp.mdp.action_hold import ActionHold

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def held_envs(env: ManagerBasedRlEnv, hold_command_name: str | None) -> torch.Tensor | None:
  """Bool mask ``(num_envs,)`` of the envs holding this control step; None when
  the term has no hold command."""
  if hold_command_name is None:
    return None
  term = env.command_manager.get_term(hold_command_name)
  assert isinstance(term, ActionHold)
  return term.hold


# End-effector control.


@dataclass(kw_only=True)
class ClippedDifferentialIKActionCfg(DifferentialIKActionCfg):
  """Delta TCP pose (dx, dy, dz, droll, dpitch, dyaw) solved by DLS IK."""

  hold_command_name: str | None = None
  interpolate_goal: bool = False
  """Relative mode only. Off: every physics substep solves toward the step's
  goal. On: substep i of the N in a control step solves toward the start pose
  advanced by i/N of the step's delta (position, and base-frame rotation
  vector), so the last substep solves toward the goal itself. The start pose is
  the TCP pose the goal was computed from. On a held step the subgoal stays on
  the kept goal."""

  def build(self, env: ManagerBasedRlEnv) -> ClippedDifferentialIKAction:
    if self.interpolate_goal and not self.use_relative_mode:
      raise ValueError("interpolate_goal needs use_relative_mode")
    return ClippedDifferentialIKAction(self, env)


class ClippedDifferentialIKAction(DifferentialIKAction):
  cfg: ClippedDifferentialIKActionCfg

  def __init__(self, cfg: ClippedDifferentialIKActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._start_pos = torch.zeros_like(self._desired_pos)
    self._start_quat = self._desired_quat.clone()
    self._goal_pos = torch.zeros_like(self._desired_pos)
    self._goal_quat = self._desired_quat.clone()
    # Scaled (dx, dy, dz, rx, ry, rz) of the current step; zero in a held env.
    self._delta = torch.zeros(self.num_envs, 6, device=self.device)
    # Physics substeps applied since the last process_actions.
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    actions = actions.clamp(-1.0, 1.0)
    hold = held_envs(self._env, self.cfg.hold_command_name)
    if hold is not None:
      pos, quat = self._desired_pos[hold], self._desired_quat[hold]
    if self.cfg.interpolate_goal:
      start_pos, start_quat = self._get_frame_pose()
      self._start_pos[:], self._start_quat[:] = start_pos, start_quat
    super().process_actions(actions)
    if hold is not None:
      self._desired_pos[hold], self._desired_quat[hold] = pos, quat
    if not self.cfg.interpolate_goal:
      return
    self._goal_pos[:], self._goal_quat[:] = self._desired_pos, self._desired_quat
    self._delta[:] = 0.0
    self._delta[:, :3] = actions[:, :3] * self.cfg.delta_pos_scale
    if self._action_dim == 6:
      self._delta[:, 3:] = actions[:, 3:] * self.cfg.delta_ori_scale
    if hold is not None:
      self._start_pos[hold], self._start_quat[hold] = pos, quat
      self._delta[hold] = 0.0
    self._substep = 0

  def apply_actions(self) -> None:
    if self.cfg.interpolate_goal:
      self._substep += 1
      substeps = self._env.cfg.decimation
      if self._substep >= substeps:
        self._desired_pos[:], self._desired_quat[:] = self._goal_pos, self._goal_quat
      else:
        pos, quat = apply_delta_pose(
          self._start_pos, self._start_quat, self._delta * (self._substep / substeps)
        )
        # A zero delta (held env) must leave the goal bit-identical.
        moving = (self._delta != 0).any(dim=-1, keepdim=True)
        self._desired_pos[:] = torch.where(moving, pos, self._goal_pos)
        self._desired_quat[:] = torch.where(moving, quat, self._goal_quat)
    super().apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._delta[env_ids] = 0.0
    self._start_pos[env_ids] = self._desired_pos[env_ids]
    self._start_quat[env_ids] = self._desired_quat[env_ids]
    self._goal_pos[env_ids] = self._desired_pos[env_ids]
    self._goal_quat[env_ids] = self._desired_quat[env_ids]


# Joint-space arm control.


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(ActionTermCfg):
  """target = q + clip(a, -1, 1) * scale, computed once per control step."""

  joint_names: tuple[str, ...]
  scale: float = 0.1
  hold_command_name: str | None = None

  def build(self, env: ManagerBasedRlEnv) -> DeltaJointPositionAction:
    return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(ActionTerm):
  cfg: DeltaJointPositionActionCfg
  _entity: Entity

  def __init__(self, cfg: DeltaJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, len(ids), device=self.device)
    self._target = torch.zeros_like(self._raw_actions)

  @property
  def action_dim(self) -> int:
    return self._raw_actions.shape[1]

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids]
    q = self._entity.data.joint_pos[:, self._joint_ids]
    target = q + actions.clamp(-1.0, 1.0) * self.cfg.scale
    target = torch.clamp(target, limits[..., 0], limits[..., 1])
    hold = held_envs(self._env, self.cfg.hold_command_name)
    self._target[:] = target if hold is None else torch.where(hold[:, None], self._target, target)

  def apply_actions(self) -> None:
    bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(self._target - bias, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


# Gripper control.


@dataclass(kw_only=True)
class GripperActionCfg(ActionTermCfg):
  """Single-dimensional gripper command on the actuated carriage joint.

  ``binary``: a >= 0 opens, a < 0 closes.
  ``absolute``: clip(a, -1, 1) maps linearly from closed (-1) to open (+1).
  ``delta``: target = q + clip(a, -1, 1) * delta_scale (a > 0 opens), clamped
  to [closed_pos, open_pos] and computed once per control step.
  """

  joint_name: str
  open_pos: float
  closed_pos: float
  mode: str = "binary"
  delta_scale: float = 0.01
  hold_command_name: str | None = None

  def build(self, env: ManagerBasedRlEnv) -> GripperAction:
    if self.mode not in ("binary", "absolute", "delta"):
      raise ValueError(f"Unknown gripper mode {self.mode!r}")
    return GripperAction(self, env)


class GripperAction(ActionTerm):
  cfg: GripperActionCfg
  _entity: Entity

  def __init__(self, cfg: GripperActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints((cfg.joint_name,), preserve_order=True)
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._target = torch.full_like(self._raw_actions, cfg.open_pos)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    if self.cfg.mode == "binary":
      opened = actions >= 0.0
      target = torch.where(
        opened,
        torch.full_like(actions, self.cfg.open_pos),
        torch.full_like(actions, self.cfg.closed_pos),
      )
    elif self.cfg.mode == "absolute":
      frac = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
      target = self.cfg.closed_pos + frac * (self.cfg.open_pos - self.cfg.closed_pos)
    else:
      q = self._entity.data.joint_pos[:, self._joint_ids]
      target = q + actions.clamp(-1.0, 1.0) * self.cfg.delta_scale
      lo, hi = sorted((self.cfg.closed_pos, self.cfg.open_pos))
      target = target.clamp(lo, hi)
    hold = held_envs(self._env, self.cfg.hold_command_name)
    self._target[:] = target if hold is None else torch.where(hold[:, None], self._target, target)

  def apply_actions(self) -> None:
    bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(self._target - bias, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
