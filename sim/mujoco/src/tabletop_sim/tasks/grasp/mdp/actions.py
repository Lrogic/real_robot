"""Action terms. Every policy input is clipped to [-1, 1] before scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# End-effector control.


@dataclass(kw_only=True)
class ClippedDifferentialIKActionCfg(DifferentialIKActionCfg):
  """Delta TCP pose (dx, dy, dz, droll, dpitch, dyaw) solved by DLS IK."""

  def build(self, env: ManagerBasedRlEnv) -> ClippedDifferentialIKAction:
    return ClippedDifferentialIKAction(self, env)


class ClippedDifferentialIKAction(DifferentialIKAction):
  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions.clamp(-1.0, 1.0))


# Joint-space arm control.


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(ActionTermCfg):
  """target = q + clip(a, -1, 1) * scale, computed once per control step."""

  joint_names: tuple[str, ...]
  scale: float = 0.1

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
    self._target[:] = torch.clamp(target, limits[..., 0], limits[..., 1])

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
  """

  joint_name: str
  open_pos: float
  closed_pos: float
  mode: str = "binary"

  def build(self, env: ManagerBasedRlEnv) -> GripperAction:
    if self.mode not in ("binary", "absolute"):
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
      self._target[:] = torch.where(
        opened,
        torch.full_like(actions, self.cfg.open_pos),
        torch.full_like(actions, self.cfg.closed_pos),
      )
    else:
      frac = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
      self._target[:] = self.cfg.closed_pos + frac * (self.cfg.open_pos - self.cfg.closed_pos)

  def apply_actions(self) -> None:
    bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(self._target - bias, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
