"""Reset events: object placement (optionally seated on the support surface by
size variant) and curriculum-scaled domain randomization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul
from tabletop_sim.tasks.grasp.mdp import curriculum, object_size

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


def reset_object_from_pose_list(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str,
  poses: tuple[tuple[float, ...], ...],
) -> None:
  """Place an object at a pose drawn uniformly from ``poses`` (world frame,
  relative to each env origin), with zero velocity."""
  env_ids = resolve_env_ids(env, env_ids)
  entity: Entity = env.scene[entity_name]
  table = torch.tensor(poses, dtype=torch.float, device=env.device)
  choice = torch.randint(0, table.shape[0], (len(env_ids),), device=env.device)
  pose = table[choice].clone()
  pose[:, :3] += env.scene.env_origins[env_ids]

  if entity.is_fixed_base:
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  else:
    state = torch.zeros(len(env_ids), 13, device=env.device)
    state[:, :7] = pose
    entity.write_root_state_to_sim(state, env_ids=env_ids)


def _sample_xy_in_disk_with_yaw(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  entity: Entity,
  radius: float,
  yaw_range_deg: float,
) -> torch.Tensor:
  """World-frame root states for :func:`reset_object_xy_in_disk_with_yaw`."""
  state = entity.data.default_root_state[env_ids].clone()
  state[:, 0:3] += env.scene.env_origins[env_ids]
  n = len(env_ids)
  angle = torch.rand(n, device=env.device) * (2.0 * math.pi)
  # Square-root keeps the sample uniform over the disk area.
  radial = radius * torch.sqrt(torch.rand(n, device=env.device))
  state[:, 0] += radial * torch.cos(angle)
  state[:, 1] += radial * torch.sin(angle)

  # Uniform yaw in [-yaw_range_deg, yaw_range_deg], converted to radians.
  yaw_range = math.radians(yaw_range_deg)
  yaw = (torch.rand(n, device=env.device) * 2.0 - 1.0) * yaw_range
  z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
  yaw_quat = quat_from_angle_axis(yaw, z_axis)
  # Left-multiply so the rotation is about the world z-axis.
  state[:, 3:7] = quat_mul(yaw_quat, state[:, 3:7])
  return state


def reset_object_xy_in_disk_with_yaw(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str,
  radius: float,
  yaw_range_deg: float,
) -> None:
  """Place an object uniformly in a disk of ``radius`` (m) around its default
  xy, and rotate it about the world z-axis by a yaw sampled uniformly from
  ``[-yaw_range_deg, yaw_range_deg]`` degrees. The yaw is applied on top of the
  default orientation. z and velocity stay at their default values."""
  env_ids = resolve_env_ids(env, env_ids)
  entity: Entity = env.scene[entity_name]
  state = _sample_xy_in_disk_with_yaw(env, env_ids, entity, radius, yaw_range_deg)
  entity.write_root_state_to_sim(state, env_ids=env_ids)


class ResetObjectOnSupport(ManagerTermBase):
  """:func:`reset_object_xy_in_disk_with_yaw`, then the root height that puts
  the lowest collision point of the env's size variant, under the sampled
  orientation, ``clearance`` above the support plane:

    root_z = env_origin_z + support_height - lowest_collision_z + clearance

  The geometry is cached per variant at init (see
  :func:`object_size.collision_points_per_variant`), so resets do no mesh work
  and never change an env's variant. mjlab runs ``forward()`` after the reset
  events and before the command captures the lift goal from this pose.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._points, self._variant = object_size.collision_points_per_variant(
      env, cfg.params["entity_name"]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str,
    radius: float,
    yaw_range_deg: float,
    support_height: float,
    clearance: float,
  ) -> None:
    env_ids = resolve_env_ids(env, env_ids).to(torch.long)
    entity: Entity = env.scene[entity_name]
    state = _sample_xy_in_disk_with_yaw(env, env_ids, entity, radius, yaw_range_deg)
    lowest = object_size.lowest_point_z(self._points[self._variant[env_ids]], state[:, 3:7])
    state[:, 2] = env.scene.env_origins[env_ids, 2] + support_height - lowest + clearance
    entity.write_root_state_to_sim(state, env_ids=env_ids)


def difficulty_scale_bounds(
  difficulty: torch.Tensor, scale_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-env multiplier bounds, interpolated from [1, 1] at d = 0 to
  ``scale_range`` at d = 1."""
  lower = 1.0 + difficulty * (scale_range[0] - 1.0)
  upper = 1.0 + difficulty * (scale_range[1] - 1.0)
  return lower, upper


def sample_difficulty_scales(
  difficulty: torch.Tensor, scale_range: tuple[float, float], num_targets: int
) -> torch.Tensor:
  """Independent uniform multipliers, ``(len(difficulty), num_targets)``, each
  drawn from its env's :func:`difficulty_scale_bounds`."""
  lower, upper = difficulty_scale_bounds(difficulty, scale_range)
  u = torch.rand((difficulty.numel(), num_targets), device=difficulty.device)
  return lower[:, None] + u * (upper - lower)[:, None]


def _position_actuator_ctrl_ids(
  env: ManagerBasedRlEnv, entity: Entity, names: tuple[str, ...]
) -> torch.Tensor:
  """Global ctrl ids of the named actuators, which must be position servos
  (gain = kp, bias = [0, -kp, -kv])."""
  local_ids, found = entity.find_actuators(names, preserve_order=True)
  if tuple(found) != tuple(names):
    raise ValueError(f"actuators {names} resolved to {found}")
  ctrl_ids = entity.indexing.ctrl_ids[local_ids].to(torch.long)
  position_ids: set[int] = set()
  for act in entity.actuators:
    if isinstance(act, BuiltinPositionActuator) or (
      isinstance(act, XmlActuator) and act.command_field == "position"
    ):
      position_ids.update(act.global_ctrl_ids.tolist())
  m = env.sim.mj_model
  for name, i in zip(names, ctrl_ids.tolist(), strict=True):
    kp, bias = m.actuator_gainprm[i, 0], m.actuator_biasprm[i]
    if (
      i not in position_ids
      or m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
      or m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE
      or not (kp > 0 and bias[0] == 0 and bias[1] == -kp)
    ):
      raise ValueError(f"actuator '{name}' is not a position actuator with kp > 0")
  return ctrl_ids


class _StiffnessDampingRandomization(ManagerTermBase):
  """Scale position-actuator kp and the embodiment's damping at reset.

  Each resetting env draws one kp multiplier per stiffness actuator and,
  independently, one damping multiplier per damping target, uniformly within
  its episode difficulty's :func:`difficulty_scale_bounds`. Values are always
  the compiled model's nominal ones times the multiplier, so samples never
  compound, and they hold until that env's next reset. kp lives in both
  ``gainprm[0]`` and ``biasprm[1]``, which are scaled together.
  """

  damping_source: Literal["joint", "actuator"]

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    entity: Entity = env.scene[cfg.params["entity_name"]]
    self._kp_ids = _position_actuator_ctrl_ids(env, entity, cfg.params["stiffness_actuators"])
    targets: tuple[str, ...] = cfg.params["damping_targets"]
    if self.damping_source == "joint":
      joint_ids, found = entity.find_joints(targets, preserve_order=True)
      if tuple(found) != tuple(targets):
        raise ValueError(f"damping joints {targets} resolved to {found}")
      self._damping_ids = entity.indexing.joint_v_adr[joint_ids].to(torch.long)
      nominal = env.sim.get_default_field("dof_damping")[self._damping_ids]
    else:
      self._damping_ids = _position_actuator_ctrl_ids(env, entity, targets)
      nominal = -env.sim.get_default_field("actuator_biasprm")[self._damping_ids, 2]
    if not bool((nominal > 0).all()):
      raise ValueError(
        f"damping_source={self.damping_source!r}: nominal damping of {targets} "
        f"is {nominal.tolist()}; every target needs positive damping to scale"
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str,
    stiffness_actuators: tuple[str, ...],
    damping_targets: tuple[str, ...],
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
  ) -> None:
    del entity_name, stiffness_actuators, damping_targets  # Resolved in __init__.
    env_ids = resolve_env_ids(env, env_ids).to(torch.long)
    d = curriculum.episode_difficulty(env, env_ids)
    kp_scale = sample_difficulty_scales(d, stiffness_scale_range, len(self._kp_ids))
    damping_scale = sample_difficulty_scales(d, damping_scale_range, len(self._damping_ids))

    model = env.sim.model
    gainprm0 = env.sim.get_default_field("actuator_gainprm")
    biasprm0 = env.sim.get_default_field("actuator_biasprm")
    rows, kp_ids = env_ids[:, None], self._kp_ids
    model.actuator_gainprm[rows, kp_ids, 0] = gainprm0[kp_ids, 0] * kp_scale
    model.actuator_biasprm[rows, kp_ids, 1] = biasprm0[kp_ids, 1] * kp_scale
    if self.damping_source == "joint":
      damping0 = env.sim.get_default_field("dof_damping")
      model.dof_damping[rows, self._damping_ids] = damping0[self._damping_ids] * damping_scale
    else:
      model.actuator_biasprm[rows, self._damping_ids, 2] = (
        biasprm0[self._damping_ids, 2] * damping_scale
      )


@requires_model_fields("actuator_gainprm", "actuator_biasprm", "dof_damping")
class RandomizeKpJointDamping(_StiffnessDampingRandomization):
  """kp on position actuators, damping on joints (``dof_damping``)."""

  damping_source = "joint"


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
class RandomizeKpActuatorKv(_StiffnessDampingRandomization):
  """kp and kv, both on position actuators."""

  damping_source = "actuator"


STIFFNESS_DAMPING_EVENTS: dict[str, type[_StiffnessDampingRandomization]] = {
  "joint": RandomizeKpJointDamping,
  "actuator": RandomizeKpActuatorKv,
}
"""Event term class per ``StiffnessDampingTargets.damping_source``."""
