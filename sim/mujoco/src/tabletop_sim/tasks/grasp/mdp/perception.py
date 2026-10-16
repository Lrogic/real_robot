"""The object pose the actor perceives: the true pose corrupted by a persistent
per-episode bias and fresh per-step noise, both curriculum-scaled.

All actor object observations read the one pose cached here; the simulated
pose, the critic, rewards, terminations and metrics never see it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import normalize, quat_apply, quat_mul
from tabletop_sim.tasks.grasp.mdp import curriculum

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

StandardNormal = Callable[[tuple[int, ...]], torch.Tensor]


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
  """Exp map: rotation vectors ``(..., 3)`` (rad) to unit quaternions (w, x, y,
  z). A zero vector gives exactly (1, 0, 0, 0)."""
  angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
  half = 0.5 * angle
  # sin(angle / 2) / angle, with its Taylor expansion near zero.
  k = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp_min(1e-6), 0.5 - angle**2 / 48.0)
  return torch.cat((torch.cos(half), rotvec * k), dim=-1)


def clipped_gaussian(z: torch.Tensor, std: torch.Tensor, clip_sigma: float) -> torch.Tensor:
  """Standard normal draws ``z`` scaled by ``std`` and clamped to
  +-``clip_sigma * std`` per component (clamped, not rejection-sampled)."""
  bound = clip_sigma * std
  return torch.clamp(z * std, -bound, bound)


def perturb_pose(
  pos_w: torch.Tensor,
  quat_w: torch.Tensor,
  base_quat_w: torch.Tensor,
  pos_offsets_b: tuple[torch.Tensor, ...],
  rotvecs_b: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
  """World object pose with base-frame errors applied.

  Positions get the sum of ``pos_offsets_b``. Rotations are left-multiplied in
  the order given, so ``rotvecs_b = (b_r, e_r)`` yields
  ``R_b' = Exp(e_r) Exp(b_r) R_b``; they turn the object about its own origin.
  World and base frames differ by ``base_quat_w``: ``Exp(R v) = R Exp(v) R^T``.
  Where every rotation vector is exactly zero the input quaternion is returned
  unchanged.
  """
  out_pos = pos_w + quat_apply(base_quat_w, torch.stack(pos_offsets_b).sum(dim=0))
  out_quat = quat_w
  for rotvec_b in rotvecs_b:
    out_quat = quat_mul(quat_from_rotvec(quat_apply(base_quat_w, rotvec_b)), out_quat)
  rotated = torch.stack([(r != 0).any(dim=-1) for r in rotvecs_b]).any(dim=0)
  out_quat = torch.where(rotated[:, None], normalize(out_quat), quat_w)
  return out_pos, out_quat


@dataclass(frozen=True)
class GaussianPoseNoise:
  """Per-axis base-frame standard deviations at difficulty 1, before clipping:
  translation in m, rotation-vector components in rad. Clip multipliers are in
  current standard deviations."""

  enabled: bool = False
  translation_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
  rotation_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
  translation_clip_sigma: float = 2.0
  rotation_clip_sigma: float = 2.0


@dataclass(kw_only=True)
class PerceivedObjectPoseCfg(CommandTermCfg):
  object_name: str
  robot_name: str = "robot"
  target_lift_height: float
  """Vertical (world z) offset of the actor goal; same as the clean goal's."""
  persistent_bias: GaussianPoseNoise = field(default_factory=GaussianPoseNoise)
  per_step: GaussianPoseNoise = field(default_factory=GaussianPoseNoise)
  # Never resampled by the timer; resets flag envs instead.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> PerceivedObjectPose:
    return PerceivedObjectPose(self, env)


class PerceivedObjectPose(CommandTerm):
  """Per env: the perceived object pose for the current control step and the
  actor goal ``perceived_initial_pos + (0, 0, target_lift_height)``.

  A command term because ``_update_command`` is the one hook that runs after
  the post-reset ``forward()`` and before observations, both in ``reset()``
  (scoped to the reset ids) and once per ``step()`` (all envs). Each call draws
  new per-step noise for the envs in scope and refreshes their cached pose;
  observations only read the cache, so repeated reads within a step agree.

  Reset flags an env (as :class:`GraspGoalCommand` does), and its next update
  draws the persistent bias, then the first per-step noise, and fixes the actor
  goal from that same corrupted pose. Flags rather than a step counter keep this
  correct when a reset and the observation share a global step.
  """

  cfg: PerceivedObjectPoseCfg

  def __init__(self, cfg: PerceivedObjectPoseCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.object_name]
    self.robot: Entity = env.scene[cfg.robot_name]
    n, dev = self.num_envs, self.device
    self.bias_pos_b = torch.zeros(n, 3, device=dev)
    self.bias_rotvec_b = torch.zeros(n, 3, device=dev)
    self.noise_pos_b = torch.zeros(n, 3, device=dev)
    self.noise_rotvec_b = torch.zeros(n, 3, device=dev)
    self.pos_w = torch.zeros(n, 3, device=dev)
    self.quat_w = torch.zeros(n, 4, device=dev)
    self.quat_w[:, 0] = 1.0
    self.actor_target_pos = torch.zeros(n, 3, device=dev)
    self._needs_capture = torch.ones(n, dtype=torch.bool, device=dev)
    self._std = {
      name: (
        torch.tensor(src.translation_std, dtype=torch.float, device=dev),
        torch.tensor(src.rotation_std, dtype=torch.float, device=dev),
      )
      for name, src in (("persistent_bias", cfg.persistent_bias), ("per_step", cfg.per_step))
    }
    self.standard_normal: StandardNormal = lambda shape: torch.randn(shape, device=dev)
    """Source of N(0, 1) draws, one ``(len(ids), 6)`` call per enabled source
    (translation then rotation columns); tests may replace it."""

  @property
  def command(self) -> torch.Tensor:
    """Perceived world pose ``[x, y, z, qw, qx, qy, qz]``."""
    return torch.cat((self.pos_w, self.quat_w), dim=-1)

  def _sample(self, source: str, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(translation, rotation vector) draws of ``source`` for ``ids``, at each
    env's episode difficulty."""
    src: GaussianPoseNoise = getattr(self.cfg, source)
    if not src.enabled:
      zeros = torch.zeros(len(ids), 3, device=self.device)
      return zeros, zeros.clone()
    d = curriculum.episode_difficulty(self._env, ids)[:, None]
    z = self.standard_normal((len(ids), 6))
    t_std, r_std = (d * s for s in self._std[source])
    return (
      clipped_gaussian(z[:, :3], t_std, src.translation_clip_sigma),
      clipped_gaussian(z[:, 3:], r_std, src.rotation_clip_sigma),
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._needs_capture[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    scope = torch.ones_like(self._needs_capture)
    if env_ids is not None:
      scope = torch.zeros_like(scope)
      scope[env_ids] = True
    capture = (self._needs_capture & scope).nonzero(as_tuple=False).squeeze(-1)
    if len(capture) > 0:
      self.bias_pos_b[capture], self.bias_rotvec_b[capture] = self._sample(
        "persistent_bias", capture
      )
    ids = scope.nonzero(as_tuple=False).squeeze(-1)
    self.noise_pos_b[ids], self.noise_rotvec_b[ids] = self._sample("per_step", ids)
    self.pos_w[ids], self.quat_w[ids] = perturb_pose(
      self.object.data.root_link_pos_w[ids],
      self.object.data.root_link_quat_w[ids],
      self.robot.data.root_link_quat_w[ids],
      (self.bias_pos_b[ids], self.noise_pos_b[ids]),
      (self.bias_rotvec_b[ids], self.noise_rotvec_b[ids]),
    )
    if len(capture) > 0:
      self.actor_target_pos[capture] = self.pos_w[capture]
      self.actor_target_pos[capture, 2] += self.cfg.target_lift_height
      self._needs_capture[capture] = False

  def _update_metrics(self) -> None:
    pass
