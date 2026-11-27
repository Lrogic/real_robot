"""Anchor-frame placement and box-symmetric goal orientations.

The anchor frame sits on the support surface: origin at a sampled point, z up,
x/y turned by the sampled yaw. Relative objects and the goal are given in it,
and world pose = ``T_world_anchor @ pose_in_anchor``. Shared by the start-state
generator and the release env.
"""

from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  quat_unique,
)

# 180-degree turns about the object's own x, y and z axes, plus the identity.
BOX_SYMMETRIES = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def symmetry_quats(symmetry: str, device: torch.device | str) -> torch.Tensor:
  """``(K, 4)`` object-frame rotations that leave the object unchanged."""
  if symmetry == "box":
    return torch.tensor(BOX_SYMMETRIES, dtype=torch.float, device=device)
  if symmetry == "none":
    return torch.tensor(BOX_SYMMETRIES[:1], dtype=torch.float, device=device)
  raise ValueError(f"unknown goal symmetry {symmetry!r}")


def sample_anchor(
  n: int,
  center: tuple[float, float],
  radius: float,
  yaw_deg: float,
  yaw_range_deg: float,
  height: float,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Anchor poses relative to the env origin: xy uniform over the disk area,
  z = ``height`` (the support plane), yaw uniform in ``yaw_deg +- yaw_range_deg``."""
  angle = torch.rand(n, device=device) * (2.0 * math.pi)
  radial = radius * torch.sqrt(torch.rand(n, device=device))
  pos = torch.zeros(n, 3, device=device)
  pos[:, 0] = center[0] + radial * torch.cos(angle)
  pos[:, 1] = center[1] + radial * torch.sin(angle)
  pos[:, 2] = height
  yaw = math.radians(yaw_deg) + (torch.rand(n, device=device) * 2.0 - 1.0) * math.radians(yaw_range_deg)
  z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
  return pos, quat_from_angle_axis(yaw, z_axis)


def compose(
  anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pose_in_anchor: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
  """``T_world_anchor @ pose_in_anchor`` for every anchor."""
  local = torch.tensor(pose_in_anchor, dtype=torch.float, device=anchor_pos.device)
  local = local.expand(anchor_pos.shape[0], 7)
  return combine_frame_transforms(anchor_pos, anchor_quat, local[:, :3], local[:, 3:])


def rotation_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Angle (rad, in [0, pi]) of the rotation between two orientations."""
  dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
  return 2.0 * torch.acos(dot)


def closest_symmetric_quat(
  goal_quat: torch.Tensor, current_quat: torch.Tensor, symmetry: str
) -> torch.Tensor:
  """The symmetric copy ``goal @ S_k`` nearest to ``current_quat``, w >= 0."""
  sym = symmetry_quats(symmetry, goal_quat.device)  # (K, 4)
  n, k = goal_quat.shape[0], sym.shape[0]
  candidates = quat_mul(
    goal_quat[:, None, :].expand(n, k, 4).reshape(-1, 4), sym[None].expand(n, k, 4).reshape(-1, 4)
  ).reshape(n, k, 4)
  angles = rotation_angle(candidates, current_quat[:, None, :].expand(n, k, 4))
  best = angles.argmin(dim=-1)
  return quat_unique(candidates[torch.arange(n, device=goal_quat.device), best])


def symmetric_rotation_distance(
  goal_quat: torch.Tensor, current_quat: torch.Tensor, symmetry: str
) -> torch.Tensor:
  """Angle (rad) from ``current_quat`` to the nearest symmetric goal orientation."""
  return rotation_angle(closest_symmetric_quat(goal_quat, current_quat, symmetry), current_quat)


def sample_ball_cap(
  n: int, radius: float, min_height: float, device: torch.device | str, tries: int = 64
) -> torch.Tensor:
  """``(n, 3)`` offsets uniform over the part of the ball of ``radius`` with
  z >= ``min_height`` (rejection sampling from the enclosing box)."""
  lo = torch.tensor([-radius, -radius, min_height], device=device)
  hi = torch.tensor([radius, radius, radius], device=device)
  out = torch.empty(n, 3, device=device)
  todo = torch.arange(n, device=device)
  while len(todo) > 0:
    cand = lo + torch.rand(len(todo), tries, 3, device=device) * (hi - lo)
    ok = torch.linalg.norm(cand, dim=-1) <= radius
    has = ok.any(dim=1)
    first = ok.float().argmax(dim=1)
    out[todo[has]] = cand[has, first[has]]
    todo = todo[~has]
  return out


def relative_rotation(q_to: torch.Tensor, q_from: torch.Tensor) -> torch.Tensor:
  """``q_to * q_from^-1``: the rotation (same frame as both) taking ``q_from`` to ``q_to``."""
  return quat_mul(q_to, quat_inv(q_from))
