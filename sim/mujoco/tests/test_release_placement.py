"""Anchor composition, box-symmetric goals and the ball-cap sampler (CPU)."""

import math

import pytest
import torch

from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul
from tabletop_sim.tasks.release.mdp import placement


def _yaw(deg: float) -> torch.Tensor:
  return quat_from_angle_axis(torch.tensor([math.radians(deg)]), torch.tensor([[0.0, 0.0, 1.0]]))


def test_compose_turns_and_shifts_the_layout():
  # The worked example: anchor at (-0.30, 0.15) with yaw 90; a point 3 cm to the
  # anchor's -y lands 3 cm to the world's +x of the anchor.
  pos, quat = placement.compose(
    torch.tensor([[-0.30, 0.15, 0.0]]), _yaw(90.0), (0.0, -0.03, 0.0375, 1.0, 0.0, 0.0, 0.0)
  )
  assert torch.allclose(pos, torch.tensor([[-0.27, 0.15, 0.0375]]), atol=1e-6)
  assert torch.allclose(quat, _yaw(90.0), atol=1e-6)


def test_sample_anchor_stays_in_the_disk_and_yaw_range():
  pos, quat = placement.sample_anchor(4000, (-0.3, 0.15), 0.05, 20.0, 30.0, 0.0, "cpu")
  r = torch.linalg.norm(pos[:, :2] - torch.tensor([-0.3, 0.15]), dim=-1)
  assert float(r.max()) <= 0.05 + 1e-6
  # Uniform over the area: about a quarter of the samples inside half the radius.
  assert 0.2 < float((r < 0.025).float().mean()) < 0.3
  yaw = torch.rad2deg(2 * torch.atan2(quat[:, 3], quat[:, 0]))
  assert float(yaw.min()) >= -10.0 - 1e-3 and float(yaw.max()) <= 50.0 + 1e-3
  assert torch.all(pos[:, 2] == 0.0)


@pytest.mark.parametrize("k", range(4))
def test_box_flips_are_zero_distance(k):
  goal = _yaw(37.0)
  flipped = quat_mul(goal, placement.symmetry_quats("box", "cpu")[k : k + 1])
  assert float(placement.symmetric_rotation_distance(goal, flipped, "box")) < 1e-3
  if k > 0:
    assert float(placement.symmetric_rotation_distance(goal, flipped, "none")) == pytest.approx(math.pi, abs=1e-3)


def test_closest_symmetric_quat_picks_the_nearest_copy():
  goal = _yaw(0.0)
  current = _yaw(170.0)  # nearest copy is the 180-deg turn about z
  chosen = placement.closest_symmetric_quat(goal, current, "box")
  assert float(placement.rotation_angle(chosen, current)) == pytest.approx(math.radians(10.0), abs=1e-4)
  assert float(chosen[0, 0]) >= 0.0


def test_ball_cap_samples_obey_radius_and_height():
  d = placement.sample_ball_cap(5000, 0.04, 0.01, "cpu")
  assert float(torch.linalg.norm(d, dim=-1).max()) <= 0.04 + 1e-6
  assert float(d[:, 2].min()) >= 0.01
  # Horizontal spread reaches near the cap's edge, sqrt(r^2 - h^2) = 3.9 cm.
  assert float(torch.linalg.norm(d[:, :2], dim=-1).max()) > 0.035
