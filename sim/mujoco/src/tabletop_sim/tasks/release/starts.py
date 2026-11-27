"""Release start-state datasets: what ``scripts/generate_release_starts.py``
writes and the release reset restores.

A dataset is a ``.pt`` file (a dict of tensors, one row per start state) and a
``<name>.meta.json`` beside it. Poses and positions are relative to the env
origin (world axes); quaternions are (w, x, y, z).

  robot_joint_pos, robot_joint_vel   (N, J)  every robot joint, ``joint_names`` order
  object_root_state                  (N, 13) pos, quat, lin vel, ang vel
  support_root_state                 (N, S, 13) relative objects, ``support_names`` order
  goal_pose                          (N, 7)  the clean goal (not symmetry-reduced)
  anchor_pose                        (N, 7)
  variant_scale                      (N,)    size-variant factor of the manipulated object
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tabletop_sim.scene.config import ConfigError
from tabletop_sim.scene.release_config import ReleaseTaskConfig, RelativePlacementConfig

FORMAT_VERSION = 1
TENSOR_KEYS = (
  "robot_joint_pos",
  "robot_joint_vel",
  "object_root_state",
  "support_root_state",
  "goal_pose",
  "anchor_pose",
  "variant_scale",
)


def meta_path(dataset: Path) -> Path:
  return dataset.with_suffix(".meta.json")


def placement_params(rp: RelativePlacementConfig) -> dict[str, Any]:
  """JSON form of the placement a dataset was generated for."""
  return {
    "objects": {name: list(pose) for name, pose in rp.objects},
    "goal_pose": list(rp.goal_pose),
    "goal_symmetry": rp.goal_symmetry,
    "anchor_center": list(rp.anchor_center),
    "anchor_radius": rp.anchor_radius,
    "anchor_yaw_deg": rp.anchor_yaw_deg,
    "anchor_yaw_range_deg": rp.anchor_yaw_range_deg,
  }


def generation_key(task: ReleaseTaskConfig) -> dict[str, Any]:
  """What a dataset must have been made with to serve ``task``."""
  return {
    "relative_placement": placement_params(task.relative_placement),
    "start": task.start.generation_params(),
  }


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def read_meta(dataset: Path) -> dict[str, Any]:
  path = meta_path(dataset)
  if not dataset.is_file() or not path.is_file():
    raise ConfigError(
      f"release start dataset {dataset} (or its {path.name}) not found; generate it with "
      "scripts/generate_release_starts.py"
    )
  return json.loads(path.read_text())


def check_meta(task: ReleaseTaskConfig, meta: dict[str, Any], scene_name: str) -> None:
  """Raise if the dataset was not generated for this task's placement and start
  settings, scene, or size variants."""
  dataset = task.start.dataset
  if meta.get("format_version") != FORMAT_VERSION:
    raise ConfigError(f"{dataset}: format version {meta.get('format_version')} != {FORMAT_VERSION}")
  want = json.loads(json.dumps(generation_key(task)))
  for key, value in want.items():
    if meta.get(key) != value:
      raise ConfigError(
        f"{dataset} was generated with {key}={meta.get(key)}, but the task has {value}; "
        "regenerate it with scripts/generate_release_starts.py"
      )
  if meta.get("scene_name") != scene_name:
    raise ConfigError(f"{dataset} was generated for scene {meta.get('scene_name')!r}, not {scene_name!r}")
  sizes = task.domain_randomization.object_size_variants
  want_scales = sorted(sizes.scales) if sizes.enabled else [1.0]
  have = set(round(s, 6) for s in meta.get("variant_scales", []))
  missing = [s for s in want_scales if round(s, 6) not in have]
  if missing:
    raise ConfigError(f"{dataset} has no start states for size scales {missing}; regenerate it")


@dataclass
class ReleaseStarts:
  """A loaded dataset, on the env device."""

  tensors: dict[str, torch.Tensor]
  joint_names: tuple[str, ...]
  support_names: tuple[str, ...]

  @property
  def num_rows(self) -> int:
    return int(self.tensors["variant_scale"].shape[0])

  def rows_for_scales(self, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per env: (first row, row count) of the rows recorded at that env's scale.
    Rows are sorted by scale at load time."""
    have = self.tensors["variant_scale"]
    values, counts = torch.unique_consecutive(have, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    match = (scales[:, None] - values[None, :]).abs() < 1e-5
    if not bool(match.any(dim=1).all()):
      missing = sorted(set(scales[~match.any(dim=1)].tolist()))
      raise ValueError(f"release start dataset has no rows at size scales {missing}")
    idx = match.float().argmax(dim=1)
    return starts[idx], counts[idx]

  @classmethod
  def load(cls, path: Path, device: torch.device | str) -> ReleaseStarts:
    data = torch.load(path, map_location="cpu", weights_only=False)
    tensors = {k: torch.as_tensor(data[k]).float() for k in TENSOR_KEYS}
    order = torch.argsort(tensors["variant_scale"], stable=True)
    tensors = {k: v[order].to(device) for k, v in tensors.items()}
    return cls(tensors, tuple(data["joint_names"]), tuple(data["support_names"]))


def save(path: Path, rows: dict[str, torch.Tensor], joint_names, support_names, meta: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {k: rows[k].detach().cpu() for k in TENSOR_KEYS}
  payload["joint_names"] = list(joint_names)
  payload["support_names"] = list(support_names)
  torch.save(payload, path)
  meta = {**meta, "format_version": FORMAT_VERSION, "rows": int(rows["variant_scale"].shape[0])}
  meta["sha256"] = file_sha256(path)
  meta_path(path).write_text(json.dumps(meta, indent=2) + "\n")
