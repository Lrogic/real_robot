"""Fixed per-env size variants of the manipulated object: each env's assigned
scale, and the collision geometry that seats its variant on the support surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from tabletop_sim.scene.builder import HULL_REF_CONTYPE, HULL_REF_PREFIX, SizeVariantEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def assigned_scale(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
  """``(num_envs,)`` size factor of each env's variant, relative to the
  object's configured scale (all ones without size variants). Fixed for the
  lifetime of the simulation."""
  cfg = env.scene[entity_name].cfg
  if not isinstance(cfg, SizeVariantEntityCfg):
    return torch.ones(env.num_envs, device=env.device)
  scales = torch.tensor(cfg.scales, dtype=torch.float, device=env.device)
  return scales[env.sim.world_to_variant[entity_name]]


def hull_reference_geoms(env: ManagerBasedRlEnv, entity_name: str) -> list[int]:
  """Ids of the entity's hull-reference geoms (see
  ``builder._add_hull_references``), after checking that none can collide."""
  m = env.sim.mj_model
  prefix = f"{entity_name}/{HULL_REF_PREFIX}"
  ids = [
    g for g in env.scene[entity_name].indexing.geom_ids.tolist()
    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)
  ]
  if ids and (m.geom_conaffinity & HULL_REF_CONTYPE).any():
    raise ValueError(f"a geom accepts contype bit {HULL_REF_CONTYPE}, reserved for hull references")
  return ids


def _per_world(array, shape: tuple[int, ...]) -> np.ndarray:
  a = array.numpy()
  if a.ndim == len(shape) - 1:
    a = a[None]
  return np.broadcast_to(a, shape)


def _hull_vertices(m: mujoco.MjModel, mesh_id: int) -> np.ndarray:
  """Vertices of a compiled mesh's convex hull (the shape MuJoCo collides
  with), in the mesh frame after MuJoCo's recentering. All vertices if the hull
  graph was not built."""
  adr, num = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
  verts = m.mesh_vert[adr : adr + num]
  graph = m.mesh_graphadr[mesh_id]
  if graph < 0:
    return verts.copy()
  hull_num = m.mesh_graph[graph]
  ids = m.mesh_graph[graph + 2 + hull_num : graph + 2 + 2 * hull_num]
  return verts[ids]


def _box_corners(half: np.ndarray) -> np.ndarray:
  signs = np.array([[1 if (i >> k) & 1 else -1 for k in range(3)] for i in range(8)])
  return signs * half


def collision_points_per_variant(
  env: ManagerBasedRlEnv, entity_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Collision geometry of each size variant, as simulated.

  Returns ``points`` ``(num_variants, P, 3)``: extreme points of every
  collision geom of the entity's root body, in the root body frame; and
  ``variant`` ``(num_envs,)``: each env's variant index. Uses each world's
  compiled mesh ids and geom offsets, so mesh scale, MuJoCo's mesh recentering
  and local geom poses are all included. Variants no env uses are NaN.
  Computed once at init; resets only index into it.
  """
  entity = env.scene[entity_name]
  m = env.sim.mj_model
  wp_model = env.sim.wp_model
  n, ngeom = env.num_envs, m.ngeom
  root = entity.indexing.root_body_id
  hull_refs = hull_reference_geoms(env, entity_name)
  geoms = [
    g for g in entity.indexing.geom_ids.tolist()
    if (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0) and g not in hull_refs
  ]
  if not geoms or any(m.geom_bodyid[g] != root for g in geoms):
    raise ValueError(f"'{entity_name}': collision geoms must all be on the root body")

  dataid = _per_world(wp_model.geom_dataid, (n, ngeom))
  size = _per_world(wp_model.geom_size, (n, ngeom, 3))
  pos = _per_world(wp_model.geom_pos, (n, ngeom, 3))
  quat = _per_world(wp_model.geom_quat, (n, ngeom, 4))
  w2v = env.sim.world_to_variant.get(entity_name)
  variant = np.zeros(n, dtype=np.int64) if w2v is None else w2v.cpu().numpy()
  metadata = entity.variant_metadata
  num_variants = 1 if metadata is None else len(metadata.variant_names)

  per_variant: dict[int, np.ndarray] = {}
  for v in np.unique(variant).tolist():
    worlds = np.flatnonzero(variant == v)
    w = worlds[0]
    for field in (dataid, size, pos, quat):
      rows = field[worlds][:, geoms]
      if not (rows == rows[:1]).all():
        raise ValueError(f"'{entity_name}': worlds of variant {v} have different geometry")
    pts = []
    for g in geoms:
      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        local = _hull_vertices(m, int(dataid[w, g]))
      elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        local = _box_corners(size[w, g])
      else:
        raise ValueError(f"'{entity_name}': unsupported collision geom type {m.geom_type[g]}")
      rot = np.empty(9)
      mujoco.mju_quat2Mat(rot, quat[w, g].astype(np.float64))
      pts.append(pos[w, g] + local @ rot.reshape(3, 3).T)
    per_variant[v] = np.concatenate(pts)

  width = max(p.shape[0] for p in per_variant.values())
  points = np.full((num_variants, width, 3), np.nan)
  for v, p in per_variant.items():
    points[v, : len(p)] = p
    points[v, len(p) :] = p[0]  # padding never lowers the minimum
  return (
    torch.tensor(points, dtype=torch.float, device=env.device),
    torch.as_tensor(variant, dtype=torch.long, device=env.device),
  )


def lowest_point_z(points: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  """Lowest world z of body-frame ``points`` ``(n, P, 3)`` under root
  orientations ``quat`` ``(n, 4)`` (w, x, y, z), relative to the root origin."""
  w, x, y, z = quat.unbind(-1)
  row = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
  return torch.einsum("npk,nk->np", points, row).amin(-1)
