"""Turn a SceneConfig into mjlab entity configs."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.entity import Entity, EntityCfg, VariantEntityCfg
from tabletop_sim.scene.config import ConfigError, ObjectConfig, ObjectSizeVariantsConfig, SceneConfig
_GEOM_TYPES = {
  "box": mujoco.mjtGeom.mjGEOM_BOX,
  "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
  "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
  "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


def _half_size(obj: ObjectConfig) -> list[float]:
  """MuJoCo geom size from the config's full-extent convention, with scale."""
  sx, sy, sz = obj.scale
  if obj.primitive == "box":
    return [obj.size[0] * sx / 2, obj.size[1] * sy / 2, obj.size[2] * sz / 2]
  if obj.primitive == "sphere":
    return [obj.size[0] * sx]
  # cylinder / capsule: radius, half-length along z.
  return [obj.size[0] * sx, obj.size[1] * sz / 2]


def _mesh_visual_kwargs(spec: mujoco.MjSpec, obj: ObjectConfig) -> dict:
  """Appearance of a mesh visual geom. A texture replaces the flat color."""
  if obj.texture_path is None:
    return {"rgba": list(obj.color)}
  tex_name = f"{obj.name}_tex"
  mat_name = f"{obj.name}_mat"
  spec.add_texture(
    name=tex_name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    file=str(obj.texture_path),
  )
  material = spec.add_material(name=mat_name)
  material.rgba = [1.0, 1.0, 1.0, 1.0]
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = tex_name
  return {"material": mat_name}


def _box_mesh(extents: tuple[float, ...]) -> tuple[list[float], list[int]]:
  """Closed, outward-wound triangle mesh of a box of full ``extents`` centred on
  the origin, as flat (vertex, face) lists."""
  corner = lambda i: np.array([1 if (i >> k) & 1 else -1 for k in range(3)])  # noqa: E731
  verts = np.array([corner(i) * np.asarray(extents) / 2 for i in range(8)])
  faces: list[int] = []
  for axis in range(3):
    a, b = (k for k in range(3) if k != axis)
    for side in (0, 1):
      quad = [
        sum(bit << k for k, bit in ((axis, side), (a, ba), (b, bb)))
        for ba, bb in ((0, 0), (1, 0), (1, 1), (0, 1))
      ]
      for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
        p0, p1, p2 = verts[list(tri)]
        outward = np.cross(p1 - p0, p2 - p0)[axis] * (1 if side else -1) > 0
        faces.extend(tri if outward else (tri[0], tri[2], tri[1]))
  return verts.ravel().tolist(), faces


def object_spec(obj: ObjectConfig, size_scale: float | None = None) -> mujoco.MjSpec:
  """MjSpec of one object. ``size_scale`` builds a size variant of a mesh
  object: every mesh is scaled by ``size_scale`` on top of ``obj.scale``, a
  ``collision_size`` box becomes an equivalent closed box mesh (mjlab variants
  can only vary mesh geoms), and an explicit ``mass`` is the scale-1 mass, so it
  scales by ``size_scale**3``."""
  if size_scale is not None and obj.primitive != "mesh":
    raise ConfigError(f"object {obj.name!r}: size variants need a mesh object")
  s = 1.0 if size_scale is None else size_scale
  scale = [k * s for k in obj.scale]
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=obj.name)
  if obj.dynamic:
    body.add_freejoint(name=f"{obj.name}_joint")

  common = dict(friction=list(obj.friction))
  if obj.mass is not None:
    common["mass"] = obj.mass * s**3
  else:
    common["density"] = obj.density

  if obj.primitive == "mesh":
    assert obj.asset_path is not None and obj.collision_asset_path is not None
    spec.add_mesh(name=f"{obj.name}_visual", file=str(obj.asset_path), scale=scale)
    body.add_geom(
      name=f"{obj.name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"{obj.name}_visual",
      contype=0,
      conaffinity=0,
      group=2,
      density=0.0,
      **_mesh_visual_kwargs(spec, obj),
    )
    if obj.collision_size and size_scale is not None:
      uservert, userface = _box_mesh(obj.collision_size)
      spec.add_mesh(
        name=f"{obj.name}_collision", uservert=uservert, userface=userface, scale=scale
      )
      collision = dict(
        name=f"{obj.name}_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{obj.name}_collision",
      )
    elif obj.collision_size:
      sx, sy, sz = obj.scale
      collision = dict(
        name=f"{obj.name}_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[
          obj.collision_size[0] * sx / 2,
          obj.collision_size[1] * sy / 2,
          obj.collision_size[2] * sz / 2,
        ],
      )
    else:
      spec.add_mesh(
        name=f"{obj.name}_collision", file=str(obj.collision_asset_path), scale=scale
      )
      collision = dict(
        name=f"{obj.name}_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{obj.name}_collision",
      )
    body.add_geom(
      **collision,
      rgba=list(obj.color[:3]) + [0.0],
      group=3,
      **common,
    )
  else:
    body.add_geom(
      name=f"{obj.name}_geom",
      type=_GEOM_TYPES[obj.primitive],
      size=_half_size(obj),
      rgba=list(obj.color),
      **common,
    )
  return spec


HULL_REF_PREFIX = "hull_ref/"
HULL_REF_CONTYPE = 1 << 30
"""Contype bit of the hull-reference geoms; no other geom may use it as conaffinity."""


@dataclass
class SizeVariantEntityCfg(VariantEntityCfg):
  """``VariantEntityCfg`` whose variant ``i`` is the object at size ``scales[i]``."""

  scales: tuple[float, ...] = ()

  def build(self) -> Entity:
    entity = super().build()
    _add_hull_references(entity)
    return entity


def _add_hull_references(entity: Entity) -> None:
  """MuJoCo only builds the convex hull and face polygons of meshes that a
  collidable geom references, and mjlab's merged template references only
  variant 0's meshes; the others are swapped in per world via
  ``geom_dataid``. Without polygons, MuJoCo Warp makes a single contact per
  box/mesh-mesh pair, so those variants spin and rock on the table. One
  massless geom per remaining collision mesh, whose contype bit no geom accepts,
  makes the compiler build the hulls without ever producing a contact."""
  metadata = entity.variant_metadata
  assert metadata is not None
  body = entity.spec.worldbody.bodies[0]
  for s_idx, slot in enumerate(metadata.slots):
    if slot.key.role != "collision":
      continue
    for v_idx in range(1, len(metadata.variant_names)):
      gs = metadata.variant_slot_specs[v_idx][s_idx]
      if gs is None:
        continue
      mesh = f"{metadata.variant_names[v_idx]}/{gs.mesh_name}"
      body.add_geom(
        name=f"{HULL_REF_PREFIX}{mesh}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh,
        contype=HULL_REF_CONTYPE,
        conaffinity=0,
        mass=0.0,
        density=0.0,
        group=5,
        rgba=[0.0, 0.0, 0.0, 0.0],
      )


class BalancedRandomAssignment:
  """World -> variant map for ``VariantEntityCfg.assignment``: as even as
  possible (counts differ by at most one, the variants getting the extra world
  chosen at random), in random world order.

  mjlab calls this from ``Simulation.__init__``, right after
  ``ManagerBasedRlEnv`` seeds the global RNGs with ``cfg.seed`` (the training
  seed), so one draw from NumPy's global RNG makes the assignment reproducible
  per seed while using a private generator for the shuffle itself.
  """

  def __init__(self, num_variants: int):
    self.num_variants = num_variants

  def __call__(self, num_envs: int) -> list[int]:
    rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    order = rng.permutation(self.num_variants)[np.arange(num_envs) % self.num_variants]
    return rng.permutation(order).tolist()


def object_entity_cfg(
  obj: ObjectConfig, size_variants: ObjectSizeVariantsConfig | None = None
) -> EntityCfg:
  pose = obj.initial_poses[0]
  init_state = EntityCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]))
  if size_variants is None or not size_variants.enabled:
    return EntityCfg(spec_fn=lambda: object_spec(obj), init_state=init_state)
  if obj.primitive != "mesh" or not obj.dynamic:
    raise ConfigError(f"object {obj.name!r}: size variants need a dynamic mesh object")
  scales = size_variants.scales
  return SizeVariantEntityCfg(
    variants={
      f"scale_{i:02d}": (lambda s=s: object_spec(obj, size_scale=s))
      for i, s in enumerate(scales)
    },
    assignment=BalancedRandomAssignment(len(scales)),
    scales=scales,
    init_state=init_state,
  )


def ground_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.add_texture(
    name="ground_grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0.1, 0.2, 0.3],
    width=300,
    height=300,
  )
  mat = spec.add_material(name="ground_mat", texrepeat=[5, 5])
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "ground_grid"
  body = spec.worldbody.add_body(name="ground")
  body.add_geom(
    name="ground_plane",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    material="ground_mat",
  )
  return spec


def scene_entity_cfgs(
  scene: SceneConfig,
  robot: EntityCfg,
  size_variants: ObjectSizeVariantsConfig | None = None,
) -> dict[str, EntityCfg]:
  """Robot first, then ground (if any), then objects in config order.
  ``size_variants`` applies to the manipulated object only."""
  base = scene.robot.base_pose
  robot.init_state.pos = tuple(base[:3])
  robot.init_state.rot = tuple(base[3:])
  entities: dict[str, EntityCfg] = {"robot": robot}
  if scene.ground_altitude is not None:
    entities["ground"] = EntityCfg(
      spec_fn=ground_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, scene.ground_altitude)),
    )
  for obj in scene.objects:
    manipulated = obj.name == scene.manipulated_object
    entities[obj.name] = object_entity_cfg(obj, size_variants if manipulated else None)
  return entities
