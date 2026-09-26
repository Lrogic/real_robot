"""Turn a SceneConfig into mjlab entity configs."""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from tabletop_sim.scene.config import ObjectConfig, SceneConfig
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


def object_spec(obj: ObjectConfig) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=obj.name)
  if obj.dynamic:
    body.add_freejoint(name=f"{obj.name}_joint")

  common = dict(friction=list(obj.friction))
  if obj.mass is not None:
    common["mass"] = obj.mass
  else:
    common["density"] = obj.density

  if obj.primitive == "mesh":
    assert obj.asset_path is not None and obj.collision_asset_path is not None
    spec.add_mesh(name=f"{obj.name}_visual", file=str(obj.asset_path), scale=list(obj.scale))
    body.add_geom(
      name=f"{obj.name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"{obj.name}_visual",
      rgba=list(obj.color),
      contype=0,
      conaffinity=0,
      group=2,
      density=0.0,
    )
    spec.add_mesh(
      name=f"{obj.name}_collision", file=str(obj.collision_asset_path), scale=list(obj.scale)
    )
    body.add_geom(
      name=f"{obj.name}_collision",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"{obj.name}_collision",
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


def object_entity_cfg(obj: ObjectConfig) -> EntityCfg:
  pose = obj.initial_poses[0]
  return EntityCfg(
    spec_fn=lambda: object_spec(obj),
    init_state=EntityCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:])),
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


def scene_entity_cfgs(scene: SceneConfig, robot: EntityCfg) -> dict[str, EntityCfg]:
  """Robot first, then ground (if any), then objects in config order."""
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
    entities[obj.name] = object_entity_cfg(obj)
  return entities
