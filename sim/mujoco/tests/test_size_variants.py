"""Fixed per-env size variants of the manipulated object."""

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply
from tabletop_sim.scene import (
  ConfigError,
  ObjectSizeVariantsConfig,
  load_scene_config,
  load_task_config,
  object_spec,
)
from tabletop_sim.scene.builder import BalancedRandomAssignment, SizeVariantEntityCfg, _box_mesh
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import (
  CONFIGS_DIR,
  DEFAULT_SCENE,
  DEFAULT_TASK,
  make_grasp_env_cfg,
  make_grasp_env_cfg_from_configs,
)
from tabletop_sim.tasks.grasp.mdp import object_size, task_state

JENGA_SCENE = CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json"
SIZE_TASK = CONFIGS_DIR / "tasks" / "grasp_size_variants_wxai.json"
JENGA_SIZE = (0.075, 0.015, 0.025)
REPRESENTATIVE = (0.9, 1.0, 1.1)
DEVICE = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_jenga = pytest.mark.skipif(
  not load_scene_config(JENGA_SCENE).manipulated.asset_path.exists(), reason="jenga mesh missing"
)


def _write(path: Path, data: dict) -> Path:
  path.write_text(json.dumps(data))
  return path


def _task(tmp_path: Path, **variants) -> Path:
  data = json.loads(SIZE_TASK.read_text())
  data["domain_randomization"]["object_size_variants"].update(variants)
  return _write(tmp_path / "task.json", data)


def _rot(axis, deg) -> np.ndarray:
  quat = np.empty(4)
  axis = np.asarray(axis, float)
  mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), math.radians(deg))
  mat = np.empty(9)
  mujoco.mju_quat2Mat(mat, quat)
  return mat.reshape(3, 3)


# Rotated, off-centre box (relative to the mesh file origin).
OFF_ROT = _rot([1, 2, 3], 35.0)
OFF_CENTRE = np.array([0.012, -0.007, 0.021])


def _box_obj(path: Path, extents, rot=np.eye(3), offset=np.zeros(3), reverse=False) -> Path:
  verts, faces = _box_mesh(tuple(extents))
  v = np.asarray(verts).reshape(-1, 3) @ rot.T + offset
  tris = np.asarray(faces).reshape(-1, 3)
  if reverse:  # same solid, different triangle order
    tris = tris[::-1]
  lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in v]
  lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tris]
  path.write_text("\n".join(lines) + "\n")
  return path


def _mesh_object(tmp_path: Path, kind: str, **extra) -> dict:
  """kind: 'box' (collision_size), 'same' (one rotated off-centre mesh for
  visual and collision) or 'separate' (visual 2% larger than its own collision
  asset)."""
  ext = (0.03, 0.02, 0.05)
  obj = {
    "name": "block",
    "primitive": "mesh",
    "dynamic": True,
    "density": 800.0,
    "friction": [0.9, 0.004, 0.0002],
    "color": [0.2, 0.4, 0.6, 1.0],
    "initial_poses": [[-0.25, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]],
  }
  if kind == "box":
    obj["asset_path"] = str(_box_obj(tmp_path / "visual.obj", JENGA_SIZE))
    obj["collision_size"] = list(JENGA_SIZE)
    obj["scale"] = [1.2, 0.8, 1.0]
  elif kind == "same":
    obj["asset_path"] = str(_box_obj(tmp_path / "visual.obj", ext, OFF_ROT, OFF_CENTRE))
  else:
    obj["asset_path"] = str(_box_obj(tmp_path / "visual.obj", np.multiply(ext, 1.02), OFF_ROT, OFF_CENTRE))
    obj["collision_asset_path"] = str(
      _box_obj(tmp_path / "collision.obj", ext, OFF_ROT, OFF_CENTRE, reverse=True)
    )
  obj.update(extra)
  return obj


def _scene_with(tmp_path: Path, obj: dict, **scene_extra) -> Path:
  data = json.loads(DEFAULT_SCENE.read_text())
  data["objects"] = [data["objects"][0], obj]
  data["manipulated_object"] = obj["name"]
  data["support_surface"] = {"object": "table"}
  data.update(scene_extra)
  return _write(tmp_path / "scene.json", data)


def _load_obj(tmp_path: Path, kind: str, **extra):
  return load_scene_config(_scene_with(tmp_path, _mesh_object(tmp_path, kind, **extra))).manipulated


# --- Configuration -----------------------------------------------------------


def test_default_scales_and_disabled_by_default():
  c = ObjectSizeVariantsConfig()
  assert not c.enabled and c.clearance == 0.0
  assert len(c.scales) == 21 and (c.scales[0], c.scales[10], c.scales[-1]) == (0.9, 1.0, 1.1)
  assert c.scales == pytest.approx([0.90 + 0.01 * i for i in range(21)], abs=1e-12)
  for path in (DEFAULT_TASK, DEFAULT_TASK.parent / "grasp_curriculum_wxai.json"):
    assert not load_task_config(path).domain_randomization.object_size_variants.enabled
  example = load_task_config(SIZE_TASK).domain_randomization.object_size_variants
  assert example.enabled and example.scales == c.scales and example.clearance == 0.0


@pytest.mark.parametrize(
  "bad, match",
  [
    ({"scales": []}, "non-empty"),
    ({"scales": [1.0, 0.0]}, "positive"),
    ({"scales": [-0.9]}, "positive"),
    ({"scales": [1.0, 1.0]}, "unique"),
    ({"scales": [True]}, "list of numbers"),
    ({"clearance": -0.001}, "clearance"),
    ({"enabled": 1}, "bool"),
    ({"scale": [1.0]}, "unknown keys"),
  ],
)
def test_size_variant_config_rejects(tmp_path, bad, match):
  data = json.loads(SIZE_TASK.read_text())
  data["domain_randomization"]["object_size_variants"] = bad
  with pytest.raises(ConfigError, match=match):
    load_task_config(_write(tmp_path / "task.json", data))


def test_support_surface_from_table_top():
  scene = load_scene_config(JENGA_SCENE)
  assert scene.support_surface.object == "table"
  assert scene.support_height() == pytest.approx(0.0)  # pose z -0.03, height 0.06


def _scene_dict(**overrides) -> dict:
  data = json.loads(DEFAULT_SCENE.read_text())
  data.update(overrides)
  return data


def test_support_surface_explicit_height_and_rotated_table(tmp_path):
  scene = load_scene_config(_write(tmp_path / "a.json", _scene_dict(support_surface={"height": 0.02})))
  assert scene.support_height() == 0.02
  # Rolled 90 deg about x: the table's 1.2 m y-extent is now vertical.
  data = _scene_dict(support_surface={"object": "table"})
  c = math.sqrt(0.5)
  data["objects"][0]["initial_poses"] = [[-0.12, 0.0, -0.6, c, c, 0.0, 0.0]]
  scene = load_scene_config(_write(tmp_path / "b.json", data))
  assert scene.support_height() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
  "support, match",
  [
    ({"object": "table", "height": 0.0}, "exactly one"),
    ({"object": "shelf"}, "not in objects"),
    ({"object": "cube"}, "manipulated"),
    ({"height": "top"}, "finite number"),
    ({"plane": 0.0}, "unknown keys"),
  ],
)
def test_support_surface_rejects(tmp_path, support, match):
  with pytest.raises(ConfigError, match=match):
    load_scene_config(_write(tmp_path / "s.json", _scene_dict(support_surface=support)))


def test_support_surface_rejects_dynamic_or_tilted(tmp_path):
  data = _scene_dict(support_surface={"object": "table"})
  data["objects"][0]["dynamic"] = True
  with pytest.raises(ConfigError, match="static box"):
    load_scene_config(_write(tmp_path / "d.json", data))
  data = _scene_dict(support_surface={"object": "table"})
  s, c = math.sin(math.radians(15)), math.cos(math.radians(15))
  data["objects"][0]["initial_poses"] = [[-0.12, 0.0, -0.03, c, s, 0.0, 0.0]]
  with pytest.raises(ConfigError, match="horizontal face"):
    load_scene_config(_write(tmp_path / "t.json", data))


@needs_jenga
def test_disabled_keeps_primitive_path():
  cfg = make_grasp_env_cfg(JENGA_SCENE, DEFAULT_TASK)
  assert all(type(e) is EntityCfg for n, e in cfg.scene.entities.items() if n != "robot")
  assert cfg.events["reset_jenga_xy"].func is mdp.reset_object_xy_in_disk_with_yaw
  model = cfg.scene.entities["jenga"].spec_fn().compile()
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "jenga_collision")
  assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX


@needs_jenga
def test_enabled_wiring_only_touches_manipulated_object():
  cfg = make_grasp_env_cfg(JENGA_SCENE, SIZE_TASK)
  ent = cfg.scene.entities["jenga"]
  assert isinstance(ent, SizeVariantEntityCfg)
  assert ent.scales == ObjectSizeVariantsConfig().scales and len(ent.variants) == 21
  assert isinstance(ent.assignment, BalancedRandomAssignment)
  assert type(cfg.scene.entities["table"]) is EntityCfg
  event = cfg.events["reset_jenga_xy"]
  assert event.func is mdp.ResetObjectOnSupport
  assert event.params["support_height"] == pytest.approx(0.0) and event.params["clearance"] == 0.0


@needs_jenga
def test_enabled_needs_mesh_object_and_support_surface():
  task = load_task_config(SIZE_TASK)
  with pytest.raises(ConfigError, match="mesh manipulated object"):
    make_grasp_env_cfg_from_configs(load_scene_config(DEFAULT_SCENE), task)
  scene = dataclasses.replace(load_scene_config(JENGA_SCENE), support_surface=None)
  with pytest.raises(ConfigError, match="support_surface"):
    make_grasp_env_cfg_from_configs(scene, task)


# --- Variant construction and physical properties ----------------------------


def _inertia_tensor(model, body=1) -> np.ndarray:
  rot = np.empty(9)
  mujoco.mju_quat2Mat(rot, model.body_iquat[body])
  rot = rot.reshape(3, 3)
  return rot @ np.diag(model.body_inertia[body]) @ rot.T


def _geom_points(model, data, gid) -> np.ndarray:
  """World (= body, at the default pose) points of a mesh or box geom."""
  if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
    mid = model.geom_dataid[gid]
    local = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
  else:
    local = object_size._box_corners(model.geom_size[gid])
  return data.geom_xpos[gid] + local @ data.geom_xmat[gid].reshape(3, 3).T


def _compiled(obj, s=None):
  model = object_spec(obj, s).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_visual")
  col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_collision")
  return model, _geom_points(model, data, vis), _geom_points(model, data, col), vis, col


def _extent(points) -> np.ndarray:
  return points.max(0) - points.min(0)


@pytest.mark.parametrize("mass", [None, 0.05])
def test_box_collision_mesh_matches_primitive_at_scale_one(tmp_path, mass):
  obj = _load_obj(tmp_path, "box", **({} if mass is None else {"mass": mass}))
  prim, _, prim_pts, _, prim_col = _compiled(obj)
  mesh, _, mesh_pts, _, mesh_col = _compiled(obj, 1.0)
  assert prim.geom_type[prim_col] == mujoco.mjtGeom.mjGEOM_BOX
  assert mesh.geom_type[mesh_col] == mujoco.mjtGeom.mjGEOM_MESH
  # Existing convention: full extents along the mesh axes, times the object scale.
  expected = np.multiply(JENGA_SIZE, obj.scale)
  np.testing.assert_allclose(_extent(prim_pts), expected, rtol=1e-6)
  np.testing.assert_allclose(_extent(mesh_pts), expected, rtol=1e-6)
  np.testing.assert_allclose(mesh_pts.mean(0), 0.0, atol=1e-9)  # same local pose
  np.testing.assert_allclose(mesh.body_mass[1], prim.body_mass[1], rtol=1e-6)  # float32 verts
  if mass is not None:
    assert mesh.body_mass[1] == pytest.approx(mass)
  np.testing.assert_allclose(mesh.body_ipos[1], prim.body_ipos[1], atol=1e-9)
  np.testing.assert_allclose(_inertia_tensor(mesh), _inertia_tensor(prim), rtol=1e-6, atol=1e-12)
  for field in ("geom_friction", "geom_condim", "geom_solref", "geom_solimp", "geom_margin",
                "geom_gap", "geom_priority", "geom_group", "geom_contype", "geom_conaffinity"):
    np.testing.assert_array_equal(getattr(mesh, field)[mesh_col], getattr(prim, field)[prim_col])


@pytest.mark.parametrize("kind", ["box", "same", "separate"])
@pytest.mark.parametrize("mass", [None, 0.05])
def test_variant_scaling_laws(tmp_path, kind, mass):
  obj = _load_obj(tmp_path, kind, **({} if mass is None else {"mass": mass}))
  ref, ref_vis, ref_col, *_ = _compiled(obj, 1.0)
  nominal = _compiled(obj)[0]
  assert ref.body_mass[1] == pytest.approx(nominal.body_mass[1], rel=1e-6)
  np.testing.assert_allclose(_inertia_tensor(ref), _inertia_tensor(nominal), rtol=1e-6, atol=1e-12)
  if kind != "box":
    np.testing.assert_allclose(ref.body_ipos[1], OFF_CENTRE, atol=1e-9)
  # Mass comes from the collision geometry only (the 'separate' visual mesh is
  # larger, so any visual contribution would show here).
  collision_volume = np.prod(np.multiply(JENGA_SIZE, obj.scale)) if kind == "box" else 0.03 * 0.02 * 0.05
  assert ref.body_mass[1] == pytest.approx(mass or 800.0 * collision_volume, rel=1e-6)
  for s in REPRESENTATIVE:
    model, vis, col, vis_id, col_id = _compiled(obj, s)
    assert model.body_mass[1] == pytest.approx(ref.body_mass[1] * s**3, rel=1e-6)
    np.testing.assert_allclose(
      _inertia_tensor(model), _inertia_tensor(ref) * s**5, rtol=1e-5, atol=1e-6 * np.abs(_inertia_tensor(ref)).max() * s**5
    )
    np.testing.assert_allclose(model.body_ipos[1], ref.body_ipos[1] * s, atol=1e-9)
    np.testing.assert_allclose(_extent(vis), _extent(ref_vis) * s, rtol=1e-6)
    np.testing.assert_allclose(_extent(col), _extent(ref_col) * s, rtol=1e-6)
    np.testing.assert_allclose(col.mean(0), ref_col.mean(0) * s, atol=1e-9)
    ratio = 1.02 if kind == "separate" else 1.0
    np.testing.assert_allclose(_extent(vis), _extent(col) * ratio, rtol=1e-6)
    assert model.geom_contype[vis_id] == 0 and model.geom_conaffinity[vis_id] == 0
    np.testing.assert_array_equal(model.geom_rgba[vis_id], ref.geom_rgba[vis_id])
    np.testing.assert_array_equal(model.geom_friction[col_id], obj.friction)


def test_box_mesh_is_closed_and_outward():
  verts, faces = _box_mesh((0.3, 0.2, 0.1))
  v = np.asarray(verts).reshape(-1, 3)
  tris = v[np.asarray(faces).reshape(-1, 3)]
  normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
  assert len(tris) == 12 and ((normals * tris.mean(1)).sum(-1) > 0).all()
  volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
  assert volume == pytest.approx(0.3 * 0.2 * 0.1)


# --- Assignment and reset geometry -------------------------------------------


def test_balanced_random_assignment_is_seeded():
  assign = BalancedRandomAssignment(21)
  np.random.seed(0)
  a = assign(42)
  np.random.seed(0)
  assert assign(42) == a
  np.random.seed(1)
  assert assign(42) != a
  assert sorted(Counter(a).values()) == [2] * 21
  assert a != sorted(a)  # not mjlab's contiguous block allocation
  np.random.seed(0)
  counts = Counter(assign(50))
  assert len(counts) == 21 and set(counts.values()) == {2, 3}
  np.random.seed(0)
  assert 0 <= assign(1)[0] < 21


def test_lowest_point_z_matches_rotated_points():
  g = torch.Generator().manual_seed(0)
  points = torch.randn(7, 30, 3, generator=g)
  quat = torch.nn.functional.normalize(torch.randn(7, 4, generator=g), dim=-1)
  expected = quat_apply(quat[:, None].expand(-1, 30, -1), points)[..., 2].amin(-1)
  torch.testing.assert_close(object_size.lowest_point_z(points, quat), expected)


# --- Simulation ---------------------------------------------------------------


def _make_env(scene, task, num_envs, seed=0) -> ManagerBasedRlEnv:
  cfg = make_grasp_env_cfg(scene, task, None, num_envs)
  cfg.seed = seed
  return ManagerBasedRlEnv(cfg, device=DEVICE)


CLEARANCE = 0.002


@pytest.fixture(scope="module")
def variant_env(tmp_path_factory):
  task = _task(tmp_path_factory.mktemp("task"), scales=list(REPRESENTATIVE), clearance=CLEARANCE)
  env = _make_env(JENGA_SCENE, task, 6)
  yield env
  env.close()


def _slot_collision_geoms(env, name):
  m = env.sim.mj_model
  return [
    g for g in env.scene[name].indexing.geom_ids.tolist()
    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{name}/mjlab/pad/")
    and "/collision/" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
  ]


def _sim_collision_points(env, name) -> list[np.ndarray]:
  """Per world: every vertex of the collision meshes that world simulates, in
  world coordinates, from the current kinematics (independent of the cached
  reset geometry)."""
  m = env.sim.mj_model
  dataid = env.sim.wp_model.geom_dataid.numpy()
  xpos = env.sim.data.geom_xpos.cpu().numpy()
  xmat = env.sim.data.geom_xmat.cpu().numpy().reshape(env.num_envs, -1, 3, 3)
  out = []
  for w in range(env.num_envs):
    pts = []
    for g in _slot_collision_geoms(env, name):
      mid = dataid[w, g]
      local = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      pts.append(xpos[w, g] + local @ xmat[w, g].T)
    out.append(np.concatenate(pts))
  return out


def _lowest_z(env, name) -> np.ndarray:
  origins = env.scene.env_origins[:, 2].cpu().numpy()
  return np.array([p[:, 2].min() for p in _sim_collision_points(env, name)]) - origins


def _variant_state(env, name):
  bid = env.scene[name].indexing.root_body_id
  return (
    env.sim.world_to_variant[name].clone(),
    torch.as_tensor(env.sim.wp_model.geom_dataid.numpy()).clone(),
    env.sim.model.body_mass[:, bid].clone(),
  )


@needs_cuda
@needs_jenga
def test_assignment_uses_env_seed_and_is_balanced(variant_env):
  env = variant_env
  w2v = env.sim.world_to_variant["jenga"]
  np.random.seed(0)
  assert w2v.tolist() == BalancedRandomAssignment(3)(6)
  assert torch.bincount(w2v).tolist() == [2, 2, 2]
  torch.testing.assert_close(
    object_size.assigned_scale(env, "jenga"),
    torch.tensor(REPRESENTATIVE, device=DEVICE)[w2v],
  )


@needs_cuda
@needs_jenga
def test_same_seed_same_assignment_new_seed_differs(tmp_path_factory):
  envs = [_make_env(JENGA_SCENE, SIZE_TASK, 42, seed) for seed in (7, 7, 8)]
  try:
    a, b, c = (e.sim.world_to_variant["jenga"].tolist() for e in envs)
    assert a == b and a != c
    assert sorted(Counter(a).values()) == [2] * 21
  finally:
    for e in envs:
      e.close()


@needs_cuda
@needs_jenga
def test_per_world_physics_follow_scale(variant_env):
  env = variant_env
  m = env.sim.mj_model
  bid = env.scene["jenga"].indexing.root_body_id
  s = object_size.assigned_scale(env, "jenga").cpu().numpy()
  mass = env.sim.model.body_mass[:, bid].cpu().numpy()
  inertia = env.sim.model.body_inertia[:, bid].cpu().numpy()
  iquat = env.sim.model.body_iquat[:, bid].cpu().numpy()
  ipos = env.sim.model.body_ipos[:, bid].cpu().numpy()
  tensors = []
  for w in range(env.num_envs):
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, iquat[w].astype(np.float64))
    tensors.append(rot.reshape(3, 3) @ np.diag(inertia[w]) @ rot.reshape(3, 3).T)
  nominal = load_scene_config(JENGA_SCENE).manipulated
  ref = object_spec(nominal).compile()  # the existing primitive-box build
  for w in range(env.num_envs):
    assert mass[w] == pytest.approx(ref.body_mass[1] * s[w] ** 3, rel=1e-5)
    np.testing.assert_allclose(tensors[w], _inertia_tensor(ref) * s[w] ** 5, rtol=1e-4, atol=1e-11)
    np.testing.assert_allclose(ipos[w], 0.0, atol=1e-7)
  # Collision box extents per world (body frame), and the jenga visual mesh agrees.
  env.reset()
  quat = env.scene["jenga"].data.root_link_quat_w.cpu().numpy().astype(np.float64)
  root = env.scene["jenga"].data.root_link_pos_w.cpu().numpy()
  vis_gid = next(
    g for g in env.scene["jenga"].indexing.geom_ids.tolist()
    if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g).endswith("/visual/0")
  )
  dataid = env.sim.wp_model.geom_dataid.numpy()
  xpos = env.sim.data.geom_xpos.cpu().numpy()
  xmat = env.sim.data.geom_xmat.cpu().numpy().reshape(env.num_envs, -1, 3, 3)
  for w, pts in enumerate(_sim_collision_points(env, "jenga")):
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, quat[w])
    to_body = lambda p: (p - root[w]) @ rot.reshape(3, 3)  # noqa: E731
    np.testing.assert_allclose(_extent(to_body(pts)), np.multiply(JENGA_SIZE, s[w]), rtol=1e-4)
    mid = dataid[w, vis_gid]
    local = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    vis = to_body(xpos[w, vis_gid] + local @ xmat[w, vis_gid].T)
    np.testing.assert_allclose(_extent(vis), np.multiply(JENGA_SIZE, s[w]), rtol=0.02)
  # Every variant's collision mesh carries the hull and polygons MuJoCo Warp
  # needs for multi-contact.
  for g in _slot_collision_geoms(env, "jenga"):
    for mid in np.unique(dataid[:, g]):
      assert m.mesh_graphadr[mid] >= 0 and m.mesh_polynum[mid] == 6


@needs_cuda
@needs_jenga
def test_reset_seats_lowest_collision_point_on_support(variant_env):
  env = variant_env
  nominal_z = env.scene["jenga"].data.default_root_state[:, 2]
  s = object_size.assigned_scale(env, "jenga")
  for _ in range(3):
    env.reset()
    np.testing.assert_allclose(_lowest_z(env, "jenga"), CLEARANCE, atol=2e-6)
    z = env.scene["jenga"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # The block stands on its 0.075 m axis: root z = s * 0.0375 + clearance.
    torch.testing.assert_close(z, s * nominal_z + CLEARANCE, atol=2e-6, rtol=0)


@needs_cuda
@needs_jenga
def test_lift_target_uses_corrected_pose(variant_env):
  env = variant_env
  env.reset()
  goal = task_state.goal(env, "grasp")
  pos = env.scene["jenga"].data.root_link_pos_w
  torch.testing.assert_close(goal.initial_object_pos, pos, atol=0, rtol=0)
  torch.testing.assert_close(goal.target_pos - pos, torch.tensor([0, 0, 0.03], device=DEVICE).expand_as(pos))
  # Auto-reset inside step(): the capture follows the corrected pose too.
  env.episode_length_buf[:] = env.max_episode_length - 1
  env.step(torch.zeros(env.num_envs, 7, device=DEVICE))
  assert (env.episode_length_buf == 0).all()
  np.testing.assert_allclose(_lowest_z(env, "jenga"), CLEARANCE, atol=2e-6)
  torch.testing.assert_close(goal.initial_object_pos, env.scene["jenga"].data.root_link_pos_w, atol=0, rtol=0)


@needs_cuda
@needs_jenga
def test_variant_survives_resets_and_subset_reset_is_local(variant_env):
  env = variant_env
  before = _variant_state(env, "jenga")
  env.reset()
  for _ in range(5):
    env.step(torch.rand(env.num_envs, 7, device=DEVICE) * 2 - 1)
  obj = env.scene["jenga"]
  keep, reset = torch.tensor([0, 2, 3, 5], device=DEVICE), torch.tensor([1, 4], device=DEVICE)
  state_before = obj.data.root_link_pose_w.clone()
  goal = task_state.goal(env, "grasp")
  initial_before, target_before = goal.initial_object_pos.clone(), goal.target_pos.clone()
  env.reset(env_ids=reset)
  torch.testing.assert_close(obj.data.root_link_pose_w[keep], state_before[keep])
  torch.testing.assert_close(goal.initial_object_pos[keep], initial_before[keep])
  torch.testing.assert_close(goal.target_pos[keep], target_before[keep])
  np.testing.assert_allclose(_lowest_z(env, "jenga")[reset.cpu().numpy()], CLEARANCE, atol=2e-6)
  torch.testing.assert_close(goal.initial_object_pos[reset], obj.data.root_link_pos_w[reset])
  for ids in (None, torch.tensor([3], device=DEVICE)):
    env.reset(env_ids=ids)
  env.episode_length_buf[:] = env.max_episode_length - 1
  env.step(torch.zeros(env.num_envs, 7, device=DEVICE))
  for a, b in zip(before, _variant_state(env, "jenga"), strict=True):
    assert torch.equal(a, b)


def _scripted_grasp(env, steps=120):
  obj = env.scene["jenga"]
  goal = task_state.goal(env, "grasp")
  tcp = env.reward_manager.get_term_cfg("reach").params["tcp_cfg"]
  for i in range(steps):
    o, t = obj.data.root_link_pos_w, task_state.tcp_pose_w(env, tcp)[0]
    if i < 30:
      target, grip = o + torch.tensor([0, 0, 0.06], device=DEVICE), 1.0
    elif i < 50:
      target, grip = o.clone(), 1.0
    elif i < 65:
      target, grip = o.clone(), -1.0
    else:
      target, grip = goal.target_pos.clone(), -1.0
    action = torch.zeros(env.num_envs, 7, device=DEVICE)
    action[:, :3] = ((target - t) / 0.01).clamp(-1, 1)
    action[:, 6] = grip
    env.step(action)


@needs_cuda
@needs_jenga
def test_resting_and_grasp_contact(variant_env):
  env = variant_env
  obj = env.scene["jenga"]
  env.reset()
  quat0 = obj.data.root_link_quat_w.clone()
  for _ in range(60):  # 3 s, the arm parked above the table
    env.step(torch.zeros(env.num_envs, 7, device=DEVICE))
  # Dropped by the clearance, then at rest on four contacts like the
  # primitive box (~0.1 mm solver penetration), without spinning or rocking.
  lowest = _lowest_z(env, "jenga")
  assert (lowest < 0).all() and (lowest > -3e-4).all()
  assert obj.data.root_link_vel_w.abs().max() < 1e-3
  assert ((obj.data.root_link_quat_w * quat0).sum(-1).abs() > 1 - 1e-6).all()

  env.reset()
  _scripted_grasp(env)
  grasp = env.reward_manager.get_term_cfg("grasp").params
  stable = env.reward_manager.get_term_cfg("stable_hold").params
  assert task_state.is_grasped(env, **grasp).all()
  assert task_state.is_lifted(env, "grasp", stable["object_cfg"], 0.01).all()


@needs_cuda
@pytest.mark.parametrize("kind", ["same", "separate"])
def test_rotated_off_centre_mesh_seated_on_support(tmp_path, kind):
  tilt = _rot([1, 0, 0], 30.0) @ _rot([0, 1, 1], 20.0)
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, tilt.ravel())
  obj = _mesh_object(tmp_path, kind, initial_poses=[[-0.25, 0.0, 0.08, *quat.tolist()]])
  scene = _scene_with(tmp_path, obj)
  env = _make_env(scene, _task(tmp_path, scales=[0.9, 1.1]), 4)
  try:
    env.reset()
    np.testing.assert_allclose(_lowest_z(env, "block"), 0.0, atol=2e-6)
    # Independently: compile the variant in isolation at the simulated pose.
    cfg_obj = load_scene_config(scene).manipulated
    s = object_size.assigned_scale(env, "block").tolist()
    pose = env.scene["block"].data.root_link_pose_w.cpu().numpy()
    for w in range(env.num_envs):
      model = object_spec(cfg_obj, s[w]).compile()
      data = mujoco.MjData(model)
      data.qpos[:7] = pose[w]
      mujoco.mj_forward(model, data)
      col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_collision")
      assert _geom_points(model, data, col)[:, 2].min() == pytest.approx(0.0, abs=2e-6)
  finally:
    env.close()
