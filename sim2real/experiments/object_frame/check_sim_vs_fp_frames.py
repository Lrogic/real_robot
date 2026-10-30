"""Temp check: compare the sim jenga body frame, the raw OBJ frame, FoundationPose's
centred-mesh frame and the wrapper's oriented-bounds (center_pose) frame."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sim/mujoco/src"))
from tabletop_sim.scene import load_scene_config  # noqa: E402
from tabletop_sim.scene.builder import object_spec  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
scene = load_scene_config(ROOT / "sim/mujoco/configs/scenes/tabletop_jenga_wxai.json")
obj = scene.manipulated
print("asset", obj.asset_path, "scale", obj.scale, "collision_size", obj.collision_size)

# --- Raw OBJ as trimesh / FoundationPose sees it.
mesh = trimesh.load(obj.asset_path, force="mesh")
v = np.asarray(mesh.vertices)
lo, hi = v.min(0), v.max(0)
print("\n[trimesh] n_verts", len(v), "AABB min", lo, "max", hi, "extent", hi - lo)
print("[trimesh] AABB centre (= FP model_center)", (lo + hi) / 2)
print("[trimesh] vertex mean", v.mean(0), " volume centroid", mesh.center_mass if mesh.is_watertight else "n/a (not watertight)")
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
print("[trimesh] oriented_bounds extents", extents)
print("[trimesh] to_origin (mesh -> OBB):\n", to_origin)
print("[trimesh] inv(to_origin) = OBB frame expressed in mesh frame:\n", np.linalg.inv(to_origin))

# --- MuJoCo: compile the object alone, body at identity.
spec = object_spec(obj)
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
b = m.body(obj.name).id
print("\n[mujoco] body xpos", d.xpos[b], "xquat", d.xquat[b], "(identity expected)")
print("[mujoco] body ipos (COM in body)", m.body_ipos[b], "iquat", m.body_iquat[b], "mass", m.body_mass[b])
for g in range(m.ngeom):
    print(f"[mujoco] geom {m.geom(g).name:16s} type {m.geom_type[g]} pos {m.geom_pos[g]} quat {m.geom_quat[g]} size {m.geom_size[g]}")
vis = m.geom(f"{obj.name}_visual").id
mid = m.geom_dataid[vis]
mv = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
# Mesh vertices placed by the geom frame back in the body frame.
R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[vis]); R = R.reshape(3, 3)
mv_body = mv @ R.T + m.geom_pos[vis]
print("[mujoco] stored mesh verts AABB (recentred):", mv.min(0), mv.max(0))
print("[mujoco] mesh verts in body frame AABB:", mv_body.min(0), mv_body.max(0))
# Match against the raw OBJ points (MuJoCo may dedupe / reorder vertices).
from scipy.spatial import cKDTree  # noqa: E402
dist, _ = cKDTree(v).query(mv_body)
print("[mujoco] max |body-frame mesh vert - nearest raw OBJ vert| =", dist.max(), "m")

# --- Full env: does root_link_pos_w equal xpos (body origin), not COM?
print("\nOBB frame origin in mesh frame:", np.linalg.inv(to_origin)[:3, 3])
print("rotation angle mesh->OBB (deg):",
      np.degrees(np.arccos(np.clip((np.trace(to_origin[:3, :3]) - 1) / 2, -1, 1))))
