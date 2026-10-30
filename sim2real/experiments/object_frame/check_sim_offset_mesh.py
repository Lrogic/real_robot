"""Temp check (wxai_mjlab env): for the off-centre, rotated mesh written by
check_fp_offset_mesh.py, is the MuJoCo body frame (what root_link_pose_w reads)
still the original OBJ frame? Builds the object with the training object_spec,
mesh collision (no collision_size), places the body at a known world pose, and
compares the compiled visual and collision geometry with T_gt applied to the
raw OBJ vertices."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "sim/mujoco/src"))
from tabletop_sim.scene.builder import object_spec  # noqa: E402
from tabletop_sim.scene.config import ObjectConfig  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
MESH = HERE / "tmp_offset_mesh/jenga_offset.obj"

obj = ObjectConfig.from_dict(
    {"name": "probe", "primitive": "mesh", "asset_path": str(MESH),
     "initial_poses": [[0, 0, 0, 1, 0, 0, 0]]},
    HERE,
)
spec = object_spec(obj)
m = spec.compile()
d = mujoco.MjData(m)

T_gt = np.eye(4)
T_gt[:3, :3] = R.from_euler("xyz", [200, 20, 40], degrees=True).as_matrix()
T_gt[:3, 3] = [0.3, -0.1, 0.2]
q_wxyz = R.from_matrix(T_gt[:3, :3]).as_quat(scalar_first=True)
d.qpos[:7] = [*T_gt[:3, 3], *q_wxyz]
mujoco.mj_forward(m, d)

b = m.body("probe").id
print("body xpos", d.xpos[b], " (set", T_gt[:3, 3], ")")
print("body COM xipos", d.xipos[b], " distance body origin -> COM [mm]:",
      1000 * np.linalg.norm(d.xipos[b] - d.xpos[b]))

raw = np.asarray(trimesh.load(MESH, force="mesh").vertices)
raw_w = raw @ T_gt[:3, :3].T + T_gt[:3, 3]
tree = cKDTree(raw_w)
for g in range(m.ngeom):
    mid = m.geom_dataid[g]
    mv = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    Rg = d.geom_xmat[g].reshape(3, 3)
    mv_w = mv @ Rg.T + d.geom_xpos[g]
    dist, _ = tree.query(mv_w)
    print(f"geom {m.geom(g).name:16s} local pos {m.geom_pos[g]} quat {m.geom_quat[g]}")
    print(f"   max distance compiled verts (world) -> T_gt * raw OBJ verts: {dist.max():.2e} m")
