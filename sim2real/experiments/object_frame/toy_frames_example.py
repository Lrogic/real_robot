"""Toy example of the FP and MuJoCo mesh frames (wxai_mjlab env, no GPU).

A 10 x 2 x 4 cm box whose OBJ origin is NOT at its centre: vertices span
x in [0.05, 0.15], y in [-0.01, 0.01], z in [0, 0.04] (metres).
"""

import mujoco
import numpy as np
import trimesh

np.set_printoptions(precision=4, suppress=True)

lo, hi = np.array([0.05, -0.01, 0.0]), np.array([0.15, 0.01, 0.04])
mesh = trimesh.creation.box(bounds=[lo, hi])
corner = np.array([0.15, 0.01, 0.04])  # one vertex, in OBJ coordinates

print("=== FoundationPose ===")
c = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
print("model_center c:", c)
print("corner in centred-mesh coords (corner - c):", corner - c)
tf_to_center = np.eye(4); tf_to_center[:3, 3] = -c
print("get_tf_to_centered_mesh():\n", tf_to_center)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
print("oriented_bounds extents:", extents)
print("to_origin:\n", to_origin)
print("inv(to_origin):\n", np.linalg.inv(to_origin))
print("corner in OBB coords (to_origin @ corner):", (to_origin @ [*corner, 1])[:3])

# A made-up FP result: OBJ frame 0.4 m in front of the camera, rotated 90 deg about camera z.
pose = np.eye(4)
pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
pose[:3, 3] = [0.0, 0.0, 0.4]
pose_last = pose @ np.linalg.inv(tf_to_center)
center_pose = pose @ np.linalg.inv(to_origin)
print("\npose (OBJ frame in camera):\n", pose)
print("pose_last (centred mesh in camera):\n", pose_last)
print("center_pose (OBB frame in camera):\n", center_pose)
for name, T, p in (("pose", pose, corner), ("pose_last", pose_last, corner - c),
                   ("center_pose", center_pose, (to_origin @ [*corner, 1])[:3])):
    print(f"corner in camera via {name:11s}:", (T @ [*p, 1])[:3])

print("\n=== MuJoCo ===")
path = "/tmp/toy_box.obj"
mesh.export(path)
spec = mujoco.MjSpec()
body = spec.worldbody.add_body(name="toy")
body.add_freejoint()
spec.add_mesh(name="toy", file=path)
body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="toy", density=1000)
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
verts = m.mesh_vert[: m.mesh_vertnum[0]]
print("stored mesh verts min/max (recentred):", verts.min(0), verts.max(0))
print("geom_pos (in body frame):", m.geom_pos[0], " geom_quat (w,x,y,z):", m.geom_quat[0])
print("body pos / quat (what root_link_pose_w reads):", d.xpos[1], d.xquat[1])
print("COM (xipos):", d.xipos[1])
Rg = d.geom_xmat[0].reshape(3, 3)
print("stored verts placed in world, min/max:", (verts @ Rg.T + d.geom_xpos[0]).min(0),
      (verts @ Rg.T + d.geom_xpos[0]).max(0))

print("\n=== collision_size box at body origin ===")
print("box spans x", -0.05, "to", 0.05, " visual mesh spans x", lo[0], "to", hi[0])
