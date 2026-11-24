"""--sim-mirror: the live episode replayed in MuJoCo, rendered from the real camera's viewpoint.

After each FoundationPose update the sim arm is set to the measured joints (read through the
policy session's arm) and every tracked object to its tracked pose; nothing is simulated
(mj_forward only). The scene is in the robot base frame: the grip model of the arm, a table
plane at z = 0, each object's run mesh with its .mtl texture, and a camera at T_base_camera
with the stream's K.
"""

from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = (Path(__file__).resolve().parents[2] / "sim/mujoco/assets/robots/trossen_wxai_base/wxai"
             / "wxai_base_real_robot_friction_grip.xml")

ARM_JOINTS = [f"joint_{i}" for i in range(6)]
CARRIAGE_JOINTS = ["left_carriage_joint", "right_carriage_joint"]


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def _texture(mesh_path: Path):
    """The diffuse texture (map_Kd) of an OBJ's .mtl, or None."""
    for mtl in mesh_path.parent.glob("*.mtl"):
        for line in mtl.read_text().splitlines():
            if line.startswith("map_Kd"):
                return mtl.parent / line.split(None, 1)[1].strip()
    return None


class SimMirror:
    def __init__(self, session, objects, colors, T_base_camera, K, width, height):
        """session: the --manipulation policy session (its .arm, and .cfg for the ready pose); objects: the tracked
        SavedObjects with meshes, in tracker order; colors: their RGB draw colors."""
        self.arm, self.X = session.arm, T_base_camera
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
        world = spec.worldbody
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[0.5, 0.5, 0.5, 1])
        world.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1])
        for obj, color in zip(objects, colors):
            spec.add_mesh(name=obj.name, file=str(obj.mesh_path.resolve()))
            body = world.add_body(name=obj.name, mocap=True)
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=obj.name,
                                 rgba=[*np.asarray(color) / 255, 1], contype=0, conaffinity=0)
            texture = _texture(obj.mesh_path.resolve())
            if texture is not None:
                spec.add_texture(name=obj.name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
                material = spec.add_material(name=obj.name)
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = obj.name
                geom.material, geom.rgba = obj.name, [1, 1, 1, 1]
        # An OpenCV camera looks along +z with y down; a MuJoCo camera along -z with y up.
        cam = world.add_camera(name="real", pos=T_base_camera[:3, 3],
                               quat=_quat(T_base_camera[:3, :3] @ np.diag([1, -1, -1])))
        cam.resolution = [width, height]
        cam.sensor_size = [width, height]  # 1 sensor unit per pixel
        cam.focal_pixel = [K[0, 0], K[1, 1]]
        cam.principal_pixel = [K[0, 2] - width / 2, height / 2 - K[1, 2]]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        joint = lambda name: self.model.joint(name).qposadr[0]
        self.arm_adr = [joint(n) for n in ARM_JOINTS]
        self.carriage_adr = [joint(n) for n in CARRIAGE_JOINTS]
        self.data.qpos[self.arm_adr] = session.cfg.ready_joint_pos
        self.data.qpos[self.carriage_adr] = session.cfg.ready_gripper
        self.mocap = [self.model.body(o.name).mocapid[0] for o in objects]
        self.renderer = mujoco.Renderer(self.model, height, width)

    def update(self, tracker) -> np.ndarray:
        """Set the sim from the arm's joints and the tracked poses; returns the RGB render."""
        try:  # the session handles driver errors; the mirror keeps the last joints
            state = self.arm.read()  # None in --manipulation-dry-run
        except Exception:
            state = None
        if state is not None:
            self.data.qpos[self.arm_adr] = state.arm_q
            self.data.qpos[self.carriage_adr] = state.gripper_q
        for mocap, tracked in zip(self.mocap, tracker.objects):
            if tracked.center_pose is not None:
                T = self.X @ tracked.center_pose @ tracked.to_origin
                self.data.mocap_pos[mocap] = T[:3, 3]
                self.data.mocap_quat[mocap] = _quat(T[:3, :3])
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="real")
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
