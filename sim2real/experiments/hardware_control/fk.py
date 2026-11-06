"""Forward kinematics of the training robot model with plain MuJoCo (CPU).

The model is the embodiment XML compiled the way training compiles it (``load_spec``) and
the scene base pose is not applied, so poses are in the robot base frame (the body that
carries joint_0), the same frame as the driver's Cartesian outputs. This is the common
TCP definition for sim and hardware trajectories: both are evaluated from joint positions
with this one model. It measures encoder-derived kinematics only; it cannot see link flex
or calibration error of the physical arm.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import settings  # noqa: F401
from common.sim import joint_addresses, load_training_robot
from geometry import make_T


class TrainingFK:
    def __init__(self, scene_path: str | Path):
        self.model, self.emb = load_training_robot(Path(scene_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.emb.tcp_site))
        if self.site_id < 0:
            raise ValueError(f"TCP site {self.emb.tcp_site!r} not in the model")
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.emb.arm_joint_names[0])
        self.base_id = int(self.model.body_parentid[self.model.jnt_bodyid[joint]])
        self.qadr = [joint_addresses(self.model, n)[0] for n in self.emb.arm_joint_names]
        self.vadr = [joint_addresses(self.model, n)[1] for n in self.emb.arm_joint_names]
        for name, value in self.emb.ready_joint_pos.items():
            self.data.qpos[joint_addresses(self.model, name)[0]] = value
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    @property
    def ready_arm(self) -> np.ndarray:
        return np.array([self.emb.ready_joint_pos[n] for n in self.emb.arm_joint_names])

    @property
    def ready_gripper(self) -> float:
        return float(self.emb.ready_joint_pos[self.emb.gripper_actuator_joint])

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xmat[self.base_id].reshape(3, 3).copy(), self.data.xpos[self.base_id].copy()

    def tcp(self, q_arm) -> np.ndarray:
        """TCP pose in the base frame for arm joint positions q_arm (rad)."""
        self.data.qpos[self.qadr] = np.asarray(q_arm, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)
        R_b, p_b = self._base()
        R_s = self.data.site_xmat[self.site_id].reshape(3, 3)
        p_s = self.data.site_xpos[self.site_id]
        return make_T(R_b.T @ R_s, R_b.T @ (p_s - p_b))

    def tcp_velocity(self, q_arm, qd_arm) -> tuple[np.ndarray, np.ndarray]:
        """(linear m/s, angular rad/s) TCP velocity in the base frame from J(q) @ qd."""
        self.tcp(q_arm)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        qd = np.asarray(qd_arm, dtype=np.float64)
        R_b, _ = self._base()
        return R_b.T @ (self._jacp[:, self.vadr] @ qd), R_b.T @ (self._jacr[:, self.vadr] @ qd)

    def describe(self) -> dict:
        return {
            "embodiment": self.emb.name,
            "xml_path": str(self.emb.xml_path),
            "tcp_site": self.emb.tcp_site,
            "tcp_site_local_pos_m": self.model.site_pos[self.site_id].tolist(),
            "base_body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_id) or "world",
            "arm_joint_names": list(self.emb.arm_joint_names),
            "mujoco_version": mujoco.__version__,
        }
