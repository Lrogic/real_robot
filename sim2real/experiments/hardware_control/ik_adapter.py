"""Use the training ``ClippedDifferentialIKAction`` with an absolute held TCP target, in
simulation and on measured hardware joint states.

What the installed mjlab 1.6.0 ``DifferentialIKAction`` does (verified in its source):

* ``process_actions(a)`` (once per policy step) reads the current site pose and writes the
  absolute goal into ``_desired_pos`` / ``_desired_quat``: ``p + a[:3] * delta_pos_scale``
  and ``exp(a[3:] * delta_ori_scale) @ R`` (base/world-frame rotation, left-multiplied).
  ``ClippedDifferentialIKAction`` only clamps ``a`` to [-1, 1] first.
* ``apply_actions()`` (every physics substep) runs ``compute_dq()``: one damped
  least-squares step toward the stored goal from the current site pose, site Jacobian and
  joint positions (joint-limit rows, no posture term with the training weights), clamps
  each joint to +-max_dq and sets ``joint_pos_target = q + dq``. Nothing is integrated
  across calls, there is no dt in the solve, and no joint-increment clip other than max_dq.

The adaptation here is ``set_held_target``: it writes an absolute goal into the same two
buffers that ``process_actions`` would have filled, and is called only at outer updates.
The DLS solve and its application are the action's own ``apply_actions``. Relative actions
are never re-applied per substep.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from geometry import make_T, quat_wxyz_to_matrix
from common.poses import quaternion_wxyz


def world_from_base(env) -> np.ndarray:
    pose = env.scene["robot"].data.root_link_pose_w[0].detach().cpu().numpy().astype(np.float64)
    return make_T(quat_wxyz_to_matrix(pose[3:7]), pose[:3])


def set_held_target(term, T_world) -> None:
    """Absolute TCP goal (world frame) into the action's desired-pose buffers."""
    T_world = np.asarray(T_world, dtype=np.float64)
    pos = torch.as_tensor(T_world[:3, 3], dtype=term._desired_pos.dtype, device=term._desired_pos.device)
    quat = torch.as_tensor(quaternion_wxyz(T_world[:3, :3]), dtype=term._desired_quat.dtype,
                           device=term._desired_quat.device)
    term._desired_pos[:] = pos
    term._desired_quat[:] = quat


def held_target(term) -> tuple[np.ndarray, np.ndarray]:
    return (term._desired_pos[0].detach().cpu().numpy().copy(), term._desired_quat[0].detach().cpu().numpy().copy())


def action_config(term) -> dict:
    c = term.cfg
    return {
        "class": f"{type(term).__module__}.{type(term).__qualname__}",
        "parent": f"{type(term).__mro__[1].__module__}.{type(term).__mro__[1].__qualname__}",
        "actuator_names": list(c.actuator_names), "frame_type": c.frame_type, "frame_name": c.frame_name,
        "use_relative_mode": c.use_relative_mode, "delta_pos_scale": c.delta_pos_scale,
        "delta_ori_scale": c.delta_ori_scale, "damping": c.damping, "max_dq": c.max_dq,
        "position_weight": c.position_weight, "orientation_weight": c.orientation_weight,
        "joint_limit_weight": c.joint_limit_weight, "posture_weight": c.posture_weight,
        "posture_target": c.posture_target,
        "soft_joint_limits_rad": [term._joint_lower[0].tolist(), term._joint_upper[0].tolist()],
    }


def self_check(ik: "TrainingIK", action=(0.4, -0.3, 0.5, 0.2, -0.6, 0.3), q_arm=None) -> dict:
    """Same state and target through two paths; the joint targets must agree.

    A: the action's own relative path, as in training: process_actions(action) then
       apply_actions() at q (kinematics refreshed by sim.forward()).
    B: this experiment's path: the same goal composed in numpy from the TCP pose at q
       (p + clip(a)*pos_scale, exp(clip(a)*rot_scale) @ R), set_target(), solve(q).
    """
    from geometry import displaced

    term, robot = ik.term, ik.robot
    q = np.asarray(q_arm if q_arm is not None else robot.data.default_joint_pos[0, ik.ids].cpu().numpy(), dtype=np.float64)
    ik._q[0] = torch.as_tensor(q.astype(np.float32))
    ik._qd.zero_()
    robot.write_joint_state_to_sim(ik._q, ik._qd, joint_ids=ik.ids)
    ik.env.sim.forward()
    a = torch.tensor([action], dtype=torch.float32, device=ik.env.device)
    term.process_actions(a)
    desired_a = held_target(term)
    term.apply_actions()
    q_a = robot.data.joint_pos_target[0, ik.ids].detach().cpu().numpy().astype(np.float64)

    d = ik.env.sim.data
    p_w = d.site_xpos[0, term._frame_id].detach().cpu().numpy().astype(np.float64)
    R_w = d.site_xmat[0, term._frame_id].detach().cpu().numpy().astype(np.float64).reshape(3, 3)
    T_base = np.linalg.inv(ik.T_world_base) @ make_T(R_w, p_w)
    c = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    T_goal = displaced(T_base, c[:3] * term.cfg.delta_pos_scale, c[3:] * term.cfg.delta_ori_scale)
    ik.set_target(T_goal)
    desired_b = held_target(term)
    q_b, _ = ik.solve(q)
    qa, qb = desired_a[1], desired_b[1]
    return {
        "action": list(action), "q_arm_rad": q.tolist(),
        "desired_position_diff_m": float(np.max(np.abs(desired_a[0] - desired_b[0]))),
        "desired_rotation_diff_rad": float(2 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))),
        "joint_target_diff_rad": float(np.max(np.abs(q_a - q_b))),
        "joint_target_action_path_rad": q_a.tolist(), "joint_target_adapter_path_rad": q_b.tolist(),
    }


class TrainingIK:
    """The training arm action evaluated on measured joint states; physics is never stepped.

    ``solve(q)`` writes q into the env's qpos (qvel too, though the solve does not use it),
    runs ``sim.forward()`` to refresh the site pose, Jacobian inputs and every other derived
    quantity at q, then calls the action's ``apply_actions()`` and reads the joint-position
    target it set. Gripper joints stay at the ready value; the TCP site is on link_6, so
    they do not affect the solve.

    Difference from training: in training only the first of the 25 substeps sees fresh
    kinematics; the other 24 use the site pose and Jacobian from before the previous
    mj_step integration (2 ms old) with the current q. Here they are always computed at the
    measured q. The measured q itself is as old as the driver's state message.
    """

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term("arm")
        self.ids = self.term._joint_ids
        self.T_world_base = world_from_base(env)
        self.max_dq = float(self.term.cfg.max_dq)
        self._q = torch.zeros(1, len(self.ids), device=env.device)
        self._qd = torch.zeros_like(self._q)

    def set_target(self, T_base) -> None:
        set_held_target(self.term, self.T_world_base @ np.asarray(T_base, dtype=np.float64))

    def solve(self, q_arm, qd_arm=None) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        self._q[0] = torch.as_tensor(np.asarray(q_arm, dtype=np.float32))
        if qd_arm is not None:
            self._qd[0] = torch.as_tensor(np.asarray(qd_arm, dtype=np.float32))
        self.robot.write_joint_state_to_sim(self._q, self._qd, joint_ids=self.ids)
        self.env.sim.forward()
        self.term.apply_actions()
        q_target = self.robot.data.joint_pos_target[0, self.ids].detach().cpu().numpy().astype(np.float64)
        dt = time.perf_counter() - t0
        dq = q_target - np.asarray(q_arm, dtype=np.float64)
        return q_target, {"compute_s": dt, "max_dq_clamped": bool(np.any(np.abs(dq) >= self.max_dq - 1e-6))}
