"""The training arm controller, mjlab's DifferentialIKAction, on measured joints. numpy + mujoco.

In sim (tabletop_sim ClippedDifferentialIKAction, mjlab 1.6.0) the policy action becomes an
absolute TCP goal once per control step (actions.ee_delta_target is that part), and on every
physics substep (sim_dt 2 ms, 25 per step) DifferentialIKAction.apply_actions runs one damped
least-squares solve toward the held goal from the current joints and sets the joint position
target q + dq. solve() is that substep, line for line:

    J^T W J + (w_lim^2 [q outside soft limits] + w_post^2 + damping^2) I) dq
        = J^T W [p_goal - p; rotvec(R_goal R^T)] + w_lim^2 r_limit + w_post^2 (q_posture - q)
    dq = clip(dq, -max_dq, max_dq),  q_target = clip(q + dq, actuator ctrlrange)

with the site Jacobian of the robot model at q. The soft limits are mjlab's: the joint range
scaled by soft_joint_pos_limit_factor about its middle. The ctrlrange clip is what MuJoCo
does to a limited position actuator's ctrl. tests/test_ik_parity.py checks it against the
mjlab action itself.

With the task's action.ee_goal_interpolation, substep i of the N in a control step solves
toward a subgoal and not the goal: the start pose (the TCP the goal was computed from)
advanced by alpha = i / N of the step's delta,

    p_sub = p_start + alpha (p_goal - p_start)
    R_sub = R(alpha rotvec(R_goal R_start^T)) R_start

set_goal(goal, start) and solve(q, alpha) are that; the caller picks alpha (ik_arm.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import matrix_to_rotvec
from .observations import make_T, rotvec_to_matrix

ARM_JOINTS = tuple(f"joint_{i}" for i in range(6))


@dataclass(frozen=True)
class IkParams:
    """DifferentialIKActionCfg fields (tabletop_sim IkDefaults) and the entity's soft limit factor."""
    damping: float
    max_dq: float
    position_weight: float
    orientation_weight: float
    joint_limit_weight: float
    posture_weight: float
    soft_joint_pos_limit_factor: float


class DifferentialIK:
    def __init__(self, model_path: Path, site: str, params: IkParams, posture_target, joints=ARM_JOINTS):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)   # fk() runs on the policy thread, solve() on the loop's
        self.params = params
        m = self.model
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
        if min(jids) < 0:
            raise ValueError(f"{model_path}: joints {joints} not all found")
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"{model_path}: site {site!r} not found")
        self.qpos_adr = m.jnt_qposadr[jids]
        self.dof_adr = m.jnt_dofadr[jids]
        lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * params.soft_joint_pos_limit_factor
        self.soft_lower, self.soft_upper = mid - half, mid + half
        self.ctrl_lower, self.ctrl_upper = self._ctrlrange(jids, lo, hi)
        self.posture_target = np.asarray(posture_target, dtype=np.float64).reshape(len(joints))
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.set_goal(np.eye(4))

    def _ctrlrange(self, jids, lo, hi):
        """ctrlrange of the position actuator on each joint (MuJoCo clamps ctrl to it); else the joint range."""
        m, mj = self.model, self._mj
        lower, upper = lo.copy(), hi.copy()
        for i, j in enumerate(jids):
            for a in range(m.nu):
                if (m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j
                        and m.actuator_ctrllimited[a]):
                    lower[i], upper[i] = m.actuator_ctrlrange[a]
        return lower, upper

    def _kinematics(self, data, q) -> None:
        data.qpos[self.qpos_adr] = q
        self._mj.mj_kinematics(self.model, data)
        self._mj.mj_comPos(self.model, data)

    def fk(self, q) -> np.ndarray:
        """TCP site pose in the robot base frame at arm joints q."""
        d = self._fk_data
        self._kinematics(d, np.asarray(q, dtype=np.float64))
        return make_T(d.site_xmat[self.site_id].reshape(3, 3).copy(), d.site_xpos[self.site_id].copy())

    def set_goal(self, T_base_goal, T_base_start=None) -> None:
        """The absolute TCP goal process_actions would store (_desired_pos / _desired_quat).

        T_base_start, the TCP pose the goal was computed from, makes subgoal() interpolate from
        it; without one every subgoal is the goal.
        """
        T = np.asarray(T_base_goal, dtype=np.float64)
        self.goal_pos, self.goal_rot = T[:3, 3].copy(), T[:3, :3].copy()
        if T_base_start is None:
            self.start_pos, self.start_rot = self.goal_pos, self.goal_rot
            self.delta_pos, self.delta_rotvec = np.zeros(3), np.zeros(3)
            return
        S = np.asarray(T_base_start, dtype=np.float64)
        self.start_pos, self.start_rot = S[:3, 3].copy(), S[:3, :3].copy()
        self.delta_pos = self.goal_pos - self.start_pos
        self.delta_rotvec = matrix_to_rotvec(self.goal_rot @ self.start_rot.T)

    def subgoal(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation) the solve at fraction alpha of the step aims at; the goal from 1 on."""
        if alpha >= 1.0:
            return self.goal_pos, self.goal_rot
        return (self.start_pos + alpha * self.delta_pos,
                rotvec_to_matrix(alpha * self.delta_rotvec) @ self.start_rot)

    def solve(self, q, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(q_target, dq) of one apply_actions() from arm joints q toward subgoal(alpha)."""
        p, q = self.params, np.asarray(q, dtype=np.float64)
        self._kinematics(self.data, q)
        pos = self.data.site_xpos[self.site_id]
        rot = self.data.site_xmat[self.site_id].reshape(3, 3)
        goal_pos, goal_rot = self.subgoal(alpha)
        pos_error = goal_pos - pos
        rot_error = matrix_to_rotvec(goal_rot @ rot.T)
        self._mj.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        jacp, jacr = self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]

        wp2, wo2 = p.position_weight ** 2, p.orientation_weight ** 2
        JTJ = wp2 * jacp.T @ jacp + wo2 * jacr.T @ jacr
        JTdx = wp2 * jacp.T @ pos_error + wo2 * jacr.T @ rot_error
        r_limit = np.minimum(self.soft_upper - q, 0.0) + np.maximum(self.soft_lower - q, 0.0)
        violated = (r_limit != 0).astype(np.float64)
        wl2 = p.joint_limit_weight ** 2
        wpost2 = p.posture_weight ** 2
        lam = max(p.damping, 1e-6)
        JTJ[np.diag_indices_from(JTJ)] += wl2 * violated + wpost2 + lam * lam
        JTdx += wl2 * violated * r_limit + wpost2 * (self.posture_target - q)
        dq = np.clip(np.linalg.solve(JTJ, JTdx), -p.max_dq, p.max_dq)
        return np.clip(q + dq, self.ctrl_lower, self.ctrl_upper), dq
