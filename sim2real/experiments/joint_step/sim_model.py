"""The training arm model stepped through the joint-step trials, with adjustable joint parameters.

One physics step (2 ms) is one loop tick: read the joints, get the trial's target, pass it
through the command model, set the position actuators' ctrl, step. The command model is the
driver's goal time as the hardware loop uses it, a new set_arm_positions every tick:

    goal_time 0      the setpoint is the target
    goal_time G > 0  SimParams.command_model:
        exp      the setpoint moves toward the target by dt / G of the remaining distance
                 each tick (a linear interpolation restarted from the setpoint every tick)
        exp_ff   exp, plus the interpolation's velocity (target - setpoint) / G fed forward
                 to the velocity loop
        ramp_ff  a new linear interpolation over G starts only when the target changes;
                 its velocity is fed forward, and stays fed forward after the
                 interpolation ends, until the next command (as measured: one call with a
                 50 ms goal time leaves the joint (target - start) / (G kp_pos) past the
                 target)
    Trials that send each target once (the _once kinds) always use ramp_ff.

The real position mode is a cascade, effort = kv (kp_pos (setpoint - q) + v_ff - qd), which
is the model's position actuator (kp = kp_pos kv) with ctrl = setpoint + v_ff / kp_pos.

SimParams overrides the model's actuator gains and joint friction, armature and damping per
arm joint, delays the commands by a whole number of ticks, and passes the actuator setpoint
through a first-order lag (timeconst, as a MuJoCo position actuator's timeconst does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from plan import TargetGenerator, Trial

N = 6


@dataclass
class SimParams:
    kp: np.ndarray | None = None            # position actuator gains; None keeps the model's
    kv: np.ndarray | None = None
    frictionloss: np.ndarray | None = None  # Nm, dry friction
    armature: np.ndarray | None = None
    damping: np.ndarray | None = None       # Nm s/rad, passive joint damping
    delay_ticks: int = 0                    # command delay
    timeconst: np.ndarray | None = None     # s, first-order lag of the actuator setpoint; None or 0 = none
    command_model: str = "exp"              # exp | exp_ff | ramp_ff (goal_time > 0 only)
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


class JointStepSim:
    def __init__(self, model_path: str, params: SimParams | None = None, gripper: float = 0.026):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(N)]
        self.qadr, self.dadr = m.jnt_qposadr[self.jids], m.jnt_dofadr[self.jids]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(N)]
        if min(self.jids) < 0 or min(self.act) < 0:
            raise ValueError(f"{model_path}: joint_0..5 or their actuators not found")
        self.gripper_act = [a for a in range(m.nu) if a not in self.act]
        self.gripper_qadr = [m.jnt_qposadr[j] for j in range(m.njnt) if j not in self.jids]
        self.gripper = gripper
        self.dt = float(m.opt.timestep)
        self.defaults = SimParams(kp=m.actuator_gainprm[self.act, 0].copy(), kv=-m.actuator_biasprm[self.act, 2].copy(),
                                  frictionloss=m.dof_frictionloss[self.dadr].copy(),
                                  armature=m.dof_armature[self.dadr].copy(), damping=m.dof_damping[self.dadr].copy())
        self.params = self.defaults
        if params is not None:
            self.set_params(params)

    def set_params(self, p: SimParams) -> None:
        m, d = self.model, self.defaults
        kp = d.kp if p.kp is None else np.asarray(p.kp, dtype=np.float64)
        kv = d.kv if p.kv is None else np.asarray(p.kv, dtype=np.float64)
        m.actuator_gainprm[self.act, 0] = kp
        m.actuator_biasprm[self.act, 1] = -kp
        m.actuator_biasprm[self.act, 2] = -kv
        m.dof_frictionloss[self.dadr] = d.frictionloss if p.frictionloss is None else p.frictionloss
        m.dof_armature[self.dadr] = d.armature if p.armature is None else p.armature
        m.dof_damping[self.dadr] = d.damping if p.damping is None else p.damping
        self.params = p

    def reset(self, q, settle_s: float = 0.5, ctrl=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.gripper_qadr] = self.gripper
        self.data.ctrl[self.act] = q if ctrl is None else ctrl
        self.data.ctrl[self.gripper_act] = self.gripper
        for _ in range(int(round(settle_s / self.dt))):
            mujoco.mj_step(self.model, self.data)

    def run_trial(self, trial: Trial, base, period: float, q_start=None) -> dict:
        """q_start: begin at rest at these joints (a measured start) with the setpoint at base,
        without settling; default settles at base."""
        base = np.asarray(base, dtype=np.float64)
        if q_start is None:
            self.reset(base)
        else:
            self.reset(q_start, settle_s=0.0, ctrl=base)
        d, p = self.data, self.params
        kp_pos = self.model.actuator_gainprm[self.act, 0] / np.maximum(-self.model.actuator_biasprm[self.act, 2], 1e-9)
        gen = TargetGenerator(trial, base[trial.joint], period)
        target, setpoint = base.copy(), base.copy()
        last_command, ramp_from, ramp_t = base.copy(), base.copy(), np.inf
        ramp_v = np.zeros(N)
        G = trial.goal_time
        queue = [base.copy()] * max(0, int(p.delay_ticks))
        tau = np.zeros(N) if p.timeconst is None else np.asarray(p.timeconst, dtype=np.float64)
        blend = np.where(tau > 0, 1.0 - np.exp(-self.dt / np.maximum(tau, 1e-9)), 1.0)
        filtered = base.copy()
        rows = {k: [] for k in ("t", "q", "qd", "effort", "target")}
        q0 = d.qpos[self.qadr].copy()
        for i in range(int(round(trial.observe_s / self.dt))):
            t = i * self.dt
            q = d.qpos[self.qadr].copy()
            target[trial.joint] = gen(t, q[trial.joint])
            for key, value in zip(rows, (t, q, d.qvel[self.dadr].copy(), d.actuator_force[self.act].copy(),
                                         target.copy())):
                rows[key].append(value)
            queue.append(target.copy())
            command = queue.pop(0)
            v_ff = 0.0
            if G <= 0:
                setpoint = command
            elif p.command_model == "ramp_ff" or not trial.resend:
                if np.any(np.abs(command - last_command) > 1e-9):
                    ramp_from, ramp_t, last_command = setpoint.copy(), 0.0, command.copy()
                    ramp_v = (command - ramp_from) / G
                ramp_t += self.dt
                v_ff = ramp_v
                setpoint = ramp_from + (command - ramp_from) * min(1.0, ramp_t / G)
            else:
                if p.command_model == "exp_ff":
                    v_ff = (command - setpoint) / G
                setpoint = setpoint + (command - setpoint) * min(1.0, self.dt / G)
            filtered = filtered + (setpoint + v_ff / kp_pos - filtered) * blend
            d.ctrl[self.act] = filtered
            mujoco.mj_step(self.model, d)
        return {**{k: np.array(v) for k, v in rows.items()}, "q_start": q0, "base": base}
