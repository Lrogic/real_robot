"""Trial list, target generator and FK safety check shared by the hardware and sim runs.

A trial moves one joint from a base pose. Its commanded target at time t since onset comes
from TargetGenerator, which both runs call once per loop tick with the measured joint, so
the hardware and the sim receive the same commands for the same measurements.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REAL_ROBOT = HERE.parents[2]
RESULTS_DIR = HERE / "results"
for _path in (REAL_ROBOT, EXPERIMENTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_PATH_KEYS = ("calibration_config", "fk_model", "sim_model")
KINDS = ("step", "ramp", "relative", "step_once", "relative_once")


def load_config(path: Path | None = None) -> dict:
    path = Path(path or HERE / "config.yaml").resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in _PATH_KEYS:
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["config_file"] = str(path)
    return cfg


@dataclass(frozen=True)
class Trial:
    pose: str
    joint: int
    kind: str           # one of KINDS
    amplitude: float    # rad, signed
    goal_time: float    # s, set_arm_positions goal time of every command
    ticks: int          # relative: number of policy ticks; else 0
    command_s: float    # time the target is still changing
    observe_s: float    # total logged time from onset

    @property
    def shape(self) -> str:
        """step | ramp | relative: how the target moves."""
        return self.kind.removesuffix("_once")

    @property
    def resend(self) -> bool:
        """True: the target is sent every loop tick. False: only when it changes."""
        return not self.kind.endswith("_once")

    @property
    def name(self) -> str:
        return (f"{self.pose}_j{self.joint}_{self.kind}_{self.amplitude * 1e3:+.0f}mrad_"
                f"gt{self.goal_time * 1e3:.0f}ms")

    def to_json(self) -> dict:
        return {**asdict(self), "name": self.name}


def build_trials(cfg: dict, poses=None, joints=None, kinds=None) -> list[Trial]:
    period = float(cfg["policy_period_s"])
    trials = []
    for pose in poses or list(cfg["poses"]):
        if pose not in cfg["poses"]:
            raise ValueError(f"unknown pose {pose!r}")
        for joint in joints if joints is not None else cfg["joints"]:
            for kind in kinds or KINDS:
                section = cfg[kind]
                for goal_time in section["goal_times"]:
                    for amplitude in section["amplitudes"]:
                        for sign in (1.0, -1.0):
                            ticks, command_s = 0, 0.0
                            if kind == "ramp":
                                command_s = period
                            elif kind.startswith("relative"):
                                ticks = max(1, min(int(section["relative_ticks"]),
                                                   int(section["relative_max_travel"] / amplitude + 1e-9)))
                                command_s = ticks * period
                            trials.append(Trial(pose, int(joint), kind, sign * float(amplitude), float(goal_time),
                                                ticks, command_s, command_s + float(section["observe_s"])))
    return trials


class TargetGenerator:
    """Commanded position of the trial's joint. Call once per loop tick."""

    def __init__(self, trial: Trial, base: float, period: float):
        self.trial, self.base, self.period = trial, float(base), float(period)
        self._tick = -1
        self._target = float(base)

    def __call__(self, t: float, measured: float) -> float:
        tr = self.trial
        if tr.shape == "step":
            return self.base + tr.amplitude
        if tr.shape == "ramp":
            return self.base + tr.amplitude * min(1.0, max(0.0, t) / self.period)
        tick = min(int(t / self.period), tr.ticks - 1)
        if tick > self._tick:               # a new policy tick: re-base on the measured joint
            self._tick = tick
            self._target = float(measured) + tr.amplitude
        return self._target

    def extreme(self) -> float:
        """Farthest target the trial can command (relative: if every tick is fully reached)."""
        tr = self.trial
        return self.base + tr.amplitude * (tr.ticks if tr.shape == "relative" else 1)


class Kinematics:
    """TCP position of the arm model at given joints (plain MuJoCo)."""

    def __init__(self, model_path: str, site: str):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(6)]
        self.qadr = self.model.jnt_qposadr[jids]
        self.joint_range = self.model.jnt_range[jids].copy()
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if min(jids) < 0 or self.site < 0:
            raise ValueError(f"{model_path}: joint_0..5 or site {site!r} not found")

    def tcp(self, q) -> np.ndarray:
        self.data.qpos[self.qadr] = q
        self._mj.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()


def check_trials(cfg: dict, trials: list[Trial], kin: Kinematics, joint_min=None, joint_max=None) -> list[str]:
    """Problems with the poses and every trial's farthest target; empty when all are safe."""
    sc = cfg["safety"]
    lo = (kin.joint_range[:, 0] if joint_min is None else np.asarray(joint_min)) + sc["joint_limit_margin"]
    hi = (kin.joint_range[:, 1] if joint_max is None else np.asarray(joint_max)) - sc["joint_limit_margin"]
    box_lo, box_hi = np.asarray(sc["workspace_min"]), np.asarray(sc["workspace_max"])

    def problems(label: str, q) -> list[str]:
        out = []
        if np.any(q < lo) or np.any(q > hi):
            out.append(f"{label}: joints {np.round(q, 3).tolist()} outside the limits with margin")
        p = kin.tcp(q)
        if np.any(p < box_lo) or np.any(p > box_hi):
            out.append(f"{label}: TCP {np.round(p, 3).tolist()} outside the workspace box")
        return out

    found = []
    for name, q in cfg["poses"].items():
        found += problems(f"pose {name}", np.asarray(q, dtype=np.float64))
    for trial in trials:
        q = np.asarray(cfg["poses"][trial.pose], dtype=np.float64).copy()
        q[trial.joint] = TargetGenerator(trial, q[trial.joint], cfg["policy_period_s"]).extreme()
        found += problems(trial.name, q)
    return found


def session_seconds(cfg: dict, trials: list[Trial]) -> float:
    return sum(t.observe_s + cfg["return_time_s"] + cfg["settle_s"] for t in trials)
