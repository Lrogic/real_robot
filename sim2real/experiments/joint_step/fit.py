"""Fit the arm model's joint parameters to a hardware joint-step session. Environment: wxai_mjlab.

    python fit.py results/hardware_<stamp>.npz [--joints 1 2] [--tag fitted]

Per joint, on that joint's goal-time-0 step and relative trials (both poses), from the
measured start position: dry friction (frictionloss), armature and passive damping are
and an actuator lag are fitted for each whole-tick command delay, with the actuator gains kept at the driver's.
The loss is the mean squared position error over time as a fraction of the amplitude.
Then the three goal-time command models are scored on the goal-time-50 ms trials.
Writes results/fit_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from plan import RESULTS_DIR, Trial, load_config
from sim_model import JointStepSim, SimParams

DELAYS = (2, 3, 4, 5)
AMPLITUDE_FLOOR = 0.005     # rad added to the amplitude in the loss, so 5 mrad trials do not dominate
MODELS = ("exp", "exp_ff", "ramp_ff")


def load(path: str):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    data = np.load(path)
    trials = []
    for t in meta["trials"]:
        tr = Trial(**{k: t[k] for k in ("pose", "joint", "kind", "amplitude", "goal_time", "ticks", "command_s",
                                        "observe_s")})
        trials.append((tr, {k: data[f"{tr.name}/{k}"] for k in ("t", "q", "q_start", "base")}))
    return meta, trials


def trial_loss(sim: JointStepSim, tr: Trial, real: dict, period: float) -> float:
    out = sim.run_trial(tr, real["base"], period, q_start=real["q_start"])
    j = tr.joint
    q_sim = np.interp(real["t"], out["t"], out["q"][:, j])
    return float(np.mean(((q_sim - real["q"][:, j]) / (abs(tr.amplitude) + AMPLITUDE_FLOOR)) ** 2))


def params_for(sim: JointStepSim, joint: int, x, delay: int, model: str = "exp") -> SimParams:
    d = sim.defaults
    f, a, c = d.frictionloss.copy(), d.armature.copy(), d.damping.copy()
    f[joint], a[joint], c[joint] = abs(x[0]), abs(x[1]), abs(x[2])
    tau = np.zeros(6)
    if len(x) > 3:
        tau[joint] = abs(x[3])
    return SimParams(frictionloss=f, armature=a, damping=c, delay_ticks=delay, command_model=model, timeconst=tau)


def fit_joint(args) -> dict:
    path, joint, cfg = args
    _, trials = load(path)
    period = cfg["policy_period_s"]
    mine = [(tr, real) for tr, real in trials if tr.joint == joint]
    plant = [(tr, real) for tr, real in mine if tr.goal_time == 0 and tr.kind in ("step", "relative")]
    gt = [(tr, real) for tr, real in mine if tr.goal_time > 0 and tr.resend]
    sim = JointStepSim(cfg["sim_model"], gripper=cfg["gripper"])
    d = sim.defaults

    def loss(x, delay, subset, model="exp"):
        sim.set_params(params_for(sim, joint, x, delay, model))
        return float(np.mean([trial_loss(sim, tr, real, period) for tr, real in subset]))

    x_default = [d.frictionloss[joint], d.armature[joint], d.damping[joint], 0.0]
    best = None
    for delay in DELAYS:
        for x0 in ([0.3, d.armature[joint], 0.05, 0.004], [1.5, d.armature[joint] * 2, 0.5, 0.012]):
            res = minimize(loss, x0, args=(delay, plant), method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-6, "maxfev": 260})
            if best is None or res.fun < best["loss"]:
                best = {"loss": float(res.fun), "delay_ticks": delay, "x": np.abs(res.x).tolist()}
    result = {"joint": joint, "plant_trials": len(plant),
              "default": {"frictionloss": x_default[0], "armature": x_default[1], "damping": x_default[2],
                          "loss": loss(x_default, 0, plant)},
              "fitted": {"frictionloss": best["x"][0], "armature": best["x"][1], "damping": best["x"][2],
                         "timeconst": best["x"][3], "delay_ticks": best["delay_ticks"], "loss": best["loss"]},
              "goal_time_models": {m: {"default_plant": loss(x_default, 0, gt, m),
                                       "fitted_plant": loss(best["x"], best["delay_ticks"], gt, m)} for m in MODELS}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--joints", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--tag", default="fitted")
    args = parser.parse_args()
    cfg = load_config()
    with Pool(len(args.joints)) as pool:
        results = pool.map(fit_joint, [(args.file, j, cfg) for j in args.joints])
    print("joint | frictionloss Nm  armature  damping  delay | loss default -> fitted (rms fraction of amplitude)")
    for r in results:
        f, d = r["fitted"], r["default"]
        print(f"  j{r['joint']}  | {d['frictionloss']:.2f} -> {f['frictionloss']:5.2f}   {d['armature']:.4f} -> {f['armature']:.4f}   "
              f"{d['damping']:.2f} -> {f['damping']:.2f}   lag {f['timeconst'] * 1e3:.1f} ms  {f['delay_ticks']} ticks | {d['loss'] ** 0.5:.3f} -> {f['loss'] ** 0.5:.3f}")
        print("        goal-time 50 ms trials, rms by command model (fitted plant): "
              + "  ".join(f"{m} {v['fitted_plant'] ** 0.5:.3f}" for m, v in r["goal_time_models"].items()))
    from common.io import write_json

    write_json(RESULTS_DIR / f"fit_{args.tag}.json", {"kind": "joint_step_fit", "hardware_file": args.file,
                                                      "joints": results})
    print(f"Saved {RESULTS_DIR / f'fit_{args.tag}.json'}")


if __name__ == "__main__":
    main()
