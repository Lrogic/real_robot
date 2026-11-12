"""The joint-step trials in the training arm model. Environment: wxai_mjlab. Run from this directory.

    python run_sim.py [--tag NAME] [--params FILE] [--poses ...] [--joints ...] [--kinds ...]

--params is a JSON of SimParams fields (per-joint lists of six, delay_ticks, command_model);
fields left out keep the model's values.

Writes results/sim_<tag>.npz and .json in the hardware result format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plan import KINDS, RESULTS_DIR, build_trials, load_config
from sim_model import JointStepSim, SimParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tag", default="training_model")
    parser.add_argument("--params", default=None)
    parser.add_argument("--poses", nargs="+", default=None)
    parser.add_argument("--joints", nargs="+", type=int, default=None)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=None)
    args = parser.parse_args()

    from common.io import write_json

    cfg = load_config(args.config)
    sim = JointStepSim(cfg["sim_model"], gripper=cfg["gripper"])
    if args.params:
        fields = json.loads(Path(args.params).read_text())
        sim.set_params(SimParams(**{k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
                                    for k, v in fields.items()}))
    if abs(sim.dt - 1.0 / cfg["loop_hz"]) > 1e-9:
        raise ValueError(f"model timestep {sim.dt} is not one loop tick (1 / {cfg['loop_hz']})")
    trials = build_trials(cfg, args.poses, args.joints, args.kinds)
    results = {}
    for trial in trials:
        data = sim.run_trial(trial, cfg["poses"][trial.pose], cfg["policy_period_s"])
        for key, value in data.items():
            results[f"{trial.name}/{key}"] = value
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS_DIR / f"sim_{args.tag}.npz", **results)
    write_json(RESULTS_DIR / f"sim_{args.tag}.json",
               {"kind": "joint_step_session", "source": "sim", "config": cfg,
                "trials": [t.to_json() for t in trials], "sim_params": sim.params.to_json()})
    print(f"Saved {len(trials)} trials to {RESULTS_DIR / f'sim_{args.tag}.npz'}")


if __name__ == "__main__":
    main()
