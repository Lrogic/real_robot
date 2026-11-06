"""Generate trial plans: absolute TCP targets built once from the shared reference pose.

The reference pose is the training TCP at the training ready joints (training-model FK).
A plan file is the single source of targets for the sim reference and every hardware
condition; results embed it and its sha256, and the comparison refuses to pair results
from different plans.

Launch from the wxai_mjlab environment (or its trossen-arm clone):

  python plans.py --suite basic                     # all single-axis plans, 3 tests x +-x,y,z
  python plans.py --test single --kind translation --axis z --sign 1
  python plans.py --test relative --goal-translation 0.03 0 0 --goal-rotation 0 0 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from settings import RESULTS_DIR, load_settings, snapshot
from geometry import axis_vector, displaced, pose_json

PLAN_KIND = "hardware_control_plan"
TESTS = ("single", "scheduled", "relative")
MOTIONS = ("translation", "rotation")
FRAMES_NOTE = (
    "Poses are the TCP in the robot base frame (training model: body carrying joint_0; "
    "hardware: driver base frame). Translations are base-frame vectors. Rotations are "
    "base-frame rotation vectors applied on the left about the TCP origin "
    "(R_target = exp(r) @ R), the training apply_delta_pose convention."
)


def plan_digest(plan: dict) -> str:
    keys = ("test", "reference", "outer_period_s", "targets", "relative", "single", "scheduled")
    body = {k: plan.get(k) for k in keys}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def scheduled_multipliers(out_steps: int, back_steps: int, hold_steps: int) -> list[int]:
    """Step multiples of the scheduled sequence: 1..out, then back by one per update, then hold."""
    if out_steps < 1 or back_steps < 0 or hold_steps < 0:
        raise ValueError("out_steps >= 1, back_steps >= 0, hold_steps >= 0")
    s = list(range(1, out_steps + 1))
    s += [out_steps - i for i in range(1, back_steps + 1)]
    s += [s[-1]] * hold_steps
    return s


def _offset(kind: str, axis: str, sign: int, amount: float) -> tuple[np.ndarray, np.ndarray]:
    v = axis_vector(axis, sign) * amount
    return (v, np.zeros(3)) if kind == "translation" else (np.zeros(3), v)


def _reference(fk) -> dict:
    q = fk.ready_arm
    return {"arm_joints_rad": q.tolist(), "gripper_m": fk.ready_gripper, "tcp": pose_json(fk.tcp(q)),
            "source": "training-model FK of EmbodimentCfg.ready_joint_pos", "model": fk.describe()}


def make_plan(fk, cfg: dict, test: str, kind: str | None = None, axis: str | None = None, sign: int | None = None,
              goal_translation=None, goal_rotation=None) -> dict:
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}")
    reference = _reference(fk)
    T_ref = fk.tcp(fk.ready_arm)
    period = 1.0 / float(cfg["timing"]["outer_hz"])
    m = cfg["motion"]
    plan = {"kind": PLAN_KIND, "test": test, "frames": FRAMES_NOTE, "outer_period_s": period,
            "reference": reference, "targets": [], "relative": None,
            "single": dict(cfg["single"]), "scheduled": dict(cfg["scheduled"]), "settings": snapshot(cfg)}
    if test in ("single", "scheduled"):
        if kind not in MOTIONS:
            raise ValueError(f"kind must be one of {MOTIONS}")
        step = m["translation_m"] if kind == "translation" else m["rotation_rad"]
        plan["motion"] = {"kind": kind, "axis": axis, "sign": sign, "step": step,
                          "direction": axis_vector(axis, sign).tolist()}
        s = cfg["scheduled"]
        mults = [1] if test == "single" else scheduled_multipliers(s["out_steps"], s["back_steps"], s["hold_steps"])
        for k, mult in enumerate(mults):
            dp, dr = _offset(kind, axis, sign, mult * step)
            plan["targets"].append({"index": k, "t_s": k * period, "step_multiple": mult,
                                    "translation_m": dp.tolist(), "rotvec_rad": dr.tolist(),
                                    "pose": pose_json(displaced(T_ref, dp, dr))})
        plan["name"] = f"{test}_{kind[:3]}_{axis}{'+' if sign > 0 else '-'}"
    else:
        r = cfg["relative"]
        if goal_translation is None and goal_rotation is None:
            if kind not in MOTIONS:
                raise ValueError("relative needs --kind/--axis/--sign or an explicit goal")
            amount = r["goal_translation_m"] if kind == "translation" else r["goal_rotation_rad"]
            dp, dr = _offset(kind, axis, sign, amount)
            plan["motion"] = {"kind": kind, "axis": axis, "sign": sign, "goal_amount": amount,
                              "direction": axis_vector(axis, sign).tolist()}
            plan["name"] = f"relative_{kind[:3]}_{axis}{'+' if sign > 0 else '-'}"
        else:
            dp = np.asarray(goal_translation if goal_translation is not None else (0, 0, 0), dtype=float)
            dr = np.asarray(goal_rotation if goal_rotation is not None else (0, 0, 0), dtype=float)
            plan["motion"] = {"kind": "combined", "goal_translation_m": dp.tolist(), "goal_rotvec_rad": dr.tolist()}
            plan["name"] = "relative_combined"
        plan["relative"] = {
            "goal": pose_json(displaced(T_ref, dp, dr)),
            "goal_translation_m": dp.tolist(), "goal_rotvec_rad": dr.tolist(),
            "step_limit": r["step_limit"],
            "max_translation_step_m": r["max_translation_step_m"],
            "max_rotation_step_rad": r["max_rotation_step_rad"],
            "position_tolerance_m": r["position_tolerance_m"],
            "orientation_tolerance_rad": r["orientation_tolerance_rad"],
            "timeout_s": r["timeout_s"], "post_goal_observe_s": r["post_goal_observe_s"],
            "step_limit_note": (
                "per_axis clips each base-frame component of the translation and rotation-vector "
                "error to +-limit, as the training action does (clip(a,-1,1) * scale). euclidean "
                "limits the translation norm and rotation angle. The deployed policy uses per_axis."
            ),
        }
    plan["sha256"] = plan_digest(plan)
    return plan


def save_plan(plan: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{plan['name']}.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    return path


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("kind") != PLAN_KIND:
        raise ValueError(f"{path} is not a {PLAN_KIND}")
    if plan_digest(plan) != plan.get("sha256"):
        raise ValueError(f"{path}: contents do not match the stored sha256 (edited after generation?)")
    return plan


def basic_suite(fk, cfg: dict) -> list[dict]:
    plans = []
    for test in TESTS:
        for kind in MOTIONS:
            for axis in "xyz":
                for sign in (1, -1):
                    plans.append(make_plan(fk, cfg, test, kind, axis, sign))
    return plans


def main() -> None:
    from fk import TrainingFK

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "plans")
    parser.add_argument("--suite", choices=("basic",), default=None)
    parser.add_argument("--test", choices=TESTS)
    parser.add_argument("--kind", choices=MOTIONS)
    parser.add_argument("--axis", choices=("x", "y", "z"))
    parser.add_argument("--sign", type=int, choices=(1, -1))
    parser.add_argument("--goal-translation", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--goal-rotation", type=float, nargs=3, default=None, metavar=("RX", "RY", "RZ"))
    args = parser.parse_args()
    cfg = load_settings(args.config)
    fk = TrainingFK(cfg["training"]["scene"])
    if args.suite:
        plans = basic_suite(fk, cfg)
    elif args.test:
        plans = [make_plan(fk, cfg, args.test, args.kind, args.axis, args.sign,
                           args.goal_translation, args.goal_rotation)]
    else:
        parser.error("give --suite or --test")
    for plan in plans:
        print(f"{save_plan(plan, args.out)}  sha256 {plan['sha256'][:12]}")


if __name__ == "__main__":
    main()
