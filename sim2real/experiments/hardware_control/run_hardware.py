"""Run plans on the WidowX AI with one control backend. Without --execute nothing connects.

Each selected trial: operator prompt, conservative joint move to the training ready pose
with a settle check (not measured), then the measured trial from the settled pose, then the
result file. A safety stop holds at the measured joints; Ctrl+C or a controller error puts
the arm and gripper in gravity compensation (manipulation.policy PolicyArm.kill). At exit
the arm is put in gravity compensation and goes idle after the support prompt.

Launch from the wxai_mjlab clone that has trossen-arm installed:

  # list what would run (no connection)
  python run_hardware.py results/plans/single_tra_z+.json --backend cartesian --goal-time 0.05 --space joint
  # one Cartesian condition on the arm
  python run_hardware.py results/plans/single_tra_z+.json --backend cartesian --goal-time 0.05 --space joint --execute
  # goal_time x interpolation-space sweep from config.yaml
  python run_hardware.py results/plans/scheduled_tra_x+.json --backend cartesian --sweep --execute
  # training IK on measured state, joint commands at 500 Hz
  python run_hardware.py results/plans/single_tra_z+.json --backend external_ik --ik-hz 500 --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from settings import load_settings, robot_identity_from_calibration, snapshot
from common.prompts import Quit, ask
from geometry import pose_from_json
from hardware import (CartesianBackend, ExternalIKBackend, HardwareRig, Safety, driver_interpolation,
                      hardware_configuration, reset_to_ready, run_trial)
from plans import load_plan
from records import RESULT_KIND, UNITS, environment_record, result_path, save_result
from timing import MonotonicClock, interval_stats

MAX_IK_HZ = 1000.0
IK_BUDGET_FRACTION = 0.5


def conditions(args, cfg) -> list[dict]:
    hc = cfg["hardware"]
    if args.backend == "cartesian":
        goal_times = args.goal_time or (hc["cartesian"]["goal_times_s"] if args.sweep else None)
        spaces = args.space or (hc["cartesian"]["interpolation_spaces"] if args.sweep else None)
        if not goal_times or not spaces:
            raise SystemExit("cartesian needs --goal-time and --space, or --sweep")
        for g in goal_times:
            driver_interpolation(g)
        return [{"backend": "cartesian", "space": s, "goal_time_s": float(g)} for g in goal_times for s in spaces]
    if args.goal_time or args.space:
        raise SystemExit("--goal-time/--space are Cartesian options; external_ik uses --ik-goal-time "
                         "(one value, default hardware.external_ik.goal_time_s)")
    rates = args.ik_hz or (hc["external_ik"]["ik_hz_list"] if args.sweep else [hc["external_ik"]["ik_hz"]])
    outer_hz = cfg["timing"]["outer_hz"]
    for r in rates:
        if not outer_hz < r <= MAX_IK_HZ:
            raise SystemExit(f"--ik-hz {r} must be in ({outer_hz}, {MAX_IK_HZ}]")
    goal_time = hc["external_ik"]["goal_time_s"] if args.ik_goal_time is None else args.ik_goal_time
    driver_interpolation(goal_time)
    return [{"backend": "external_ik", "ik_hz": float(r), "goal_time_s": float(goal_time)} for r in rates]


def describe(cond: dict) -> str:
    if cond["backend"] == "cartesian":
        return (f"cartesian space={cond['space']} goal_time={cond['goal_time_s']:g}s "
                f"({driver_interpolation(cond['goal_time_s'])})")
    return f"external_ik {cond['ik_hz']:g} Hz goal_time={cond['goal_time_s']:g}s"


def precheck(plan: dict, sc: dict) -> list[str]:
    """Plan targets against the configured workspace and offset limits (no joint limits yet)."""
    T_ref = pose_from_json(plan["reference"]["tcp"])
    safety = Safety(sc, T_ref, np.full(6, -np.inf), np.full(6, np.inf))
    poses = [pose_from_json(t["pose"]) for t in plan["targets"]]
    if plan["relative"]:
        poses.append(pose_from_json(plan["relative"]["goal"]))
    return [r for r in (safety.target(T) for T in poses) if r]


def build_ik(cfg: dict, device: str):
    from ik_adapter import TrainingIK, self_check
    from sim_reference import build_training_env, training_controller_record

    env = build_training_env(cfg, device)
    ik = TrainingIK(env)
    check = self_check(ik)
    if check["joint_target_diff_rad"] > 1e-5:
        raise SystemExit(f"IK adapter self-check failed: {check}")
    q0 = env.scene["robot"].data.default_joint_pos[0, ik.ids].cpu().numpy()
    rng = np.random.default_rng(0)
    times = []
    for _ in range(300):
        _, info = ik.solve(q0 + rng.normal(0, 0.01, 6))
        times.append(info["compute_s"])
    timing = interval_stats(times[50:])
    print(f"Training IK on {device}: self-check joint-target difference {check['joint_target_diff_rad']:.2e} rad; "
          f"solve median {timing['median'] * 1e3:.3f} ms, p95 {timing['p95'] * 1e3:.3f} ms")
    return ik, {"self_check": check, "solve_time_s": timing, "device": device,
                "training_controller": training_controller_record(env)}


def operator_prompt(index: int, total: int, label: str, run_all: bool) -> str:
    if run_all:
        return "run"
    while True:
        text = ask(f"\n[{index}/{total}] {label}\n  Enter = reset and run, s = skip, a = run all remaining, "
                   "q = quit: ").strip().lower()
        if text in ("", "a"):
            return "all" if text == "a" else "run"
        if text in ("s", "q"):
            return {"s": "skip", "q": "quit"}[text]


def confirm_far_start(message: str) -> None:
    if ask(f"{message}\nClear the path, then type 'yes' to move it to the ready pose: ").strip().lower() != "yes":
        raise Quit()


def summary_line(trial: dict) -> str:
    ts = trial.get("timing_summary", {})
    loop, outer, state = ts.get("loop", {}), ts.get("outer", {}), ts.get("state", {})
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
    return (f"  stop: {trial['stop_reason']}; loop {fmt(loop.get('achieved_hz'))}/{loop.get('requested_hz')} Hz, "
            f"skipped slots {loop.get('skipped_slots')}; outer {fmt(outer.get('achieved_hz'))} Hz, skipped "
            f"{outer.get('skipped_indices')}; fresh state {fmt(state.get('fresh_rate_hz'))} Hz, repeated reads "
            f"{state.get('repeated_reads')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", type=Path, nargs="+")
    parser.add_argument("--backend", choices=("cartesian", "external_ik"), required=True)
    parser.add_argument("--goal-time", type=float, nargs="+", default=None, help="Cartesian goal_time(s), s")
    parser.add_argument("--space", choices=("joint", "cartesian"), nargs="+", default=None,
                        help="Cartesian interpolation space(s)")
    parser.add_argument("--ik-hz", type=float, nargs="+", default=None, help="external IK/send rate(s)")
    parser.add_argument("--ik-goal-time", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="use the condition lists in config.yaml")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default=None, help="external IK device (default config)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="connect and move the arm")
    args = parser.parse_args()

    cfg = load_settings(args.config)
    hc = cfg["hardware"]
    plans = [load_plan(p) for p in args.plans]
    conds = conditions(args, cfg)
    trials = [(p, c) for _ in range(args.repeats) for p in plans for c in conds]
    problems = False
    print(f"{len(trials)} trial(s):")
    for i, (plan, cond) in enumerate(trials, 1):
        issues = precheck(plan, hc["safety"])
        problems |= bool(issues)
        print(f"  {i:3d}. {plan['name']:<22s} {describe(cond)}" + (f"  REJECTED: {issues}" if issues else ""))
    if problems:
        raise SystemExit("Some plans violate the safety limits in config.yaml; nothing was run.")
    ik, ik_record = (None, None)
    if args.backend == "external_ik":
        ik, ik_record = build_ik(cfg, args.device or hc["external_ik"]["device"])
        p95 = ik_record["solve_time_s"]["p95"]
        too_fast = sorted({c["ik_hz"] for c in conds if p95 > IK_BUDGET_FRACTION / c["ik_hz"]})
        if too_fast:
            raise SystemExit(f"IK solve p95 {p95 * 1e3:.2f} ms exceeds {IK_BUDGET_FRACTION:.0%} of the period at "
                             f"{too_fast} Hz (the rest is for the state read and send). Use a lower --ik-hz "
                             "or a faster --device; nothing was run.")
    if not args.execute:
        print("\nDry run: nothing connected. Add --execute to run these on the arm.")
        return

    from fk import TrainingFK

    fk = TrainingFK(cfg["training"]["scene"])
    identity = robot_identity_from_calibration(cfg)
    clock = MonotonicClock()
    rc, sc = hc["reset"], hc["safety"]
    with HardwareRig(identity, hc["gripper_limit_margin_m"]) as rig:
        hw_config = hardware_configuration(rig.driver)
        joint_min, joint_max = rig.arm_joint_limits()
        run_all = False
        try:
            for i, (plan, cond) in enumerate(trials, 1):
                choice = operator_prompt(i, len(trials), f"{plan['name']}  {describe(cond)}", run_all)
                if choice == "quit":
                    break
                if choice == "skip":
                    continue
                run_all = run_all or choice == "all"
                T_ref = pose_from_json(plan["reference"]["tcp"])
                safety = Safety(sc, T_ref, joint_min, joint_max)
                if cond["backend"] == "cartesian":
                    backend = CartesianBackend(rig, cond["space"], cond["goal_time_s"], hc["cartesian"]["poll_hz"])
                else:
                    backend = ExternalIKBackend(rig, ik, safety, cond["goal_time_s"], cond["ik_hz"])
                ready = np.asarray(plan["reference"]["arm_joints_rad"])
                reset = reset_to_ready(rig, ready, plan["reference"]["gripper_m"], rc, clock, confirm_far_start)
                if not reset["settled"]:
                    print(f"Did not settle at the ready pose: {reset}. Stopping.")
                    break
                trial = run_trial(rig, backend, plan, safety, fk, clock, stale_s=sc["stale_state_s"],
                                  max_trial_s=sc["max_trial_s"], stop_goal_time_s=rc["stop_goal_time_s"])
                result = {
                    "kind": RESULT_KIND, "source": "hardware", "backend": backend.name,
                    "condition": backend.condition(), "plan": plan, "plan_sha256": plan["sha256"],
                    "plan_name": plan["name"], "test": plan["test"], "settings": snapshot(cfg), "units": UNITS,
                    "environment": environment_record(), "robot_identity": identity,
                    "hardware_configuration": hw_config, "tcp_model": fk.describe(), "reset": reset,
                    "external_ik": ik_record, **trial,
                }
                path = save_result(result, result_path("hardware", plan["name"], backend.condition()["tag"]))
                print(summary_line(trial) + f"\n  saved {path}")
                if trial.get("interrupted") or rig.killed:
                    print("The arm was stopped; ending the session.")
                    break
                if str(trial["stop_reason"]).startswith("safety"):
                    run_all = False
        except (KeyboardInterrupt, Quit):
            rig.kill("keyboard interrupt / quit outside a trial")


if __name__ == "__main__":
    main()
