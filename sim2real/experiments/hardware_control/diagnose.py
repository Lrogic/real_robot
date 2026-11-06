"""Read-only timing diagnostic: driver state-message rate, read cost, and training-IK cost.

Hardware part: connects like the policy deployment's state reader (arm and gripper in
gravity compensation, zero external effort, never position-commanded), reads the
controller configuration, then polls get_robot_output() at --poll-hz for --duration s.
A message is fresh when its header id is new; repeated ids are counted separately. Reports
the header-timestamp interval and the host first-seen interval of fresh messages (median,
p95, max), skipped ids, and the read call duration. Guide or support the arm by hand; it
goes idle after the support prompt at exit.

IK part (--with-ik or --ik-only): builds the training env, runs the adapter self-check, and
times TrainingIK.solve (state sync + sim.forward + the action's apply_actions + readback)
and training-model FK. --ik-only needs no hardware.

  python diagnose.py                       # hardware only (fp_robot or the clone env)
  python diagnose.py --with-ik             # hardware + IK timing (clone env)
  python diagnose.py --ik-only --device cpu
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from settings import RESULTS_DIR, load_settings, robot_identity_from_calibration, snapshot
from records import environment_record, save_result, to_jsonable
from timing import Deadlines, MonotonicClock, StateTracker, interval_stats


def hardware_timing(cfg: dict, duration_s: float, poll_hz: float) -> dict:
    from manipulation.policy.robot_state import RobotStateReader
    from hardware import hardware_configuration

    identity = robot_identity_from_calibration(cfg)
    clock = MonotonicClock()
    tracker = StateTracker()
    calls, ids = [], []
    with RobotStateReader(identity) as reader:
        driver = reader.driver
        config = hardware_configuration(driver)
        print(f"Polling get_robot_output() at {poll_hz:g} Hz for {duration_s:g} s ...")
        t0 = clock.now()
        sched = Deadlines(1.0 / poll_hz, t0)
        while clock.now() - t0 < duration_s:
            a = clock.now()
            out = driver.get_robot_output()
            b = clock.now()
            hid, ts = int(out.header.id), int(out.header.timestamp)
            tracker.add(hid, ts, b)
            calls.append(b - a)
            ids.append(hid)
            _, due, _ = sched.advance(clock.now())
            clock.sleep_until(due)
        elapsed = clock.now() - t0
    reads = len(calls)
    state = tracker.summary(elapsed)
    return {"identity": identity, "hardware_configuration": config, "poll_hz_requested": poll_hz,
            "duration_s": elapsed, "reads": reads, "poll_hz_achieved": reads / elapsed,
            "skipped_poll_slots": sched.skipped_total, "read_call_s": interval_stats(calls), "state": state,
            "mode": "gravity compensation (external_effort 0); no position commands"}


def ik_timing(cfg: dict, device: str, samples: int) -> dict:
    from fk import TrainingFK
    from ik_adapter import TrainingIK, self_check
    from sim_reference import build_training_env

    env = build_training_env(cfg, device)
    ik = TrainingIK(env)
    check = self_check(ik)
    fk = TrainingFK(cfg["training"]["scene"])
    q0 = fk.ready_arm
    ik.set_target(fk.tcp(q0))
    rng = np.random.default_rng(0)
    qs = q0 + rng.normal(0.0, 0.01, (samples, 6))
    solve, fk_t = [], []
    for i, q in enumerate(qs):
        _, info = ik.solve(q)
        a = time.perf_counter()
        fk.tcp(q)
        fk_t.append(time.perf_counter() - a)
        if i >= 50:
            solve.append(info["compute_s"])
    return {"device": device, "self_check": check, "solve_s": interval_stats(solve), "fk_s": interval_stats(fk_t),
            "note": "solve = write measured q, sim.forward(), ClippedDifferentialIKAction.apply_actions(), read target"}


def recommend(report: dict, rates) -> dict:
    """Largest configured IK rate whose period covers p95 read + p95 IK + FK, with 50% headroom.
    The joint-command send cost cannot be measured read-only; it is assumed equal to the read cost."""
    hw, ik = report.get("hardware"), report.get("ik")
    if not hw or not ik:
        return {"note": "needs both hardware and IK timing"}
    read = hw["read_call_s"]["p95"]
    cost = 2 * read + ik["solve_s"]["p95"] + ik["fk_s"]["p95"]
    ok = [r for r in sorted(rates, reverse=True) if 1.0 / r >= 1.5 * cost]
    fresh = hw["state"].get("fresh_rate_hz")
    return {"estimated_tick_cost_s": cost, "sustainable_configured_rates_hz": ok,
            "recommended_ik_hz": ok[0] if ok else None, "fresh_state_rate_hz": fresh,
            "note": "rates above the fresh state rate reuse state samples on some ticks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--poll-hz", type=float, default=None)
    parser.add_argument("--with-ik", action="store_true")
    parser.add_argument("--ik-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_settings(args.config)
    dc = cfg["diagnostic"]
    report = {"kind": "hardware_control_diagnostic", "settings": snapshot(cfg)}
    if not args.ik_only:
        report["hardware"] = hardware_timing(cfg, args.duration or dc["duration_s"], args.poll_hz or dc["poll_hz"])
    if args.with_ik or args.ik_only:
        report["ik"] = ik_timing(cfg, args.device or cfg["hardware"]["external_ik"]["device"], dc["ik_samples"])
    report["environment"] = environment_record()
    report["recommendation"] = recommend(report, cfg["hardware"]["external_ik"]["ik_hz_list"])

    ms = lambda s: "n/a" if s is None else f"{s * 1e3:.3f} ms"  # noqa: E731
    if "hardware" in report:
        h, st = report["hardware"], report["hardware"]["state"]
        print(f"\nDriver {h['hardware_configuration'].get('driver_version')}, controller "
              f"{h['hardware_configuration'].get('controller_version')}")
        print(f"Reads: {h['reads']} at {h['poll_hz_achieved']:.0f} Hz; call median {ms(h['read_call_s']['median'])}, "
              f"p95 {ms(h['read_call_s']['p95'])}, max {ms(h['read_call_s']['max'])}")
        print(f"Fresh messages: {st['fresh_reads']} ({st.get('fresh_rate_hz', 0):.1f} Hz), repeated reads "
              f"{st['repeated_reads']}, skipped ids {st['skipped_message_ids']}")
        for key in ("header_interval_s", "host_first_seen_interval_s"):
            s = st[key]
            print(f"  {key}: median {ms(s['median'])}, p95 {ms(s['p95'])}, max {ms(s['max'])}")
    if "ik" in report:
        k = report["ik"]
        print(f"IK ({k['device']}): solve median {ms(k['solve_s']['median'])}, p95 {ms(k['solve_s']['p95'])}, "
              f"max {ms(k['solve_s']['max'])}; FK p95 {ms(k['fk_s']['p95'])}; self-check joint-target diff "
              f"{k['self_check']['joint_target_diff_rad']:.2e} rad")
    print(f"Recommendation: {report['recommendation']}")
    path = RESULTS_DIR / "diagnostics" / f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_result(to_jsonable(report), path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
