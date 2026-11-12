"""Single-joint step tests on the real arm. Environment: wxai_mjlab_hw. Run from this directory.

    python run_hardware.py                       # dry run: list the trials and the safety check
    python run_hardware.py --execute             # every pose, joint and test (about 12 min)
    python run_hardware.py --execute --joints 1 2 --poses ready --kinds step

Dry run unless --execute is passed. With --execute the arm moves to each pose and one joint
at a time is commanded by up to 0.12 rad from it; clear the table first. The loop reads the
robot state and sends set_arm_positions (all six joints, non-blocking) at loop_hz, as the
policy's sim IK loop does, and logs every fresh state message. The _once kinds send each new
target a single time and leave the driver's interpolation to finish.

A safety violation stops at the measured joints and ends the session. Ctrl+C or a driver
error puts the arm in gravity compensation. At the end the arm returns to the ready pose,
then goes to gravity compensation and waits for Enter before going idle: support it.

Results: results/hardware_<stamp>.npz (samples of every trial) and .json (trials, config,
driver configuration).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import numpy as np

from plan import (KINDS, RESULTS_DIR, Kinematics, TargetGenerator, build_trials, check_trials, load_config,
                  session_seconds)


class SafetyStop(Exception):
    pass


def sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0.0003:
        time.sleep(remaining - 0.0003)
    while time.monotonic() < deadline:
        pass


class Session:
    def __init__(self, cfg: dict, arm, kin: Kinematics):
        self.cfg, self.arm, self.kin = cfg, arm, kin
        self.period = 1.0 / float(cfg["loop_hz"])
        self.sc = cfg["safety"]
        self.last_id, self.last_fresh = None, time.monotonic()

    def read(self):
        """(arm q, qd, effort, external effort, header id, fresh) from one driver message."""
        with self.arm.lock:
            out = self.arm.driver.get_robot_output()
        a = out.joint.arm
        hid = int(out.header.id)
        fresh = hid != self.last_id
        now = time.monotonic()
        if fresh:
            self.last_id, self.last_fresh = hid, now
        elif now - self.last_fresh > self.sc["max_state_age_s"]:
            raise SafetyStop(f"no new robot state for {now - self.last_fresh:.3f} s")
        return (np.array(a.positions, dtype=np.float64), np.array(a.velocities, dtype=np.float64),
                np.array(a.efforts, dtype=np.float64), np.array(a.external_efforts, dtype=np.float64), hid, fresh)

    def send(self, q, goal_time: float) -> None:
        with self.arm.lock:
            self.arm._check()
            self.arm.driver.set_arm_positions([float(v) for v in q], float(goal_time), False)

    def check(self, q, qd, target) -> None:
        sc = self.sc
        error = float(np.max(np.abs(q - target)))
        if error > sc["max_tracking_error"]:
            raise SafetyStop(f"tracking error {error:.3f} rad")
        speed = float(np.max(np.abs(qd)))
        if speed > sc["max_joint_speed"]:
            raise SafetyStop(f"joint speed {speed:.2f} rad/s")
        z = float(self.kin.tcp(q)[2])
        if z < sc["min_measured_tcp_z"]:
            raise SafetyStop(f"measured TCP z {z:.3f} m")

    def move_to(self, q_goal, move_time: float, settle: float) -> np.ndarray:
        """Joint move (driver trajectory), then hold; returns the measured joints at the end."""
        q_goal = np.asarray(q_goal, dtype=np.float64)
        self.send(q_goal, move_time)
        end = time.monotonic() + move_time + settle
        q = None
        while time.monotonic() < end:
            q, qd, _, _, _, _ = self.read()
            if np.max(np.abs(qd)) > self.sc["max_joint_speed"]:
                raise SafetyStop(f"joint speed {np.max(np.abs(qd)):.2f} rad/s during a pose move")
            time.sleep(0.005)
        return q

    def run_trial(self, trial, base: np.ndarray) -> dict:
        gen = TargetGenerator(trial, base[trial.joint], self.cfg["policy_period_s"])
        target = base.copy()
        rows = {k: [] for k in ("t", "q", "qd", "effort", "external_effort", "target", "header_id")}
        q0 = self.read()[0]
        onset = time.monotonic()
        tick, late, sent = 0, 0, None
        while True:
            now = time.monotonic()
            t = now - onset
            if t >= trial.observe_s:
                break
            q, qd, effort, ext, hid, fresh = self.read()
            target[trial.joint] = gen(t, q[trial.joint])
            self.check(q, qd, target)
            if trial.resend or sent is None or abs(target[trial.joint] - sent) > 1e-12:
                self.send(target, trial.goal_time)
                sent = float(target[trial.joint])
            if fresh:
                for key, value in zip(rows, (t, q, qd, effort, ext, target.copy(), hid)):
                    rows[key].append(value)
            tick += 1
            due = onset + tick * self.period
            if time.monotonic() > due + self.period:        # more than a period late: no catch-up
                late += 1
                tick = int((time.monotonic() - onset) / self.period) + 1
                due = onset + tick * self.period
            sleep_until(due)
        return {**{k: np.array(v) for k, v in rows.items()}, "q_start": q0, "base": base.copy(),
                "late_ticks": late}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--execute", action="store_true", help="command the arm (default: dry run)")
    parser.add_argument("--poses", nargs="+", default=None)
    parser.add_argument("--joints", nargs="+", type=int, default=None)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    kin = Kinematics(cfg["fk_model"], cfg["site"])
    trials = build_trials(cfg, args.poses, args.joints, args.kinds)
    problems = check_trials(cfg, trials, kin)
    print(f"{len(trials)} trials, about {session_seconds(cfg, trials) / 60:.1f} min plus pose moves")
    for pose in dict.fromkeys(t.pose for t in trials):
        q = np.asarray(cfg["poses"][pose], dtype=np.float64)
        print(f"  pose {pose}: joints {q.tolist()}, TCP {np.round(kin.tcp(q), 3).tolist()}")
    if problems:
        print("UNSAFE, nothing will run:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("Safety check passed: every target's TCP is inside the workspace box and the joint limits.")
    if not args.execute:
        for trial in trials[:6]:
            print("  ", trial.name)
        print("   ...\nDry run. Pass --execute to command the arm.")
        return

    import yaml

    from common.hardware import robot_identity
    from manipulation.policy.arm import PolicyArm
    sys.path.insert(0, str(RESULTS_DIR.parents[1] / "hardware_control"))
    from hardware import hardware_configuration

    identity = robot_identity(yaml.safe_load(open(cfg["calibration_config"])))
    answer = input(f"\nThe arm at {identity['ip']} will move to the poses above and step single joints.\n"
                   "Clear the table and stay at the arm. Type 'yes' to start: ")
    if answer.strip().lower() != "yes":
        print("Not started.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results, done, stop_reason = {}, [], "completed"
    with PolicyArm(identity, 0.002) as arm:
        session = Session(cfg, arm, kin)
        configuration = hardware_configuration(arm.driver)
        limits = arm.driver.get_joint_limits()[:-1]
        problems = check_trials(cfg, trials, kin, [l.position_min for l in limits], [l.position_max for l in limits])
        if problems:
            print("UNSAFE against the driver's joint limits:\n  " + "\n  ".join(problems))
            return
        ready = np.asarray(cfg["poses"]["ready"], dtype=np.float64)
        try:
            arm.move_to_ready(ready, cfg["gripper"], 4.0)
            session.move_to(ready, 4.0, 1.0)
            current_pose = None
            for index, trial in enumerate(trials):
                base = np.asarray(cfg["poses"][trial.pose], dtype=np.float64)
                if trial.pose != current_pose:
                    q_now = session.read()[0]
                    move = max(cfg["pose_move_min_s"],
                               float(np.max(np.abs(base - q_now))) / cfg["pose_move_speed_rad_s"])
                    print(f"Moving to pose {trial.pose} over {move:.1f} s")
                    session.move_to(base, move, 1.0)
                    current_pose = trial.pose
                else:
                    session.move_to(base, cfg["return_time_s"], cfg["settle_s"])
                data = session.run_trial(trial, base)
                moved = data["q"][-1, trial.joint] - data["q_start"][trial.joint]
                print(f"[{index + 1}/{len(trials)}] {trial.name}: moved {moved * 1e3:+.1f} mrad, "
                      f"{len(data['t'])} samples, {data['late_ticks']} late ticks")
                for key, value in data.items():
                    results[f"{trial.name}/{key}"] = value
                done.append(trial.to_json())
        except SafetyStop as stop:
            stop_reason = f"safety stop: {stop}"
            print(f"\n!!! {stop_reason}. Holding at the measured joints.")
            arm.hold_at(arm.read().arm_q, cfg["safety"]["stop_goal_time_s"])
            time.sleep(1.0)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            arm.kill("Ctrl+C")
        except Exception as error:
            stop_reason = f"error: {error!r}"
            arm.kill(stop_reason)
            raise
        finally:
            if done:
                from common.io import write_json

                RESULTS_DIR.mkdir(exist_ok=True)
                np.savez_compressed(RESULTS_DIR / f"hardware_{stamp}.npz", **results)
                write_json(RESULTS_DIR / f"hardware_{stamp}.json",
                           {"kind": "joint_step_session", "source": "hardware", "stop_reason": stop_reason,
                            "config": cfg, "trials": done, "robot_identity": identity,
                            "hardware_configuration": configuration,
                            "units": {"t": "s since trial onset (host monotonic clock)", "q": "rad",
                                      "qd": "rad/s", "effort": "Nm", "target": "rad"}})
                print(f"Saved {len(done)} trials to {RESULTS_DIR / f'hardware_{stamp}.npz'} ({stop_reason})")
        if stop_reason == "completed":
            print("Returning to the ready pose.")
            try:
                session.move_to(ready, 4.0, 0.5)
            except SafetyStop as stop:
                print(f"!!! {stop} on the way back")


if __name__ == "__main__":
    main()
