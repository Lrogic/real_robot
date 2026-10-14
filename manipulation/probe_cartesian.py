"""Three small Cartesian tests to find why the tool drifted away from its command.

Run from real_robot/vision in the fp_robot env. The arm starts in gravity compensation; guide it
to a ready pose when asked (it refuses to lock if a joint is near a limit):

    PYTHONPATH=.. python -m manipulation.probe_cartesian [--step-m 0.02] [--calibration PATH]

    1. hold      re-send the current pose at control_hz for --hold-s (the controller's way of sending)
    2. single    one 2 cm downward move with a single goal (like the Trossen cartesian_position demo), then back
    3. streamed  the same move as control_hz small non-blocking goals (like the controller), then back

Each test waits for Enter. Ctrl+C at any time = kill (gravity compensation). A test also kills
the arm if the tool leaves the commanded segment by more than --abort-m. Results are printed and
saved to --out (JSON) for comparison.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calibration.transforms import make_T, rotation_between_deg

from .arm import ManipulatorArm
from .grasp import ManipulationConfig, joint_limit_margins, step_toward
from .session import find_calibration, robot_config

SETTLE_S = 0.5
SAMPLE_HZ = 50.0


def distance_to_segment(p, a, b) -> float:
    ab = b - a
    s = 0.0 if not ab.any() else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


class Probe:
    def __init__(self, arm: ManipulatorArm, cfg: ManipulationConfig, abort_m: float):
        self.arm, self.cfg, self.abort_m = arm, cfg, abort_m
        self.dt = 1.0 / cfg.control_hz

    def sample(self, samples, start_T, goal_T, commanded_T):
        state = self.arm.measured_state()
        T = state.T_base_ee
        deviation = distance_to_segment(T[:3, 3], start_T[:3, 3], goal_T[:3, 3])
        samples.append({"t": time.monotonic(), "T_measured": T.tolist(), "T_commanded": commanded_T.tolist(),
                        "joints": state.arm_positions.tolist(), "deviation_m": deviation})
        if deviation > self.abort_m:
            self.arm.kill(f"tool {deviation * 100:.1f} cm off the commanded segment")
            raise SystemExit("Aborted: the tool left the commanded segment")

    def poll(self, samples, seconds, start_T, goal_T, commanded_T):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.sample(samples, start_T, goal_T, commanded_T)
            time.sleep(1.0 / SAMPLE_HZ)

    def hold(self, start_T, seconds):
        samples = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tick = time.monotonic()
            self.arm.send_tool_pose(start_T, self.dt)
            self.sample(samples, start_T, start_T, start_T)
            time.sleep(max(0.0, self.dt - (time.monotonic() - tick)))
        self.poll(samples, SETTLE_S, start_T, start_T, start_T)
        return samples

    def single(self, start_T, goal_T):
        """One goal; its duration keeps the average speed at the configured limit (>= 2 s)."""
        dist = np.linalg.norm(goal_T[:3, 3] - start_T[:3, 3])
        duration = max(2.0, dist / self.cfg.max_lin_speed_m_s)
        samples = []
        self.arm.send_tool_pose(goal_T, duration)
        self.poll(samples, duration + SETTLE_S, start_T, goal_T, goal_T)
        return samples

    def streamed(self, start_T, goal_T):
        samples, command, reached = [], start_T, False
        while not reached:
            tick = time.monotonic()
            command, reached = step_toward(command, goal_T, self.dt, self.cfg)
            self.arm.send_tool_pose(command, self.dt)
            self.sample(samples, start_T, goal_T, command)
            time.sleep(max(0.0, self.dt - (time.monotonic() - tick)))
        self.poll(samples, SETTLE_S, start_T, goal_T, goal_T)
        return samples


def summarize(name, samples, start_T, goal_T, limits) -> dict:
    T_end = np.asarray(samples[-1]["T_measured"])
    joints = np.array([s["joints"] for s in samples])
    margins = np.degrees(np.min([joint_limit_margins(j, limits) for j in joints], axis=0))
    result = {
        "test": name,
        "goal_minus_start_cm": ((goal_T[:3, 3] - start_T[:3, 3]) * 100).round(2).tolist(),
        "measured_minus_start_cm": ((T_end[:3, 3] - start_T[:3, 3]) * 100).round(2).tolist(),
        "final_error_cm": ((T_end[:3, 3] - goal_T[:3, 3]) * 100).round(2).tolist(),
        "final_rotation_error_deg": round(rotation_between_deg(T_end[:3, :3], goal_T[:3, :3]), 2),
        "max_deviation_from_segment_cm": round(max(s["deviation_m"] for s in samples) * 100, 2),
        "joint_change_deg": np.degrees(joints[-1] - joints[0]).round(2).tolist(),
        "min_joint_limit_margin_deg": margins.round(1).tolist(),
    }
    print(f"\n[{name}]")
    for key, value in result.items():
        if key != "test":
            print(f"  {key:32s} {value}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="T_base_camera.json, only for the robot ip/model/end effector (default: newest)")
    parser.add_argument("--manipulation-config", type=Path, default=None)
    parser.add_argument("--step-m", type=float, default=0.02, help="downward move for tests 2 and 3")
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--abort-m", type=float, default=0.02,
                        help="kill if the tool is this far from the commanded segment")
    parser.add_argument("--out", type=Path, default=Path(f"probe_cartesian_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    args = parser.parse_args()
    if not 0 < args.step_m <= 0.05:
        parser.error("--step-m must be in (0, 0.05]")

    cfg = ManipulationConfig.load(args.manipulation_config)
    cal = json.loads(find_calibration(args.calibration).read_text())
    results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: str(v) for k, v in vars(args).items()},
               "config": cfg.__dict__, "tests": [], "samples": {}}

    with ManipulatorArm(robot_config(cal, cfg), cfg.gripper_open_margin_m, cfg.gripper_open_time_s,
                        cfg.gripper_close_effort) as arm:
        results["arm"] = arm.info
        print("Ctrl+C at any time = KILL (gravity compensation).")
        try:
            time.sleep(cfg.gripper_open_time_s)
            while True:
                input("\nGuide the arm to a ready pose (elbow bent, gripper level, clear space), "
                      "hold it still, then press Enter to lock it ...")
                refusal = arm.engage(cfg.min_start_joint_margin_deg)
                if refusal is None:
                    break
                print(refusal)
            probe = Probe(arm, cfg, args.abort_m)
            start_T = arm.measured_state().T_base_ee
            down_T = make_T(start_T[:3, :3], start_T[:3, 3] - [0.0, 0.0, args.step_m])
            print(f"Start tool position {np.round(start_T[:3, 3], 4)} m, joints "
                  f"{np.degrees(arm.measured_state().arm_positions).round(1)} deg")
            if down_T[2, 3] < cfg.min_tool_z_m:
                raise SystemExit(f"Moving down {args.step_m} m would cross the floor min_tool_z_m "
                                 f"{cfg.min_tool_z_m}; park the arm higher")

            tests = [
                ("1_hold", lambda: probe.hold(start_T, args.hold_s), start_T, start_T),
                ("2_single_down", lambda: probe.single(start_T, down_T), start_T, down_T),
                ("2_single_back", lambda: probe.single(down_T, start_T), down_T, start_T),
                ("3_streamed_down", lambda: probe.streamed(start_T, down_T), start_T, down_T),
                ("3_streamed_back", lambda: probe.streamed(down_T, start_T), down_T, start_T),
            ]
            for name, run, a, b in tests:
                if not name.endswith("_back"):
                    input(f"\nPress Enter to run {name} ...")
                begin = arm.measured_state().T_base_ee
                if np.linalg.norm(begin[:3, 3] - a[:3, 3]) > args.abort_m:
                    raise SystemExit(f"{name}: arm is not at the expected start; stopping")
                samples = run()
                results["tests"].append(summarize(name, samples, a, b, arm.joint_limits))
                results["samples"][name] = samples
        finally:
            args.out.write_text(json.dumps(results, indent=1))
            print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
