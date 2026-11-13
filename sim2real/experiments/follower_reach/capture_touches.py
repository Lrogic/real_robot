"""Touch a fixed height at several reaches with an arm in gravity compensation, and fit height
against reach.

The manipulator reads the table about 4 mm higher for every 10 cm of reach
(docs/block_height_offset.md, sections 6 and 8). This checks whether another arm (by default the
follower, which carries the D405) shows the same pattern. Touch the top of the upright jenga
block (or anything else of fixed height) at different reaches. If the arm were exact, every touch
would read the same height. A trend in reach is the same error as the manipulator's.

The arm stays in gravity compensation (zero external effort); nothing is position-commanded.
A live line shows the joints, the tool position, reach and how far the tool axis is from
pointing straight down. Hold the arm still on the touch and press Enter: the script averages
--average-s of readings and saves them. Type a label before Enter to store it with the sample.
q (or Ctrl-C) finishes; the arm then goes idle, so SUPPORT IT.

Keep the touching part and the wrist orientation the same for every touch (for example the
carriage face, tool pointing straight down). The tool frame is the gripper's fingertip frame
of the configured end effector, so without fingers it sits in the air at a fixed offset from
the flange. A fixed offset only shifts every height by the same amount, but it changes with
wrist orientation, so the flange position is saved as well.

MOVING THE FOLLOWER MOVES THE CAMERA. Afterwards, re-park it before tracking
(python -m calibration.reset_follower calibration/data/sessions/20261009_combined from vision/,
or run.sh --setup-track ... --camera parked), or T_base_camera no longer holds.

--camera opens a live window of the D405 colour image (stop the tracking pipeline first) with
the tool position overlaid and any ChArUco corners marked, and saves the frame with each
capture. In the window, Space or Enter captures without a label and q finishes.

Launch from the fp_robot environment:

  /opt/conda-envs/fp_robot/bin/python real_robot/sim2real/experiments/follower_reach/capture_touches.py
  ... --camera           # live camera window, frame saved with each capture
  ... --arm robot        # the manipulator instead (the calibration config's robot section)
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROBOT))
sys.path.insert(0, str(REAL_ROBOT / "vision"))
sys.path.insert(0, str(EXPERIMENTS))

import trossen_arm  # noqa: E402

from calibration.board import DEFAULT_CONFIG as CALIBRATION_CONFIG  # noqa: E402
from calibration.board import load_config as load_calibration_config  # noqa: E402
from common.hardware import end_effector_record, hold_arm_with_gravity_compensation, release_to_idle  # noqa: E402
from common.io import write_json  # noqa: E402
from common.poses import rotvec_to_matrix  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STILL_MM = 0.5          # warn if the tool moved more than this during the averaging window


def identity(calibration: dict, section: str) -> dict:
    cfg = calibration.get(section) or {}
    missing = [k for k in ("ip", "model", "end_effector") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"{section}.{', '.join(missing)} missing in the calibration config")
    return {k: cfg[k] for k in ("ip", "model", "end_effector")}


def read(driver, n_arm: int) -> dict:
    out = driver.get_robot_output()
    q = np.array(out.joint.arm.positions, dtype=float)
    cart = np.array(out.cartesian.positions, dtype=float)
    if len(q) != n_arm or len(cart) != 6:
        raise RuntimeError(f"unexpected robot output: {len(q)} joints, {len(cart)} Cartesian values")
    return {"t": time.time(), "header_id": int(out.header.id), "q": q, "tool": cart}


def flange_position(tool6, t_flange_tool) -> np.ndarray:
    """Flange origin in the base frame from the tool pose and the flange-to-tool transform."""
    R_tool = rotvec_to_matrix(tool6[3:])
    R_ft = rotvec_to_matrix(t_flange_tool[3:])
    R_flange = R_tool @ R_ft.T
    return tool6[:3] - R_flange @ np.asarray(t_flange_tool[:3])


def tool_down_deg(tool6) -> float:
    """Angle between the tool x axis (through the fingers) and straight down."""
    x_axis = rotvec_to_matrix(tool6[3:])[:, 0]
    return float(np.degrees(np.arccos(np.clip(-x_axis[2], -1.0, 1.0))))


def status(r: dict) -> str:
    p = r["tool"][:3] * 1000
    reach = np.hypot(p[0], p[1])
    az = np.degrees(np.arctan2(p[1], p[0]))
    q = " ".join(f"{v:6.1f}" for v in np.degrees(r["q"]))
    return (f"\rq(deg) {q} | tool x {p[0]:6.1f} y {p[1]:6.1f} z {p[2]:6.1f} mm | reach {reach:5.0f} mm "
            f"az {az:4.0f} deg | tool {tool_down_deg(r['tool']):4.1f} deg from down   ")


class LiveCamera:
    """D405 colour stream in an OpenCV window, with the tool position and ChArUco corners drawn."""

    WINDOW = "follower camera (Space/Enter: capture, q: finish)"

    def __init__(self, width=640, height=480, fps=30):
        import cv2
        import pyrealsense2 as rs
        from calibration.board import BoardConfig, load_config, make_board, make_detector
        from calibration.detection import detect_raw

        self.cv2, self.detect_raw = cv2, detect_raw
        self.detector = make_detector(make_board(BoardConfig.from_dict(load_config()["board"])))
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)
        self.frame = None
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def update(self, reading: dict) -> str | None:
        """Show the newest frame. Returns "" for a capture key, "q" to finish, else None."""
        cv2 = self.cv2
        frames = self.pipe.poll_for_frames()
        if frames:
            color = frames.get_color_frame()
            if color:
                self.frame = np.asanyarray(color.get_data()).copy()
        if self.frame is None:
            return None
        view = self.frame.copy()
        corners, ids, _, _ = self.detect_raw(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), self.detector)
        for c in np.asarray(corners).reshape(-1, 2):
            cv2.circle(view, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
        p = reading["tool"][:3] * 1000
        lines = [f"tool x {p[0]:.0f} y {p[1]:.0f} z {p[2]:.0f} mm",
                 f"reach {np.hypot(p[0], p[1]):.0f} mm  az {np.degrees(np.arctan2(p[1], p[0])):.0f} deg",
                 f"board corners: {len(ids)}"]
        for i, text in enumerate(lines):
            cv2.putText(view, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(view, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10, 32):
            return ""
        if key == ord("q"):
            return "q"
        return None

    def save(self, path: Path) -> str | None:
        if self.frame is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cv2.imwrite(str(path), self.frame)
        return str(path)

    def close(self):
        try:
            self.pipe.stop()
        finally:
            self.cv2.destroyAllWindows()


def wait_for_line(driver, n_arm: int, camera: LiveCamera | None = None, period_s: float = 0.1) -> str:
    """Show the live status (and camera window) until a line is typed or a capture key is hit."""
    last_status = 0.0
    while True:
        reading = read(driver, n_arm)
        if time.monotonic() - last_status >= period_s:
            print(status(reading), end="", flush=True)
            last_status = time.monotonic()
        if camera is not None:
            key = camera.update(reading)
            if key is not None:
                return key
        ready, _, _ = select.select([sys.stdin], [], [], 0.01 if camera is not None else period_s)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                raise EOFError
            return line.strip()


def capture(driver, n_arm: int, average_s: float, t_flange_tool, label: str, index: int) -> dict:
    rows = []
    end = time.monotonic() + average_s
    while time.monotonic() < end:
        rows.append(read(driver, n_arm))
        time.sleep(0.01)
    q = np.array([r["q"] for r in rows])
    tool = np.array([r["tool"] for r in rows])
    pos = tool[:, :3]
    mean_tool = tool.mean(axis=0)
    mean_tool[3:] = tool[len(tool) // 2, 3:]          # rotvecs do not average; take the middle one
    p = mean_tool[:3]
    moved_mm = float(np.ptp(pos, axis=0).max() * 1000)
    return {
        "sample": index,
        "label": label,
        "host_time_iso": datetime.fromtimestamp(rows[0]["t"]).astimezone().isoformat(),
        "readings": len(rows),
        "arm_joint_positions_rad": q.mean(axis=0).tolist(),
        "tool_m_rotvec_rad": mean_tool.tolist(),
        "tool_z_std_mm": float(pos[:, 2].std() * 1000),
        "moved_mm": moved_mm,
        "flange_m": flange_position(mean_tool, t_flange_tool).tolist(),
        "reach_m": float(np.hypot(p[0], p[1])),
        "azimuth_deg": float(np.degrees(np.arctan2(p[1], p[0]))),
        "tool_from_down_deg": tool_down_deg(mean_tool),
    }


def summarize(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    print("\n  #  label                 reach mm   az deg   tool z mm   flange z mm   tool from down")
    for s in samples:
        print(f"{s['sample']:3d}  {s['label'][:20]:<20} {s['reach_m'] * 1000:8.0f} {s['azimuth_deg']:8.0f} "
              f"{s['tool_m_rotvec_rad'][2] * 1000:11.1f} {s['flange_m'][2] * 1000:13.1f} {s['tool_from_down_deg']:12.1f}")
    if len(samples) < 3:
        print("Need 3 or more touches for a fit.")
        return None
    r = np.array([s["reach_m"] for s in samples])
    fits = {}
    for key, z in (("tool", [s["tool_m_rotvec_rad"][2] for s in samples]), ("flange", [s["flange_m"][2] for s in samples])):
        z = np.array(z) * 1000
        slope, intercept = np.polyfit(r, z, 1)
        res = z - (slope * r + intercept)
        fits[key] = {"slope_mm_per_m": float(slope), "intercept_mm": float(intercept),
                     "residual_rms_mm": float(np.sqrt((res ** 2).mean())), "spread_mm": float(np.ptp(z))}
        print(f"{key:>6} z vs reach: {slope:+.1f} mm per metre, intercept {intercept:.1f} mm, "
              f"residual rms {fits[key]['residual_rms_mm']:.1f} mm, spread {np.ptp(z):.1f} mm")
    print("The manipulator gave about +32 to +46 mm per metre (docs/block_height_offset.md).")
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("follower", "robot"), default="follower",
                        help="calibration config section to connect to (robot = the manipulator)")
    parser.add_argument("--calibration-config", type=Path, default=CALIBRATION_CONFIG)
    parser.add_argument("--average-s", type=float, default=0.5, help="readings averaged per touch, s")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--camera", action="store_true", help="live D405 window; saves the frame with each capture")
    args = parser.parse_args()

    ident = identity(load_calibration_config(args.calibration_config), args.arm)
    driver = trossen_arm.TrossenArmDriver()
    output = args.output or RESULTS_DIR / f"touches_{args.arm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = None
    camera = None
    try:
        driver.configure(getattr(trossen_arm.Model, ident["model"]),
                         getattr(trossen_arm.StandardEndEffector, ident["end_effector"]), ident["ip"], False)
        ee = driver.get_end_effector()
        t_flange_tool = [float(v) for v in ee.t_flange_tool]
        print(f"Connected to the {args.arm} ({ident['model']}, {ident['end_effector']}) at {ident['ip']}.")
        if input("SUPPORT THE ARM. Enter puts it in gravity compensation (q to quit): ").strip().lower() == "q":
            return
        n_arm = hold_arm_with_gravity_compensation(driver)
        payload = {
            "experiment": "follower_reach",
            "arm": args.arm,
            "configuration": {**ident, "driver_version": driver.get_driver_version(),
                              "controller_version": driver.get_controller_version(),
                              "arm_control": "external_effort 0 (gravity compensation); no position commands",
                              "end_effector_properties": end_effector_record(ee),
                              "average_s": args.average_s},
            "units": "positions in m in the arm's own base frame, rotations as rotvec rad, joints rad",
            "samples": [],
        }
        if args.camera:
            camera = LiveCamera()
        print(f"Gravity compensation on. Saving to {output}\n"
              "Hold still on a touch and press Enter (type a label first if you like); q finishes.")
        while True:
            line = wait_for_line(driver, n_arm, camera)
            if line.lower() in ("q", "quit"):
                break
            s = capture(driver, n_arm, args.average_s, t_flange_tool, line, len(payload["samples"]) + 1)
            if camera is not None:
                s["image"] = camera.save(output.with_suffix("") / f"{s['sample']:02d}.png")
            payload["samples"].append(s)
            write_json(output, payload)
            warn = f"  MOVED {s['moved_mm']:.1f} mm while averaging, redo it?" if s["moved_mm"] > STILL_MM else ""
            print(f"\nSaved #{s['sample']} {s['label']!r}: reach {s['reach_m'] * 1000:.0f} mm, "
                  f"tool z {s['tool_m_rotvec_rad'][2] * 1000:.1f} mm, flange z {s['flange_m'][2] * 1000:.1f} mm, "
                  f"tool {s['tool_from_down_deg']:.1f} deg from down{warn}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        if camera is not None:
            camera.close()
        if payload is not None:
            payload["fit"] = summarize(payload["samples"])
            write_json(output, payload)
            print(f"{len(payload['samples'])} touches in {output}")
        release_to_idle(driver)


if __name__ == "__main__":
    main()
