"""Guided ChArUco sample collection with the manipulator in gravity compensation.

    python -m calibration.collect --follower PATH|unrecorded
                                  --grav-comp normal|board-adjusted
                                  [--session data/sessions/NAME] [--resume]

The follower (camera arm) stays parked. Pass a profile from record_follower, or
unrecorded if that pose was not saved. The operator guides the manipulator by hand,
holds it still with the board settled, and presses:
    c  capture a sample (robot must be still; a fresh image is taken after the keypress)
    d  delete (reject) the last sample
    g  toggle the guidance panel
    q  quit (you will be asked to support the arm before it goes limp)
Requires a passing data/board_verification.json (see verify_board).
"""

import argparse
import json
import platform
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .board import DEFAULT_DATA_DIR, BoardConfig, board_metadata, load_config, make_board, make_detector, \
    same_board
from .detection import detect, draw_detection, quiet_opencv
from .diversity import analyze, nearest_duplicate, summary_lines
from .record_follower import load_follower_choice
from .robot import joint_motion
from .session import Session, load_sample, read_json, sample_E_C

WINDOW = "hand-eye collection"
PANEL_W = 520
GREEN, RED, YELLOW, WHITE, GREY = (80, 220, 80), (60, 60, 255), (0, 220, 255), (235, 235, 235), (160, 160, 160)

GUIDANCE = [
    "Collect ~20-30 distinct poses:",
    "- vary position and distance in the workspace",
    "- tilt 15-45 deg about at least TWO axes",
    "- include some in-plane rotations",
    "- sharp views, board covering a good area",
    "Hold still, let the board settle, then 'c'.",
    "Check cardboard is flat and not slipping.",
    "If the mount changes: restart the session.",
]


def same_follower(saved: dict, chosen: dict) -> bool:
    if saved.get("status") != chosen["status"]:
        return False
    if chosen["status"] != "recorded":
        return True
    a = np.asarray(saved["joint_positions_rad"], dtype=np.float64)
    b = np.asarray(chosen["joint_positions_rad"], dtype=np.float64)
    return a.shape == b.shape and np.allclose(a, b)


def check_verification(path: Path, board_meta: dict, skip: bool) -> dict:
    if skip:
        print("WARNING: board verification skipped")
        return {"skipped": True}
    if not path.exists():
        raise SystemExit(f"{path} not found: run verify_board first (or --skip-verification)")
    v = read_json(path)
    if not v.get("passed"):
        raise SystemExit(f"{path} did not pass; fix the board config or reprint")
    if not same_board(v["board"], board_meta):
        raise SystemExit(f"{path} was made for a different board definition; rerun verify_board")
    return v


class Collector:
    def __init__(self, config, session, stream, robot, board_cfg, camera_arm_joints):
        self.config, self.session, self.stream, self.robot = config, session, stream, robot
        self.cfg = board_cfg
        self.board = make_board(board_cfg)
        self.detector = make_detector(self.board)
        self.board_meta = board_metadata(board_cfg)
        self.camera_arm_joints = camera_arm_joints
        self.q, self.cap, self.div = config["quality"], config["capture"], config["diversity"]
        self.message, self.message_color = "ready", WHITE
        self.show_guidance = True
        self.reload()

    def reload(self):
        self.samples = [load_sample(d) for d in self.session.sample_dirs()]
        pairs = [sample_E_C(s) for s in self.samples]
        self.Es = [p[0] for p in pairs]
        self.Cs = [p[1] for p in pairs]
        self.div_result = analyze(self.Es, self.Cs, [s["detection"]["metrics"] for s in self.samples], self.div,
                                  ids=[s["id"] for s in self.samples])

    def set_message(self, text, color):
        self.message, self.message_color = text, color
        print(text)

    def capture(self, t_key: float):
        cap = self.cap
        still, states, still_stats = self.robot.wait_still(cap["still_window_s"], cap["still_joint_vel_rad_s"],
                                                           cap["still_joint_range_rad"])
        if not still:
            return self.set_message(f"not captured: arm moving (drift {still_stats['max_joint_drift_rad_s']:.4f} "
                                    f"rad/s, range {still_stats['max_joint_range_rad']:.4f} rad)", RED)
        before = states[-1]
        not_before = max(t_key + cap["settle_s"], before.host_time_s)
        frame = self.stream.grab_fresh(not_before, cap["fresh_frames"], cap["fresh_timeout_s"])
        after = self.robot.read_state()
        delta = float(np.abs(after.arm_positions - before.arm_positions).max())
        if delta > cap["max_joint_delta_rad"]:
            return self.set_message(f"not captured: arm moved during the image (delta {delta:.4f} rad)", RED)
        det = detect(frame.image, self.detector, self.board, self.cfg, self.stream.K, self.stream.dist, self.q)
        if not det.accepted:
            return self.set_message("not captured: " + "; ".join(det.reasons), RED)

        index = self.session.next_index()
        sample = {
            "id": index,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "keypress_time_s": t_key,
            "image_file": "image.png",
            "image_shape": list(frame.image.shape),
            "camera_frame": frame.to_json(),
            "intrinsics": {"K": self.stream.K.tolist(), "dist": self.stream.dist.tolist(),
                           "distortion_model": self.stream.info["intrinsics"]["distortion_model"],
                           "stream": self.stream.info["stream"]},
            "detection": det.to_json(),
            "robot": {"state_before_image": before.to_json(), "state_after_image": after.to_json(),
                      "stillness": still_stats, "joint_delta_rad": delta,
                      "T_base_ee": after.T_base_ee.tolist(),
                      "T_base_ee_from": "state_after_image (measured, FK)"},
            "camera_arm_joint_positions_rad": self.camera_arm_joints,
            "board": self.board_meta,
        }
        self.session.save_sample(index, frame.image, sample)
        self.reload()
        dup = nearest_duplicate(self.Es, self.div)
        if dup is not None:
            self.set_message(f"saved #{index}, but it nearly duplicates #{self.samples[dup]['id']} "
                             "('d' to delete)", YELLOW)
        else:
            m = det.metrics
            self.set_message(f"saved #{index}: {m['num_corners']} corners, {m['reproj_rms_px']:.2f} px, "
                             f"tilt {m['tilt_deg']:.0f} deg", GREEN)

    def delete_last(self):
        dirs = self.session.sample_dirs()
        if not dirs:
            return self.set_message("no samples to delete", YELLOW)
        moved = self.session.reject(dirs[-1])
        self.reload()
        self.set_message(f"moved #{moved.name} to rejected/", YELLOW)

    def panel(self, height, det, motion):
        p = np.full((height, PANEL_W, 3), 30, np.uint8)
        y = 28

        def line(text, color=WHITE, scale=0.55, gap=24):
            nonlocal y
            cv2.putText(p, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += gap

        lo, hi = self.div["target_samples"]
        line(f"samples {len(self.samples)}  (target {lo}-{hi})", WHITE, 0.7, 30)
        still = (motion["max_joint_range_rad"] < self.cap["still_joint_range_rad"]
                 and motion["max_joint_drift_rad_s"] < self.cap["still_joint_vel_rad_s"])
        line(f"arm {'still' if still else 'MOVING'}  range {motion['max_joint_range_rad'] * 1000:.1f} mrad  "
             f"drift {motion['max_joint_drift_rad_s'] * 1000:.1f} mrad/s", GREEN if still else YELLOW)
        m = det.metrics
        line(f"markers {m.get('num_markers', 0)}/{self.cfg.num_markers}  corners {m.get('num_corners', 0)}/"
             f"{self.cfg.num_corners}  rows {m.get('corner_rows', 0)} cols {m.get('corner_cols', 0)}")
        if "reproj_rms_px" in m:
            line(f"reproj {m['reproj_rms_px']:.2f}px  ambig {m['ambiguity_ratio']:.2f}  spread {m['spread_ratio']:.2f}")
            line(f"area {m['hull_fraction'] * 100:.1f}%  sharp {m['sharpness']:.0f}  dist {m['distance_m']:.2f}m")
            line(f"tilt {m['tilt_deg']:.0f} (X {m['tilt_about_x_deg']:+.0f}, Y {m['tilt_about_y_deg']:+.0f})  "
                 f"in-plane {m['inplane_deg']:+.0f} deg")
        line("DETECTION OK" if det.accepted else "REJECT: " + (det.reasons[0] if det.reasons else ""),
             GREEN if det.accepted else RED)
        for r in det.reasons[1:3]:
            line("        " + r, RED)
        y += 6
        line(self.message[:62], self.message_color)
        if len(self.message) > 62:
            line(self.message[62:124], self.message_color)
        y += 6
        line("diversity", GREY)
        for s in summary_lines(self.div_result)[1:]:
            line(s, WHITE, 0.5, 22)
        for w in self.div_result["warnings"]:
            for k in range(0, len(w), 60):
                line(("! " if k == 0 else "  ") + w[k:k + 60], YELLOW, 0.48, 20)
        if self.show_guidance:
            y += 6
            for g in GUIDANCE:
                line(g, GREY, 0.48, 20)
        p[height - 36:] = 30
        y = height - 14
        line("c capture  d delete last  g guidance  q quit", WHITE, 0.5)
        return p

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        recent = deque()
        while True:
            frame = self.stream.read()
            det = detect(frame.image, self.detector, self.board, self.cfg, self.stream.K, self.stream.dist, self.q)
            recent.append(self.robot.read_state())
            while recent[-1].host_time_s - recent[0].host_time_s > self.cap["still_window_s"]:
                recent.popleft()
            motion = joint_motion(list(recent))
            vis = draw_detection(frame.image, det, self.stream.K, self.stream.dist)
            cv2.imshow(WINDOW, np.hstack([vis, self.panel(vis.shape[0], det, motion)]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                t_key = time.time()
                self.set_message("capturing: hold still...", YELLOW)
                cv2.imshow(WINDOW, np.hstack([vis, self.panel(vis.shape[0], det, motion)]))
                cv2.waitKey(1)
                self.capture(t_key)
            elif key == ord("d"):
                self.delete_last()
            elif key == ord("g"):
                self.show_guidance = not self.show_guidance
            elif key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--session", type=Path, default=None,
                   help="session directory (default data/sessions/<timestamp>)")
    p.add_argument("--resume", action="store_true", help="add samples to an existing session")
    p.add_argument("--follower", required=True,
                   help='follower profile JSON from record_follower, or "unrecorded"')
    p.add_argument("--grav-comp", required=True, choices=["normal", "board-adjusted"],
                   help="normal: stock gripper model; board-adjusted: include robot.board_payload")
    p.add_argument("--verification", type=Path, default=DEFAULT_DATA_DIR / "board_verification.json")
    p.add_argument("--skip-verification", action="store_true")
    args = p.parse_args()
    quiet_opencv()

    from .camera import ColorStream
    from .robot import GravityCompManipulator

    config = load_config(args.config)
    board_cfg = BoardConfig.from_dict(config["board"])
    meta = board_metadata(board_cfg)
    verification = check_verification(args.verification, meta, args.skip_verification)
    follower = load_follower_choice(args.follower)
    cam_arm = follower["joint_positions_rad"]
    if follower["status"] == "unrecorded":
        print("NOTE: follower pose unrecorded; the result is tied to wherever that arm was parked")
    else:
        print(f"Follower profile {follower['profile_path']}")
        print(f"  joints rad: {cam_arm}")
    root = args.session or DEFAULT_DATA_DIR / "sessions" / time.strftime("%Y%m%d_%H%M%S")

    board_adjusted = args.grav_comp == "board-adjusted"
    with ColorStream(config["camera"]) as stream, \
            GravityCompManipulator(config["robot"], board_adjusted=board_adjusted) as robot:
        input("Press Enter to close the gripper...")
        robot.close_gripper()
        session_meta = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "convention": "column vectors; T_destination_source; metres. X = T_base_camera, "
                          "C_i = T_camera_board, E_i = T_base_ee, Y = T_ee_board; X @ C_i = E_i @ Y",
            "board": meta,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "python": sys.version.split()[0],
            "host": platform.node(),
            "camera": stream.info,
            "robot": robot.info,
            "follower": follower,
            "camera_arm": {"joint_positions_rad": cam_arm, "connected": False,
                           "status": follower["status"], "profile_path": follower["profile_path"],
                           "note": "parked for the whole session; result valid only at this pose"},
            "capture": config["capture"], "quality": config["quality"], "diversity": config["diversity"],
            "board_verification": verification,
        }
        if args.resume:
            session = Session(root)
            old = session.meta
            if not same_board(old["board"], meta):
                raise SystemExit("Session was collected with a different board definition")
            if not np.allclose(old["camera"]["intrinsics"]["K"], stream.K) or \
                    old["camera"]["stream"] != stream.info["stream"]:
                raise SystemExit("Camera stream/intrinsics differ from the session; start a new session")
            if old["robot"]["end_effector"] != robot.info["end_effector"]:
                raise SystemExit("End-effector configuration differs from the session")
            previous_grav = old["robot"].get("grav_comp")
            if previous_grav is not None and previous_grav != robot.info["grav_comp"]:
                raise SystemExit(f"Session used --grav-comp {previous_grav}; pass the same mode")
            previous = old.get("follower")
            if previous is not None and not same_follower(previous, follower):
                raise SystemExit("Follower profile differs from this session; pass the same "
                                 "--follower value (or unrecorded, if that is what was used)")
            print(f"Resuming {root} with {len(session.sample_dirs())} samples. "
                  "Only resume if the board mount and follower have NOT changed.")
        else:
            session = Session.create(root, session_meta)
            print(f"New session {root}")
        print(json.dumps({"stream": stream.info["stream"], "K": stream.info["intrinsics"]["K"],
                          "distortion": stream.info["intrinsics"]["distortion_model"]}, indent=None))
        Collector(config, session, stream, robot, board_cfg, cam_arm).run()
    print(f"Session saved in {root}. Next: review / calibrate --session {root}")


if __name__ == "__main__":
    main()
