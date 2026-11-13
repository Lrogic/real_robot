"""Two-arm ChArUco collection: the camera arm (follower) and the board-holding manipulator both move.

    python -m calibration.collect_dual --grav-comp normal|board-adjusted [--session data/sessions/NAME] [--resume]
                                       [--follower-hold idle|position] [--manipulator-hold idle|position]

Solved afterwards by calibrate_dual into T_base_camera(q) = Z @ F(q) @ X for any follower pose.

Work in stations. For each one:
    f  free the follower (gravity compensation), guide it to a camera pose, f again to lock it there
    then move the manipulator (gravity compensation, board taped in the closed gripper) through
    4-6 board poses, pressing c at each. A capture holds the manipulator where it is, waits for both
    arms to be still, takes a fresh image, and frees it again.
Both arms hold in position mode. check_follower_play found that a held arm can sit 0.5-1 deg away
from what its joints report (gear play), and idle mode hid about half as much of that, but idle
starts with no holding torque and builds it only as the arm moves: switching a loaded arm from
gravity compensation into idle lets it DROP (seen on both arms at extended poses, 2026-11-13).
--follower-hold / --manipulator-hold idle remain for lightly loaded poses only.
Aim for about 8 stations: 3-4 around the pose the camera will be used at most, the rest at the edges
of the follower workspace (left, right, near, far, high), plus the parked pose.
    c  capture (only with the follower locked)    d  delete the last sample
    g  toggle guidance                             q  quit (support both arms when asked)

Each sample keeps everything needed to replay it later: both arms' joints before and after the
image, their FK poses, the station, the capture order, the image, intrinsics and detection.
Writes data/sessions/<timestamp>_dual/ (git-ignored). Requires a passing board verification.
Runs in the fp_robot env (trossen_arm, pyrealsense2, OpenCV).
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .board import DEFAULT_DATA_DIR, BoardConfig, board_metadata, load_config, make_board, make_detector, same_board
from .collect import check_verification
from .detection import detect, draw_detection, quiet_opencv
from .robot import joint_motion, wait_still_all
from .session import Session, load_sample

WINDOW = "two-arm hand-eye collection"
PANEL_W = 520
GREEN, RED, YELLOW, WHITE, GREY = (80, 220, 80), (60, 60, 255), (0, 220, 255), (235, 235, 235), (160, 160, 160)
CONVENTION = ("column vectors; T_destination_source; metres. E_i @ Y = Z @ F_i @ X @ C_i with E_i = T_base_ee "
              "(manipulator FK), F_i = T_fbase_fee (follower FK), C_i = T_camera_board, X = T_fee_camera, "
              "Z = T_base_fbase, Y = T_ee_board. T_base_camera(q) = Z @ F(q) @ X")
GUIDANCE = [
    "Per station: f, move the follower, f to lock it.",
    "Then 4-6 board poses with the manipulator, c each:",
    "- tilt the board 15-45 deg about TWO axes",
    "- vary distance and position in the view",
    "~8 stations: 3-4 near the common camera pose,",
    "the rest at the workspace edges, plus parked.",
    "Turn and tilt the follower between stations too.",
    "If the board shifts in the gripper: restart.",
]


class DualCollector:
    def __init__(self, config, session, stream, manipulator, follower, board_cfg, manipulator_hold="position"):
        self.config, self.session, self.stream = config, session, stream
        self.manip, self.follower = manipulator, follower
        self.manip_hold = manipulator_hold
        self.cfg = board_cfg
        self.board = make_board(board_cfg)
        self.detector = make_detector(self.board)
        self.board_meta = board_metadata(board_cfg)
        self.q, self.cap = config["quality"], config["capture"]
        self.message, self.message_color = "ready", WHITE
        self.show_guidance = True
        self.reload()
        self.station = max((s["follower"]["station"] for s in self.samples), default=-1) + 1

    def reload(self):
        self.samples = [load_sample(d) for d in self.session.sample_dirs()]

    def set_message(self, text, color):
        self.message, self.message_color = text, color
        print(text)

    def toggle_follower(self):
        if self.follower.locked:
            self.follower.free()
            self.set_message("follower FREE: guide it, then f to lock (support it)", YELLOW)
        else:
            self.follower.lock()
            if any(s["follower"]["station"] == self.station for s in self.samples):
                self.station += 1                     # a new follower pose; an unused station number is reused
            self.set_message(f"follower locked: station {self.station}", GREEN)

    def capture(self, t_key: float):
        if not self.follower.locked:
            return self.set_message("not captured: lock the follower first (f)", RED)
        cap = self.cap
        try:
            self.manip.hold_position(mode=self.manip_hold)
            time.sleep(cap["settle_s"])
            arms = {"manipulator": self.manip, "follower": self.follower}
            still, states, stats = wait_still_all(arms, cap["still_window_s"], cap["still_joint_vel_rad_s"],
                                                  cap["still_joint_range_rad"])
            if not still:
                moving = ", ".join(f"{n} drift {s['max_joint_drift_rad_s']:.4f} rad/s range "
                                   f"{s['max_joint_range_rad']:.4f} rad" for n, s in stats.items())
                return self.set_message(f"not captured: arms not still ({moving})", RED)
            before = {n: s[-1] for n, s in states.items()}
            not_before = max(t_key + cap["settle_s"], max(b.host_time_s for b in before.values()))
            frame = self.stream.grab_fresh(not_before, cap["fresh_frames"], cap["fresh_timeout_s"])
            after = {n: a.read_state() for n, a in arms.items()}
        finally:
            self.manip.release_arm()
        delta = {n: float(np.abs(after[n].arm_positions - before[n].arm_positions).max()) for n in arms}
        if max(delta.values()) > cap["max_joint_delta_rad"]:
            return self.set_message(f"not captured: an arm moved during the image {delta}", RED)
        det = detect(frame.image, self.detector, self.board, self.cfg, self.stream.K, self.stream.dist, self.q)
        if not det.accepted:
            return self.set_message("not captured: " + "; ".join(det.reasons), RED)

        index = self.session.next_index()
        m, f = after["manipulator"], after["follower"]
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
            "robot": {"state_before_image": before["manipulator"].to_json(), "state_after_image": m.to_json(),
                      "stillness": stats["manipulator"], "joint_delta_rad": delta["manipulator"],
                      "T_base_ee": m.T_base_ee.tolist(), "T_base_ee_from": "state_after_image (measured, FK)",
                      "mode_at_image": f"arm {self.manip_hold} mode holding its joints; gripper external_effort closed"},
            "follower": {"station": self.station,
                         "state_before_image": before["follower"].to_json(), "state_after_image": f.to_json(),
                         "stillness": stats["follower"], "joint_delta_rad": delta["follower"],
                         "T_fbase_fee": f.T_base_ee.tolist(), "T_fbase_fee_from": "state_after_image (measured, FK)",
                         "mode_at_image": self.follower.info["mode_at_image"]},
            "replay": {"order": index, "station": self.station,
                       "follower_arm_positions_rad": f.arm_positions.tolist(),
                       "manipulator_arm_positions_rad": m.arm_positions.tolist(),
                       "manipulator_gripper_position_m": m.gripper_position,
                       "note": f"joints measured after the image; follower held in {self.follower.hold_mode} "
                               f"mode, manipulator in {self.manip_hold} mode"},
            "board": self.board_meta,
        }
        self.session.save_sample(index, frame.image, sample)
        self.reload()
        n_station = sum(s["follower"]["station"] == self.station for s in self.samples)
        dm = det.metrics
        self.set_message(f"saved #{index} (station {self.station}: {n_station}): {dm['num_corners']} corners, "
                         f"{dm['reproj_rms_px']:.2f} px, tilt {dm['tilt_deg']:.0f} deg", GREEN)

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

        counts = {}
        for s in self.samples:
            counts[s["follower"]["station"]] = counts.get(s["follower"]["station"], 0) + 1
        full = sum(n >= 3 for n in counts.values())
        line(f"samples {len(self.samples)}   stations with 3+ samples: {full} (aim ~8)", WHITE, 0.6, 28)
        line(f"follower {'LOCKED' if self.follower.locked else 'FREE'}   station {self.station}: "
             f"{counts.get(self.station, 0)} samples", GREEN if self.follower.locked else YELLOW)
        for name, mo in motion.items():
            still = (mo["max_joint_range_rad"] < self.cap["still_joint_range_rad"]
                     and mo["max_joint_drift_rad_s"] < self.cap["still_joint_vel_rad_s"])
            line(f"{name} {'still' if still else 'MOVING'}  range {mo['max_joint_range_rad'] * 1000:.1f} mrad",
                 GREEN if still else YELLOW)
        m = det.metrics
        line(f"corners {m.get('num_corners', 0)}/{self.cfg.num_corners}  markers {m.get('num_markers', 0)}")
        if "reproj_rms_px" in m:
            line(f"reproj {m['reproj_rms_px']:.2f}px  dist {m['distance_m']:.2f}m  tilt {m['tilt_deg']:.0f} "
                 f"(X {m['tilt_about_x_deg']:+.0f}, Y {m['tilt_about_y_deg']:+.0f})")
        line("DETECTION OK" if det.accepted else "REJECT: " + (det.reasons[0] if det.reasons else ""),
             GREEN if det.accepted else RED)
        y += 6
        line(self.message[:62], self.message_color)
        if len(self.message) > 62:
            line(self.message[62:124], self.message_color)
        y += 6
        line("per station: " + " ".join(f"{k}:{n}" for k, n in sorted(counts.items())), GREY, 0.5, 22)
        if self.show_guidance:
            y += 6
            for g in GUIDANCE:
                line(g, GREY, 0.48, 20)
        p[height - 36:] = 30
        y = height - 14
        line("f free/lock follower  c capture  d delete  g guide  q quit", WHITE, 0.5)
        return p

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        recent = {"manipulator": [], "follower": []}
        arms = {"manipulator": self.manip, "follower": self.follower}
        while True:
            frame = self.stream.read()
            det = detect(frame.image, self.detector, self.board, self.cfg, self.stream.K, self.stream.dist, self.q)
            motion = {}
            for name, arm in arms.items():
                recent[name].append(arm.read_state())
                while recent[name][-1].host_time_s - recent[name][0].host_time_s > self.cap["still_window_s"]:
                    recent[name].pop(0)
                motion[name] = joint_motion(recent[name])
            vis = draw_detection(frame.image, det, self.stream.K, self.stream.dist)
            cv2.imshow(WINDOW, np.hstack([vis, self.panel(vis.shape[0], det, motion)]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                t_key = time.time()
                self.set_message("capturing: hold still...", YELLOW)
                cv2.imshow(WINDOW, np.hstack([vis, self.panel(vis.shape[0], det, motion)]))
                cv2.waitKey(1)
                self.capture(t_key)
            elif key == ord("f"):
                self.toggle_follower()
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
    p.add_argument("--session", type=Path, default=None, help="default data/sessions/<timestamp>_dual")
    p.add_argument("--resume", action="store_true", help="add samples to an existing session (board not re-taped)")
    p.add_argument("--grav-comp", required=True, choices=["normal", "board-adjusted"],
                   help="manipulator gravity compensation: stock gripper model, or including robot.board_payload")
    p.add_argument("--follower-hold", choices=["idle", "position"], default="position",
                   help="how the locked follower holds its pose (idle drops it at extended poses)")
    p.add_argument("--manipulator-hold", choices=["idle", "position"], default="position",
                   help="how the manipulator holds during each capture (idle drops a loaded arm; see the docstring)")
    p.add_argument("--verification", type=Path, default=DEFAULT_DATA_DIR / "board_verification.json")
    p.add_argument("--skip-verification", action="store_true")
    args = p.parse_args()
    quiet_opencv()

    from .camera import ColorStream
    from .robot import FollowerArm, GravityCompManipulator

    config = load_config(args.config)
    board_cfg = BoardConfig.from_dict(config["board"])
    meta = board_metadata(board_cfg)
    verification = check_verification(args.verification, meta, args.skip_verification)
    root = args.session or DEFAULT_DATA_DIR / "sessions" / (time.strftime("%Y%m%d_%H%M%S") + "_dual")

    with ColorStream(config["camera"]) as stream, \
            GravityCompManipulator(config["robot"], board_adjusted=args.grav_comp == "board-adjusted") as manip:
        input("Seat the taped board in the manipulator's gripper, then press Enter to close it...")
        manip.close_gripper()
        input(f"The follower (camera arm) is about to be locked in {args.follower_hold} mode where it is. Press Enter...")
        with FollowerArm(config["follower"], hold_mode=args.follower_hold) as follower:
            session_meta = {
                "kind": "dual_arm",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "convention": CONVENTION,
                "board": meta,
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "python": sys.version.split()[0],
                "host": platform.node(),
                "camera": stream.info,
                "robot": manip.info,
                "follower": follower.info,
                "capture": config["capture"], "quality": config["quality"],
                "hold_modes": {"follower": args.follower_hold, "manipulator": args.manipulator_hold},
                "board_verification": verification,
            }
            if args.resume:
                session = Session(root)
                old = session.meta
                if old.get("kind") != "dual_arm":
                    raise SystemExit(f"{root} is not a two-arm session")
                if not same_board(old["board"], meta):
                    raise SystemExit("Session was collected with a different board definition")
                if not np.allclose(old["camera"]["intrinsics"]["K"], stream.K) or \
                        old["camera"]["stream"] != stream.info["stream"]:
                    raise SystemExit("Camera stream/intrinsics differ from the session; start a new session")
                for arm in ("robot", "follower"):
                    if old[arm]["end_effector"] != (manip.info if arm == "robot" else follower.info)["end_effector"]:
                        raise SystemExit(f"{arm} end-effector configuration differs from the session")
                if old.get("hold_modes", {"follower": "position", "manipulator": "position"}) != session_meta["hold_modes"]:
                    raise SystemExit(f"Session used hold modes {old.get('hold_modes')}; pass the same "
                                     "--follower-hold / --manipulator-hold")
                print(f"Resuming {root} with {len(session.sample_dirs())} samples. "
                      "Only resume if the board has NOT been re-taped or moved in the gripper.")
            else:
                session = Session.create(root, session_meta)
                print(f"New session {root}")
            DualCollector(config, session, stream, manip, follower, board_cfg, args.manipulator_hold).run()
    print(f"Session saved in {root}. Next: ./run.sh calibrate-dual --session {root}")


if __name__ == "__main__":
    main()
