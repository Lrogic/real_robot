"""Live RealSense tracking of one clicked object with an HF SAM video tracker.

    python -m experiments.segmentation.live --model sam2-small [--tracking-fps 15]

Space freezes the init frame; click the object (right click: background, drag: box), Enter
confirms the previewed mask and starts tracking from a fresh session. q/Esc quits.
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from fp_capture.camera import CameraSettings, RealSenseStream
from fp_capture.drawing import overlay_mask
from fp_capture.run_folder import format_foundationpose_mask

from .camera_thread import LatestFrameReader
from .metrics import (MEMORY_DEFINITION, TIMING_DEFINITION, RunLog, reset_peak_memory,
                      software_versions, write_json)
from .models import MODELS
from .prompt_ui import InitMaskUI, draw_text_strip
from .render import annotate
from .tracker import HfVideoTracker

RUNS_DIR = Path(__file__).resolve().parent / "runs"
WINDOW = "HF SAM live tracking"
QUIT_KEYS = (ord("q"), 27)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--tracking-fps", type=float, default=30.0,
                        help="Maximum tracking-request rate; each request takes the newest unprocessed frame")
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Tracking requests on a disposable warm-up session (default: 5)")
    parser.add_argument("--num-seconds", type=float, default=None, help="Stop tracking after this long")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Run folder (default: {RUNS_DIR}/live_<model>_<time>)")
    args = parser.parse_args()
    if min(args.width, args.height, args.camera_fps) <= 0 or args.tracking_fps <= 0:
        parser.error("width, height, camera-fps, tracking-fps must be > 0")
    if args.tracking_fps > args.camera_fps:
        parser.error("--tracking-fps cannot exceed --camera-fps")
    if args.warmup_steps < 0 or args.settle_frames < 0:
        parser.error("--warmup-steps and --settle-frames must be >= 0")
    return args


def window_closed() -> bool:
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def show(image_rgb) -> int:
    cv2.imshow(WINDOW, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


def pick_init_frame(reader: LatestFrameReader):
    seq = 0
    frame = None
    while True:
        newest = reader.latest(newer_than=seq, timeout=0.05)
        if newest is not None:
            frame, seq = newest, newest.seq
            hint = draw_text_strip(frame.rgb.shape[1], ["Space: freeze this frame for initialization",
                                                        "q/Esc: quit"])
            key = show(np.vstack([frame.rgb, hint]))
        else:
            key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and frame is not None:
            return frame
        if key in QUIT_KEYS or (frame is not None and window_closed()):
            return None


def save_init(out_dir: Path, rgb, mask) -> None:
    cv2.imwrite(str(out_dir / "init_frame.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "init_mask.png"), format_foundationpose_mask(mask))
    cv2.imwrite(str(out_dir / "init_mask_overlay.png"), cv2.cvtColor(overlay_mask(rgb, mask), cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or RUNS_DIR / f"live_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    tracker = HfVideoTracker(args.model)
    settings = CameraSettings(args.width, args.height, args.camera_fps, args.settle_frames, undistort=False)
    log = RunLog(args.model)
    meta = {"mode": "live", "model_info": tracker.loaded.describe(),
            "camera": {"width": args.width, "height": args.height, "fps": args.camera_fps,
                       "settle_frames": args.settle_frames, "undistort": False,
                       "stream": "fp_capture RealSenseStream (color + aligned depth; depth unused)"},
            "tracking_fps": args.tracking_fps,
            "frame_policy": "each request takes the newest camera frame not yet processed, at most "
                            "tracking_fps requests per second; no queue, no duplicates",
            "timing_definition": TIMING_DEFINITION, "memory_definition": MEMORY_DEFINITION,
            "peak_reset_point": "after model load, warm-up, and mask previews; immediately before "
                                "the trial's init request"}
    period = 1.0 / args.tracking_fps
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        with RealSenseStream(settings) as stream, LatestFrameReader(stream) as reader:
            init = pick_init_frame(reader)
            if init is None:
                return 0
            warmup_s = tracker.warmup(init.rgb, args.warmup_steps)
            meta["warmup"] = {"steps": args.warmup_steps, "seconds": warmup_s,
                              "policy": "disposable session on the frozen init frame with a synthetic "
                                        "centered-box mask; init + `steps` requests, all excluded"}
            mask = InitMaskUI(init.rgb, partial(tracker.preview, init.rgb),
                              f"{args.model}: select the object", WINDOW).run()
            if mask is None:
                print("No mask confirmed; quitting")
                return 0
            save_init(out_dir, init.rgb, mask)

            reset_peak_memory()
            first = tracker.start(init.rgb, mask)
            log.add("init", first, tracker, source_idx=init.timing.frame_number,
                    camera_timestamp_ms=init.timing.camera_timestamp_ms)
            last_seq, last_number = init.seq, init.timing.frame_number
            log.loop_started = next_tick = time.perf_counter()
            while True:
                now = time.perf_counter()
                if args.num_seconds is not None and now - log.loop_started >= args.num_seconds:
                    break
                frame = None
                if now >= next_tick:
                    frame = reader.latest(newer_than=last_seq, timeout=0.005)
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    next_tick = max(next_tick + period, now)
                    result = tracker.step(frame.rgb)
                    skipped = frame.timing.frame_number - last_number - 1
                    last_seq, last_number = frame.seq, frame.timing.frame_number
                    log.add("track", result, tracker, source_idx=frame.timing.frame_number,
                            camera_timestamp_ms=frame.timing.camera_timestamp_ms, skipped=skipped)
                    status = (f"cam frame {frame.timing.frame_number} | proc {result.frame_idx} | "
                              f"{result.elapsed_s * 1000:.0f} ms | skipped {skipped} | q: quit")
                    key = show(annotate(frame.rgb, result.mask, args.model, result.presence, status))
                if key in QUIT_KEYS or window_closed():
                    break
            log.loop_ended = time.perf_counter()
            meta["camera_reader"] = {"frames_received": reader.received,
                                     "frames_missed_before_reader": reader.camera_frames_missed}
    finally:
        cv2.destroyAllWindows()
        tracker.close()
        if log.records:
            log.write_frames_csv(out_dir / "frames.csv")
            summary = log.summary()
            write_json(out_dir / "summary.json", {**meta, "software": software_versions(), "summary": summary})
            log.print_report(summary)
            if "camera_reader" in meta:
                print(f"  camera frames received {meta['camera_reader']['frames_received']}, "
                      f"missed before reader {meta['camera_reader']['frames_missed_before_reader']}")
            print(f"Run folder {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
