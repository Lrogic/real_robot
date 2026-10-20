#!/usr/bin/env python3
"""Live RealSense click-init SAM video tracking (comparison stage 1).

Run from real_robot/vision with the conda env matching --model (sam21 or sam3):

    cd real_robot/vision
    conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-small
    conda run -n sam3 python experiments/segmentation_tests/test_sam_tracking.py --model sam31
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# vision/ must be on sys.path before fp_capture imports (run from real_robot/vision).
_VISION = Path(__file__).resolve().parents[2]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from fp_capture.camera import RealSenseStream, CameraSettings
from fp_capture.drawing import overlay_mask

from experiments.segmentation_tests.init_ui import InitPromptUI, WINDOW
from experiments.segmentation_tests.metrics import CudaTimer, FrameStats, RunMetrics
from experiments.segmentation_tests.trackers import MODEL_CHOICES, build_tracker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CHOICES.keys()))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override default checkpoint path")
    camera = parser.add_argument_group("camera")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.add_argument("--fps", type=int, default=30)
    camera.add_argument("--settle-frames", type=int, default=30)
    camera.add_argument("--undistort", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=5,
                        help="Frames excluded from steady-state timing after tracking starts")
    parser.add_argument("--stats-every", type=float, default=5.0)
    return parser.parse_args()


def print_startup(meta: dict) -> None:
    print("SAM tracking experiment (stage 1)")
    print(f"  model:           {meta['model']}")
    print(f"  checkpoint:      {meta['checkpoint']}")
    if "config" in meta:
        print(f"  config:          {meta['config']}")
    print(f"  device:          {meta['device']}")
    print(f"  implementation:  {meta['implementation']}")


def draw_tracking_overlay(rgb, out, video_frame_idx, presence_line: str) -> np.ndarray:
    vis = overlay_mask(rgb, out.mask)
    h = rgb.shape[0]
    status = "no mask" if out.mask is None or out.mask_area == 0 else f"area={out.mask_area}"
    lines = [
        f"frame={video_frame_idx}  {status}",
        "r=reset trial, q/Esc=quit",
    ]
    if presence_line:
        lines.append(presence_line)
    y = 24
    for line in lines:
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22
    return vis


def presence_string(presence) -> str:
    if not presence:
        return ""
    return "presence: " + ", ".join(f"{k}={v:.3f}" for k, v in presence.items())


def wait_for_freeze(stream, window_title: str) -> np.ndarray:
    """Show live feed until user presses Space to freeze."""
    print("Live preview — press Space to freeze for initialization, q to quit.")
    frozen = None
    while True:
        rgb, _, _ = stream.read()
        display = rgb.copy()
        cv2.putText(
            display,
            "Space=freeze frame for clicks, q=quit",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_title, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frozen = rgb.copy()
            break
        if key in (ord("q"), 27) or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit(0)
    return frozen


def run_trial(tracker, stream, args, metrics: RunMetrics) -> bool:
    """One init + track loop. Returns False if user quits entirely."""
    frozen = wait_for_freeze(stream, WINDOW)
    tracker.begin_session(frozen)

    def predict(points, labels, _clear):
        return tracker.preview_prompt(points, labels, clear_old=True)

    ui = InitPromptUI(predict)
    ui.set_frame(frozen)
    if not ui.run():
        tracker.reset_session()
        return True
    with torch.inference_mode():
        init_out = tracker.finalize_init()
    print(f"Init committed on video frame 0 (mask area {init_out.mask_area})")

    last_camera_frame = None
    trial_metrics = RunMetrics(warmup_frames=args.warmup_frames)
    started = time.monotonic()
    last_report = started
    frame_counter = 0

    while True:
        loop_t0 = time.perf_counter()
        rgb, _, timing = stream.read()
        skipped = 0
        if last_camera_frame is not None:
            skipped = max(timing.frame_number - last_camera_frame - 1, 0)
        last_camera_frame = timing.frame_number

        timer = CudaTimer()
        timer.start()
        with torch.inference_mode():
            out = tracker.track_frame(rgb)
        infer_s = timer.stop()

        presence_line = presence_string(out.presence)
        vis = draw_tracking_overlay(rgb, out, out.video_frame_idx, presence_line)
        cv2.imshow(WINDOW, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        loop_s = time.perf_counter() - loop_t0
        warmup = frame_counter < args.warmup_frames
        sample = FrameStats(
            frame_number=timing.frame_number,
            video_frame_idx=out.video_frame_idx,
            mask_area=out.mask_area,
            skipped_camera_frames=skipped,
            wait_s=timing.wait_s,
            prepare_s=timing.prepare_s,
            infer_s=infer_s,
            loop_s=loop_s,
            warmup=warmup,
        )
        trial_metrics.record(sample)
        metrics.record(sample)
        frame_counter += 1

        now = time.monotonic()
        if now - last_report >= args.stats_every:
            trial_metrics.report("recent", now - last_report, args.fps)
            trial_metrics = RunMetrics(warmup_frames=args.warmup_frames)
            last_report = now

        if key == ord("r"):
            print("Reset — new initialization trial.")
            tracker.reset_session()
            return True
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
    return False


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height, fps must be positive")

    expected_env = MODEL_CHOICES[args.model]["env"]
    print(f"Note: run this script in the conda env '{expected_env}'.")

    tracker, meta = build_tracker(args.model, args.checkpoint)
    print_startup(meta)

    settings = CameraSettings(args.width, args.height, args.fps, args.settle_frames, args.undistort)
    metrics = RunMetrics(warmup_frames=args.warmup_frames)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        with RealSenseStream(settings) as stream:
            print(f"Camera {settings.width}x{settings.height} @ {settings.fps} fps "
                  f"({'undistorted' if settings.undistort else 'distorted'})")
            while run_trial(tracker, stream, args, metrics):
                pass
    finally:
        tracker.reset_session()
        cv2.destroyAllWindows()
        if metrics.samples:
            elapsed = sum(s.loop_s for s in metrics.samples)
            metrics.report("overall", elapsed, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
