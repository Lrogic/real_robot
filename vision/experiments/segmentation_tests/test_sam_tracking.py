#!/usr/bin/env python3
"""Live RealSense click-init SAM tracking (comparison stage 1)."""

import os

# SAM2/SAM3 call tqdm inside propagate_in_video once per live frame.
os.environ["TQDM_DISABLE"] = "1"

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_VISION = Path(__file__).resolve().parents[2]
if str(_VISION) not in sys.path:
    sys.path.insert(0, str(_VISION))

from fp_capture.camera import RealSenseStream, CameraSettings
from fp_capture.drawing import overlay_mask

from experiments.segmentation_tests.init_ui import InitPromptUI, WINDOW
from experiments.segmentation_tests.metrics import CudaTimer, FrameStats, RunMetrics
from experiments.segmentation_tests.screen_hints import draw_setup_banner, draw_status_bar_bottom
from experiments.segmentation_tests.trackers import MODEL_CHOICES, TrackOutput, build_tracker
from experiments.segmentation_tests.ui_display import show_rgb, snap_window_to_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CHOICES.keys()))
    parser.add_argument("--checkpoint", type=Path, default=None)
    camera = parser.add_argument_group("camera")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.add_argument("--fps", type=int, default=30)
    camera.add_argument("--settle-frames", type=int, default=30)
    camera.add_argument("--undistort", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--stats-every", type=float, default=5.0)
    parser.add_argument(
        "--tracking-fps",
        type=float,
        default=None,
        help="Max SAM inference rate (Hz). Default: same as --fps (every camera frame). "
        "Lower values skip tracker steps; e.g. camera 60 fps with --tracking-fps 15 runs "
        "inference on every 4th frame and holds the last mask on others.",
    )
    return parser.parse_args()


def tracking_stride(camera_fps: int, tracking_fps: float | None) -> int:
    if tracking_fps is None or tracking_fps <= 0 or tracking_fps >= camera_fps:
        return 1
    return max(1, int(round(camera_fps / tracking_fps)))


def print_startup(meta: dict) -> None:
    print("SAM live tracking")
    print(f"  model:          {meta['model']}")
    print(f"  checkpoint:     {meta['checkpoint']}")
    if "config" in meta:
        print(f"  config:         {meta['config']}")
    if "use_fa3" in meta:
        print(f"  use_fa3:        {meta['use_fa3']}")
    print(f"  device:         {meta['device']}")
    print(f"  implementation: {meta['implementation']}")


def capture_init_frame(stream) -> np.ndarray:
    snapped = False
    while True:
        rgb, _, _ = stream.read()
        if not snapped:
            snap_window_to_frame(WINDOW, rgb)
            snapped = True
        display = draw_setup_banner(
            rgb,
            "Pick init frame",
            [
                "Hold the scene still, then press i or Space to use this frame",
                "q / Esc: quit",
            ],
        )
        show_rgb(WINDOW, display, snap=False)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("i"), ord(" ")):
            return rgb.copy()
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit(0)


def presence_string(presence) -> str:
    if not presence:
        return ""
    return ", ".join(f"{k}={v:.3f}" for k, v in presence.items())


def mask_status_line(out: TrackOutput) -> str:
    if out.mask is None or out.mask_area == 0:
        return "no mask"
    return f"mask area {out.mask_area}"


def draw_tracking_view(rgb: np.ndarray, out: TrackOutput, mask_held: bool = False) -> np.ndarray:
    vis = overlay_mask(rgb, out.mask)
    hold = "  |  mask hold (no new inference)" if mask_held else ""
    lines = [
        f"frame {out.video_frame_idx}  |  {mask_status_line(out)}{hold}",
    ]
    scores = presence_string(out.presence)
    if scores:
        lines.append(scores)
    lines.append("r: new trial   q / Esc: quit")
    return draw_status_bar_bottom(vis, lines)


def run_trial(tracker, stream, args, metrics: RunMetrics) -> bool:
    init_rgb = capture_init_frame(stream)
    tracker.begin_session(init_rgb)

    def predict(points, labels, _clear):
        return tracker.preview_prompt(points, labels, clear_old=True)

    ui = InitPromptUI(predict)
    ui.set_frame(init_rgb)
    if not ui.run():
        tracker.reset_session()
        return True
    with torch.inference_mode():
        init_out = tracker.finalize_init()
    stride = tracking_stride(args.fps, args.tracking_fps)
    effective_tracking_fps = args.fps / stride
    if stride > 1:
        target = args.tracking_fps if args.tracking_fps is not None else effective_tracking_fps
        print(
            f"Tracking stride {stride} (~{effective_tracking_fps:.1f} Hz inference, "
            f"target {target:g} Hz, camera {args.fps} fps)."
        )
    print(f"Tracking started (init mask area {init_out.mask_area}).")
    snap_window_to_frame(WINDOW, init_rgb)

    last_camera_frame = None
    trial_metrics = RunMetrics(warmup_frames=args.warmup_frames)
    last_report = time.monotonic()
    frame_counter = 0
    last_out = init_out

    while True:
        loop_t0 = time.perf_counter()
        rgb, _, timing = stream.read()
        skipped = 0
        if last_camera_frame is not None:
            skipped = max(timing.frame_number - last_camera_frame - 1, 0)
        last_camera_frame = timing.frame_number

        run_track = (frame_counter % stride) == 0
        infer_s = 0.0
        if run_track:
            timer = CudaTimer()
            timer.start()
            with torch.inference_mode():
                last_out = tracker.track_frame(rgb)
            infer_s = timer.stop()
        out = last_out

        vis = draw_tracking_view(rgb, out, mask_held=not run_track)
        show_rgb(WINDOW, vis, snap=False)
        key = cv2.waitKey(1) & 0xFF

        loop_s = time.perf_counter() - loop_t0
        sample = FrameStats(
            frame_number=timing.frame_number,
            video_frame_idx=out.video_frame_idx,
            mask_area=out.mask_area,
            skipped_camera_frames=skipped,
            wait_s=timing.wait_s,
            prepare_s=timing.prepare_s,
            infer_s=infer_s,
            loop_s=loop_s,
            warmup=frame_counter < args.warmup_frames,
            ran_inference=run_track,
        )
        trial_metrics.record(sample)
        metrics.record(sample)
        frame_counter += 1

        now = time.monotonic()
        if now - last_report >= args.stats_every:
            scores = presence_string(out.presence)
            extra = f"  ({scores})" if scores else ""
            print(f"  frame {out.video_frame_idx} area={out.mask_area}{extra}")
            trial_metrics.report("recent", now - last_report, args.fps)
            trial_metrics = RunMetrics(warmup_frames=args.warmup_frames)
            last_report = now

        if key == ord("r"):
            tracker.reset_session()
            return True
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
    return False


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height, fps must be positive")
    if args.tracking_fps is not None:
        if args.tracking_fps <= 0:
            raise SystemExit("--tracking-fps must be positive")
        if args.tracking_fps > args.fps:
            raise SystemExit("--tracking-fps cannot be greater than camera --fps")

    print(f"Use conda env: {MODEL_CHOICES[args.model]['env']}")
    tracker, meta = build_tracker(args.model, args.checkpoint)
    print_startup(meta)

    settings = CameraSettings(args.width, args.height, args.fps, args.settle_frames, args.undistort)
    metrics = RunMetrics(warmup_frames=args.warmup_frames)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        with RealSenseStream(settings) as stream:
            print(f"Camera {settings.width}x{settings.height} @ {settings.fps} fps")
            while run_trial(tracker, stream, args, metrics):
                pass
    finally:
        tracker.reset_session()
        cv2.destroyAllWindows()
        if metrics.samples:
            metrics.report("overall", sum(s.loop_s for s in metrics.samples), args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
