"""Live tracking loop for the track stage: FoundationPose every frame, optional recovery and arm hooks.

track_live reads the newest RealSense frameset, tracks every object, draws the window, and
reports timing; hooks (--manipulation) and recovery (--recovery) plug into each iteration.
"""

import time

import cv2
import numpy as np

from .foundationpose.pose_variation_log import PoseVariationLog
from .recovery.sam_recovery import LiveFrame
from .run_folder import POSE_VARIATION_FILE, VIDEO_FILE

WINDOW = "FoundationPose multi-object (q or Esc to quit)"


def trackable_objects(run):
    objects = []
    for obj in run.objects:
        if obj.mesh_path is None:
            print(f"Skipping {obj.name}: no mesh")
        elif not np.any((run.capture.depth > 0) & obj.mask):
            print(f"Skipping {obj.name}: no valid depth inside its mask")
        else:
            objects.append(obj)
    return objects


def check_live_camera(saved_K, saved_shape, saved_undistorted, stream):
    live_shape = (stream.settings.height, stream.settings.width)
    if live_shape != saved_shape:
        raise ValueError(f"Live resolution {live_shape[::-1]} differs from saved "
                         f"{saved_shape[::-1]}; pass the same --width/--height as setup")
    if stream.settings.undistort != saved_undistorted:
        state = lambda on: "undistorted (--undistort)" if on else "distorted (default)"
        raise ValueError(f"Run was captured {state(saved_undistorted)} but tracking is "
                         f"{state(stream.settings.undistort)}; use the same setting as setup, or rerun setup")
    if not np.allclose(saved_K, stream.K, rtol=1e-3, atol=0.5):
        raise ValueError("Camera intrinsics changed since setup; rerun setup")


def print_stats(label, samples, elapsed_s, camera_fps):
    if not samples:
        return

    def mean_ms(key):
        return 1000 * np.mean([sample[key] for sample in samples])

    processed_fps = len(samples) / max(elapsed_s, 1e-6)
    skipped = sum(sample["skipped"] for sample in samples)
    camera_span_s = (samples[-1]["camera_timestamp_ms"] - samples[0]["camera_timestamp_ms"]) / 1000
    camera_frames = samples[-1]["frame_number"] - samples[0]["frame_number"]
    camera_rate = (f", observed camera ~{camera_frames / camera_span_s:.1f} fps"
                   if camera_span_s > 0 and camera_frames > 0 else "")
    print(f"[{label}] {len(samples)} processed in {elapsed_s:.1f}s: "
          f"{processed_fps:.1f} fps (camera configured {camera_fps} fps{camera_rate}), "
          f"{skipped} camera frame numbers skipped")
    fp_totals = [s["fp_s"] for s in samples]
    print(f"  Mean per frame: FP all objects {mean_ms('fp_s'):.1f} ms "
          f"(p95 {1000 * np.percentile(fp_totals, 95):.1f}), "
          f"camera wait {mean_ms('wait_s'):.1f} ms, "
          f"align/convert {mean_ms('prepare_s'):.1f} ms, "
          f"whole loop {mean_ms('loop_s'):.1f} ms")
    per_object = ", ".join(
        f"{name} {1000 * np.mean([s['fp_per_object'][name] for s in samples]):.1f} ms"
        for name in samples[0]["fp_per_object"])
    print(f"  Mean FP per object: {per_object}")


def _even(n):
    """MPEG-4 frame sizes must be multiples of two."""
    return n + (n & 1)


def _fit_frame(frame, height, width):
    """Place frame on a fixed-size canvas. Status strips grow (the episode key adds a line),
    and VideoWriter drops every frame whose size differs from the one it was opened with."""
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    canvas = np.full((height, width, frame.shape[2]), 20, dtype=np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    return canvas


def write_video(frames, output_path, elapsed_seconds):
    if not frames:
        print("No frames were displayed; video was not saved")
        return
    height = _even(max(frame.shape[0] for frame in frames))
    width = _even(max(frame.shape[1] for frame in frames))
    # FP may process substantially fewer frames than the camera's configured FPS.
    # This average makes playback approximately match the time spent tracking.
    playback_fps = len(frames) / max(elapsed_seconds, 1e-6)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             playback_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    failed = 0
    try:
        for frame in frames:
            if writer.write(_fit_frame(frame, height, width)) is False:
                failed += 1
    finally:
        writer.release()
    written = len(frames) - failed
    padded = sum(frame.shape[0] != height or frame.shape[1] != width for frame in frames)
    note = f", {padded} padded to {width}x{height}" if padded else ""
    print(f"Saved {written} buffered frames to {output_path} at {playback_fps:.2f} fps{note}")
    if failed:
        print(f"Video writer dropped {failed} of {len(frames)} frames")


def show_now(vis) -> int:
    """Display an RGB frame immediately (used while recovery blocks the loop); returns the key."""
    cv2.imshow(WINDOW, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


def track_live(args, tracker, stream, hooks=None, recovery=None):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if hooks is not None:
        cv2.setMouseCallback(WINDOW, hooks.on_mouse)
    video_frames = [] if args.save_video else None
    pose_log = PoseVariationLog() if args.log_variation else None
    all_samples, window_samples = [], []
    last_frame_number = None
    # Duration starts with the first live tracking frame, after model setup.
    started_at = time.monotonic()
    last_report_at = started_at
    deadline = started_at + args.num_seconds if args.num_seconds else None
    quit_pending = False

    def show_during_recovery(vis):
        # A key pressed here (e.g. kill) is not seen by the loop's own waitKey.
        nonlocal quit_pending
        key = show_now(vis)
        if key != 0xFF:
            quit_pending |= hooks.on_key(key) if hooks is not None else key in (ord("q"), 27)

    if recovery is not None:
        recovery.start_loop()
    try:
        while deadline is None or time.monotonic() < deadline:
            loop_started = time.perf_counter()
            rgb, depth, timing = stream.read()
            frame = LiveFrame(rgb, depth, stream.K, timing.frame_number,
                              timing.camera_timestamp_ms, time.monotonic())
            fp_per_object = tracker.track(rgb, depth, stream.K, args.track_refine_iter)
            if pose_log is not None:
                pose_log.record(time.monotonic() - started_at, tracker)
            if recovery is not None:
                recovery.after_fp(frame, show_during_recovery)
            vis = tracker.draw(rgb, stream.K)
            if hooks is not None:
                hooks.update(tracker, stream.K, timing)
                vis = hooks.draw(vis)
            if recovery is not None:
                vis = recovery.draw(vis, frame)
            if hooks is not None and hasattr(hooks, "draw_strip"):
                vis = np.vstack([vis, hooks.draw_strip(vis.shape[1])])
            display_frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW, display_frame)
            if video_frames is not None:
                video_frames.append(display_frame)
            key = cv2.waitKey(1) & 0xFF

            skipped = (max(timing.frame_number - last_frame_number - 1, 0)
                       if last_frame_number is not None else 0)
            last_frame_number = timing.frame_number
            sample = {"fp_s": sum(fp_per_object.values()), "fp_per_object": fp_per_object,
                      "wait_s": timing.wait_s, "prepare_s": timing.prepare_s,
                      "frame_number": timing.frame_number,
                      "camera_timestamp_ms": timing.camera_timestamp_ms,
                      "loop_s": time.perf_counter() - loop_started, "skipped": skipped}
            all_samples.append(sample)
            window_samples.append(sample)
            now = time.monotonic()
            if now - last_report_at >= args.stats_every:
                print_stats("recent", window_samples, now - last_report_at, args.fps)
                if recovery is not None:
                    recovery.report(now - last_report_at)
                window_samples.clear()
                last_report_at = now
            quit_requested = quit_pending or (hooks.on_key(key) if hooks is not None else key in (ord("q"), 27))
            if quit_requested or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if hooks is not None:
            hooks.close()
        elapsed_seconds = time.monotonic() - started_at
        cv2.destroyAllWindows()
        print_stats("overall", all_samples, elapsed_seconds, args.fps)
        if recovery is not None:
            recovery.summary([s["fp_s"] for s in all_samples], elapsed_seconds)
        if video_frames is not None:
            write_video(video_frames, args.save_path / VIDEO_FILE, elapsed_seconds)
        if pose_log is not None:
            pose_log.write(args.save_path / POSE_VARIATION_FILE)
    print(f"Processed {len(all_samples)} live frames")
