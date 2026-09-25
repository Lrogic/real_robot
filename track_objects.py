"""Track every saved object that has a mesh, live, with FoundationPose.

Reads a run folder written by segment_objects.py, registers each object on the
saved RGB-D frame using its mask, then tracks all of them sequentially on each
live frame. Objects without a mesh are skipped. Run in the fp_robot env.

    python track_objects.py --save-path foundationpose_runs/run1
    python track_objects.py --save-path foundationpose_runs/run1 --num-seconds 30 --save-video
    python track_objects.py --save-path foundationpose_runs/run1 --fp-mode batched

Do not move the camera or objects between setup and the start of tracking.
Press q or Esc in the window to stop.
"""

import sys
import tempfile
import time

import cv2
import numpy as np

from fp_capture.camera import RealSenseStream
from fp_capture.cli import camera_settings, parse_pipeline_args, pipeline_parser
from fp_capture.run_folder import VIDEO_FILE, load_run

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


def check_live_camera(saved_K, saved_shape, stream):
    live_shape = (stream.settings.height, stream.settings.width)
    if live_shape != saved_shape:
        raise ValueError(f"Live resolution {live_shape[::-1]} differs from saved "
                         f"{saved_shape[::-1]}; pass the same --width/--height as setup")
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


def write_video(frames, output_path, elapsed_seconds):
    if not frames:
        print("No frames were displayed; video was not saved")
        return
    height, width = frames[0].shape[:2]
    # FP may process substantially fewer frames than the camera's configured FPS.
    # This average makes playback approximately match the time spent tracking.
    playback_fps = len(frames) / max(elapsed_seconds, 1e-6)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             playback_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    print(f"Saved {len(frames)} buffered frames to {output_path} at {playback_fps:.2f} fps")


def track_live(args, tracker, stream):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    video_frames = [] if args.save_video else None
    all_samples, window_samples = [], []
    last_frame_number = None
    # Duration starts with the first live tracking frame, after model setup.
    started_at = time.monotonic()
    last_report_at = started_at
    deadline = started_at + args.num_seconds if args.num_seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            loop_started = time.perf_counter()
            rgb, depth, timing = stream.read()
            fp_per_object = tracker.track(rgb, depth, stream.K, args.track_refine_iter)
            display_frame = cv2.cvtColor(tracker.draw(rgb, stream.K), cv2.COLOR_RGB2BGR)
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
                window_samples.clear()
                last_report_at = now
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        elapsed_seconds = time.monotonic() - started_at
        cv2.destroyAllWindows()
        print_stats("overall", all_samples, elapsed_seconds, args.fps)
        if video_frames is not None:
            write_video(video_frames, args.save_path / VIDEO_FILE, elapsed_seconds)
    print(f"Processed {len(all_samples)} live frames")


def main():
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    try:
        run = load_run(args.save_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    objects = trackable_objects(run)
    if not objects:
        print(f"Nothing to track in {args.save_path}: no saved object has a usable mesh")
        return 0

    from fp_capture.tracking import MultiObjectTracker

    with tempfile.TemporaryDirectory(prefix="fp_debug_") as debug_dir:
        tracker = MultiObjectTracker(objects, args.foundationpose_dir, debug_dir,
                                     batched=args.fp_mode == "batched",
                                     batched_render=args.fp_mode == "batched-render")
        print(f"FoundationPose tracking mode: {args.fp_mode}")
        tracker.register(objects, run.capture.rgb, run.capture.depth, run.capture.K,
                         args.est_refine_iter)
        with RealSenseStream(camera_settings(args)) as stream:
            check_live_camera(run.capture.K, run.capture.depth.shape, stream)
            track_live(args, tracker, stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
