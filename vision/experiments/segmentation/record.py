"""Record clean RealSense RGB video, then create its init mask on the saved video's first frame.

    python -m experiments.segmentation.record [--width 640 --height 480 --camera-fps 30]

Space starts recording, Space/q/Esc stops. The video is lossless FFV1 (video.mkv) with no
annotations; recording.json and frames.csv hold settings and per-frame camera metadata.
Afterwards the saved file is reopened and its first decoded frame is used for mask selection.
"""

import argparse
import csv
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from fp_capture.camera import CameraSettings, RealSenseStream

from .init_mask import create_init_mask
from .metrics import write_json
from .models import MODELS
from .prompt_ui import draw_text_strip
from .video_io import SOURCE_FOURCC, array_sha256, iter_frames, open_writer

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
WINDOW = "RealSense recording"
VIDEO_FILE = "video.mkv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop recording after this long")
    parser.add_argument("--mask-model", choices=list(MODELS), default="sam3",
                        help="Model for the init-mask preview (default: sam3)")
    parser.add_argument("--no-mask", action="store_true", help="Skip init-mask creation")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Recording folder (default: {RECORDINGS_DIR}/<time>)")
    args = parser.parse_args()
    if min(args.width, args.height, args.camera_fps) <= 0 or args.settle_frames < 0:
        parser.error("width, height, camera-fps must be > 0 and settle-frames >= 0")
    return args


def _writer_loop(writer: cv2.VideoWriter, frames: queue.Queue) -> None:
    while True:
        bgr = frames.get()
        if bgr is None:
            return
        writer.write(bgr)


def record(args, video: Path):
    settings = CameraSettings(args.width, args.height, args.camera_fps, args.settle_frames, undistort=False)
    rows, first_rgb = [], None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    with RealSenseStream(settings) as stream:
        recording, started = False, None
        writer = frames = thread = None
        while True:
            rgb, _depth, timing = stream.read()
            host_time = time.time()
            if recording:
                frames.put(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                previous = rows[-1]["camera_frame_number"] if rows else None
                rows.append({"index": len(rows), "camera_frame_number": timing.frame_number,
                             "camera_timestamp_ms": timing.camera_timestamp_ms, "host_time_s": host_time,
                             "skipped_camera_frames": 0 if previous is None else timing.frame_number - previous - 1})
                if first_rgb is None:
                    first_rgb = rgb.copy()
                text = [f"REC {len(rows)} frames ({time.perf_counter() - started:.1f} s)", "Space/q/Esc: stop"]
            else:
                text = ["Space: start recording", "q/Esc: quit"]
            # Annotations go only to the preview window, never to the recording.
            cv2.imshow(WINDOW, cv2.cvtColor(np.vstack([rgb, draw_text_strip(rgb.shape[1], text)]), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            timed_out = recording and args.max_seconds is not None and time.perf_counter() - started >= args.max_seconds
            closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if not recording and key == ord(" "):
                writer = open_writer(video, SOURCE_FOURCC, args.camera_fps, (args.width, args.height))
                frames = queue.Queue()
                thread = threading.Thread(target=_writer_loop, args=(writer, frames), daemon=True)
                thread.start()
                recording, started = True, time.perf_counter()
            elif (recording and key in (ord(" "), ord("q"), 27)) or timed_out or (recording and closed):
                break
            elif not recording and (key in (ord("q"), 27) or closed):
                break
    cv2.destroyAllWindows()
    if writer is not None:
        frames.put(None)
        thread.join()
        writer.release()
    return settings, rows, first_rgb


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or RECORDINGS_DIR / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    video = out_dir / VIDEO_FILE
    settings, rows, first_rgb = record(args, video)
    if not rows:
        print("Nothing recorded")
        return 1

    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decoded, decoded_first = 0, None
    for _, rgb in iter_frames(video):
        if decoded_first is None:
            decoded_first = rgb
        decoded += 1
    first_matches = decoded_first is not None and np.array_equal(decoded_first, first_rgb)
    span_s = (rows[-1]["camera_timestamp_ms"] - rows[0]["camera_timestamp_ms"]) / 1000
    write_json(out_dir / "recording.json", {
        "video": VIDEO_FILE, "codec": SOURCE_FOURCC, "lossless": True,
        "width": settings.width, "height": settings.height, "camera_fps": settings.fps,
        "settle_frames": settings.settle_frames, "undistort": settings.undistort, "color_order": "RGB (stored BGR in file)",
        "frames_written": len(rows), "frames_decoded": decoded,
        "first_decoded_frame_matches_capture": first_matches,
        "first_decoded_frame_sha256": array_sha256(decoded_first) if decoded_first is not None else None,
        "skipped_camera_frames": sum(r["skipped_camera_frames"] for r in rows),
        "camera_timestamp_span_s": span_s,
        "measured_fps": (len(rows) - 1) / span_s if span_s > 0 else None,
        "container_fps_note": "frames are stored at the nominal camera FPS; true capture times are in frames.csv",
    })
    print(f"Saved {video}: {len(rows)} frames written, {decoded} decoded, "
          f"{sum(r['skipped_camera_frames'] for r in rows)} camera frames skipped, "
          f"first frame lossless: {first_matches}")
    if decoded != len(rows) or not first_matches:
        print("ERROR: the saved video does not match what was captured")
        return 1
    if not args.no_mask:
        create_init_mask(video, args.mask_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
