"""Saved binary initialization mask for a video's first decoded frame, plus a metadata sidecar.

    <video>.init_mask.png   one-channel uint8 0/255 (lossless), first decoded frame's resolution
    <video>.init_mask.json  frame index 0, dimensions, source-video identity, mask model

A mask is reused only if its sidecar matches the video exactly; anything else is rejected
(rerun with --recreate-mask to replace it). Masks are never resized.
"""

import json
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from fp_capture.drawing import overlay_mask
from fp_capture.run_folder import format_foundationpose_mask

from .prompt_ui import InitMaskUI
from .video_io import array_sha256, file_sha256, first_frame, video_info

FORMAT_VERSION = 1
WINDOW = "SAM init mask"


class InitMaskError(RuntimeError):
    pass


def mask_paths(video: Path) -> Tuple[Path, Path]:
    return (video.with_name(video.name + ".init_mask.png"),
            video.with_name(video.name + ".init_mask.json"))


def video_identity(video: Path, first: Optional[np.ndarray] = None) -> dict:
    first = first_frame(video) if first is None else first
    info = video_info(video)
    return {
        "name": video.name,
        "size_bytes": video.stat().st_size,
        "sha256": file_sha256(video),
        "first_frame_sha256": array_sha256(first),
        "fps": info.fps,
        "container_frame_count": info.container_frame_count,
    }


def save_init_mask(video: Path, mask: np.ndarray, first: np.ndarray, mask_model: dict) -> Path:
    if mask.shape != first.shape[:2]:
        raise InitMaskError(f"Mask {mask.shape} does not match the first frame {first.shape[:2]}")
    png, sidecar = mask_paths(video)
    if not cv2.imwrite(str(png), format_foundationpose_mask(mask)):
        raise InitMaskError(f"Could not write {png}")
    overlay = overlay_mask(first, mask)
    cv2.imwrite(str(png.with_name(video.name + ".init_mask_overlay.png")),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    height, width = mask.shape
    sidecar.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "frame_index": 0,
        "height": height,
        "width": width,
        "mask_area": int(mask.sum()),
        "mask_sha256": array_sha256(mask),
        "video": video_identity(video, first),
        "mask_model": mask_model,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2) + "\n")
    print(f"Saved init mask {png} (+ {sidecar.name})")
    return png


def load_init_mask(video: Path) -> Tuple[np.ndarray, dict]:
    """Load and validate the video's saved mask; raise InitMaskError if missing or stale."""
    png, sidecar = mask_paths(video)
    if not png.is_file() and not sidecar.is_file():
        raise FileNotFoundError(f"No init mask for {video}")
    if not png.is_file() or not sidecar.is_file():
        raise InitMaskError(f"Incomplete init mask: need both {png.name} and {sidecar.name}")
    meta = json.loads(sidecar.read_text())
    raw = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint8:
        raise InitMaskError(f"{png} is not a one-channel uint8 image")
    if not np.isin(raw, (0, 255)).all():
        raise InitMaskError(f"{png} is not binary (values other than 0/255)")
    mask = raw > 0

    first = first_frame(video)
    identity = video_identity(video, first)
    problems = []
    if meta.get("format_version") != FORMAT_VERSION:
        problems.append(f"format_version {meta.get('format_version')} != {FORMAT_VERSION}")
    if meta.get("frame_index") != 0:
        problems.append(f"frame_index {meta.get('frame_index')} != 0")
    if mask.shape != first.shape[:2] or (meta.get("height"), meta.get("width")) != first.shape[:2]:
        problems.append(f"mask {mask.shape} / sidecar {(meta.get('height'), meta.get('width'))} "
                        f"vs first frame {first.shape[:2]}")
    for key in ("sha256", "size_bytes", "first_frame_sha256", "fps"):
        if meta.get("video", {}).get(key) != identity[key]:
            problems.append(f"video {key} {meta.get('video', {}).get(key)} != {identity[key]}")
    if meta.get("mask_sha256") != array_sha256(mask):
        problems.append("mask PNG content differs from the sidecar's mask_sha256")
    if not mask.any() or mask.all():
        problems.append("mask is entirely background or entirely foreground")
    if problems:
        raise InitMaskError(f"Init mask {png} does not match {video}:\n  - " + "\n  - ".join(problems)
                            + "\nRerun with --recreate-mask to replace it.")
    return mask, meta


def create_init_mask(video: Path, mask_model_key: str) -> bool:
    """Interactive: prompt on the first decoded frame with mask_model_key; save on confirm."""
    from .tracker import HfVideoTracker

    first = first_frame(video)
    tracker = HfVideoTracker(mask_model_key)
    ui = InitMaskUI(first, partial(tracker.preview, first),
                    f"{video.name} frame 0 | mask model {mask_model_key}", WINDOW)
    try:
        mask = ui.run()
    finally:
        cv2.destroyAllWindows()
    if mask is None:
        print("Init mask not confirmed; nothing saved")
        return False
    save_init_mask(video, mask, first, tracker.loaded.describe())
    return True
