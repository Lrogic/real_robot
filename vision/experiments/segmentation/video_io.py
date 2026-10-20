"""Incremental video decoding, deterministic frame subsampling, and video writers."""

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

SOURCE_FOURCC = "FFV1"   # lossless; clean recordings
ANNOTATED_FOURCC = "mp4v"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    container_frame_count: int  # as reported by the container; decoding is authoritative


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    return capture


def video_info(path: Path) -> VideoInfo:
    capture = _open(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"{path} reports no frame rate")
        return VideoInfo(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(fps),
                         int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def iter_frames(path: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (source index, HxWx3 uint8 RGB) one decoded frame at a time."""
    capture = _open(path)
    try:
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def first_frame(path: Path) -> np.ndarray:
    for _, rgb in iter_frames(path):
        return rgb
    raise RuntimeError(f"{path} has no decodable frames")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class FrameSelector:
    """Pick source frames for a requested tracking rate from source index and FPS only.

    Source frame i has timestamp i / source_fps. Tracking tick k is at k / tracking_fps, and
    the processed frame for tick k is the first source frame at or after it, i.e. frame i is
    selected iff floor(i * tracking_fps / source_fps) > floor((i - 1) * tracking_fps / source_fps).
    Frame 0 (the init frame) is always selected. With tracking_fps=None every frame is selected.
    The choice depends on nothing else, so every model gets the same frame indices.
    """

    def __init__(self, source_fps: float, tracking_fps: Optional[float]):
        if tracking_fps is not None:
            if tracking_fps <= 0:
                raise ValueError("--tracking-fps must be positive")
            if tracking_fps > source_fps + 1e-9:
                raise ValueError(f"--tracking-fps {tracking_fps} exceeds the source FPS {source_fps}")
        self.source_fps = source_fps
        self.tracking_fps = tracking_fps
        self._ratio = (None if tracking_fps is None else
                       Fraction(tracking_fps).limit_denominator(100000)
                       / Fraction(source_fps).limit_denominator(100000))

    def selected(self, index: int) -> bool:
        if self._ratio is None or index == 0:
            return True
        return math.floor(index * self._ratio) > math.floor((index - 1) * self._ratio)

    def describe(self) -> dict:
        return {"source_fps": self.source_fps, "tracking_fps": self.tracking_fps,
                "policy": "all source frames" if self._ratio is None else
                "first source frame at or after each tick k / tracking_fps (frame i at i / source_fps)"}


def open_writer(path: Path, fourcc: str, fps: float, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size_wh)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a {fourcc} writer for {path}")
    return writer
