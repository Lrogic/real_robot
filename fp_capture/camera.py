"""RealSense capture using the same stream setup as realsense_foundationpose.py."""

from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class CameraSettings:
    width: int = 640
    height: int = 480
    fps: int = 30
    settle_frames: int = 30


@dataclass(frozen=True)
class Capture:
    rgb: np.ndarray    # HxWx3 uint8, RGB order
    depth: np.ndarray  # HxW float32, metres, aligned to color
    K: np.ndarray      # 3x3 float64 color intrinsics


def _intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                    dtype=np.float64)


def capture_frame(settings: CameraSettings) -> Capture:
    """Start the camera, discard settle frames, and return one aligned RGB-D frame."""
    config = rs.config()
    config.enable_stream(rs.stream.depth, settings.width, settings.height,
                         rs.format.z16, settings.fps)
    config.enable_stream(rs.stream.color, settings.width, settings.height,
                         rs.format.bgr8, settings.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        K = _intrinsics(profile)
        print(f"Discarding {settings.settle_frames} frames...")
        for _ in range(settings.settle_frames):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense did not return both color and aligned depth")
        # Camera delivers BGR8; downstream code expects RGB and depth in metres.
        rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        return Capture(rgb=rgb, depth=depth, K=K)
    finally:
        pipeline.stop()
