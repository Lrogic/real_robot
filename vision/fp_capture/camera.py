"""RealSense capture using the same stream setup as realsense_foundationpose.py.

By default frames are passed through as RealSense delivers them (distorted). With
undistort=True they are undistorted after depth-to-color alignment, so K is an exact
pinhole model for both RGB and depth (FoundationPose takes no distortion coefficients).
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration.camera import MAX_MODEL_MISMATCH_PX, opencv_model_mismatch_px


@dataclass(frozen=True)
class CameraSettings:
    width: int = 640
    height: int = 480
    fps: int = 30
    settle_frames: int = 30
    undistort: bool = False  # True: undistort RGB and depth so the pinhole K is exact


@dataclass(frozen=True)
class Capture:
    rgb: np.ndarray    # HxWx3 uint8, RGB order
    depth: np.ndarray  # HxW float32, metres, aligned to color
    K: np.ndarray      # 3x3 float64 intrinsics; pinhole model only if undistorted
    undistorted: bool = False


@dataclass(frozen=True)
class FrameTiming:
    frame_number: int
    camera_timestamp_ms: float
    wait_s: float     # blocked in wait_for_frames
    prepare_s: float  # align + undistort + color/depth conversion


class RealSenseStream:
    """Aligned RGB-D stream. Use as a context manager; settle frames are discarded on entry."""

    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._undistort_maps = None
        self.K = None

    def __enter__(self) -> "RealSenseStream":
        s = self.settings
        config = rs.config()
        config.enable_stream(rs.stream.depth, s.width, s.height, rs.format.z16, s.fps)
        config.enable_stream(rs.stream.color, s.width, s.height, rs.format.bgr8, s.fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        try:
            self._align = rs.align(rs.stream.color)
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                              dtype=np.float64)
            if self.settings.undistort:
                dist = np.array(intr.coeffs[:5], dtype=np.float64)
                mismatch = opencv_model_mismatch_px(intr, self.K, dist)
                if mismatch > MAX_MODEL_MISMATCH_PX:
                    raise RuntimeError(f"Distortion model {intr.model} differs from OpenCV's by {mismatch:.2f} px")
                # Same K out as in, so a few raw pixels at the image border fall outside the undistorted frame.
                self._undistort_maps = cv2.initUndistortRectifyMap(
                    self.K, dist, None, self.K, (intr.width, intr.height), cv2.CV_16SC2)
            print(f"Discarding {s.settle_frames} frames...")
            for _ in range(s.settle_frames):
                self._pipeline.wait_for_frames()
        except BaseException:
            self._pipeline.stop()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        self._pipeline.stop()

    def read(self):
        """Return (rgb, depth_m, FrameTiming) for the next aligned frame."""
        started = time.perf_counter()
        frames = self._pipeline.wait_for_frames()
        after_wait = time.perf_counter()
        frames = self._align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense did not return both color and aligned depth")
        # Camera delivers BGR8; downstream code expects RGB and depth in metres.
        bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        if self._undistort_maps is not None:
            map1, map2 = self._undistort_maps
            bgr = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR)
            # Nearest keeps every depth a measured value; interpolation invents points across edges and holes.
            depth_raw = cv2.remap(depth_raw, map1, map2, cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = depth_raw.astype(np.float32) * self._depth_scale
        timing = FrameTiming(frame_number=color_frame.get_frame_number(),
                             camera_timestamp_ms=color_frame.get_timestamp(),
                             wait_s=after_wait - started,
                             prepare_s=time.perf_counter() - after_wait)
        return rgb, depth, timing


def capture_frame(settings: CameraSettings) -> Capture:
    """Start the camera, discard settle frames, and return one aligned RGB-D frame."""
    with RealSenseStream(settings) as stream:
        rgb, depth, _ = stream.read()
        return Capture(rgb=rgb, depth=depth, K=stream.K, undistorted=settings.undistort)
