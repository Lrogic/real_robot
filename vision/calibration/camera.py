"""Color-only RealSense stream with native intrinsics and fresh-frame capture.

Images are used exactly as delivered by the color stream (no resize, no rectification),
so the reported intrinsics and distortion apply directly to saved images.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

# RealSense coefficients are [k1, k2, p1, p2, k3], OpenCV's 5-element order. The D405 color
# stream reports inverse_brown_conrady, whose librealsense projection matches OpenCV's model;
# this is verified numerically per stream rather than assumed from the model name.
MAX_MODEL_MISMATCH_PX = 0.1


def opencv_model_mismatch_px(intr, K, dist) -> float:
    """Max pixel difference between librealsense and cv2.projectPoints over the image."""
    u, v = np.meshgrid(np.linspace(0, intr.width - 1, 17), np.linspace(0, intr.height - 1, 11))
    rays = np.stack([(u.ravel() - intr.ppx) / intr.fx, (v.ravel() - intr.ppy) / intr.fy,
                     np.ones(u.size)], axis=1)
    rs_px = np.array([rs.rs2_project_point_to_pixel(intr, [float(c) for c in p]) for p in rays])
    cv_px = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    return float(np.linalg.norm(rs_px - cv_px, axis=1).max())


@dataclass
class Frame:
    image: np.ndarray            # HxWx3 uint8 BGR, as delivered
    frame_number: int
    timestamp_ms: float          # device/global timestamp
    timestamp_domain: str
    arrival_ms: float            # host wall clock (ms) when the frame reached the host
    arrival_from_metadata: bool

    def to_json(self) -> dict:
        return {"frame_number": self.frame_number, "timestamp_ms": self.timestamp_ms,
                "timestamp_domain": self.timestamp_domain, "host_arrival_ms": self.arrival_ms,
                "host_arrival_from_metadata": self.arrival_from_metadata}


def list_color_profiles(device) -> list:
    profiles = set()
    for sensor in device.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.color:
                v = p.as_video_stream_profile()
                profiles.add((v.width(), v.height(), v.fps(), str(p.format()).split(".")[-1]))
    return sorted(profiles)


def find_device(serial=None):
    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError("No RealSense device found")
    for d in devices:
        if serial is None or d.get_info(rs.camera_info.serial_number) == str(serial):
            return d
    raise RuntimeError(f"RealSense {serial} not found")


def choose_profile(available: list, cam_cfg: dict):
    wanted = [(cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"])]
    if cam_cfg.get("fallback"):
        fb = cam_cfg["fallback"]
        wanted.append((fb["width"], fb["height"], fb["fps"]))
    have = {(w, h, f) for w, h, f, fmt in available if fmt == "bgr8"}
    for i, w in enumerate(wanted):
        if w in have:
            return w, i > 0
    lines = "\n".join(f"  {w}x{h} @ {f} fps {fmt}" for w, h, f, fmt in available)
    raise RuntimeError(f"None of {wanted} is available as bgr8. Color profiles:\n{lines}\n"
                       "Set camera.width/height/fps or camera.fallback in config.yaml")


class ColorStream:
    def __init__(self, cam_cfg: dict):
        self.cfg = cam_cfg
        self.pipeline = None
        self.info = {}
        self.K = None
        self.dist = None

    def __enter__(self) -> "ColorStream":
        device = find_device(self.cfg.get("serial"))
        available = list_color_profiles(device)
        (w, h, fps), is_fallback = choose_profile(available, self.cfg)
        if is_fallback:
            print(f"Preferred {self.cfg['width']}x{self.cfg['height']}@{self.cfg['fps']} unavailable; "
                  f"using fallback {w}x{h}@{fps}. Available: {available}")
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        try:
            vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = vsp.get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            self.dist = np.array(intr.coeffs[:5], dtype=np.float64)
            mismatch = opencv_model_mismatch_px(intr, self.K, self.dist)
            if mismatch > MAX_MODEL_MISMATCH_PX:
                raise RuntimeError(f"Distortion model {intr.model} differs from OpenCV's by {mismatch:.2f} px")

            def info(key):
                return device.get_info(key) if device.supports(key) else None

            self.info = {
                "device_name": info(rs.camera_info.name),
                "serial": serial,
                "firmware": info(rs.camera_info.firmware_version),
                "usb_type": info(rs.camera_info.usb_type_descriptor),
                "librealsense": rs.__version__ if hasattr(rs, "__version__") else None,
                "stream": {"stream": "color", "width": w, "height": h, "fps": fps, "format": "bgr8",
                           "is_fallback": is_fallback, "resized": False, "rectified": False},
                "available_color_profiles": [list(p) for p in available],
                "intrinsics": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                               "ppx": intr.ppx, "ppy": intr.ppy, "K": self.K.tolist(),
                               "distortion_model": str(intr.model).split(".")[-1],
                               "realsense_coeffs": list(intr.coeffs),
                               "opencv_dist": self.dist.tolist(),
                               "opencv_model_mismatch_px": mismatch},
                "frame": "D405 color optical frame (x right, y down, z forward)",
            }
            for _ in range(self.cfg.get("warmup_frames", 30)):
                self.pipeline.wait_for_frames()
        except BaseException:
            self.pipeline.stop()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        self.pipeline.stop()

    def read(self) -> Frame:
        frames = self.pipeline.wait_for_frames()
        received_ms = time.time() * 1000.0
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense returned no color frame")
        from_meta = color.supports_frame_metadata(rs.frame_metadata_value.time_of_arrival)
        arrival = float(color.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)) if from_meta \
            else received_ms
        return Frame(image=np.asanyarray(color.get_data()).copy(),
                     frame_number=int(color.get_frame_number()),
                     timestamp_ms=float(color.get_timestamp()),
                     timestamp_domain=str(color.get_frame_timestamp_domain()).split(".")[-1],
                     arrival_ms=arrival, arrival_from_metadata=from_meta)

    def grab_fresh(self, not_before_s: float, count: int, timeout_s: float) -> Frame:
        """Discard buffered frames that arrived before not_before_s, then return the last of `count` fresh ones."""
        deadline = time.time() + timeout_s
        fresh = 0
        last_number = None
        while time.time() < deadline:
            f = self.read()
            if f.arrival_ms < not_before_s * 1000.0 or f.frame_number == last_number:
                continue
            last_number = f.frame_number
            fresh += 1
            if fresh >= count:
                return f
        raise TimeoutError("No fresh camera frame within timeout")
