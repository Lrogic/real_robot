"""ChArUco detection, board pose (T_camera_board) and quality checks."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import BoardConfig
from .transforms import T_from_rvec


@dataclass
class Detection:
    charuco_corners: np.ndarray        # (N, 2) px
    charuco_ids: np.ndarray            # (N,)
    marker_corners: np.ndarray         # (M, 4, 2) px
    marker_ids: np.ndarray             # (M,)
    rvec: np.ndarray = None            # board -> camera
    tvec: np.ndarray = None
    metrics: dict = field(default_factory=dict)
    accepted: bool = False
    reasons: list = field(default_factory=list)

    @property
    def T_camera_board(self):
        return None if self.rvec is None else T_from_rvec(self.rvec, self.tvec)

    def to_json(self) -> dict:
        return {
            "charuco_corners_px": self.charuco_corners.tolist(),
            "charuco_ids": self.charuco_ids.tolist(),
            "marker_corners_px": self.marker_corners.tolist(),
            "marker_ids": self.marker_ids.tolist(),
            "rvec_camera_board": None if self.rvec is None else self.rvec.tolist(),
            "tvec_camera_board_m": None if self.tvec is None else self.tvec.tolist(),
            "T_camera_board": None if self.rvec is None else self.T_camera_board.tolist(),
            "metrics": self.metrics,
            "accepted": self.accepted,
            "reject_reasons": self.reasons,
        }


def _flat(a, shape_tail):
    if a is None or len(a) == 0:
        return np.zeros((0, *shape_tail), dtype=np.float64)
    return np.asarray(a, dtype=np.float64).reshape(-1, *shape_tail)


def _ids(a):
    return np.zeros(0, dtype=np.int32) if a is None else np.asarray(a, dtype=np.int32).ravel()


def detect_raw(gray, detector):
    """Run ChArUco detection; returns (charuco_corners, charuco_ids, marker_corners, marker_ids)."""
    cc, ci, mc, mi = detector.detectBoard(gray)
    return _flat(cc, (2,)), _ids(ci), _flat(mc, (4, 2)), _ids(mi)


def estimate_pose(board, corners, ids, K, dist):
    """IPPE on the planar board, then LM refinement. Returns rvec, tvec, rms_px, ambiguity_ratio."""
    obj, img = board.matchImagePoints(corners.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1))
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        return None
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    rvec, tvec = rvecs[order[0]].reshape(3, 1), tvecs[order[0]].reshape(3, 1)
    ambiguity = float(errs[order[0]] / errs[order[1]]) if n > 1 and errs[order[1]] > 0 else 0.0
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return rvec.ravel(), tvec.ravel(), rms, ambiguity


def pose_angles(R, t) -> dict:
    """Board orientation relative to the camera, for quality and diversity reporting (deg)."""
    n = R[:, 2] * (1.0 if R[2, 2] >= 0 else -1.0)
    view = t / max(np.linalg.norm(t), 1e-9)
    return {
        "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(n @ view), 0.0, 1.0)))),
        "tilt_about_x_deg": float(np.degrees(np.arctan2(n[1], n[2]))),
        "tilt_about_y_deg": float(np.degrees(np.arctan2(n[0], n[2]))),
        "inplane_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
    }


def compute_metrics(gray, det: Detection, cfg: BoardConfig, board, K, dist) -> dict:
    h, w = gray.shape[:2]
    m = {"num_markers": int(len(det.marker_ids)), "num_corners": int(len(det.charuco_ids))}
    if len(det.charuco_ids) == 0:
        return m
    rows, cols = cfg.corner_row_col(det.charuco_ids)
    m["corner_rows"] = int(len(np.unique(rows)))
    m["corner_cols"] = int(len(np.unique(cols)))
    if len(det.charuco_ids) >= 3:
        board_xy = np.stack([cols, rows], axis=1).astype(np.float64)
        ev = np.linalg.eigvalsh(np.cov(board_xy.T))
        m["spread_ratio"] = float(np.sqrt(max(ev[0], 0.0) / ev[1])) if ev[1] > 0 else 0.0
        hull = cv2.convexHull(det.charuco_corners.astype(np.float32))
        m["hull_fraction"] = float(cv2.contourArea(hull) / (w * h))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        m["sharpness"] = float(lap[mask > 0].var()) if np.any(mask) else 0.0
    if len(det.charuco_ids) >= 4 and K is not None:
        pose = estimate_pose(board, det.charuco_corners, det.charuco_ids, K, dist)
        if pose is not None:
            det.rvec, det.tvec, m["reproj_rms_px"], m["ambiguity_ratio"] = pose
            T = det.T_camera_board
            m["distance_m"] = float(np.linalg.norm(det.tvec))
            m.update(pose_angles(T[:3, :3], T[:3, 3]))
    return m


def accept(m: dict, q: dict):
    reasons = []

    def need(key, ok, msg):
        if key not in m:
            reasons.append(f"no {key}")
        elif not ok(m[key]):
            reasons.append(msg.format(m[key]))

    need("num_corners", lambda v: v >= q["min_corners"], f"{{}} corners < {q['min_corners']}")
    need("corner_rows", lambda v: v >= q["min_corner_rows"], f"{{}} corner rows < {q['min_corner_rows']}")
    need("corner_cols", lambda v: v >= q["min_corner_cols"], f"{{}} corner cols < {q['min_corner_cols']}")
    need("spread_ratio", lambda v: v >= q["min_spread_ratio"], "near-collinear corners (spread {:.2f})")
    need("hull_fraction", lambda v: v >= q["min_hull_fraction"], "board too small ({:.3f} of image)")
    need("sharpness", lambda v: v >= q["min_sharpness"], "blurry (sharpness {:.0f})")
    need("reproj_rms_px", lambda v: v <= q["max_reproj_rms_px"], "reprojection {:.2f} px too high")
    need("ambiguity_ratio", lambda v: v <= q["max_ambiguity_ratio"], "ambiguous pose (ratio {:.2f})")
    return not reasons, reasons


def detect(image_bgr, detector, board, cfg: BoardConfig, K, dist, quality: dict) -> Detection:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    det = Detection(*detect_raw(gray, detector))
    det.metrics = compute_metrics(gray, det, cfg, board, K, dist)
    det.accepted, det.reasons = accept(det.metrics, quality)
    return det


def quiet_opencv():
    """Silence per-frame OpenCV warnings (e.g. drawFrameAxes endpoints off-screen at close range)."""
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    elif hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(2)


def draw_detection(image_bgr, det: Detection, K, dist, axis_len_m=0.04):
    vis = image_bgr.copy()
    if len(det.marker_ids):
        cv2.aruco.drawDetectedMarkers(vis, det.marker_corners.reshape(-1, 1, 4, 2).astype(np.float32),
                                      det.marker_ids.reshape(-1, 1))
    if len(det.charuco_ids):
        cv2.aruco.drawDetectedCornersCharuco(vis, det.charuco_corners.reshape(-1, 1, 2).astype(np.float32),
                                             det.charuco_ids.reshape(-1, 1), (0, 255, 255))
    if det.rvec is not None:
        cv2.drawFrameAxes(vis, K, dist, det.rvec, det.tvec, axis_len_m, 2)
    return vis
