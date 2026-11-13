"""Two-arm hand-eye solve: camera on the follower, ChArUco board on the manipulator.

Measurement model (column vectors, T_destination_source, metres):

    E_i @ Y = Z @ F_i @ X @ C_i

    E_i = T_base_ee     manipulator FK (driver Cartesian pose)        measured
    F_i = T_fbase_fee   follower FK (driver Cartesian pose)            measured
    C_i = T_camera_board                                               observed (ChArUco corners)
    X   = T_fee_camera  camera on the follower's tool frame            unknown
    Z   = T_base_fbase  follower base in the manipulator base frame    unknown
    Y   = T_ee_board    board on the manipulator's tool frame          unknown

Then T_base_camera(q_follower) = Z @ F(q_follower) @ X.

Samples are grouped in stations: one follower pose, several board poses. The starting point
is two linear AX = ZB solves (Shah's method, plain numpy):
    per station k:     E_i @ Y = W_k @ C_i          gives W_k = T_base_camera at that station, and Y
    across stations:   F_k @ X = inv(Z) @ W_k       gives X and Z
which needs at least 3 stations with 3 or more board poses each, and rotation about at least two
different axes in both arms. All three transforms are then refined together by minimising the
corner reprojection error over every sample (scipy least_squares, robust loss).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from .transforms import inv_T, make_T, mean_T, rotation_between_deg, transform_points


@dataclass
class Obs:
    """One sample: both arms' FK, the camera intrinsics and the detected board corners."""
    id: int
    station: int
    E: np.ndarray            # T_base_ee (manipulator)
    F: np.ndarray            # T_fbase_fee (follower)
    K: np.ndarray
    dist: np.ndarray
    corners_px: np.ndarray   # (N, 2) detected ChArUco corners
    corner_ids: np.ndarray   # (N,) their IDs
    C: np.ndarray = None     # T_camera_board from PnP on this sample (for the start point and 3D checks)
    meta: dict = field(default_factory=dict)


def project_so3(M) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def solve_axzb(As, Bs):
    """Linear least-squares X, Z with A_i @ X = Z @ B_i (Shah 2013). Needs rotation about 2+ axes."""
    rows = []
    for A, B in zip(As, Bs):
        rows.append(np.hstack([np.kron(np.eye(3), A[:3, :3]), -np.kron(B[:3, :3].T, np.eye(3))]))
    _, _, Vt = np.linalg.svd(np.vstack(rows))
    v = Vt[-1]
    RX, RZ = v[:9].reshape(3, 3, order="F"), v[9:].reshape(3, 3, order="F")
    det = np.linalg.det(RX)
    scale = np.sign(det) * abs(det) ** (-1.0 / 3.0)
    RX, RZ = project_so3(RX * scale), project_so3(RZ * scale)
    M = np.vstack([np.hstack([A[:3, :3], -np.eye(3)]) for A in As])
    b = np.concatenate([RZ @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t = np.linalg.lstsq(M, b, rcond=None)[0]
    return make_T(RX, t[:3]), make_T(RZ, t[3:])


def axzb_residual(As, Bs, X, Z):
    """Mean rotation (deg) and translation (mm) disagreement of A_i X against Z B_i."""
    rot = [rotation_between_deg((A @ X)[:3, :3], (Z @ B)[:3, :3]) for A, B in zip(As, Bs)]
    trans = [np.linalg.norm((A @ X)[:3, 3] - (Z @ B)[:3, 3]) * 1000 for A, B in zip(As, Bs)]
    return float(np.mean(rot)), float(np.mean(trans))


def initial_guess(obs: list, min_per_station: int = 3):
    """Start point (X, Z, Y) and a per-station report, from the two linear solves."""
    stations = {}
    for o in obs:
        stations.setdefault(o.station, []).append(o)
    Ws, Fs, Ys, report = [], [], [], {}
    for k, group in sorted(stations.items()):
        if len(group) < min_per_station:
            report[k] = {"samples": len(group), "used": False, "reason": f"fewer than {min_per_station} samples"}
            continue
        Y_k, W_k = solve_axzb([o.E for o in group], [o.C for o in group])
        rot, trans = axzb_residual([o.E for o in group], [o.C for o in group], Y_k, W_k)
        report[k] = {"samples": len(group), "used": True, "eye_to_hand_rot_deg": rot, "eye_to_hand_trans_mm": trans}
        Ws.append(W_k)
        Ys.append(Y_k)
        Fs.append(mean_T([o.F for o in group]))
    if len(Ws) < 3:
        raise ValueError(f"need 3 or more follower poses with {min_per_station}+ board poses each; have {len(Ws)}")
    X, Zinv = solve_axzb(Fs, Ws)
    return X, inv_T(Zinv), mean_T(Ys), report


def _pack(X, Z, Y):
    return np.concatenate([np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]] for T in (X, Z, Y)])


def _unpack(v):
    return [make_T(cv2.Rodrigues(v[6 * i:6 * i + 3])[0], v[6 * i + 3:6 * i + 6]) for i in range(3)]


def predicted_C(o: Obs, X, Z, Y) -> np.ndarray:
    return inv_T(Z @ o.F @ X) @ o.E @ Y


def reprojection(o: Obs, X, Z, Y, board_pts) -> np.ndarray:
    """(N, 2) predicted minus detected corner pixels."""
    C = predicted_C(o, X, Z, Y)
    proj, _ = cv2.projectPoints(board_pts[o.corner_ids], cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], o.K, o.dist)
    return proj.reshape(-1, 2) - o.corners_px


def refine(obs: list, X, Z, Y, board_pts, loss: str = "soft_l1", f_scale_px: float = 1.0):
    def res(v):
        Xv, Zv, Yv = _unpack(v)
        return np.concatenate([reprojection(o, Xv, Zv, Yv, board_pts).ravel() for o in obs])
    sol = least_squares(res, _pack(X, Z, Y), loss=loss, f_scale=f_scale_px, x_scale="jac")
    X, Z, Y = _unpack(sol.x)
    return X, Z, Y, {"success": bool(sol.success), "message": str(sol.message), "nfev": int(sol.nfev),
                     "cost": float(sol.cost)}


def sample_errors(o: Obs, X, Z, Y, board_pts) -> dict:
    """Reprojection RMS (px), and board corners through the manipulator vs through the camera (mm).

    The 3D number is what a tracked object would be off by: the corners placed by the camera
    chain Z F X C (with C from this image alone) against where the manipulator says they are.
    """
    px = np.linalg.norm(reprojection(o, X, Z, Y, board_pts), axis=1)
    out = {"id": o.id, "station": o.station, "reproj_rms_px": float(np.sqrt(np.mean(px ** 2)))}
    if o.C is not None:
        pts = board_pts[o.corner_ids]
        d = np.linalg.norm(transform_points(o.E @ Y, pts) - transform_points(Z @ o.F @ X @ o.C, pts), axis=1) * 1000
        out.update(corner_rms_mm=float(np.sqrt(np.mean(d ** 2))), corner_max_mm=float(d.max()))
    return out


def summarize(errors: list) -> dict:
    if not errors:
        return {}
    px = np.array([e["reproj_rms_px"] for e in errors])
    mm = np.array([e.get("corner_rms_mm", np.nan) for e in errors])
    return {"samples": len(errors), "reproj_rms_px": float(np.sqrt(np.mean(px ** 2))), "reproj_max_px": float(px.max()),
            "corner_rms_mm": float(np.sqrt(np.nanmean(mm ** 2))), "corner_max_mm": float(np.nanmax(mm))}


def outliers(errors: list, abs_mm: float, floor_mm: float = 2.0) -> list:
    """Samples over abs_mm, or well above the rest (median + 3 MAD) and over floor_mm.

    A single image's board pose alone scatters the corner number by up to about 1 mm (synthetic, 0.2 px
    corner noise), so the floor keeps that from being flagged; a slipped board gives several mm.
    """
    mm = np.array([e.get("corner_rms_mm", 0.0) for e in errors])
    med = np.median(mm)
    mad = 1.4826 * np.median(np.abs(mm - med))
    return [e["id"] for e, v in zip(errors, mm) if v > abs_mm or (v > med + 3 * mad and v > floor_mm)]
