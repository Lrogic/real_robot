"""Solve X = T_base_camera from a saved session with OpenCV's hand-eye solvers (offline).

    python -m calibration.calibrate --session data/sessions/NAME [--method auto|TSAI|...] [--exclude 3,7]
                                    [--redetect] [--out PATH]

Measurement model (column vectors, T_destination_source, metres):
    X @ C_i = E_i @ Y,   E_i = T_base_ee (measured FK), C_i = T_camera_board (PnP),
    X = T_base_camera,   Y = T_ee_board (fixed, unknown board mount).
Equivalently A_ij @ X = X @ B_ij with A_ij = E_i @ inv(E_j), B_ij = C_i @ inv(C_j).

Solvers:
  * cv2.calibrateHandEye in the eye-to-hand arrangement: pass inv(E_i) = T_ee_base as
    "gripper2base" and C_i as "target2cam"; the returned "cam2gripper" is then T_base_camera.
    Methods TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS.
  * cv2.calibrateRobotWorldHandEye solves A_i @ Xw = Zg @ B_i with A_i = "world2cam",
    B_i = "base2gripper". Writing E_i @ Y = X @ C_i in that form: A_i = E_i, B_i = C_i,
    so the returned "base2world" is Y and "gripper2cam" is X. Methods SHAH, LI.
Needs OpenCV 4.x with these functions (sam3 env; OpenCV 5.0's Python bindings lack them).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .board import BoardConfig, board_corners_m, board_metadata, load_config, make_board, same_board
from .diversity import analyze, summary_lines
from .session import Session, load_sample, sample_E_C
from .transforms import inv_T, make_T, mean_T, rotation_between_deg, transform_points

HAND_EYE = ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"]
ROBOT_WORLD = ["SHAH", "LI"]


def _Rt(Ts):
    return [T[:3, :3].copy() for T in Ts], [T[:3, 3].reshape(3, 1).copy() for T in Ts]


def solve(method, Es, Cs):
    """Return (X = T_base_camera, Y = T_ee_board or None) using the named OpenCV method."""
    if method in HAND_EYE:
        R_g2b, t_g2b = _Rt([inv_T(E) for E in Es])
        R_t2c, t_t2c = _Rt(Cs)
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=getattr(cv2, f"CALIB_HAND_EYE_{method}"))
        return make_T(R, t), None
    if method in ROBOT_WORLD:
        R_w2c, t_w2c = _Rt(Es)
        R_b2g, t_b2g = _Rt(Cs)
        R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
            R_w2c, t_w2c, R_b2g, t_b2g, method=getattr(cv2, f"CALIB_ROBOT_WORLD_HAND_EYE_{method}"))
        return make_T(R_g2c, t_g2c), make_T(R_b2w, t_b2w)
    raise ValueError(f"unknown method {method}")


def residuals(X, Es, Cs, corners, Ks, dists) -> dict:
    """Consistency of X: scatter of Y_i = inv(E_i) X C_i and board-corner disagreement."""
    Ys = [inv_T(E) @ X @ C for E, C in zip(Es, Cs)]
    Y = mean_T(Ys)
    rot = np.array([rotation_between_deg(Y[:3, :3], Yi[:3, :3]) for Yi in Ys])
    trans = np.array([np.linalg.norm(Yi[:3, 3] - Y[:3, 3]) for Yi in Ys]) * 1000
    rms, mx, px = [], [], []
    for E, C, K, d in zip(Es, Cs, Ks, dists):
        p_cam = transform_points(X @ C, corners)
        p_rob = transform_points(E @ Y, corners)
        err = np.linalg.norm(p_cam - p_rob, axis=1) * 1000
        rms.append(float(np.sqrt(np.mean(err ** 2))))
        mx.append(float(err.max()))
        T_cam_board_pred = inv_T(X) @ E @ Y
        proj_pred, _ = cv2.projectPoints(corners, cv2.Rodrigues(T_cam_board_pred[:3, :3])[0],
                                         T_cam_board_pred[:3, 3], K, d)
        proj_meas, _ = cv2.projectPoints(corners, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], K, d)
        px.append(float(np.sqrt(np.mean(np.sum((proj_pred - proj_meas).reshape(-1, 2) ** 2, axis=1)))))
    rms = np.array(rms)
    return {
        "T_ee_board_mean": Y.tolist(),
        "Y_rot_scatter_deg": {"mean": float(rot.mean()), "max": float(rot.max())},
        "Y_trans_scatter_mm": {"mean": float(trans.mean()), "max": float(trans.max())},
        "corner_rms_mm": float(np.sqrt(np.mean(rms ** 2))),
        "corner_max_mm": float(max(mx)),
        "per_sample_corner_rms_mm": rms.tolist(),
        "per_sample_corner_max_mm": mx,
        "per_sample_reproj_px": px,
        "per_sample_Y_rot_deg": rot.tolist(),
        "per_sample_Y_trans_mm": trans.tolist(),
    }


def leave_one_out(method, X, Es, Cs) -> dict:
    if len(Es) < 4:
        return {}
    rot, trans = [], []
    for k in range(len(Es)):
        Xk, _ = solve(method, Es[:k] + Es[k + 1:], Cs[:k] + Cs[k + 1:])
        rot.append(rotation_between_deg(X[:3, :3], Xk[:3, :3]))
        trans.append(float(np.linalg.norm(Xk[:3, 3] - X[:3, 3]) * 1000))
    return {"rot_deg": {"mean": float(np.mean(rot)), "max": float(np.max(rot))},
            "trans_mm": {"mean": float(np.mean(trans)), "max": float(np.max(trans))},
            "per_sample_rot_deg": rot, "per_sample_trans_mm": trans}


def outliers(res, ids, abs_mm) -> list:
    r = np.asarray(res["per_sample_corner_rms_mm"])
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med))
    return [ids[i] for i in range(len(r)) if r[i] > abs_mm or (mad > 0 and r[i] > med + 3 * mad)]


def load_samples(session, config, exclude, redetect_images):
    samples = [load_sample(d) for d in session.sample_dirs()]
    samples = [s for s in samples if s["id"] not in exclude]
    if redetect_images:
        from .review import redetect
        for s in samples:
            s["detection"] = redetect(s, config).to_json()
    dropped = [s["id"] for s in samples if not s["detection"]["accepted"] or s["detection"]["T_camera_board"] is None]
    if dropped:
        print(f"Skipping samples whose detection is rejected: {dropped}")
    return [s for s in samples if s["id"] not in dropped]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--method", default=None, help="override calibration.method in config")
    p.add_argument("--exclude", default="", help="comma-separated sample IDs to leave out")
    p.add_argument("--redetect", action="store_true", help="re-run detection on raw images first")
    p.add_argument("--out", type=Path, default=None, help="default <session>/T_base_camera.json")
    args = p.parse_args()

    if not hasattr(cv2, "calibrateHandEye") or not hasattr(cv2, "calibrateRobotWorldHandEye"):
        raise SystemExit(f"OpenCV {cv2.__version__} lacks the hand-eye solvers; run in the sam3 env (run.sh calibrate)")

    config = load_config(args.config)
    session = Session(args.session)
    smeta = session.meta
    cfg = BoardConfig.from_dict(smeta["board"])
    if not same_board(smeta["board"], board_metadata(BoardConfig.from_dict(config["board"]))):
        print("NOTE: config.yaml board differs from the session; using the session's board definition")
    exclude = {int(v) for v in args.exclude.split(",") if v.strip()}
    samples = load_samples(session, {**config, "board": smeta["board"]}, exclude, args.redetect)
    if len(samples) < 3:
        raise SystemExit(f"Need at least 3 usable samples, have {len(samples)}")
    if len(samples) < 10:
        print(f"WARNING: only {len(samples)} samples; 20-30 recommended")

    ids = [s["id"] for s in samples]
    Es, Cs = map(list, zip(*(sample_E_C(s) for s in samples)))
    Ks = [np.asarray(s["intrinsics"]["K"]) for s in samples]
    dists = [np.asarray(s["intrinsics"]["dist"]) for s in samples]
    corners = board_corners_m(make_board(cfg))

    div = analyze(Es, Cs, [s["detection"]["metrics"] for s in samples], config["diversity"], ids=ids)
    print(f"\n{len(samples)} samples: {ids}")
    for line in summary_lines(div)[1:]:
        print("  " + line)
    for w in div["warnings"]:
        print("  ! " + w)

    results = {}
    for method in HAND_EYE + ROBOT_WORLD:
        try:
            X, Y_solver = solve(method, Es, Cs)
        except cv2.error as e:
            print(f"{method}: failed ({e.msg if hasattr(e, 'msg') else e})")
            continue
        if not np.all(np.isfinite(X)):
            print(f"{method}: non-finite result")
            continue
        res = residuals(X, Es, Cs, corners, Ks, dists)
        results[method] = {"T_base_camera": X.tolist(),
                           "T_ee_board_solver": None if Y_solver is None else Y_solver.tolist(),
                           "residuals": res, "leave_one_out": leave_one_out(method, X, Es, Cs),
                           "outlier_samples": outliers(res, ids, config["calibration"]["outlier_corner_rms_mm"])}
    if not results:
        raise SystemExit("All solvers failed")

    wanted = (args.method or config["calibration"]["method"]).upper()
    if wanted == "AUTO":
        chosen = min(results, key=lambda m: results[m]["residuals"]["corner_rms_mm"])
    elif wanted in results:
        chosen = wanted
    else:
        raise SystemExit(f"method {wanted} unavailable; have {list(results)}")
    X = np.asarray(results[chosen]["T_base_camera"])

    print(f"\n{'method':<11} {'corner rms':>10} {'max':>7} {'Y rot':>7} {'Y trans':>8} {'LOO rot':>8} "
          f"{'LOO trans':>9} {'vs chosen':>16}")
    for m, r in results.items():
        Xm = np.asarray(r["T_base_camera"])
        res, loo = r["residuals"], r["leave_one_out"]
        d_rot = rotation_between_deg(X[:3, :3], Xm[:3, :3])
        d_t = np.linalg.norm(Xm[:3, 3] - X[:3, 3]) * 1000
        r["vs_chosen"] = {"rot_deg": d_rot, "trans_mm": float(d_t)}
        print(f"{m + ('*' if m == chosen else ''):<11} {res['corner_rms_mm']:8.2f}mm {res['corner_max_mm']:5.1f}mm "
              f"{res['Y_rot_scatter_deg']['max']:6.2f}d {res['Y_trans_scatter_mm']['max']:6.1f}mm "
              f"{loo.get('rot_deg', {}).get('max', float('nan')):7.2f}d "
              f"{loo.get('trans_mm', {}).get('max', float('nan')):7.1f}mm "
              f"{d_rot:6.2f}d {d_t:6.1f}mm")

    best = results[chosen]
    print(f"\nChosen: {chosen} ({'auto: lowest corner RMS' if wanted == 'AUTO' else 'configured'})")
    print("T_base_camera =\n" + np.array2string(X, precision=5, suppress_small=True))
    print(f"camera position in base frame: {np.round(X[:3, 3], 4)} m; optical axis (z) in base: "
          f"{np.round(X[:3, 2], 3)}")
    Y = np.asarray(best["residuals"]["T_ee_board_mean"])
    print(f"board mount T_ee_board translation: {np.round(Y[:3, 3], 4)} m (sanity: should be near the gripper)")
    per = best["residuals"]["per_sample_corner_rms_mm"]
    print("per-sample corner RMS [mm]: " + ", ".join(f"#{i}:{v:.1f}" for i, v in zip(ids, per)))
    if best["outlier_samples"]:
        print(f"! outlier samples {best['outlier_samples']}: inspect with review, then rerun with --exclude")
    spread = max(r["vs_chosen"]["trans_mm"] for r in results.values())
    if spread > 10:
        print(f"! solvers disagree by up to {spread:.0f} mm: motion diversity is probably insufficient")

    out = args.out or session.root / "T_base_camera.json"
    with open(out, "w") as f:
        json.dump({
            "T_base_camera": X.tolist(),
            "T_ee_board": Y.tolist(),
            "method": chosen,
            "method_selection": "auto (lowest corner RMS)" if wanted == "AUTO" else "configured",
            "convention": "column vectors; T_destination_source maps source coords into destination; metres. "
                          "X @ C_i = E_i @ Y; use T_base_object = T_base_camera @ T_camera_object",
            "camera_frame": smeta["camera"].get("frame", "D405 color optical frame"),
            "valid_for": "camera arm parked at the recorded pose; D405 color optical frame (any color resolution "
                         "of the same sensor; use that stream's own intrinsics)",
            "camera_arm_joint_positions_rad": smeta["camera_arm"]["joint_positions_rad"],
            "samples_used": ids,
            "excluded": sorted(exclude),
            "diversity": div,
            "results": results,
            "session": str(session.root.resolve()),
            "session_meta": smeta,
            "board": board_metadata(cfg),
            "calibration_opencv_version": cv2.__version__,
            "collection_opencv_version": smeta.get("opencv_version"),
            "redetected": args.redetect,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
