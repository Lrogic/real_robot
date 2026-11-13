"""Solve the camera-on-follower calibration from a collect_dual session (offline).

    python -m calibration.calibrate_dual --session data/sessions/NAME_dual [--mount calibrate|model]
                                         [--exclude 3,7] [--holdout-every 5] [--no-station-check]
                                         [--compare PATH] [--out PATH]

Finds X = T_fee_camera (camera on the follower's tool frame), Z = T_base_fbase (follower base in the
manipulator base frame) and Y = T_ee_board (board in the gripper), so that for any follower pose

    T_base_camera = Z @ F(q) @ X,   F(q) = the follower driver's Cartesian pose (same end_effector config)

Model, start point and refinement are described in calibration/dual_solver.py.

--mount model takes X from the follower model (sim/mujoco/assets/robots/trossen_wxai/wxai_follower.xml:
the wrist_cam camera relative to the ee_gripper_link site, which is the driver's tool frame for the
wxai_v0_base end effector) and solves only Z and Y. On session 20261113_130351_dual the model mount
gave 5.75 mm rms at unseen follower poses against 6.44 mm with the mount solved: with the joints' play
in the data, the extra six parameters fit noise (docs/camera_calibration.md). The script checks the
model's site against the driver's tool poses in the session and stops if they disagree.

What it reports
    fit        reprojection (px) and board-corner disagreement (mm) on the samples used in the fit.
    held out   the same on every Nth sample, left out of the fit (--holdout-every, 0 to use all).
    stations   leave-one-station-out: refit without each follower pose, then test on it. This is
               the error to expect at a follower pose the calibration has not seen.
    parked     T_base_camera predicted at the October parked follower pose, against the October
               eye-to-hand result (--compare), as object shift at 0.3-0.5 m in front of the camera.
The corner disagreement (mm) is the closest number to a tracked object's error: the board corners
placed through the camera chain Z F X C against the manipulator's Y E.

Runs in the vision env: needs scipy and OpenCV (not the hand-eye functions), and MuJoCo for --mount model.
Writes <session>/T_follower_camera.json.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .board import BoardConfig, board_corners_m, board_metadata, load_config, make_board, same_board
from .check_depth import DEFAULT_CALIBRATION
from .dual_solver import Obs, initial_guess, initial_guess_fixed_mount, outliers, refine, sample_errors, summarize
from .session import Session, load_sample, read_json
from .transforms import inv_T, make_T, rotation_between_deg, transform_points

REAL_ROBOT = Path(__file__).resolve().parents[2]
FOLLOWER_MODEL = REAL_ROBOT / "sim" / "mujoco" / "assets" / "robots" / "trossen_wxai" / "wxai_follower.xml"
MODEL_SITE, MODEL_CAMERA = "ee_gripper_link", "wrist_cam"
MAX_TOOL_FRAME_MISMATCH = (0.05, 0.5)      # deg, mm: model site against the driver's tool pose


def load_obs(session: Session, exclude: set) -> list:
    obs, skipped = [], []
    for d in session.sample_dirs():
        s = load_sample(d)
        det = s["detection"]
        if s["id"] in exclude:
            continue
        if not det["accepted"] or det["T_camera_board"] is None:
            skipped.append(s["id"])
            continue
        obs.append(Obs(id=s["id"], station=int(s["follower"]["station"]),
                       E=np.asarray(s["robot"]["T_base_ee"]), F=np.asarray(s["follower"]["T_fbase_fee"]),
                       K=np.asarray(s["intrinsics"]["K"]), dist=np.asarray(s["intrinsics"]["dist"]),
                       corners_px=np.asarray(det["charuco_corners_px"]), corner_ids=np.asarray(det["charuco_ids"]),
                       C=np.asarray(det["T_camera_board"]),
                       meta={"follower_q": s["follower"]["state_after_image"]["arm_positions_rad"]}))
    if skipped:
        print(f"Skipping samples whose detection is rejected: {skipped}")
    return obs


class FollowerModel:
    """Forward kinematics of the follower's MuJoCo model: the tool site and the camera."""

    def __init__(self, path: Path, site: str = MODEL_SITE, camera: str = MODEL_CAMERA):
        import mujoco
        self.mujoco, self.path = mujoco, Path(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.site, self.camera = self.model.site(site).id, self.model.camera(camera).id
        self.site_name, self.camera_name = site, camera

    def _poses(self, q):
        m, d = self.model, self.data
        d.qpos[:] = 0
        d.qpos[:6] = q
        self.mujoco.mj_kinematics(m, d)
        self.mujoco.mj_camlight(m, d)
        T_site = make_T(d.site_xmat[self.site].reshape(3, 3).copy(), d.site_xpos[self.site].copy())
        # MuJoCo cameras look along -z with y up; the OpenCV optical frame looks along +z with y down.
        T_cam = make_T(d.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), d.cam_xpos[self.camera].copy())
        return T_site, T_cam

    def mount(self) -> np.ndarray:
        """T_fee_camera: the camera in the tool-site frame (a rigid offset, the same at any joints)."""
        T_site, T_cam = self._poses(np.zeros(6))
        return inv_T(T_site) @ T_cam

    def tool_frame_mismatch(self, obs) -> tuple:
        """Worst (deg, mm) between the model site at the recorded joints and the driver's tool pose."""
        rot, trans = [0.0], [0.0]
        for o in obs:
            D = inv_T(self._poses(o.meta["follower_q"])[0]) @ o.F
            rot.append(rotation_between_deg(np.eye(3), D[:3, :3]))
            trans.append(float(np.linalg.norm(D[:3, 3]) * 1000))
        return max(rot), max(trans)


def fit(obs, board_pts, X_fixed=None):
    if X_fixed is None:
        X, Z, Y, init_report = initial_guess(obs)
    else:
        X = X_fixed
        Z, Y, init_report = initial_guess_fixed_mount(obs, X)
    X, Z, Y, info = refine(obs, X, Z, Y, board_pts, fix_X=X_fixed is not None)
    return X, Z, Y, init_report, info


def station_check(obs, board_pts, X0, Z0, Y0, fix_X=False) -> dict:
    """Leave one station out: refit on the rest (from the full fit), test on the left-out station."""
    out = {}
    for k in sorted({o.station for o in obs}):
        train = [o for o in obs if o.station != k]
        test = [o for o in obs if o.station == k]
        if len({o.station for o in train}) < (1 if fix_X else 3):
            continue
        X, Z, Y, _ = refine(train, X0, Z0, Y0, board_pts, fix_X=fix_X)
        out[k] = summarize([sample_errors(o, X, Z, Y, board_pts) for o in test])
    return out


def compare_parked(X, Z, path: Path) -> dict | None:
    """T_base_camera at the October parked follower pose, predicted vs the October eye-to-hand result."""
    if not path or not Path(path).exists():
        return None
    old = read_json(path)
    try:
        F_park = np.asarray(old["session_meta"]["follower"]["profile"]["state"]["T_base_ee"])
    except (KeyError, TypeError):
        return None
    W_old = np.asarray(old["T_base_camera"])
    W_new = Z @ F_park @ X
    u, v, w = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(-0.08, 0.08, 5), np.linspace(0.3, 0.5, 3))
    pts = np.c_[u.ravel(), v.ravel(), w.ravel()]                       # points in front of the camera
    shift = transform_points(W_new, pts) - transform_points(W_old, pts)
    return {"calibration": str(path), "rotation_deg": rotation_between_deg(W_old[:3, :3], W_new[:3, :3]),
            "camera_translation_mm": float(np.linalg.norm(W_new[:3, 3] - W_old[:3, 3]) * 1000),
            "object_shift_mean_mm": float(np.linalg.norm(shift, axis=1).mean() * 1000),
            "object_shift_max_mm": float(np.linalg.norm(shift, axis=1).max() * 1000),
            "object_shift_base_mean_mm": (shift.mean(axis=0) * 1000).tolist(),
            "note": "F at the parked pose is the follower's idle reading from October; idle vs held moved "
                    "objects about 0.5 mm (check_follower_hold)"}


def fmt(s: dict) -> str:
    if not s:
        return "-"
    return (f"{s['samples']:3d} samples  reproj {s['reproj_rms_px']:.2f} px (max {s['reproj_max_px']:.2f})  "
            f"corners {s['corner_rms_mm']:.2f} mm (max {s['corner_max_mm']:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--mount", choices=["calibrate", "model"], default="calibrate",
                   help="calibrate: solve the camera mount X; model: take it from --follower-model and solve Z, Y only")
    p.add_argument("--follower-model", type=Path, default=FOLLOWER_MODEL, help="MuJoCo model for --mount model")
    p.add_argument("--exclude", default="", help="comma-separated sample IDs to leave out")
    p.add_argument("--holdout-every", type=int, default=5, help="hold out every Nth sample for testing (0: none)")
    p.add_argument("--no-station-check", action="store_true", help="skip the leave-one-station-out refits")
    p.add_argument("--compare", type=Path, default=DEFAULT_CALIBRATION, help="parked eye-to-hand T_base_camera.json")
    p.add_argument("--out", type=Path, default=None, help="default <session>/T_follower_camera.json")
    args = p.parse_args()

    config = load_config(args.config)
    session = Session(args.session)
    smeta = session.meta
    if smeta.get("kind") != "dual_arm":
        raise SystemExit(f"{args.session} is not a collect_dual session")
    cfg = BoardConfig.from_dict(smeta["board"])
    if not same_board(smeta["board"], board_metadata(BoardConfig.from_dict(config["board"]))):
        print("NOTE: config.yaml board differs from the session; using the session's board definition")
    board_pts = board_corners_m(make_board(cfg))
    exclude = {int(v) for v in args.exclude.split(",") if v.strip()}
    obs = load_obs(session, exclude)
    if args.holdout_every > 0:
        held = [o for i, o in enumerate(obs) if i % args.holdout_every == args.holdout_every - 1]
        held_ids = {o.id for o in held}
        used = [o for o in obs if o.id not in held_ids]
    else:
        held, used = [], obs
    stations = {}
    for o in used:
        stations[o.station] = stations.get(o.station, 0) + 1
    print(f"{len(obs)} samples, {len(used)} in the fit, {len(held)} held out; "
          f"fit samples per station: {dict(sorted(stations.items()))}")

    X_fixed, mount_info = None, {"source": "calibrated"}
    if args.mount == "model":
        model = FollowerModel(args.follower_model)
        rot, trans = model.tool_frame_mismatch(obs)
        if rot > MAX_TOOL_FRAME_MISMATCH[0] or trans > MAX_TOOL_FRAME_MISMATCH[1]:
            raise SystemExit(f"{model.path}: site {model.site_name} differs from the driver's tool pose by up to "
                             f"{rot:.3f} deg, {trans:.2f} mm; is the follower end effector {smeta['follower']['end_effector']}?")
        X_fixed = model.mount()
        mount_info = {"source": "model", "model": str(model.path), "site": model.site_name, "camera": model.camera_name,
                      "site_vs_driver_tool_max": {"rot_deg": rot, "trans_mm": trans}}
        print(f"Camera mount from {model.path.name} ({model.camera_name} in {model.site_name}); model site matches the "
              f"driver's tool pose to {rot:.3f} deg, {trans:.2f} mm. Solving the follower base and board mount only.")
    X, Z, Y, init_report, info = fit(used, board_pts, X_fixed)
    errs_fit = [sample_errors(o, X, Z, Y, board_pts) for o in used]
    errs_held = [sample_errors(o, X, Z, Y, board_pts) for o in held]
    per_station = {k: summarize([e for e in errs_fit + errs_held if e["station"] == k]) for k in sorted(stations)}
    bad = outliers(errs_fit + errs_held, config["calibration"]["outlier_corner_rms_mm"])
    loso = {} if args.no_station_check else station_check(used, board_pts, X, Z, Y, fix_X=X_fixed is not None)
    parked = compare_parked(X, Z, args.compare)

    print(f"\nrefinement: {info['message']} ({info['nfev']} evaluations)")
    print(f"fit       {fmt(summarize(errs_fit))}")
    print(f"held out  {fmt(summarize(errs_held))}")
    print("per station (all its samples, with the full fit):")
    for k, s in per_station.items():
        print(f"  {k:3d}  {fmt(s)}")
    if loso:
        print("leave one station out (the error at a follower pose the fit has not seen):")
        for k, s in loso.items():
            print(f"  {k:3d}  {fmt(s)}")
        print(f"  overall corners {np.sqrt(np.mean([s['corner_rms_mm'] ** 2 for s in loso.values()])):.2f} mm rms")
    if parked:
        print(f"October parked pose: predicted T_base_camera differs by {parked['rotation_deg']:.2f} deg, camera "
              f"{parked['camera_translation_mm']:.1f} mm; objects 0.3-0.5 m ahead shift "
              f"{parked['object_shift_mean_mm']:.1f} mm mean (max {parked['object_shift_max_mm']:.1f})")
    print(f"camera on the follower tool ({mount_info['source']}): {np.round(X[:3, 3] * 1000, 1).tolist()} mm; "
          f"follower base in the manipulator frame: {np.round(Z[:3, 3] * 1000, 1).tolist()} mm; "
          f"board in the gripper: {np.round(Y[:3, 3] * 1000, 1).tolist()} mm")
    if bad:
        print(f"! outlier samples {bad}: inspect them (board slipping? motion?), then rerun with --exclude")

    out = args.out or session.root / "T_follower_camera.json"
    with open(out, "w") as f:
        json.dump({
            "T_fee_camera": X.tolist(),
            "T_base_fbase": Z.tolist(),
            "T_ee_board": Y.tolist(),
            "mount": mount_info,
            "usage": "T_base_camera = T_base_fbase @ F @ T_fee_camera, F = the follower's driver Cartesian pose "
                     "(T_fbase_fee) with the same model and end_effector configuration; then "
                     "T_base_object = T_base_camera @ T_camera_object",
            "follower": smeta["follower"],
            "convention": smeta["convention"],
            "camera_frame": smeta["camera"].get("frame", "D405 color optical frame"),
            "fit": summarize(errs_fit), "held_out": summarize(errs_held), "per_station": per_station,
            "leave_one_station_out": loso, "parked_comparison": parked, "outlier_samples": bad,
            "per_sample": errs_fit + errs_held, "held_out_ids": [o.id for o in held],
            "initial_guess": init_report, "refinement": info,
            "excluded": sorted(exclude), "session": str(session.root.resolve()), "board": board_metadata(cfg),
            "opencv_version": cv2.__version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
