"""Refit the camera calibrations with joint offsets on the manipulator, to test sag_map's finding in 3D.

sag_map.py (docs/arm_joint_offsets.md, section 3) found that the arm's height error fits constant
joint offsets best: shoulder true = reading + 1.71 deg, elbow true = reading - 0.55 deg. A ruler only
sees height. The calibration sessions saw the ChArUco board on the manipulator's gripper in 3D at
70 (parked camera) and 83 (camera on the moving follower) poses, with the manipulator's joints. Their
fits assume the manipulator's FK is exact. Joint offsets would make the board's position through the
arm disagree with the camera by a pose-dependent amount that one rigid camera placement cannot absorb.

So this refits each calibration with offsets o on chosen manipulator joints as extra unknowns:

  parked   T_base_camera @ C_i       = FK(q_i + o) @ T_ee_board
  dual     Z @ F_i @ X_model @ C_i   = FK(q_i + o) @ T_ee_board    (X_model: the follower model's camera
                                                                     mount, as the current calibration)

minimising the detected corners' reprojection error (pixels, robust loss), as the calibration's own
refinement does. FK is the MuJoCo model's tool site (driver tool frame); its fixed offset to the
driver's recorded tool pose is absorbed by T_ee_board. Joint 0 is left out: a base-yaw offset is the
same as rotating the camera about the base z axis. Each variant is scored on the fit and by 10-fold
cross-validation (held-out corner error, px and mm), and the offsets get a 1-sigma uncertainty from the
fit's Jacobian.

The sessions are only read. Results go to sim2real/sag_test/results/refit_calibration_offsets.json.

Env vision (OpenCV, scipy, mujoco), from any directory:

  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/refit_calibration_offsets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

REAL_ROBOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REAL_ROBOT / "vision"))

from calibration.board import BoardConfig, board_corners_m, make_board  # noqa: E402
from calibration.calibrate_dual import FollowerModel, FOLLOWER_MODEL  # noqa: E402
from calibration.session import Session, load_sample, read_json  # noqa: E402
from calibration.transforms import inv_T, make_T, mean_T, transform_points  # noqa: E402

SESSIONS = REAL_ROBOT / "vision" / "calibration" / "data" / "sessions"
PARKED = SESSIONS / "20261009_combined"
DUAL = SESSIONS / "20261113_130351_dual"
OUTPUT = Path(__file__).resolve().parent / "results" / "refit_calibration_offsets.json"
SAG_MAP_OFFSETS_DEG = {1: 1.71, 2: -0.55}      # docs/arm_joint_offsets.md section 3 (true = reading + o)
VARIANTS = {"none": [], "shoulder": [1], "shoulder+elbow": [1, 2], "shoulder+elbow+wrist": [1, 2, 3],
            "joints 1-5": [1, 2, 3, 4, 5]}
LOSS, F_SCALE_PX = "soft_l1", 1.0
FOLDS = 10
# The parked follower joints (20261009_combined session.json camera_arm), filled in at load.
PARKED_FOLLOWER_Q = None


class ArmFK:
    """Tool-site pose of the WXAI MuJoCo model (the driver's tool frame for wxai_v0_base)."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(FOLLOWER_MODEL))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("ee_gripper_link").id

    def __call__(self, q) -> np.ndarray:
        d = self.data
        d.qpos[:] = 0
        d.qpos[:6] = q
        self.mj.mj_kinematics(self.model, d)
        return make_T(d.site_xmat[self.site].reshape(3, 3).copy(), d.site_xpos[self.site].copy())


def parked_follower_joints() -> np.ndarray:
    """The follower joints of the parked calibration (read only)."""
    meta = read_json(PARKED / "session.json")
    for key in ("camera_arm", "follower"):
        block = meta.get(key) or {}
        for name in ("arm_positions_rad", "joint_positions_rad", "joints_rad", "parked_joints_rad"):
            if name in block:
                return np.asarray(block[name][:6], dtype=float)
    raise SystemExit(f"no parked follower joints in {PARKED / 'session.json'}: keys {list(meta)}")


def board_points(session_dir: Path) -> np.ndarray:
    meta = read_json(session_dir / "session.json")["board"]
    names = {f.name for f in fields(BoardConfig)}
    return board_corners_m(make_board(BoardConfig(**{k: v for k, v in meta.items() if k in names})))


def load(session_dir: Path, fk: ArmFK, dual: bool) -> list[dict]:
    obs = []
    for d in Session(session_dir).sample_dirs():
        s = load_sample(d)
        det = s["detection"]
        if not det["accepted"] or det["T_camera_board"] is None:
            continue
        q = np.asarray(s["robot"]["state_after_image"]["arm_positions_rad"], dtype=float)
        E = np.asarray(s["robot"]["T_base_ee"])
        o = {"id": s["id"], "q": q, "E_driver": E, "K": np.asarray(s["intrinsics"]["K"]),
             "dist": np.asarray(s["intrinsics"]["dist"]), "px": np.asarray(det["charuco_corners_px"]),
             "ids": np.asarray(det["charuco_ids"]), "C": np.asarray(det["T_camera_board"])}
        if dual:
            o["F"] = np.asarray(s["follower"]["T_fbase_fee"])
            o["qf"] = np.asarray(s["follower"]["state_after_image"]["arm_positions_rad"], dtype=float)
            o["station"] = int(s["follower"]["station"])
        obs.append(o)
    # FK model against the driver's recorded tool pose: a constant offset is fine (T_ee_board takes it).
    D = [inv_T(fk(o["q"])) @ o["E_driver"] for o in obs]
    spread = max(np.linalg.norm(Di[:3, 3] - D[0][:3, 3]) for Di in D) * 1000
    print(f"{session_dir.name}: {len(obs)} samples; model FK vs driver tool pose: "
          f"{np.linalg.norm(D[0][:3, 3]) * 1000:.2f} mm, varying by {spread:.3f} mm across samples")
    if spread > 0.5:
        raise SystemExit("the model's tool frame does not track the driver's; FK would be wrong")
    return obs


def pose_vec(T):
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def vec_pose(v):
    return make_T(cv2.Rodrigues(np.asarray(v[:3], dtype=float))[0], v[3:6])


class Problem:
    """Parked: unknowns T_base_camera, T_ee_board, offsets. Dual: Z = T_base_fbase, T_ee_board, offsets."""

    def __init__(self, obs, board_pts, fk, joints, dual: bool, X_mount=None, fixed_off=None, follower_off=None):
        self.obs, self.pts, self.fk, self.joints, self.dual, self.X_mount = obs, board_pts, fk, joints, dual, X_mount
        self.fixed_off = np.zeros(6) if fixed_off is None else np.asarray(fixed_off, dtype=float)
        self.follower_off = np.zeros(6) if follower_off is None else np.asarray(follower_off, dtype=float)

    def follower_pose(self, o):
        """T_fbase_fee from the follower's joints on the model (plus any follower offsets)."""
        return self.fk(o["qf"] + self.follower_off)

    def camera_in_base(self, o, A):
        return A @ self.follower_pose(o) @ self.X_mount if self.dual else A

    def unpack(self, v):
        off = self.fixed_off.copy()
        off[self.joints] += v[12:]
        return vec_pose(v[:6]), vec_pose(v[6:12]), off

    def residual_px(self, o, A, Y, off):
        C = inv_T(self.camera_in_base(o, A)) @ self.fk(o["q"] + off) @ Y
        proj, _ = cv2.projectPoints(self.pts[o["ids"]], cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], o["K"], o["dist"])
        return proj.reshape(-1, 2) - o["px"]

    def corner_mm(self, o, A, Y, off):
        pts = self.pts[o["ids"]]
        through_arm = transform_points(self.fk(o["q"] + off) @ Y, pts)
        through_camera = transform_points(self.camera_in_base(o, A) @ o["C"], pts)
        return np.linalg.norm(through_arm - through_camera, axis=1) * 1000

    def start(self, obs):
        """No-offset start: camera from the per-sample hand-eye relation, averaged; then the mount."""
        A0 = getattr(self, "A0", None)
        Ys = [inv_T(self.fk(o["q"])) @ self.camera_in_base(o, A0) @ o["C"] for o in obs]
        return np.r_[pose_vec(A0), pose_vec(mean_T(Ys)), np.zeros(len(self.joints))]

    def solve(self, obs, v0):
        def f(v):
            A, Y, off = self.unpack(v)
            return np.concatenate([self.residual_px(o, A, Y, off).ravel() for o in obs])
        r = least_squares(f, v0, loss=LOSS, f_scale=F_SCALE_PX, x_scale="jac")
        return r


def initial_camera(problem: Problem, obs, session_dir: Path, dual: bool):
    """Start from the session's own calibration result (read only)."""
    if dual:
        cal = read_json(session_dir.parent / "20261113_130351_dual_model_all" / "T_follower_camera.json")
        return np.asarray(cal["T_base_fbase"])
    return np.asarray(read_json(session_dir / "T_base_camera.json")["T_base_camera"])


def evaluate(problem: Problem, v, obs) -> dict:
    A, Y, off = problem.unpack(v)
    px = np.concatenate([np.linalg.norm(problem.residual_px(o, A, Y, off), axis=1) for o in obs])
    mm = np.concatenate([problem.corner_mm(o, A, Y, off) for o in obs])
    return {"px_rms": float(np.sqrt(np.mean(px ** 2))), "mm_rms": float(np.sqrt(np.mean(mm ** 2))),
            "mm_median": float(np.median(mm))}


def offsets_sigma_deg(result, n_off: int) -> np.ndarray:
    """1-sigma of the offsets from the Jacobian at the solution, scaled by the residual variance."""
    J, r = result.jac, result.fun
    dof = max(len(r) - J.shape[1], 1)
    cov = np.linalg.pinv(J.T @ J) * (r @ r) / dof
    return np.degrees(np.sqrt(np.diag(cov)[-n_off:])) if n_off else np.array([])


def folds(obs, dual: bool):
    """10-fold split; for the dual session, by follower station when there are enough of them."""
    if dual:
        stations = sorted({o["station"] for o in obs})
        if len(stations) >= 4:
            return [[i for i, o in enumerate(obs) if o["station"] == s] for s in stations]
    idx = np.arange(len(obs))
    return [list(idx[k::FOLDS]) for k in range(FOLDS)]


def run_session(name, session_dir, fk, dual: bool) -> dict:
    obs = load(session_dir, fk, dual)
    pts = board_points(session_dir)
    X_mount = FollowerModel(FOLLOWER_MODEL).mount() if dual else None
    out = {"session": str(session_dir), "samples": len(obs), "variants": {}}
    split = folds(obs, dual)
    print(f"\n{name}: held-out by {'follower station' if dual and len(split) != FOLDS else f'{FOLDS}-fold'}"
          f" ({len(split)} folds)")
    print(f"  {'variant':<22}{'fit px':>8}{'fit mm':>8}{'held px':>9}{'held mm':>9}   offsets, true = reading + o (deg, 1 sigma)")
    for vname, joints in VARIANTS.items():
        prob = Problem(obs, pts, fk, joints, dual, X_mount)
        prob.A0 = initial_camera(prob, obs, session_dir, dual)
        res = prob.solve(obs, prob.start(obs))
        fit = evaluate(prob, res.x, obs)
        sig = offsets_sigma_deg(res, len(joints))
        held = []
        for test in split:
            train = [o for i, o in enumerate(obs) if i not in set(test)]
            r_k = prob.solve(train, res.x)
            held.append((evaluate(prob, r_k.x, [obs[i] for i in test]), len(test)))
        n = sum(k for _, k in held)
        held_px = float(np.sqrt(sum(h["px_rms"] ** 2 * k for h, k in held) / n))
        held_mm = float(np.sqrt(sum(h["mm_rms"] ** 2 * k for h, k in held) / n))
        _, _, off = prob.unpack(res.x)
        offs = {f"joint_{j}": {"deg": float(np.degrees(off[j])), "sigma_deg": float(s)} for j, s in zip(joints, sig)}
        out["variants"][vname] = {"joints": joints, "fit": fit, "held_out_px_rms": held_px, "held_out_mm_rms": held_mm,
                                  "offsets": offs, "solver_status": int(res.status)}
        text = ", ".join(f"j{j} {np.degrees(off[j]):+.2f}±{s:.2f}" for j, s in zip(joints, sig))
        print(f"  {vname:<22}{fit['px_rms']:8.2f}{fit['mm_rms']:8.2f}{held_px:9.2f}{held_mm:9.2f}   {text}")
    # The sag_map offsets applied to the recorded joints (not fitted), camera and board re-solved.
    o_sag = np.zeros(6)
    for j, deg in SAG_MAP_OFFSETS_DEG.items():
        o_sag[j] = np.radians(deg)
    cases = {"sag_map offsets, manipulator": (o_sag, None)}
    if dual:
        cases["sag_map offsets, both arms"] = (o_sag, o_sag)
    out["fixed_offsets"] = {}
    base = None
    for cname, (man, fol) in [("no offsets", (None, None)), *cases.items()]:
        prob = Problem(obs, pts, fk, [], dual, X_mount, fixed_off=man, follower_off=fol)
        prob.A0 = initial_camera(prob, obs, session_dir, dual)
        res = prob.solve(obs, prob.start(obs))
        fit = evaluate(prob, res.x, obs)
        held = []
        for test in split:
            train = [o for i, o in enumerate(obs) if i not in set(test)]
            r_k = prob.solve(train, res.x)
            held.append((evaluate(prob, r_k.x, [obs[i] for i in test]), len(test)))
        n = sum(k for _, k in held)
        held_px = float(np.sqrt(sum(h["px_rms"] ** 2 * k for h, k in held) / n))
        held_mm = float(np.sqrt(sum(h["mm_rms"] ** 2 * k for h, k in held) / n))
        A, Y, _ = prob.unpack(res.x)
        rec = {"fit": fit, "held_out_px_rms": held_px, "held_out_mm_rms": held_mm,
               "T_ee_board": Y.tolist(), "manipulator_offsets_deg": np.degrees(prob.fixed_off).tolist(),
               "follower_offsets_deg": np.degrees(prob.follower_off).tolist()}
        if dual:
            rec["T_base_fbase"] = A.tolist()
            rec["T_fee_camera"] = X_mount.tolist()
            q_park = PARKED_FOLLOWER_Q
            rec["T_base_camera_at_parked_follower"] = (A @ fk(q_park + prob.follower_off) @ X_mount).tolist()
        else:
            rec["T_base_camera"] = A.tolist()
        out["fixed_offsets"][cname] = rec
        if base is None:
            base = rec
        print(f"  {cname:<32}{fit['px_rms']:8.2f}{fit['mm_rms']:8.2f}{held_px:9.2f}{held_mm:9.2f}")
    out["original_calibration"] = (
        {"T_base_fbase": prob.A0.tolist()} if dual
        else {"T_base_camera": read_json(session_dir / "T_base_camera.json")["T_base_camera"]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=("parked", "dual"), default=None)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    fk = ArmFK()
    global PARKED_FOLLOWER_Q
    PARKED_FOLLOWER_Q = parked_follower_joints()
    result = {"sag_map_offsets_deg": SAG_MAP_OFFSETS_DEG, "loss": LOSS, "f_scale_px": F_SCALE_PX}
    if args.only in (None, "parked"):
        result["parked"] = run_session("parked camera (20261009_combined)", PARKED, fk, dual=False)
    if args.only in (None, "dual"):
        result["dual"] = run_session("camera on the follower (20261113_130351_dual)", DUAL, fk, dual=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
