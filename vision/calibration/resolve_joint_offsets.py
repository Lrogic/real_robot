"""Re-solve the camera calibrations in use for arms that report corrected joints (joint_offsets.py).

No new capture: the same images and recorded joints, with each arm's poses taken from the MuJoCo
model at the corrected joints (true = reading + offset, docs/arm_joint_offsets.md). Writes two new
session folders beside the originals, which are only read:

  data/sessions/20261113_130351_dual_model_all_joint_offsets_both_arms/
      The two-arm (move mode) calibration, solved by calibrate_dual.py exactly as the one in use
      (20261113_130351_dual_model_all: camera mount from the follower model, all 83 samples), with
      --manipulator-offsets-deg and --follower-offsets-deg. T_follower_camera.json, session.json
      (copy), samples -> the original's.
  data/sessions/20261009_combined_joint_offsets_both_arms/
      The parked eye-to-hand calibration, re-solved with the manipulator's offsets (the follower does
      not move in it, so its offsets do not enter). T_base_camera.json and session.json carry the
      parked follower joints converted to corrected readings (new = old + offset), so the follower
      returns to the same physical pose. Solved by least squares on the corner reprojection, as
      calibrate_dual's refinement; the original came from OpenCV's SHAH hand-eye solve (calibrate.py,
      OpenCV 4 env), and without offsets the two agree to about 1 mm at the block.

Each folder gets a README.md saying what it is. Run from vision/ (env vision):

  python -m calibration.resolve_joint_offsets
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .board import BoardConfig, board_corners_m, make_board
from .calibrate_dual import FOLLOWER_MODEL, MAX_TOOL_FRAME_MISMATCH, FollowerModel
from .joint_offsets import OFFSETS_DEG, corrected_joints, offsets_rad
from .session import Session, load_sample, read_json
from .transforms import inv_T, make_T, mean_T, rotation_between_deg, transform_points

SESSIONS = Path(__file__).resolve().parent / "data" / "sessions"
PARKED_SRC = SESSIONS / "20261009_combined"
DUAL_SRC = SESSIONS / "20261113_130351_dual"
DUAL_IN_USE = SESSIONS / "20261113_130351_dual_model_all"
SUFFIX = "_joint_offsets_both_arms"
PARKED_OUT = SESSIONS / (PARKED_SRC.name + SUFFIX)
DUAL_OUT = SESSIONS / (DUAL_IN_USE.name + SUFFIX)
OFFSETS_ARG = ",".join(f"{j}:{d}" for j, d in OFFSETS_DEG.items())


def new_folder(path: Path, samples_from: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; remove it to re-solve")
    path.mkdir()
    for sub in ("samples", "rejected"):
        if (samples_from / sub).exists():
            os.symlink(os.path.relpath(samples_from / sub, path), path / sub)


def offsets_text() -> str:
    return ", ".join(f"joint_{j} (true = reading {'+' if d >= 0 else '-'} {abs(d):.2f} deg)"
                     for j, d in OFFSETS_DEG.items())


# ---------------------------------------------------------------- parked
def parked_obs(model: FollowerModel) -> list[dict]:
    obs, worst = [], (0.0, 0.0)
    for d in Session(PARKED_SRC).sample_dirs():
        s = load_sample(d)
        det = s["detection"]
        if not det["accepted"] or det["T_camera_board"] is None:
            continue
        q = np.asarray(s["robot"]["state_after_image"]["arm_positions_rad"], dtype=float)
        D = inv_T(model.site_pose(q)) @ np.asarray(s["robot"]["T_base_ee"])
        worst = (max(worst[0], rotation_between_deg(np.eye(3), D[:3, :3])),
                 max(worst[1], float(np.linalg.norm(D[:3, 3]) * 1000)))
        obs.append({"id": s["id"], "E": model.site_pose(corrected_joints(q)), "C": np.asarray(det["T_camera_board"]),
                    "K": np.asarray(s["intrinsics"]["K"]), "dist": np.asarray(s["intrinsics"]["dist"]),
                    "px": np.asarray(det["charuco_corners_px"]), "ids": np.asarray(det["charuco_ids"])})
    if worst[0] > MAX_TOOL_FRAME_MISMATCH[0] or worst[1] > MAX_TOOL_FRAME_MISMATCH[1]:
        raise SystemExit(f"model tool site differs from the recorded manipulator poses by {worst}")
    print(f"parked: {len(obs)} samples; model matched the recorded tool poses to {worst[0]:.3f} deg, {worst[1]:.2f} mm")
    return obs


def solve_parked(obs, pts, X0):
    Y0 = mean_T([inv_T(o["E"]) @ X0 @ o["C"] for o in obs])

    def unpack(v):
        return make_T(cv2.Rodrigues(v[:3])[0], v[3:6]), make_T(cv2.Rodrigues(v[6:9])[0], v[9:12])

    def res(v):
        X, Y = unpack(v)
        out = []
        for o in obs:
            C = inv_T(X) @ o["E"] @ Y
            proj, _ = cv2.projectPoints(pts[o["ids"]], cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], o["K"], o["dist"])
            out.append((proj.reshape(-1, 2) - o["px"]).ravel())
        return np.concatenate(out)

    v0 = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3], cv2.Rodrigues(Y0[:3, :3])[0].ravel(), Y0[:3, 3]]
    r = least_squares(res, v0, loss="soft_l1", f_scale=1.0, x_scale="jac")
    X, Y = unpack(r.x)
    px = np.linalg.norm(res(r.x).reshape(-1, 2), axis=1)
    mm = np.concatenate([np.linalg.norm(transform_points(o["E"] @ Y, pts[o["ids"]])
                                        - transform_points(X @ o["C"], pts[o["ids"]]), axis=1) * 1000 for o in obs])
    return X, Y, {"reproj_rms_px": float(np.sqrt(np.mean(px ** 2))), "corner_rms_mm": float(np.sqrt(np.mean(mm ** 2))),
                  "corner_max_mm": float(mm.max()), "samples": len(obs), "solver": r.message}


def convert_follower_joints(block: dict | None) -> None:
    if block and block.get("joint_positions_rad") is not None:
        block["joint_positions_rad_as_recorded"] = block["joint_positions_rad"]
        block["joint_positions_rad"] = corrected_joints(block["joint_positions_rad"]).tolist()


def write_parked(model: FollowerModel) -> Path:
    original = read_json(PARKED_SRC / "T_base_camera.json")
    session = read_json(PARKED_SRC / "session.json")
    pts = board_corners_m(make_board(BoardConfig.from_dict(session["board"])))
    obs = parked_obs(model)
    X, Y, quality = solve_parked(obs, pts, np.asarray(original["T_base_camera"]))
    X_old = np.asarray(original["T_base_camera"])
    block = np.array([0.42, 0.0, 0.0375, 1.0])
    shift = (X @ inv_T(X_old) @ block)[:3] - block[:3]
    print(f"parked: corner error {quality['corner_rms_mm']:.2f} mm rms, {quality['reproj_rms_px']:.2f} px; camera moved "
          f"{np.linalg.norm(X[:3, 3] - X_old[:3, 3]) * 1000:.1f} mm, {rotation_between_deg(X[:3, :3], X_old[:3, :3]):.2f} deg; "
          f"a block at 0.42 m now reads {np.round(shift * 1000, 1).tolist()} mm from before")

    new_folder(PARKED_OUT, PARKED_SRC)
    q_old = original["camera_arm_joint_positions_rad"]
    q_new = corrected_joints(q_old).tolist()
    for blk in (session.get("follower"), session.get("camera_arm")):
        convert_follower_joints(blk)
    session["joint_offsets_note"] = (f"Copied from {PARKED_SRC.name}; follower joint_positions_rad converted to corrected "
                                     f"readings (new = old + offset; {offsets_text()}), the original kept as "
                                     "joint_positions_rad_as_recorded.")
    (PARKED_OUT / "session.json").write_text(json.dumps(session, indent=2))

    out = dict(original)
    meta = json.loads(json.dumps(original["session_meta"]))
    for blk in (meta.get("follower"), meta.get("camera_arm")):
        convert_follower_joints(blk)
    out.update({
        "T_base_camera": X.tolist(), "T_ee_board": Y.tolist(),
        "method": "least squares on the ChArUco corner reprojection (soft_l1, 1 px), started from the SHAH result",
        "method_selection": "resolve_joint_offsets.py",
        "camera_arm_joint_positions_rad": q_new, "camera_arm_joint_positions_rad_as_recorded": q_old,
        "valid_for": ("camera arm parked at the recorded pose, with both arms reporting corrected joints (driver "
                      "position_offset set by calibration/joint_offsets.py); D405 color optical frame"),
        "joint_offsets": {"convention": "true joint angle = reading + offset", "manipulator_deg": OFFSETS_DEG,
                          "follower_deg": OFFSETS_DEG, "reference": "docs/arm_joint_offsets.md",
                          "follower_note": ("does not enter the solve (the follower is parked); its parked joints are "
                                            "converted so it returns to the same physical pose (its elbow target is "
                                            "below the old 0 deg limit; joint_offsets.py moves the limits with the "
                                            "offsets so it is reachable).")},
        "results": {"this_solve": quality,
                    "note": f"results of the original SHAH solve are in {PARKED_SRC.name}/T_base_camera.json"},
        "session": str(PARKED_OUT), "session_meta": meta, "source_calibration": str(PARKED_SRC / "T_base_camera.json"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    path = PARKED_OUT / "T_base_camera.json"
    path.write_text(json.dumps(out, indent=2))
    (PARKED_OUT / "README.md").write_text(f"""# {PARKED_OUT.name}

The parked eye-to-hand calibration of `{PARKED_SRC.name}`, re-solved for arms that report corrected
shoulder and elbow angles: {offsets_text()} (docs/arm_joint_offsets.md). Same images and joints
(`samples` links to the original); nothing re-captured.

- Valid only with both arms' `position_offset` set (`python -m calibration.joint_offsets --apply`).
  The calibration in `{PARKED_SRC.name}` is valid only without.
- The manipulator's poses come from the MuJoCo model at the corrected joints. Solved by least
  squares on the corner reprojection, started from the original SHAH result.
- The follower's parked joints in `session.json` and `T_base_camera.json` are converted to
  corrected readings (new = old + offset; originals kept as `*_as_recorded`). The elbow's converted
  target ({np.degrees(q_new[2]):+.2f} deg) is below the old 0 deg limit; `joint_offsets.py --apply`
  moves the limits with the offsets, so it is reachable.
- Fit: {quality['corner_rms_mm']:.2f} mm rms board-corner disagreement, {quality['reproj_rms_px']:.2f} px
  ({quality['samples']} samples). A block at 0.42 m reads {np.round(shift * 1000, 1).tolist()} mm
  (x, y, z) from the original calibration's reading.

Made by `vision/calibration/resolve_joint_offsets.py` on {time.strftime('%Y-%m-%d')}.
""")
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------- dual
def write_dual(parked_path: Path) -> Path:
    new_folder(DUAL_OUT, DUAL_SRC)
    shutil.copy2(DUAL_IN_USE / "session.json", DUAL_OUT / "session.json")
    out = DUAL_OUT / "T_follower_camera.json"
    cmd = [sys.executable, "-m", "calibration.calibrate_dual", "--session", str(DUAL_OUT), "--mount", "model",
           "--holdout-every", "0", "--manipulator-offsets-deg", OFFSETS_ARG, "--follower-offsets-deg", OFFSETS_ARG,
           "--compare", str(parked_path), "--out", str(out)]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    res = read_json(out)
    old = read_json(DUAL_IN_USE / "T_follower_camera.json")
    Z, Z_old = np.asarray(res["T_base_fbase"]), np.asarray(old["T_base_fbase"])
    loso = res.get("leave_one_station_out") or {}
    loso_mm = float(np.sqrt(np.mean([s["corner_rms_mm"] ** 2 for s in loso.values()]))) if loso else float("nan")
    loso_old = old.get("leave_one_station_out") or {}
    loso_old_mm = float(np.sqrt(np.mean([s["corner_rms_mm"] ** 2 for s in loso_old.values()]))) if loso_old else float("nan")
    tilt = np.degrees(np.arccos(np.clip(Z[2, 2], -1, 1)))
    (DUAL_OUT / "README.md").write_text(f"""# {DUAL_OUT.name}

The two-arm (move mode) calibration `{DUAL_IN_USE.name}` re-solved for arms that report corrected
shoulder and elbow angles: {offsets_text()}, on both arms (docs/arm_joint_offsets.md). Same session
`{DUAL_SRC.name}` (`samples` links to it), same settings: camera mount from the follower model, all
samples (`calibrate_dual.py --mount model --holdout-every 0`), plus `--manipulator-offsets-deg
{OFFSETS_ARG} --follower-offsets-deg {OFFSETS_ARG}`.

- Valid only with both arms' `position_offset` set (`python -m calibration.joint_offsets --apply`).
  `{DUAL_IN_USE.name}` is valid only without.
- Follower base in the manipulator frame: {np.round(Z[:3, 3] * 1000, 1).tolist()} mm, tilted
  {tilt:.2f} deg (in use: {np.round(Z_old[:3, 3] * 1000, 1).tolist()} mm,
  {np.degrees(np.arccos(np.clip(Z_old[2, 2], -1, 1))):.2f} deg). Measured on the table: about 970 mm
  apart, level, same height.
- Fit {res['fit']['corner_rms_mm']:.2f} mm rms (in use {old['fit']['corner_rms_mm']:.2f}); leave one
  follower pose out {loso_mm:.2f} mm (in use {loso_old_mm:.2f}).

Made by `vision/calibration/resolve_joint_offsets.py` on {time.strftime('%Y-%m-%d')}.
""")
    print(f"follower base {np.round(Z[:3, 3] * 1000, 1).tolist()} mm, tilt {tilt:.2f} deg; leave one station out "
          f"{loso_mm:.2f} mm (in use {loso_old_mm:.2f})")
    return out


def main() -> None:
    print(f"Offsets: {offsets_text()}, both arms")
    model = FollowerModel(FOLLOWER_MODEL)
    parked = write_parked(model)
    write_dual(parked)


if __name__ == "__main__":
    main()
