"""Run MV-SAM3D on all of a session's views and upstream SAM 3D on each view alone, then compare.

Both use the D405 depth instead of a depth model, the same seed and the same step counts.

- MV-SAM3D: its own run_inference_weighted.py (default weighting: stage 1 and stage 2 entropy),
  from ~/research/MV-SAM3D with that checkout on PYTHONPATH so its sam3d_objects fork shadows the
  upstream one. Its Depth Anything 3 input, da3_output.npz, is written from the D405 depth and
  T_base_camera instead (extrinsics are world-to-camera with the manipulator base as the world;
  the default entropy weighting does not use them). The pose it outputs is in view 0's camera.
- SAM 3D: single_view.py in a separate process with upstream's package (the two packages share
  a name, so they cannot load in one process).

Session folder (from capture.py), plus what this adds:

    <session>/view_<i>/...                    the captures
    <session>/mvsam3d_input/images/<i>.png    MV-SAM3D's input layout, with da3_output.npz
    <session>/mvsam3d_input/object/<i>.png    RGBA, alpha = mask
    <session>/results/mvsam3d/                result.glb, result.ply, params.npz, meta.json, inference.log
    <session>/results/sam3d_view<i>/          the same per single view
    <session>/comparison.png, metrics.json    from compare.py

Run in the sam3d env (any cwd):

    /opt/conda-envs/sam3d/bin/python vision/experiments/sam3d_multiview/reconstruct.py \
        vision/foundationpose_runs/mv_jenga1 [--seed 42] [--views 0 1 2]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[3]
SAM3D_DIR = RESEARCH / "sam-3d-objects"
MVSAM3D_DIR = RESEARCH / "MV-SAM3D"
SAM3D_PYTHON = "/opt/conda-envs/sam3d/bin/python"
HF_HUB_CACHE = "/bigdata/luka/models/hf"
TORCH_HOME = "/bigdata/luka/cache/torch"
OBJECT = "object"  # MV-SAM3D mask folder name


def masked_views(session: Path):
    views = sorted((int(p.name.split("_")[1]) for p in session.glob("view_*") if (p / "obj1/mask.png").is_file()))
    for p in session.glob("view_*"):
        if not (p / "obj1/mask.png").is_file():
            print(f"Skipping {p.name}: no obj1/mask.png")
    return views


def write_mvsam3d_input(session: Path, views, out: Path):
    """MV-SAM3D's scene layout and da3_output.npz from the D405 captures; image i is view views[i]."""
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / OBJECT).mkdir()
    pointmaps, extrinsics, intrinsics, files = [], [], [], []
    for i, v in enumerate(views):
        view = session / f"view_{v}"
        bgr = cv2.imread(str(view / "rgb.png"))
        mask = cv2.imread(str(view / "obj1/mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        depth = np.load(view / "depth.npy").astype(np.float32)
        K = np.array(json.loads((view / "camera_intrinsics.json").read_text())["K"], dtype=np.float32)
        T_base_camera = np.array(json.loads((view / "camera_pose.json").read_text())["T_base_camera"])
        image_path = out / "images" / f"{i}.png"
        cv2.imwrite(str(image_path), bgr)
        cv2.imwrite(str(out / OBJECT / f"{i}.png"), np.dstack([bgr, (mask * 255).astype(np.uint8)]))
        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]
        z = np.where(depth > 0, depth, np.nan)
        # OpenCV camera frame: the fork flips to PyTorch3D axes itself (upstream does not).
        pointmaps.append(np.stack([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], 0).astype(np.float32))
        extrinsics.append(np.linalg.inv(T_base_camera).astype(np.float32))
        intrinsics.append(K)
        files.append(str(image_path))
    np.savez(out / "da3_output.npz", pointmaps_sam3d=np.stack(pointmaps), extrinsics=np.stack(extrinsics),
             intrinsics=np.stack(intrinsics), image_files=np.array(files))


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_CACHE=HF_HUB_CACHE, TORCH_HOME=TORCH_HOME, **(extra or {}))
    return env


def run_mvsam3d(session: Path, views, args) -> None:
    scene = session / "mvsam3d_input"
    write_mvsam3d_input(session, views, scene)
    out = session / "results" / "mvsam3d"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "inference.log"
    cmd = [SAM3D_PYTHON, "run_inference_weighted.py", "--input_path", str(scene.resolve()), "--mask_prompt", OBJECT,
           "--da3_output", str((scene / "da3_output.npz").resolve()), "--seed", str(args.seed),
           "--stage1_steps", str(args.stage1_steps), "--stage2_steps", str(args.stage2_steps)]
    print(f"MV-SAM3D on views {views} (log: {log})", flush=True)
    start = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=MVSAM3D_DIR, env=clean_env({"PYTHONPATH": str(MVSAM3D_DIR)}),
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    seconds = time.time() - start
    match = re.findall(r"All output files saved to: (\S+)", log.read_text())
    if not match:
        raise RuntimeError(f"MV-SAM3D wrote no output; see {log}")
    result = MVSAM3D_DIR / match[-1]
    for name in ("result.glb", "result.ply", "params.npz"):
        shutil.copy2(result / name, out / name)
    (out / "meta.json").write_text(json.dumps({"method": "mvsam3d", "views": views, "reference_view": views[0],
                                               "seed": args.seed, "source": str(result),
                                               "wall_s_including_model_load": round(seconds, 1)}, indent=2))
    print(f"MV-SAM3D done in {seconds:.0f} s -> {out}", flush=True)


def run_single_views(session: Path, views, args) -> None:
    cmd = [SAM3D_PYTHON, str(HERE / "single_view.py"), str(session.resolve()), str((session / "results").resolve()),
           "--views", *map(str, views), "--seed", str(args.seed),
           "--stage1-steps", str(args.stage1_steps), "--stage2-steps", str(args.stage2_steps)]
    print(f"SAM 3D on each of views {views}", flush=True)
    subprocess.run(cmd, cwd=SAM3D_DIR, env=clean_env(), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--views", type=int, nargs="+", default=None, help="Default: every masked view")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=25, help="SAM 3D's pipeline default")
    parser.add_argument("--stage2-steps", type=int, default=25, help="SAM 3D's pipeline default")
    parser.add_argument("--skip", choices=("mvsam3d", "sam3d"), nargs="*", default=[],
                        help="Reuse existing results for these")
    args = parser.parse_args()

    views = args.views or masked_views(args.session)
    if len(views) < 2:
        parser.error(f"need at least 2 masked views, found {views}")
    if "mvsam3d" not in args.skip:
        run_mvsam3d(args.session, views, args)
    if "sam3d" not in args.skip:
        run_single_views(args.session, views, args)
    subprocess.run([SAM3D_PYTHON, str(HERE / "compare.py"), str(args.session.resolve())],
                   cwd=SAM3D_DIR, env=clean_env(), check=True)


if __name__ == "__main__":
    sys.exit(main())
