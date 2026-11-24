"""Upstream SAM 3D Objects on each view of a session alone, with the D405 depth as the pointmap.

Called by reconstruct.py; it must run with upstream's sam3d_objects (no MV-SAM3D on PYTHONPATH)
and from the sam-3d-objects checkout (checkpoints/hf, notebook/inference.py):

    cd ~/research/sam-3d-objects && /opt/conda-envs/sam3d/bin/python \
        <this file> SESSION OUT_DIR --views 0 1 2 --seed 42

Writes OUT_DIR/sam3d_view<i>/result.glb, result.ply, params.npz in the layout MV-SAM3D's
run_inference_weighted.py uses, so compare.py reads both the same way. Inference uses
notebook/inference.py's settings (vertex colours, no mesh postprocess or texture baking, layout
post-optimization on, the pipeline's default step counts unless --stage1-steps/--stage2-steps).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append("notebook")
from inference import Inference  # noqa: E402
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera  # noqa: E402


def pytorch3d_pointmap(view: Path) -> np.ndarray:
    """HxWx3 float32 in PyTorch3D camera axes (upstream does not flip a passed pointmap), NaN = no depth."""
    depth = np.load(view / "depth.npy").astype(np.float32)
    K = np.array(json.loads((view / "camera_intrinsics.json").read_text())["K"])
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = np.where(depth > 0, depth, np.nan)
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return (pts @ camera_to_pytorch3d_camera().rotation[0].numpy()).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--views", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    args = parser.parse_args()

    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
    for i in args.views:
        view = args.session / f"view_{i}"
        rgb = cv2.cvtColor(cv2.imread(str(view / "rgb.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(view / "obj1/mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        start = time.time()
        o = inference._pipeline.run(
            inference.merge_mask_to_rgba(rgb, mask), None, args.seed, stage1_only=False,
            with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=True,
            use_vertex_color=True, stage1_inference_steps=args.stage1_steps,
            stage2_inference_steps=args.stage2_steps, pointmap=torch.from_numpy(pytorch3d_pointmap(view)))
        seconds = time.time() - start
        out = args.out / f"sam3d_view{i}"
        out.mkdir(parents=True, exist_ok=True)
        o["glb"].export(str(out / "result.glb"))
        o["gs"].save_ply(str(out / "result.ply"))
        np.savez(out / "params.npz", **{k: o[k].detach().cpu().numpy() for k in ("rotation", "translation", "scale", "coords")})
        (out / "meta.json").write_text(json.dumps({"method": "sam3d", "views": [i], "reference_view": i,
                                                   "seed": args.seed, "inference_s": round(seconds, 1)}, indent=2))
        print(f"sam3d view {i}: {o['coords'].shape[0]} voxels, {seconds:.1f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
