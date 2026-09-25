"""Offline check of --fp-mode batched-render against FoundationPose's own code.

Uses a run folder's saved frame (no camera). Registers its objects, then:
  1. crops: one batched-render call vs FoundationPose's make_crop_data_batch per
     object, on a mix of meshes (run meshes, a retextured copy, a vertex-coloured box)
  2. poses: sequential vs batched vs batched-render from the same registered state
  3. timing: all three modes for 1-8 objects

    python check_batched_render.py --save-path ~/research/foundationpose_runs/run7
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from fp_capture.cli import DEFAULT_FOUNDATIONPOSE_DIR
from fp_capture.run_folder import load_run
from fp_capture.tracking import MultiObjectTracker


def timed(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


def pose_error(a, b):
    translation_mm = np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return translation_mm, np.degrees(np.arccos(cos))


def extra_estimators(template, estimater):
    """A retextured copy of the first mesh and a vertex-coloured box, sharing its networks."""
    base = template.mesh_ori
    image = np.asarray(base.visual.material.image.convert("RGB"))
    recoloured = Image.fromarray(255 - image[::-1]).resize((768, 512))
    retextured = trimesh.Trimesh(base.vertices * 0.8, base.faces, process=False,
                                 visual=trimesh.visual.TextureVisuals(uv=base.visual.uv,
                                                                      image=recoloured))
    box = trimesh.creation.box(extents=(0.06, 0.03, 0.02))
    box.visual.vertex_colors = np.random.default_rng(0).integers(0, 255, (len(box.vertices), 4))
    made = []
    for mesh in (retextured, box):
        made.append(estimater.FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=template.scorer, refiner=template.refiner, glctx=template.glctx,
            debug=0, debug_dir=template.debug_dir))
    return made


def check_crops(estimators, poses, rgb_t, depth_t, xyz_map, K):
    import learning.training.predict_pose_refine as fp_refine
    from fp_capture.batched_render import MeshBank, make_crops_batched

    refiner, cfg = estimators[0].refiner, estimators[0].refiner.cfg
    bank = MeshBank(estimators)
    ours = make_crops_batched(refiner, bank, estimators[0].glctx, poses, rgb_t, depth_t, xyz_map, K)
    worst = {}
    for i, est in enumerate(estimators):
        ref = fp_refine.make_crop_data_batch(
            cfg.input_resize, poses[i:i + 1], est.mesh, rgb_t, depth_t, K,
            crop_ratio=cfg["crop_ratio"], normal_map=None, xyz_map=xyz_map, cfg=cfg,
            glctx=est.glctx, mesh_tensors=est.mesh_tensors, dataset=refiner.dataset,
            mesh_diameter=est.diameter)
        for field in ("rgbAs", "xyz_mapAs", "rgbBs", "xyz_mapBs", "tf_to_crops"):
            diff = (getattr(ours, field)[i:i + 1] - getattr(ref, field)).abs().max().item()
            worst[field] = max(worst.get(field, 0.0), diff)
        coverage = (ref.rgbAs.abs().sum(1) > 0).float().mean().item()
        print(f"  object {i} ({'texture' if 'tex' in est.mesh_tensors else 'vertex colour'}, "
              f"diameter {est.diameter * 100:.1f} cm): rendered pixels {coverage:.0%}")
    for field, diff in worst.items():
        print(f"  max abs difference {field:12s} {diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    run = load_run(args.save_path)
    objects = [o for o in run.objects if o.mesh_path is not None]
    rgb, depth, K = run.capture.rgb, run.capture.depth, run.capture.K
    with tempfile.TemporaryDirectory(prefix="fp_check_") as debug_dir:
        tracker = MultiObjectTracker(objects, DEFAULT_FOUNDATIONPOSE_DIR, debug_dir)
        import estimater
        import Utils as fp_utils
        from fp_capture.batched_refine import track_batched
        from fp_capture.batched_render import MeshBank, track_batched_render

        tracker.register(objects, rgb, depth, K, 5)
        estimators = [t.estimator for t in tracker.objects]
        registered = [e.pose_last.clone() for e in estimators]

        print("\n1. Crops: one batched-render call vs make_crop_data_batch per object")
        extras = extra_estimators(estimators[0], estimater)
        mixed = estimators + extras
        offsets = [np.eye(4) for _ in mixed]
        for i, offset in enumerate(offsets[len(estimators):]):
            offset[:3, :3] = trimesh.transformations.euler_matrix(0.4 * (i + 1), 0.7, 0.2)[:3, :3]
            offset[0, 3] = 0.03 * (i + 1)
        poses = torch.stack([registered[i % len(registered)].reshape(4, 4) @
                             torch.as_tensor(offsets[i], dtype=torch.float, device="cuda")
                             for i in range(len(mixed))])
        depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth_t = fp_utils.bilateral_filter_depth(fp_utils.erode_depth(depth_t, radius=2, device="cuda"),
                                                  radius=2, device="cuda")
        xyz_map = fp_utils.depth2xyzmap_batch(
            depth_t[None], torch.as_tensor(K, dtype=torch.float, device="cuda")[None], zfar=np.inf)[0]
        rgb_t = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        print(" run meshes only:")
        check_crops(estimators, poses[:len(estimators)], rgb_t, depth_t, xyz_map, K)
        print(" mixed meshes:")
        check_crops(mixed, poses, rgb_t, depth_t, xyz_map, K)

        bank = MeshBank(estimators)
        modes = {
            "sequential": lambda ests, b: [e.track_one(rgb=rgb, depth=depth, K=K,
                                                       iteration=args.iterations) for e in ests],
            "batched": lambda ests, b: track_batched(ests, rgb, depth, K, args.iterations),
            "batched-render": lambda ests, b: track_batched_render(ests, b, rgb, depth, K,
                                                                   args.iterations),
        }

        def reset():
            for est, pose in zip(estimators, registered):
                est.pose_last = pose.clone()

        print("\n2. Poses vs sequential, from the same registered state")
        for steps in (1, 30):
            results = {}
            for name, step in modes.items():
                reset()
                for _ in range(steps):
                    out = step(estimators, bank)
                results[name] = out
            for name in ("batched", "batched-render"):
                errors = [pose_error(a, b) for a, b in zip(results["sequential"], results[name])]
                print(f"  {steps:2d} steps, {name:14s}: max {max(e[0] for e in errors):.3f} mm, "
                      f"{max(e[1] for e in errors):.3f} deg")

        print(f"\n3. Timing per frame ({args.iterations} refinement iterations)")
        print("  objects | sequential | batched  | batched-render")
        for n in (1, 2, 4, 8):
            subset = [estimators[i % len(estimators)] for i in range(n)]
            subset_bank = MeshBank(subset)
            row = []
            for name, step in modes.items():
                reset()
                row.append(timed(lambda: step(subset, subset_bank), 100 if n <= 2 else 40))
            print(f"  {n:7d} | {row[0]:7.2f} ms | {row[1]:6.2f} ms | {row[2]:6.2f} ms")
        mixed_bank = MeshBank(mixed)
        for est, pose in zip(mixed, poses):
            est.pose_last = pose[None].clone()
        row = [timed(lambda: step(mixed, mixed_bank), 100) for step in modes.values()]
        print(f"  {len(mixed)} mixed | {row[0]:7.2f} ms | {row[1]:6.2f} ms | {row[2]:6.2f} ms"
              "   (run meshes + retextured copy + vertex-coloured box)")


if __name__ == "__main__":
    main()
