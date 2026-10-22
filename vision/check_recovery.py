"""Offline check of --recovery on a run folder's saved frame (GPU, no camera). Run in vision.

    python check_recovery.py --save-path ~/research/foundationpose_runs/undistort/run1

1. Silhouette convention: FP's centred mesh at pose_last vs the original mesh at the
   returned pose (must agree), vs the original mesh at center_pose (the wrong frame).
2. Every --fp-mode: silhouette IoU against the saved mask after registration and tracking.
3. register_one: restores state on unusable input, leaves other objects untouched.
4. SAM under FoundationPose's process-wide CUDA default tensor type, with and without
   the cpu_default_tensors guard.
5. End to end: SamRecovery with the real model and FP on the saved frame (duplicated as a
   second object). One object's FP pose is pushed off; it alone must trigger after
   failure_limit SAM observations, re-register from its SAM mask, and match again,
   while the SAM session keeps its frame numbering.
"""

import argparse
import contextlib
import io
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from fp_capture.cli import DEFAULT_FOUNDATIONPOSE_DIR
from fp_capture.recovery_config import load_recovery_config
from fp_capture.recovery_state import mask_iou
from fp_capture.run_folder import SavedObject, load_run


def duplicated_objects(run):
    objects = [o for o in run.objects if o.mesh_path is not None]
    if len(objects) == 1:
        # A second object, so ID mapping and per-object independence are exercised.
        objects.append(replace(objects[0], name="obj2"))
    return objects


def render_with(tracker, utils, mesh_tensors, pose, K, H, W):
    import torch
    _, depth, _ = utils.nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(
        pose, device="cuda", dtype=torch.float).reshape(1, 4, 4), glctx=tracker.objects[0].estimator.glctx,
        mesh_tensors=mesh_tensors)
    return (depth[0] > 0).cpu().numpy()


def check_convention(tracker, rgb, depth, K, mask):
    import Utils as utils
    H, W = depth.shape
    t = tracker.objects[0]
    est = t.estimator
    centred = tracker.render_silhouettes([0], K, H, W)[0]
    original_pose = (est.pose_last.reshape(4, 4) @ est.get_tf_to_centered_mesh()).cpu().numpy()
    original = render_with(tracker, utils, utils.make_mesh_tensors(est.mesh_ori), original_pose, K, H, W)
    wrong = render_with(tracker, utils, utils.make_mesh_tensors(est.mesh_ori), t.center_pose, K, H, W)
    out = {"centred_vs_original_pixels_differing": int((centred != original).sum()),
           "centred_area": int(centred.sum()),
           "iou_vs_saved_mask": mask_iou(centred, mask),
           "iou_center_pose_render_vs_correct": mask_iou(wrong, centred),
           "model_center_m": est.model_center.tolist()}
    print(f"[convention] {out}")
    return out


def check_modes(objects, run, fp_dir, iterations):
    from fp_capture.tracking import MultiObjectTracker
    rgb, depth, K = run.capture.rgb, run.capture.depth, run.capture.K
    out = {}
    with tempfile.TemporaryDirectory() as debug:
        for mode in ("sequential", "batched", "batched-render"):
            tracker = MultiObjectTracker(objects, fp_dir, debug, batched=mode == "batched",
                                         batched_render=mode == "batched-render")
            tracker.register(objects, rgb, depth, K, 5)
            after_register = tracker.render_silhouettes(range(len(objects)), K, *depth.shape)
            for _ in range(10):
                tracker.track(rgb, depth, K, iterations)
            after_track = tracker.render_silhouettes(range(len(objects)), K, *depth.shape)
            out[mode] = {"iou_after_register": [mask_iou(s, o.mask) for s, o in zip(after_register, objects)],
                         "iou_after_10_tracks": [mask_iou(s, o.mask) for s, o in zip(after_track, objects)]}
            print(f"[mode {mode}] {out[mode]}")
            if mode == "sequential":
                out["convention"] = check_convention(tracker, rgb, depth, K, objects[0].mask)
                out["register_one"] = check_register_one(tracker, rgb, depth, K, objects)
            del tracker
    return out


def check_register_one(tracker, rgb, depth, K, objects):
    from fp_capture.tracking import RegistrationError
    est0, est1 = tracker.objects[0].estimator, tracker.objects[1].estimator
    before0, before1 = est0.pose_last, est1.pose_last
    center0 = tracker.objects[0].center_pose.copy()
    failed = False
    try:
        tracker.register_one(0, rgb, np.zeros_like(depth), K, objects[0].mask, 5)
    except RegistrationError as error:
        failed = True
        message = str(error)
    unchanged = est0.pose_last is before0 and np.array_equal(tracker.objects[0].center_pose, center0)
    tracker.register_one(0, rgb, depth, K, objects[0].mask, 5)
    out = {"zero_depth_raises": failed, "zero_depth_message": message if failed else None,
           "state_unchanged_after_failure": unchanged,
           "success_updates_pose_last": est0.pose_last is not before0,
           "other_object_untouched": est1.pose_last is before1}
    print(f"[register_one] {out}")
    return out


def check_default_tensor_guard(cfg, rgb, mask):
    import torch
    from experiments.segmentation.models import load_model
    from experiments.segmentation.tracker import HfVideoTracker
    from fp_capture.recovery import cpu_default_tensors

    results = {}
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # as FoundationPose leaves it
    try:
        load_model(cfg.sam_model, cache_dir=cfg.hf_cache)
        results["load_under_fp_cuda_default"] = "ok"
    except Exception as error:
        results["load_under_fp_cuda_default"] = repr(error)[:200]
    with cpu_default_tensors():
        loaded = load_model(cfg.sam_model, cache_dir=cfg.hf_cache)
    load_result = results.pop("load_under_fp_cuda_default")
    for name, guarded in (("cpu_default", None), ("fp_cuda_default_guarded", True),
                          ("fp_cuda_default_unguarded", False)):
        torch.set_default_tensor_type(torch.FloatTensor if guarded is None else torch.cuda.FloatTensor)
        tracker = HfVideoTracker(cfg.sam_model, loaded=loaded)
        ctx = cpu_default_tensors() if guarded else contextlib.nullcontext()
        try:
            with ctx:
                tracker.start(rgb, mask)
                steps = [tracker.step(rgb) for _ in range(3)]
            results[name] = {"masks": np.stack([s.mask for s in steps]),
                             "logits": [s.presence_logit for s in steps]}
        except Exception as error:
            results[name] = {"error": repr(error)[:300]}
        still_cuda = torch.empty(0).is_cuda
        results[name]["default_after_is_cuda"] = still_cuda
    torch.set_default_tensor_type(torch.FloatTensor)
    ref = results["cpu_default"]
    out = {"load_under_fp_cuda_default": load_result}
    for name, r in results.items():
        entry = {"default_after_is_cuda": r["default_after_is_cuda"]}
        if "error" in r:
            entry["error"] = r["error"]
        else:
            entry["identical_to_cpu_default"] = bool(np.array_equal(r["masks"], ref["masks"])
                                                     and r["logits"] == ref["logits"])
        out[name] = entry
    print(f"[default tensor type] {out}")
    return out, loaded


def check_end_to_end(cfg, loaded, objects, run, fp_dir, iterations, offset_m):
    import torch
    from experiments.segmentation.tracker import HfVideoTracker
    from fp_capture.recovery import LiveFrame, SamRecovery
    from fp_capture.tracking import MultiObjectTracker

    rgb, depth, K = run.capture.rgb, run.capture.depth, run.capture.K
    with tempfile.TemporaryDirectory() as tmp:
        tracker = MultiObjectTracker(objects, fp_dir, tmp)
        tracker.register(objects, rgb, depth, K, 5)
        sam = HfVideoTracker(cfg.sam_model, loaded=loaded, sync_timing=False)
        log_path = Path(tmp) / "recovery_log.jsonl"
        recovery = SamRecovery(cfg, tracker, objects, 5, log_path, {"check": True}, sam=sam)
        recovery.initialize(rgb, objects, "saved frame")
        recovery.start_loop()
        recovery.schedule.period = 0.0  # SAM on every iteration for this check
        # Push only the last object's pose sideways; it must trigger, the others must not.
        victim = len(objects) - 1
        est = tracker.objects[victim].estimator
        est.pose_last = est.pose_last.clone()
        est.pose_last.reshape(4, 4)[0, 3] += offset_m
        history = []
        for n in range(cfg.failure_limit + 3):
            frame = LiveFrame(rgb, depth, K, n, n * 16.7, time.monotonic())
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                recovery.after_fp(frame, lambda _vis: None)
            history.append({"frame": n, "sam_frame_idx": sam._next_idx - 1,
                            "states": [(s.name, round(s.last_iou or -1, 3), s.mismatch_count, s.registrations)
                                       for s in recovery.states],
                            "printed": printed.getvalue().strip().splitlines()})
            # FP keeps tracking between SAM observations, but must not undo the offset here.
            if n == 0:
                after_offset_iou = recovery.states[victim].last_iou
        recovery.close()
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        regs = [e for e in events if e["event"] == "registration"]
        out = {
            "offset_m": offset_m,
            "victim": objects[victim].name,
            "victim_iou_after_offset": after_offset_iou,
            "registrations": [(e["name"], e["frame_number"], e["ok"], round(e["ms"])) for e in regs],
            "victim_final_iou": recovery.states[victim].last_iou,
            "others_registrations": [s.registrations for s in recovery.states[:victim]],
            "sam_frame_indices": [h["sam_frame_idx"] for h in history],
            "sam_storage": sam.storage_counts(),
            "history": history,
        }
        print(f"[end to end] " + json.dumps({k: v for k, v in out.items() if k != "history"}))
        for h in history:
            for line in h["printed"]:
                print("   ", line)
        del tracker
        torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--sam-model", default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--offset-m", type=float, default=0.05, help="Sideways FP pose error to inject")
    parser.add_argument("--output", type=Path, default=None, help="Write the report as JSON")
    args = parser.parse_args()

    cfg = load_recovery_config(None, sam_model=args.sam_model)
    run = load_run(args.save_path)
    objects = duplicated_objects(run)
    print(f"Run {args.save_path}: objects {[o.name for o in objects]}, undistorted {run.capture.undistorted}")
    report = {"run": str(args.save_path), "sam_model": cfg.sam_model}
    report["modes"] = check_modes(objects, run, DEFAULT_FOUNDATIONPOSE_DIR, args.iterations)
    report["default_tensor_type"], loaded = check_default_tensor_guard(cfg, run.capture.rgb, objects[0].mask)
    report["end_to_end"] = check_end_to_end(cfg, loaded, objects, run, DEFAULT_FOUNDATIONPOSE_DIR,
                                            args.iterations, args.offset_m)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
