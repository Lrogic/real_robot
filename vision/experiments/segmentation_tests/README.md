# SAM live tracking (stage 1)

Click-initialize one object on a frozen RealSense frame, then track with **stateful video memory** (no FoundationPose, no detectors, no auto re-init).

## Environment

| `--model` | Conda env | Default checkpoint |
|-----------|-----------|-------------------|
| `sam21-small` | `sam21` | `/bigdata/luka/models/sam2.1/sam2.1_hiera_small.pt` |
| `sam21-base-plus` | `sam21` | `/bigdata/luka/models/sam2.1/sam2.1_hiera_base_plus.pt` |
| `sam21-large` | `sam21` | `/bigdata/luka/models/sam2.1/sam2.1_hiera_large.pt` |
| `sam31` | `sam3` | `/bigdata/luka/models/sam3.1/sam3.1_multiplex.pt` |

SAM 2.1 uses `sam2.build_sam2_video_predictor` with the official 2.1 Hiera configs. SAM 3.1 uses `build_sam3_multiplex_video_predictor` (multiplex checkpoint with point prompts), not the SAM3 image builder or Hugging Face video class.

## Launch

```bash
cd /home/luka/research/real_robot/vision

# Picks sam21 vs sam3 from --model:
./experiments/segmentation_tests/run_sam_tracking.sh --model sam21-small

conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-small
conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-base-plus
conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-large

./experiments/segmentation_tests/run_sam_tracking.sh --model sam31
```

SAM 3.1 enables Flash Attention 3 only when `flash_attn_interface` is installed **and** the GPU is Hopper (compute **9.x**, e.g. H100). Other GPUs (Ampere/Ada, Blackwell, etc.) use `use_fa3=False` (PyTorch SDPA). Do not run `--model sam31` from the `sam21` env (use `run_sam_tracking.sh` or `conda activate sam3`).

Optional: `--checkpoint /path/to.pt`, `--width`, `--height`, `--fps`, `--tracking-fps`, `--undistort`, `--warmup-frames`, `--stats-every`.

`--tracking-fps` caps how often SAM runs on the live stream (default: same as `--fps`). Example: `--fps 60 --tracking-fps 15` runs inference every 4th camera frame; skipped frames still show the live image with the last mask held (`mask hold` in the status bar). Only frames that run inference are appended to tracker state.

## Interaction

Setup steps show a short hint bar in the window; tracking shows only the mask overlay (stats in the terminal).

1. Live feed → **i** or **Space** picks the current frame for clicks.
2. Left / right click = positive / negative; **Enter** starts tracking (keep the scene still while clicking).
3. **r** = new object, **q** = quit.

Video frame 0 is the frozen init frame (prompt + memory commit). Live camera frames append as frame 1, 2, … and are not re-prompted.

## Streaming adapters

- **SAM 2.1**: one new frame per step, then official `propagate_in_video(..., max_frame_num_to_track=1)` with default `offload_video_to_cpu` / `offload_state_to_cpu`. Pixel buffers and past outputs outside SAM2’s memory horizon are dropped after each step.
- **SAM 3.1**: appends one frame and calls multiplex `propagate_in_video` for that index only.

Skipped camera frame numbers are reported when inference falls behind the camera.

## Scores shown

- SAM 2.1: `object_score_logit` from `object_score_logits` (higher ⇒ model thinks object is present).
- SAM 3.1: `detector_score` (`out_probs`) and `sam2_tracker_score` (`out_sam2_probs`) when returned by the multiplex postprocessor.

Empty masks are shown as **no mask**; they are not treated as ground-truth absence.

## Manual test checklist

1. Normal object motion in view.
2. Camera motion (pan/tilt) while the object is visible.
3. Partial occlusion.
4. Full occlusion, then reveal.
5. Object leaves the field of view and returns.
6. Return near a similar-looking object without new clicks.

Recovery should rely on native tracking memory only.

## Structural checks (no camera)

```bash
cd /home/luka/research/real_robot/vision
conda run -n sam21 python -c "from experiments.segmentation_tests.trackers import build_tracker; print('sam21 import ok')"
conda run -n sam3 python -c "from experiments.segmentation_tests.trackers import build_tracker; print('sam3 import ok')"
conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --help
conda run -n sam3 python experiments/segmentation_tests/test_sam_tracking.py --help
```

GPU, RealSense, and full model load are not exercised by the import/`--help` checks.
