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

conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-small
conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-base-plus
conda run -n sam21 python experiments/segmentation_tests/test_sam_tracking.py --model sam21-large

conda run -n sam3 python experiments/segmentation_tests/test_sam_tracking.py --model sam31
```

Optional: `--checkpoint /path/to.pt`, `--width`, `--height`, `--fps`, `--undistort`, `--warmup-frames`, `--stats-every`.

## Interaction

1. Live preview → **Space** freezes a frame.
2. On the frozen frame: left = positive, right = negative, **r** = clear prompts, **Enter** = start tracking (stay still while clicking).
3. Tracking: translucent mask + contour; **r** = new trial, **q** / Esc = quit.

Video frame 0 is the frozen init frame (prompt + memory commit). Live camera frames append as frame 1, 2, … and are not re-prompted.

## Streaming adapters

- **SAM 2.1**: grows `inference_state["images"]` and runs `SAM2VideoPredictor._run_single_frame_inference` per new frame (same memory path as `propagate_in_video`).
- **SAM 3.1**: appends preprocessed frames to the multiplex `input_batch` and calls `propagate_in_video` for one frame index at a time.

Algorithmic tracking memory is preserved; camera frames are not queued (skipped frame numbers are reported).

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
