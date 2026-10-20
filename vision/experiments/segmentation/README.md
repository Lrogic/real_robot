# HF SAM video-tracker comparison

Compares Hugging Face Transformers SAM video trackers on identical inputs, live from the
RealSense or on recorded video. One object is selected once on frame 0; after that each
model tracks on its own with one persistent streaming session. There are no further clicks,
detections, re-initializations, or resets. FoundationPose is not involved yet.

| `--model`    | Checkpoint                         | Classes |
|--------------|------------------------------------|---------|
| `sam2-small` | `facebook/sam2.1-hiera-small`      | `Sam2VideoModel` / `Sam2VideoProcessor` |
| `sam2-base`  | `facebook/sam2.1-hiera-base-plus` (SAM 2.1 Base+) | `Sam2VideoModel` / `Sam2VideoProcessor` |
| `sam3`       | `facebook/sam3` (tracker weights only) | `Sam3TrackerVideoModel` / `Sam3TrackerVideoProcessor` |

Weights load offline from the HF cache `/bigdata/luka/models/hf`, pinned to the cached
`refs/main` revision, which is recorded in every summary. All modes use float32 weights,
CUDA bfloat16 autocast, `torch.inference_mode()`, and float32 session frames.

## Environment

```bash
conda activate sam-video-hf
cd /home/luka/research/real_robot/vision     # all commands below run from here
```

## Live tracking

```bash
python -m experiments.segmentation.live --model sam2-small
python -m experiments.segmentation.live --model sam3 --tracking-fps 10 --camera-fps 30 --width 640 --height 480
```

1. Space freezes the current camera frame as the init frame.
2. Left click marks the object, right click marks background, and left drag draws a box.
   `u` undoes and `r` resets. The previewed mask updates after each click.
3. Enter confirms the mask. It is refused while there is no non-empty mask. Tracking then
   starts from a fresh session conditioned on the confirmed mask. Previews run in
   throwaway sessions, so they never advance the trial.
4. While tracking, the window shows each processed camera frame with its predicted mask,
   the model name, and the presence score. `q` or Esc quits; `--num-seconds` stops automatically.

RGB is used as delivered by the camera, without undistortion, just like the recording
helper. The camera runs through `fp_capture.camera.RealSenseStream` (color plus aligned depth;
depth is unused) on a background thread that keeps only the newest frame.

## Recording a video and creating its init mask

```bash
python -m experiments.segmentation.record                       # 640x480 @ 30, mask with sam3
python -m experiments.segmentation.record --width 848 --height 480 --camera-fps 60 --mask-model sam2-base
```

Space starts recording, and Space/q/Esc stops it. The output folder
`experiments/segmentation/recordings/<time>/` contains:

- `video.mkv`: lossless FFV1 with no annotations.
- `recording.json`: settings, frame counts, measured FPS, and number of skipped camera frames.
- `frames.csv`: per-frame RealSense frame number, camera timestamp, host time, and skipped frames.

The saved file is then reopened and checked: the decoded frame count must match, and the
first decoded frame must be bit-identical to the capture. That first decoded frame is used
for click selection. The confirmed mask is saved next to the video (see [Init mask](#init-mask)).
Use `--no-mask` to skip mask creation.

## Recorded-video tracking

```bash
# one model -> runs/replay_<video>_<time>/annotated_sam2-small.mp4
python -m experiments.segmentation.replay --video experiments/segmentation/recordings/<time>/video.mkv --model sam2-small

# all models, sequentially, one process each -> runs/.../comparison.mp4 (one panel per model)
python -m experiments.segmentation.replay --video .../video.mkv --model all
python -m experiments.segmentation.replay --video .../video.mkv --model all --tracking-fps 10
```

Frames are decoded one at a time and passed individually to the same `HfVideoTracker` used
live. There is no preloaded video, no offline propagation API, no backward pass, and no
access to future frames. Processing never sleeps to match playback speed. With
`--model all`, each model runs in its own spawned process, one after another, so GPU
memory is fully released between models. Each worker writes its predictions (bit-packed
host masks and presence logits) to disk. The panel video is built only after every model
has finished.

If the video has no init mask, the first decoded frame is shown for selection with
`--mask-model`, which defaults to `sam3`. The mask is saved and then reused unchanged by
every model; each worker records the mask's sha256. A stale or mismatched mask is
rejected; `--recreate-mask` replaces it interactively.

## Tracking FPS and subsampling

- **Live** (`--tracking-fps`, default 30, must be ≤ `--camera-fps`): at most this many
  requests per second. Each request takes the newest camera frame not yet processed, so
  nothing queues and no frame is processed twice. If no fresh frame is ready, the loop
  polls the UI and waits. `skipped` is the gap in RealSense frame numbers since the previous
  processed frame. It includes frames passed over on purpose at lower tracking rates and
  frames missed while inference was running. `frames_missed_before_reader` in the summary
  counts frame numbers that never reached the process. The first request's gap covers
  warm-up and prompting after the frozen frame, so it is reported separately as
  `camera_frames_between_init_and_first_track`; keep the scene still until tracking starts.
- **Recorded** (`--tracking-fps`, default every frame): source frame *i* has timestamp
  *i* / source_fps, taken from the container. The frame processed for tick *k* is the first
  source frame at or after *k* / tracking_fps, and frame 0 is always processed. The selection
  depends only on frame indices and the two rates, so every model gets the same frames.
  Example: at 30 → 10 FPS the processed frames are 0, 3, 6, …
- **Playback of outputs**: videos play at the source FPS with one output frame per source
  frame. Each panel shows the newest prediction at or before that frame and is labelled
  `inferred`, `held from src N`, or `init`.

## Init mask

The mask is saved next to the video as three files:

- `<video>.init_mask.png`: one-channel uint8 0/255 at the first frame's resolution.
- `<video>.init_mask.json`: the sidecar.
- `<video>.init_mask_overlay.png`: a preview image.

The sidecar records `frame_index` 0, height and width, the mask's sha256, and the model used
to create the mask, with its revision. To identify the source video it stores the file
sha256, file size, sha256 of the first decoded frame, FPS, and container frame count.
Loading checks all of these and that the mask is strictly binary. The mask is never resized.

## Output files

Run folders go under `experiments/segmentation/runs/`, which is git-ignored:

| File | Content |
|------|---------|
| `frames.csv` (live) or `<model>/frames.csv` (replay) | One row per request: `kind`, `processed_idx`, `source_idx` (video index, or RealSense frame number when live), `tracking_ms`, `presence`, `presence_logit`, `mask_area`, CUDA allocated/reserved/peak, CPU RSS, session storage counts, camera timestamp, skipped frames. |
| `summary.json` | Model ID and revision, classes, precision, software and GPU versions, camera or video settings, frame-selection policy, warm-up policy, peak-reset point, timing and memory definitions, per-model statistics. Replay adds the worker timeline (PIDs, start/stop times) and checks for sequential execution and identical frame indices. |
| `<model>/predictions.npz` | Replay only: `source_idx`, `processed_idx`, `presence_logit`, `masks_packed`, `mask_shape`. |
| `annotated_<model>.mp4` / `comparison.mp4` | Replay only: annotated video or model panels. |
| `init_frame.png`, `init_mask*.png` | Live only: the frozen frame and the confirmed mask. |

At exit, each model prints: the number of timed requests, mean, median, p90, and max
tracking time, processed FPS, skipped camera frames (live), init time, number of
empty-mask requests, and first/last memory. Processed FPS is timed requests divided by
the wall time of the tracking loop, which includes waiting, decoding, UI, and logging.

## Timing

`tracking_ms` covers one tracking request. The clock (`time.perf_counter()`) starts after
`torch.cuda.synchronize()` with the RGB uint8 frame in host memory. The interval contains:

1. HF preprocessing: upload, resize to 1024 (SAM 2.1) or 1008 (SAM 3), normalize. This runs on the GPU.
2. One streamed `model(inference_session, frame_idx, frame)` call.
3. Mask upsampling to the input size and thresholding at 0.
4. Session pruning (below).
5. The blocking host copy of the mask and the object-score logit, followed by a final `torch.cuda.synchronize()`.

Excluded: camera waiting, decoding, prompting and previews, initialization, warm-up, UI
rendering, terminal logging, and video writing. The frame-0 init request runs the same path
but is reported separately (`init_ms`) and is not part of the statistics.

**Warm-up:** after the model loads, a disposable session runs on the init frame with a
synthetic centered-box mask: one init request plus `--warmup-steps` tracking requests
(default 5). All of these are excluded. The trial then starts in a new session.

## Memory

Each request records:

- `cuda_allocated` and `cuda_reserved`: current `torch.cuda.memory_allocated()` and `memory_reserved()`.
- `cuda_peak_allocated`: `torch.cuda.max_memory_allocated()`, the high-water mark since
  `reset_peak_memory_stats()`. The reset happens once per model run: after model load,
  warm-up, and previews, immediately before the trial's init request.
- `cpu_rss`: VmRSS from `/proc/self/status`.

**Session storage.** In Transformers 5.20.2, the HF streaming session never frees anything.
Each frame adds a 12 MiB float32 input frame and about 3 MiB of outputs, about 15 MiB per
frame of GPU memory in total. The model only ever reads the init (conditioning) frame, the
last 6 frames' memories (`num_maskmem` − 1), and the last 15 frames' object pointers
(`max_object_pointers_in_encoder` − 1). After each step, `HfVideoTracker` deletes stored
input frames older than the current one and outputs older than that 15-frame window. This
edits the session's dictionaries directly, outside the documented API, but it changes no
model input. `validate.py` checks that pruned and unpruned sessions give bit-identical masks
and logits. Storage counts are logged per request: steady state is 1 frame and 15 outputs.
Replay workers also keep bit-packed host masks (about 38 KB per frame at 640×480) in CPU
memory for the output video.

## Presence score

`presence` is `sigmoid(object_score_logits)`, the model's native object-presence output.
It is not a calibrated mask accuracy. It can be wrong, and it can stay high while the model
follows a distractor. An empty mask or a low score does not stop tracking; the session
keeps running so the model's own recovery can work. On the init frame, the HF mask-prompt
path sets the score from the mask itself (sigmoid(10) ≈ 1.0), so it is not a prediction.

## Validation

```bash
python -m experiments.segmentation.validate                          # about 2 min
python -m experiments.segmentation.validate --sustained-frames 3000 --sustained-model all --skip-pruning
```

The validation writes a synthetic FFV1 video with a known frame-0 mask. In it, a striped
target leaves the image for about 80 frames and comes back, and a static distractor is
present. It checks:

- pruned and unpruned sessions give identical results;
- a mismatched init mask is rejected;
- `replay --model all`, with and without `--tracking-fps 10`, runs workers sequentially in
  separate processes, gives identical frame indices and the same mask hash for all models,
  and tracks every selected frame through empty masks;
- session storage stays bounded, and memory stays flat once the 15-frame window is full
  (`settled_from_proc_20`).

The report is `runs/validation_<time>/validation.json`. The validation does not exercise the
camera or the mouse UI.
