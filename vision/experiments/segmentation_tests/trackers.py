"""Stateful live-video trackers for SAM 2.1 and SAM 3.1 multiplex."""

from __future__ import annotations

import logging
import os

# One-frame propagate steps must not print a tqdm line each camera tick.
os.environ.setdefault("TQDM_DISABLE", "1")

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

OBJ_ID = 1


@dataclass
class TrackOutput:
    """One tracked frame."""

    video_frame_idx: int
    mask: Optional[np.ndarray]  # HxW bool, same resolution as input RGB
    mask_area: int
    presence: Optional[Dict[str, float]]  # model-native scores when available


class LiveVideoTracker(ABC):
    """Interface shared by SAM2 and SAM3.1 adapters (recorded video later)."""

    implementation: str

    @abstractmethod
    def reset_session(self) -> None:
        """Drop tracking state; model weights stay loaded."""

    @abstractmethod
    def begin_session(self, init_rgb: np.ndarray) -> None:
        """Start a new trial with a frozen initialization frame (video index 0)."""

    @abstractmethod
    def preview_prompt(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
        clear_old: bool,
    ) -> TrackOutput:
        """Segment on the init frame only (no temporal memory commit)."""

    @abstractmethod
    def finalize_init(self) -> TrackOutput:
        """Commit prompts to temporal memory; returns mask on frame 0."""

    @abstractmethod
    def track_frame(self, rgb: np.ndarray) -> TrackOutput:
        """Append one live frame and track (must not re-process frame 0)."""


# ----- SAM 2.1 ----------------------------------------------------------------


class _GrowingFrameStore:
    """Frame list compatible with SAM2 video state; drops pixels outside the memory horizon."""

    def __init__(self) -> None:
        self._frames: Dict[int, torch.Tensor] = {}
        self._count = 0

    def append(self, frame: torch.Tensor) -> int:
        idx = self._count
        self._count += 1
        self._frames[idx] = frame
        return idx

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._frames[idx]

    def prune_before(self, min_idx: int, keep: set) -> None:
        for key in list(self._frames.keys()):
            if key < min_idx and key not in keep:
                del self._frames[key]


def _sam2_preprocess(rgb: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    import cv2

    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return (img - mean) / std


def _sam2_memory_horizon(predictor) -> int:
    stride = predictor.memory_temporal_stride_for_eval
    slots = max(predictor.num_maskmem - 1, 1)
    return slots * stride + predictor.max_obj_ptrs_in_encoder + 4


def _sam2_prune_state(predictor, state: dict, frame_idx: int) -> None:
    """Drop frames and outputs SAM2 will not read again (same indices as memory attention)."""
    obj_idx = 0
    if obj_idx not in state["output_dict_per_obj"]:
        return
    obj_out = state["output_dict_per_obj"][obj_idx]
    keep = set(obj_out["cond_frame_outputs"].keys())
    oldest = max(0, frame_idx - _sam2_memory_horizon(predictor))
    for key in list(obj_out["non_cond_frame_outputs"].keys()):
        if key < oldest and key not in keep:
            del obj_out["non_cond_frame_outputs"][key]
    tracked = state["frames_tracked_per_obj"].get(obj_idx, {})
    for key in list(tracked.keys()):
        if key < oldest and key not in keep:
            del tracked[key]
    images = state["images"]
    if isinstance(images, _GrowingFrameStore):
        images.prune_before(oldest, keep)


def _mask_from_sam2_logits(video_res_masks: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """First object mask at original video resolution."""
    if video_res_masks is None or video_res_masks.numel() == 0:
        return None
    logits = video_res_masks[0, 0] if video_res_masks.dim() >= 3 else video_res_masks[0]
    return (logits > 0).detach().cpu().numpy()


class Sam21LiveTracker(LiveVideoTracker):
    implementation = "sam2.SAM2VideoPredictor.propagate_in_video (live frames, default CPU offload)"

    def __init__(self, predictor, device: torch.device):
        self._predictor = predictor
        self._device = device
        self._state = None
        self._next_frame_idx = 1
        self._offload_video = True
        self._offload_state = True

    def reset_session(self) -> None:
        self._state = None
        self._next_frame_idx = 1

    def begin_session(self, init_rgb: np.ndarray) -> None:
        h, w = init_rgb.shape[:2]
        pixel_device = torch.device("cpu") if self._offload_video else self._device
        frame = _sam2_preprocess(init_rgb, self._predictor.image_size, pixel_device)
        store = _GrowingFrameStore()
        store.append(frame)
        storage_device = torch.device("cpu") if self._offload_state else self._device
        inference_state = {
            "images": store,
            "num_frames": 1,
            "offload_video_to_cpu": self._offload_video,
            "offload_state_to_cpu": self._offload_state,
            "video_height": h,
            "video_width": w,
            "device": self._device,
            "storage_device": storage_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": __import__("collections").OrderedDict(),
            "obj_idx_to_id": __import__("collections").OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        self._predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        self._state = inference_state
        self._next_frame_idx = 1

    def preview_prompt(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
        clear_old: bool,
    ) -> TrackOutput:
        assert self._state is not None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            _, _, video_res_masks = self._predictor.add_new_points_or_box(
                self._state,
                frame_idx=0,
                obj_id=OBJ_ID,
                points=list(points),
                labels=list(labels),
                clear_old_points=clear_old,
                normalize_coords=True,
            )
        mask = _mask_from_sam2_logits(video_res_masks)
        return TrackOutput(0, mask, int(mask.sum()) if mask is not None else 0, None)

    def finalize_init(self) -> TrackOutput:
        assert self._state is not None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            self._predictor.propagate_in_video_preflight(self._state)
        out = self._read_mask_from_state(0)
        self._next_frame_idx = 1
        return out

    def _append_rgb(self, rgb: np.ndarray) -> int:
        assert self._state is not None
        pixel_device = torch.device("cpu") if self._offload_video else self._device
        frame = _sam2_preprocess(rgb, self._predictor.image_size, pixel_device)
        idx = self._state["images"].append(frame)
        self._state["num_frames"] = len(self._state["images"])
        self._next_frame_idx = idx + 1
        return idx

    def _read_mask_from_state(self, frame_idx: int) -> TrackOutput:
        obj_idx = 0
        obj_output_dict = self._state["output_dict_per_obj"][obj_idx]
        current_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
        if current_out is None:
            current_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if current_out is None:
            return TrackOutput(frame_idx, None, 0, None)
        presence = {
            "object_score_logit": float(current_out["object_score_logits"].detach().cpu().item())
        }
        _, video_res_masks = self._predictor._get_orig_video_res_output(
            self._state, current_out["pred_masks"]
        )
        mask = _mask_from_sam2_logits(video_res_masks)
        area = int(mask.sum()) if mask is not None else 0
        return TrackOutput(frame_idx, mask, area, presence)

    def track_frame(self, rgb: np.ndarray) -> TrackOutput:
        frame_idx = self._append_rgb(rgb)
        video_res_masks = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            for out_idx, _obj_ids, masks in self._predictor.propagate_in_video(
                self._state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                assert out_idx == frame_idx
                video_res_masks = masks
        mask = _mask_from_sam2_logits(video_res_masks)
        state_out = self._read_mask_from_state(frame_idx)
        if mask is None and state_out.mask is not None:
            mask = state_out.mask
        presence = state_out.presence
        _sam2_prune_state(self._predictor, self._state, frame_idx)
        area = int(mask.sum()) if mask is not None else 0
        return TrackOutput(frame_idx, mask, area, presence)


# ----- SAM 3.1 multiplex ------------------------------------------------------


def _rgb_to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _quiet_sam3_live_logs() -> None:
    """SAM3 logs INFO on every one-frame propagate; keep the live HUD terminal readable."""
    for name in (
        "sam3",
        "sam3.model.sam3_base_predictor",
        "sam3.model.sam3_multiplex_tracking",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)
        for handler in log.handlers:
            handler.setLevel(logging.WARNING)


def _sam31_strip_tail_propagation_actions(state: dict) -> None:
    """Multiplex marks single-frame propagate as done and next step becomes fetch-only.

    Live append-one-frame tracking must run SAM2 partial propagation each tick, not fetch.
    """
    history = state.get("action_history", [])
    while history and "propagation" in history[-1].get("type", ""):
        history.pop()


def _sam31_mask_to_display(mask: np.ndarray, display_h: int, display_w: int) -> np.ndarray:
    """Map SAM3 masks (orig_height x orig_width) onto numpy RGB (H x W).

    load_resource_as_video_frames swaps PIL width/height into orig_height/orig_width,
    so model masks are shaped (display_w, display_h) before transpose.
    """
    import cv2

    m = np.squeeze(mask)
    if m.ndim != 2:
        raise ValueError(f"expected 2D mask, got {m.shape}")
    if m.shape == (display_w, display_h):
        m = m.T
    if m.shape != (display_h, display_w):
        m = cv2.resize(
            m.astype(np.uint8),
            (display_w, display_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return m.astype(bool)


def _sam31_mask_from_outputs(outputs: dict, height: int, width: int) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    masks = outputs.get("out_binary_masks")
    if masks is None or len(masks) == 0:
        return None, None
    # Track object OBJ_ID if present, else first mask.
    obj_ids = outputs.get("out_obj_ids", [])
    idx = 0
    if len(obj_ids):
        try:
            idx = list(obj_ids).index(OBJ_ID)
        except ValueError:
            idx = 0
    m = _sam31_mask_to_display(np.asarray(masks[idx]), height, width)
    presence = {}
    probs = outputs.get("out_probs")
    if probs is not None and len(probs) > idx:
        presence["detector_score"] = float(probs[idx])
    sam2_probs = outputs.get("out_sam2_probs")
    if sam2_probs is not None and len(sam2_probs) > idx:
        presence["sam2_tracker_score"] = float(sam2_probs[idx])
    return m.astype(bool), presence or None


class Sam31LiveTracker(LiveVideoTracker):
    implementation = "sam3.build_sam3_multiplex_video_predictor + streaming frame append"

    def __init__(self, predictor):
        self._predictor = predictor
        self._model = predictor.model
        self._state = None
        self._session_id: Optional[str] = None
        self._next_frame_idx = 1
        self._h = 0
        self._w = 0

    def reset_session(self) -> None:
        if self._session_id is not None:
            try:
                self._predictor.close_session(self._session_id)
            except Exception:
                pass
        self._state = None
        self._session_id = None
        self._next_frame_idx = 1

    def begin_session(self, init_rgb: np.ndarray) -> None:
        pil = _rgb_to_pil(init_rgb)
        self._state = self._model.init_state(
            resource_path=[pil],
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        self._display_h, self._display_w = init_rgb.shape[:2]
        self._session_id = "live"
        self._predictor._all_inference_states[self._session_id] = {
            "state": self._state,
            "session_id": self._session_id,
            "start_time": 0.0,
            "last_use_time": 0.0,
        }
        self._next_frame_idx = 1

    def preview_prompt(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
        clear_old: bool,
    ) -> TrackOutput:
        # orig_height/orig_width follow io_utils swap: height->width, width->height in pixels.
        rel_points = [
            [x / self._display_w, y / self._display_h] for x, y in points
        ]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = self._predictor.add_prompt(
                session_id=self._session_id,
                frame_idx=0,
                points=rel_points,
                point_labels=list(labels),
                clear_old_points=clear_old,
                obj_id=OBJ_ID,
                rel_coordinates=True,
            )
        outputs = _sam31_postprocess_predictor_outputs(
            result["outputs"], self._display_h, self._display_w
        )
        mask, presence = outputs
        return TrackOutput(0, mask, int(mask.sum()) if mask is not None else 0, presence)

    def finalize_init(self) -> TrackOutput:
        # add_prompt already ran SAM2 preflight; do not call propagate_in_video here
        # (it poisons action_history and the next live frame uses propagation_fetch).
        if 0 not in self._state.get("cached_frame_outputs", {}):
            return TrackOutput(0, None, 0, None)
        _, outputs = self._model.fetch_and_process_single_frame_results(self._state, 0)
        mask, presence = _sam31_postprocess_predictor_outputs(
            outputs, self._display_h, self._display_w
        )
        self._next_frame_idx = 1
        return TrackOutput(0, mask, int(mask.sum()) if mask is not None else 0, presence)

    def _append_frame(self, rgb: np.ndarray) -> int:
        from sam3.model.data_misc import BatchedDatapoint, BatchedPointer, FindStage, NestedTensor, convert_my_tensors
        from sam3.model.sam3_multiplex_tracking import recursive_to

        pil = _rgb_to_pil(rgb)
        images, orig_h, orig_w = __import__(
            "sam3.model.io_utils", fromlist=["load_resource_as_video_frames"]
        ).load_resource_as_video_frames(
            resource_path=[pil],
            image_size=self._model.image_size,
            offload_video_to_cpu=False,
            img_mean=self._model.image_mean,
            img_std=self._model.image_std,
            async_loading_frames=False,
        )
        frame_tensor = images[0]
        batch = self._state["input_batch"]
        new_tensors = torch.cat([batch.img_batch.tensors, frame_tensor.unsqueeze(0)], dim=0)
        batch.img_batch = NestedTensor(tensors=new_tensors, mask=None)

        stage_id = self._state["num_frames"]
        dummy_ptrs = BatchedPointer(stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[])
        stage = FindStage(
            img_ids=[stage_id],
            img_ids_np=np.array([stage_id]),
            text_ids=[0],
            input_boxes=[torch.zeros(258)],
            input_boxes_before_embed=[torch.empty(0, 4)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, 257)],
            input_points_before_embed=[torch.empty(0, 3)],
            input_points_mask=[torch.empty(0)],
            ptrs=dummy_ptrs,
            ptrs_seg=dummy_ptrs,
            object_ids=[],
        )
        stage = convert_my_tensors(stage)
        device = self._state["device"]
        stage = recursive_to(stage, device, non_blocking=True)
        batch.find_inputs.append(stage)
        batch.find_targets.append(None)
        batch.find_metadatas.append(None)
        self._state["previous_stages_out"].append(None)
        self._state["per_frame_raw_point_input"].append(None)
        self._state["per_frame_raw_box_input"].append(None)
        self._state["per_frame_visual_prompt"].append(None)
        self._state["per_frame_geometric_prompt"].append(None)
        self._state["per_frame_cur_step"].append(0)
        self._state["num_frames"] += 1
        for sam2_state in self._state.get("sam2_inference_states", []):
            sam2_state["num_frames"] = self._state["num_frames"]
        # Multiplex _build_sam2_output returns {} if this frame is missing from cache
        # (only frame 0 exists after click-init). Seed an empty entry so SAM2 refined
        # masks on new live frames are merged instead of dropped.
        cache = self._state.setdefault("cached_frame_outputs", {})
        if stage_id not in cache:
            cache[stage_id] = {}
        idx = self._next_frame_idx
        self._next_frame_idx += 1
        return idx

    def track_frame(self, rgb: np.ndarray) -> TrackOutput:
        frame_idx = self._append_frame(rgb)
        _sam31_strip_tail_propagation_actions(self._state)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction="forward",
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            )
            last = None
            for item in gen:
                last = item
        if last is None:
            return TrackOutput(frame_idx, None, 0, None)
        h, w = rgb.shape[:2]
        mask, presence = _sam31_postprocess_predictor_outputs(last["outputs"], h, w)
        return TrackOutput(frame_idx, mask, int(mask.sum()) if mask is not None else 0, presence)


def _sam31_postprocess_predictor_outputs(outputs: Any, height: int, width: int):
    if isinstance(outputs, dict) and "out_binary_masks" in outputs:
        return _sam31_mask_from_outputs(outputs, height, width)
    # Model-native dict before numpy conversion
    if hasattr(outputs, "keys"):
        converted = {}
        for key in ("out_binary_masks", "out_obj_ids", "out_probs", "out_sam2_probs"):
            if key in outputs:
                val = outputs[key]
                converted[key] = val.cpu().numpy() if torch.is_tensor(val) else val
        return _sam31_mask_from_outputs(converted, height, width)
    return None, None


# ----- Factory ----------------------------------------------------------------

def _resolve_sam31_use_fa3() -> Tuple[bool, Optional[str]]:
    """FA3 wheels ship Hopper (SM90) kernels only — not Ampere/Ada/Blackwell."""
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        return False, "flash_attn_interface not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    props = torch.cuda.get_device_properties(0)
    if props.major != 9:
        return False, (
            f"FA3 kernels target Hopper only (compute 9.x); "
            f"{props.name} is {props.major}.{props.minor}"
        )
    return True, None


def _conda_env_matches(expected: str, active: str) -> bool:
    if not active:
        return True
    if active == expected:
        return True
    # conda sometimes sets CONDA_DEFAULT_ENV to the env prefix, e.g. /opt/conda-envs/sam3
    return Path(active).name == expected


def _require_conda_env(model_name: str) -> None:
    expected = MODEL_CHOICES[model_name]["env"]
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    if not _conda_env_matches(expected, active):
        raise RuntimeError(
            f"--model {model_name} requires conda env '{expected}' (active: '{active}'). "
            f"Run: ./experiments/segmentation_tests/run_sam_tracking.sh --model {model_name}"
        )


MODEL_CHOICES = {
    "sam21-small": {
        "env": "sam21",
        "checkpoint": Path("/bigdata/luka/models/sam2.1/sam2.1_hiera_small.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam21-base-plus": {
        "env": "sam21",
        "checkpoint": Path("/bigdata/luka/models/sam2.1/sam2.1_hiera_base_plus.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam21-large": {
        "env": "sam21",
        "checkpoint": Path("/bigdata/luka/models/sam2.1/sam2.1_hiera_large.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
    "sam31": {
        "env": "sam3",
        "checkpoint": Path("/bigdata/luka/models/sam3.1/sam3.1_multiplex.pt"),
    },
}


def build_tracker(model_name: str, checkpoint: Optional[Path]) -> Tuple[LiveVideoTracker, dict]:
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown --model {model_name!r}; choose from {list(MODEL_CHOICES)}")
    _require_conda_env(model_name)
    spec = MODEL_CHOICES[model_name]
    ckpt = checkpoint or spec["checkpoint"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SAM video tracking")

    device = torch.device("cuda")
    meta = {
        "model": model_name,
        "checkpoint": str(ckpt),
        "device": str(device),
    }

    if model_name.startswith("sam21"):
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(spec["config"], str(ckpt), device=device)
        tracker = Sam21LiveTracker(predictor, device)
        meta["implementation"] = tracker.implementation
        meta["config"] = spec["config"]
        return tracker, meta

    if model_name == "sam31":
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        use_fa3, fa3_skip = _resolve_sam31_use_fa3()
        if not use_fa3:
            print(f"SAM 3.1: building with use_fa3=False ({fa3_skip}).")
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(ckpt),
            async_loading_frames=False,
            warm_up=False,
            compile=False,
            use_fa3=use_fa3,
        )
        _quiet_sam3_live_logs()
        tracker = Sam31LiveTracker(predictor)
        meta["implementation"] = tracker.implementation
        meta["use_fa3"] = use_fa3
        return tracker, meta

    raise AssertionError("unreachable")
