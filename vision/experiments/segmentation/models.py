"""Supported Hugging Face SAM video trackers, loaded identically for every mode."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

HF_CACHE = Path("/bigdata/luka/models/hf")
# Weights come only from the local cache, so a run never silently picks up a new upstream revision.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

WEIGHT_DTYPE = torch.float32
AUTOCAST_DTYPE = torch.bfloat16
SESSION_DTYPE = torch.float32  # dtype of the session's stored input frames


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    description: str
    family: str  # "sam2" or "sam3"


MODELS = {
    "sam2-small": ModelSpec("sam2-small", "facebook/sam2.1-hiera-small", "SAM 2.1 Hiera Small", "sam2"),
    "sam2-base": ModelSpec("sam2-base", "facebook/sam2.1-hiera-base-plus", "SAM 2.1 Hiera Base+", "sam2"),
    "sam3": ModelSpec("sam3", "facebook/sam3", "SAM 3 tracker", "sam3"),
}
# Alternative names accepted by load_model (the FoundationPose recovery config uses these).
MODEL_ALIASES = {"sam2-base-plus": "sam2-base"}


def resolve_model_key(key: str) -> str:
    key = MODEL_ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}; expected one of {sorted([*MODELS, *MODEL_ALIASES])}")
    return key


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: object
    processor: object
    revision: str
    device: torch.device
    load_s: float

    def describe(self) -> dict:
        return {
            "key": self.spec.key,
            "repo_id": self.spec.repo_id,
            "description": self.spec.description,
            "revision": self.revision,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "weight_dtype": str(WEIGHT_DTYPE),
            "autocast_dtype": str(AUTOCAST_DTYPE),
            "session_dtype": str(SESSION_DTYPE),
            "device": str(self.device),
            "load_s": self.load_s,
        }


def cached_revision(repo_id: str, cache_dir: Path = HF_CACHE) -> str:
    """Commit hash that refs/main points to in the local HF cache."""
    ref = cache_dir / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"{repo_id} is not in the HF cache {cache_dir} (missing {ref})")
    return ref.read_text().strip()


def load_model(key: str, device: str = "cuda", cache_dir: Path = HF_CACHE) -> LoadedModel:
    from transformers import (Sam2VideoModel, Sam2VideoProcessor,
                              Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)

    key = resolve_model_key(key)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16 autocast")
    spec = MODELS[key]
    model_cls, processor_cls = {
        "sam2": (Sam2VideoModel, Sam2VideoProcessor),
        "sam3": (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor),
    }[spec.family]
    revision = cached_revision(spec.repo_id, cache_dir)
    started = time.perf_counter()
    common = dict(cache_dir=cache_dir, revision=revision, local_files_only=True)
    model = model_cls.from_pretrained(spec.repo_id, dtype=WEIGHT_DTYPE, **common)
    model = model.to(device).eval()
    processor = processor_cls.from_pretrained(spec.repo_id, **common)
    torch.cuda.synchronize()
    return LoadedModel(spec, model, processor, revision, torch.device(device),
                       time.perf_counter() - started)
