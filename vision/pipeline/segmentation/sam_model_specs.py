"""Supported Hugging Face SAM video-tracker checkpoints and their accepted names (no torch import)."""

from dataclasses import dataclass


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
SAM_MODEL_NAMES = (*MODELS, *MODEL_ALIASES)


def resolve_model_key(key: str) -> str:
    key = MODEL_ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}; expected one of {sorted(SAM_MODEL_NAMES)}")
    return key
