"""Recovery settings: pipeline/recovery/recovery.yaml (or --recovery-config), overridden by CLI flags."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..segmentation.sam_model_specs import SAM_MODEL_NAMES

DEFAULT_RECOVERY_CONFIG = Path(__file__).resolve().parent / "recovery.yaml"


@dataclass(frozen=True)
class RecoveryConfig:
    source: Path
    camera_width: int
    camera_height: int
    camera_fps: int
    sam_model: str
    sam_fps: float
    hf_cache: Path
    warmup_steps: int
    iou_threshold: float
    failure_limit: int

    def describe(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_recovery_config(path: Optional[Path], sam_model: Optional[str] = None,
                         sam_fps: Optional[float] = None, iou_threshold: Optional[float] = None,
                         failure_limit: Optional[int] = None,
                         hf_cache: Optional[Path] = None) -> RecoveryConfig:
    """Read the YAML file, apply any non-None override, and validate the result."""
    import yaml

    path = path or DEFAULT_RECOVERY_CONFIG
    with open(path) as f:
        c = yaml.safe_load(f)
    try:
        camera, sam, recovery = c["camera"], c["sam"], c["recovery"]
        cfg = RecoveryConfig(
            source=Path(path),
            camera_width=int(camera["width"]), camera_height=int(camera["height"]),
            camera_fps=int(camera["fps"]),
            sam_model=str(sam_model if sam_model is not None else sam["model"]),
            sam_fps=float(sam_fps if sam_fps is not None else sam["tracking_fps"]),
            hf_cache=Path(hf_cache if hf_cache is not None else sam["hf_cache"]),
            warmup_steps=int(sam["warmup_steps"]),
            iou_threshold=float(iou_threshold if iou_threshold is not None else recovery["iou_threshold"]),
            failure_limit=int(failure_limit if failure_limit is not None else recovery["failure_limit"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing or malformed setting ({error})") from error
    if cfg.sam_model not in SAM_MODEL_NAMES:
        raise ValueError(f"SAM model {cfg.sam_model!r} is not one of {list(SAM_MODEL_NAMES)}")
    if min(cfg.camera_width, cfg.camera_height, cfg.camera_fps) <= 0:
        raise ValueError("recovery camera width, height, fps must be > 0")
    if cfg.sam_fps <= 0:
        raise ValueError("SAM tracking fps must be > 0")
    if not 0.0 < cfg.iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    if cfg.failure_limit < 1:
        raise ValueError("failure limit must be >= 1")
    if cfg.warmup_steps < 0:
        raise ValueError("sam.warmup_steps must be >= 0")
    return cfg
