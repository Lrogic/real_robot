"""Per-object SAM-vs-FoundationPose mismatch bookkeeping (pure numpy, no GPU)."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two same-shape bool masks; two empty masks give 1.0."""
    if a.dtype != bool or b.dtype != bool:
        raise ValueError(f"IoU needs bool masks, got {a.dtype} and {b.dtype}")
    if a.shape != b.shape:
        raise ValueError(f"IoU needs same-shape masks, got {a.shape} and {b.shape}")
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return np.count_nonzero(a & b) / union


class Verdict(Enum):
    SAM_EMPTY = "sam_empty"   # object lost; counter reset, no comparison
    MATCH = "match"
    MISMATCH = "mismatch"
    TRIGGER = "trigger"       # mismatch that reached the failure limit


@dataclass
class ObjectRecoveryState:
    """One object's counter; observe() must be called exactly once per new SAM observation."""
    name: str
    sam_id: int
    iou_threshold: float
    failure_limit: int
    mismatch_count: int = 0
    sam_empty: bool = False
    last_iou: Optional[float] = None
    last_presence: Optional[float] = None
    last_frame_number: Optional[int] = None
    last_observed_at: Optional[float] = None
    registrations: int = 0
    failed_registrations: int = 0
    observations: int = 0
    empty_observations: int = 0
    mismatches: int = 0

    def observe(self, sam_mask: np.ndarray, fp_silhouette: Optional[np.ndarray], presence: Optional[float],
                frame_number: int, observed_at: float) -> Verdict:
        """Update from one SAM mask and the same-frame FP silhouette (needed unless SAM is empty)."""
        self.observations += 1
        self.last_presence = presence
        self.last_frame_number = frame_number
        self.last_observed_at = observed_at
        if not sam_mask.any():
            self.empty_observations += 1
            self.mismatch_count = 0
            self.sam_empty = True
            self.last_iou = None
            return Verdict.SAM_EMPTY
        self.sam_empty = False
        if fp_silhouette is None:
            raise ValueError("A nonempty SAM mask needs the same-frame FP silhouette")
        self.last_iou = mask_iou(sam_mask, fp_silhouette)
        if self.last_iou >= self.iou_threshold:
            self.mismatch_count = 0
            return Verdict.MATCH
        self.mismatches += 1
        self.mismatch_count += 1
        return Verdict.TRIGGER if self.mismatch_count >= self.failure_limit else Verdict.MISMATCH

    def registered(self, succeeded: bool) -> None:
        """Clear the counter after a re-registration attempt; count the outcome."""
        self.mismatch_count = 0
        if succeeded:
            self.registrations += 1
        else:
            self.failed_registrations += 1


class RateSchedule:
    """Requests at least 1/rate_hz apart on a monotonic clock: never faster, no catch-up after stalls."""

    def __init__(self, rate_hz: float, start: float):
        self.period = 1.0 / rate_hz
        self.next_due = start

    def due(self, now: float) -> bool:
        if now < self.next_due:
            return False
        self.next_due = now + self.period
        return True
