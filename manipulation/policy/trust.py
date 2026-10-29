"""Per-frame trust in the tracked object pose, from the SAM recovery verdicts.

  verified       SAM compared this exact frame and matched FoundationPose.
  unverified_ok  no SAM comparison on this frame; nothing has gone wrong since the last match.
  untrusted      a mismatch, an empty SAM mask, a re-registration (ok or failed), recovery
                 disabled, or no match yet; lasts until the next match.

observe() must be called once per tracking-loop iteration, after recovery.after_fp, so a
verdict on a frame that is not an observation tick is not missed.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple


class Trust(Enum):
    VERIFIED = "verified"
    UNVERIFIED_OK = "unverified_ok"
    UNTRUSTED = "untrusted"


class TrustTracker:
    def __init__(self):
        self.state = Trust.UNTRUSTED
        self.last_match_frame: Optional[int] = None
        self._disabled_reported = False

    def observe(self, recovery, index: int, frame_number: int) -> Tuple[Trust, List[dict]]:
        """Trust for this frame and the recovery events seen on it (dicts with an 'event' key)."""
        from pipeline.recovery.recovery_state import Verdict

        events: List[dict] = []
        previous = self.state
        rec = recovery.states[index]
        if recovery.disabled_reason is not None:
            if not self._disabled_reported:
                events.append({"event": "recovery_disabled", "reason": recovery.disabled_reason})
                self._disabled_reported = True
            state = Trust.UNTRUSTED
        elif rec.last_registration_frame == frame_number:
            events.append({"event": "reregistration", "ok": rec.last_registration_ok,
                           "registrations": rec.registrations, "failed": rec.failed_registrations})
            state = Trust.UNTRUSTED
        elif rec.last_frame_number == frame_number:
            verdict = rec.last_verdict
            if verdict is Verdict.MATCH:
                state = Trust.VERIFIED
                self.last_match_frame = frame_number
            else:
                kind = "sam_empty" if verdict is Verdict.SAM_EMPTY else "sam_mismatch"
                events.append({"event": kind, "verdict": verdict.value, "iou": rec.last_iou,
                               "presence": rec.last_presence, "mismatch_count": rec.mismatch_count})
                state = Trust.UNTRUSTED
        else:
            state = Trust.UNTRUSTED if previous is Trust.UNTRUSTED else Trust.UNVERIFIED_OK
        if (previous is Trust.UNTRUSTED) != (state is Trust.UNTRUSTED):
            events.append({"event": "trust_change", "from": previous.value, "to": state.value})
            if state is Trust.VERIFIED:
                events.append({"event": "tracking_resumed", "iou": rec.last_iou})
        self.state = state
        return state, events
