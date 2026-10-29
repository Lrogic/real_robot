"""Control state machine and command gate of the policy session. Pure logic: the session does
the arm calls each transition asks for.

  gravity          startup; arm and gripper in gravity compensation, never commanded
  moving_to_ready  episode key: position mode, joint move to the training ready pose
  await_go         at the ready pose, episode armed / active; ready key starts commands
  running          one command per policy tick (shadow: computed and logged, not sent)
  hold_stale       frame too old or too far after the previous one; resumes on its own
  hold_tracking    object pose untrusted (any SAM verdict other than a match, re-registration),
                   no observation or action, target outside the workspace, or a driver error;
                   resumes on the ready key once the pose is trusted again
  holding          episode key while commanding: stopped at the measured pose, episode ended;
                   the episode key moves to the ready pose again
  killed           kill key: gravity compensation until restart

In shadow mode the arm is never moved: the episode key arms the episode directly and stopping
returns to gravity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .trust import Trust

GRAVITY, MOVING, AWAIT_GO, RUNNING = "gravity", "moving_to_ready", "await_go", "running"
HOLD_STALE, HOLD_TRACKING, HOLDING, KILLED = "hold_stale", "hold_tracking", "holding", "killed"
CONTROL_STATES = (RUNNING, HOLD_STALE, HOLD_TRACKING)
EPISODE_STATES = (MOVING, AWAIT_GO, *CONTROL_STATES)

# press_episode() results
DO_READY, DO_ARM, DO_STOP, DO_NOTHING = "ready", "arm", "stop", "nothing"


@dataclass
class Gate:
    """What a policy tick knows before sending its command."""
    trust: Trust
    frame_age_s: Optional[float]
    frame_gap_s: Optional[float]
    action_error: Optional[str] = None      # no observation, non-finite action, ...
    in_workspace: Optional[bool] = None     # of the commanded TCP; None when there is no target


@dataclass
class Decision:
    send: bool
    reason: Optional[str] = None            # why nothing is sent this tick


@dataclass
class Controller:
    mode: str
    control_period_s: float
    slow_time_dilation: int
    step_goal_time_s: float
    max_frame_age_s: float
    max_frame_gap_s: float
    keys: dict
    state: str = GRAVITY
    hold_reason: Optional[str] = None
    step_pending: bool = False
    events: List[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg) -> "Controller":
        c = cfg.control
        return cls(mode=c.mode, control_period_s=cfg.control_period_s, slow_time_dilation=c.slow_time_dilation,
                   step_goal_time_s=c.step_goal_time_s, max_frame_age_s=c.max_frame_age_s,
                   max_frame_gap_s=c.max_frame_gap_s, keys={"episode": cfg.episode_key, **c.keys()})

    # ----- timing -----------------------------------------------------------------------

    @property
    def shadow(self) -> bool:
        return self.mode == "shadow"

    @property
    def tick_period_s(self) -> float:
        """Policy tick period: slow mode ticks slow_time_dilation times less often."""
        return self.control_period_s * (self.slow_time_dilation if self.mode == "slow" else 1)

    @property
    def goal_time_s(self) -> float:
        """Duration over which the driver interpolates one Cartesian goal."""
        if self.mode == "step":
            return self.step_goal_time_s
        return self.tick_period_s

    # ----- keys -------------------------------------------------------------------------

    def press_episode(self) -> str:
        if self.state in (GRAVITY, HOLDING):
            if self.shadow:
                self._set(AWAIT_GO, "episode key (shadow: no ready move)")
                return DO_ARM
            self._set(MOVING, "episode key")
            return DO_READY
        if self.state in EPISODE_STATES:
            self._set(GRAVITY if self.shadow else HOLDING, "episode key")
            return DO_STOP
        return DO_NOTHING

    def ready_reached(self) -> None:
        if self.state == MOVING:
            self._set(AWAIT_GO, "at the ready pose")

    def press_go(self, episode_active: bool, trust: Trust) -> Optional[str]:
        """Start or resume commands; returns why not instead."""
        if self.state not in (AWAIT_GO, HOLD_TRACKING):
            return f"'{self.keys['ready']}' does nothing while {self.state}"
        if not episode_active:
            return "waiting for a trusted object pose to start the episode"
        if trust is Trust.UNTRUSTED:
            return "object pose untrusted: wait for a SAM match"
        self.step_pending = False
        self._set(RUNNING, "ready key")
        return None

    def press_step(self) -> Optional[str]:
        if self.mode != "step":
            return f"'{self.keys['step']}' only works in step mode (mode {self.mode})"
        if self.state != RUNNING:
            return f"'{self.keys['step']}' needs running (now {self.state})"
        self.step_pending = True
        return None

    def kill(self, reason: str) -> None:
        if self.state != KILLED:
            self._set(KILLED, reason)

    # ----- per tick ---------------------------------------------------------------------

    def tick(self, gate: Gate) -> Decision:
        if self.state not in CONTROL_STATES:
            return Decision(False, None)
        if self.state == HOLD_TRACKING:
            return Decision(False, self.hold_reason)
        reason = self._latched_reason(gate)
        if reason is not None:
            self._set(HOLD_TRACKING, reason)
            return Decision(False, reason)
        reason = self._stale_reason(gate)
        if reason is not None:
            if self.state != HOLD_STALE or reason != self.hold_reason:
                self._set(HOLD_STALE, reason)
            return Decision(False, reason)
        if self.state != RUNNING:
            self._set(RUNNING, "frames fresh again")
        if self.shadow:
            return Decision(False, "shadow mode")
        if self.mode == "step":
            if not self.step_pending:
                return Decision(False, f"step mode: press '{self.keys['step']}'")
            self.step_pending = False
        return Decision(True, None)

    def report_send_error(self, error: str) -> None:
        self._set(HOLD_TRACKING, f"driver: {error}")

    # ----- helpers ----------------------------------------------------------------------

    def _latched_reason(self, gate: Gate) -> Optional[str]:
        if gate.trust is Trust.UNTRUSTED:
            return "object pose untrusted (SAM)"
        if gate.action_error is not None:
            return gate.action_error
        if gate.in_workspace is False:
            return "target TCP outside the workspace"
        return None

    def _stale_reason(self, gate: Gate) -> Optional[str]:
        if gate.frame_age_s is not None and gate.frame_age_s > self.max_frame_age_s:
            return f"frame age {1000 * gate.frame_age_s:.0f} ms > {1000 * self.max_frame_age_s:.0f} ms"
        if gate.frame_gap_s is not None and gate.frame_gap_s > self.max_frame_gap_s:
            return f"frame gap {1000 * gate.frame_gap_s:.0f} ms > {1000 * self.max_frame_gap_s:.0f} ms"
        return None

    def _set(self, state: str, reason: str) -> None:
        self.events.append({"event": "control", "from": self.state, "to": state, "reason": reason})
        self.state = state
        self.hold_reason = reason if state in (HOLD_STALE, HOLD_TRACKING) else None

    def drain_events(self) -> List[dict]:
        events, self.events = self.events, []
        return events
