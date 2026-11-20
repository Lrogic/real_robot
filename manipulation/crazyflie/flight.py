"""Guarded hover for the Crazyflie session (--manipulation crazyflie --crazyflie-fly).

The drone flies on its own Kalman estimate, which fuses the FoundationPose positions the session
sends. This module only takes off, hovers in place, lands and stops, with the high-level commander,
and watches for anything that makes the estimate unsafe to fly on. A thread ticks at TICK_HZ, so
the checks keep running while SAM recovery or a re-registration blocks the tracking loop.

States: grounded -> arming -> takeoff -> hover -> landing -> grounded; killed from anywhere.

Takeoff (t) is refused unless positions are being sent at min_send_hz, the drone's estimate is
within max_estimate_error_m of the tracked position with variance under max_position_variance_m2,
the battery reads at least min_battery_v, and the drone's supervisor (log supervisor.info, bits as
in cflib.crazyflie.supervisor) reports a recent state without locked, crashed or tumbled. The
takeoff point is the mean tracked position over the last half second; the drone climbs
hover_height_m above it and holds that point. After the arming request, takeoff is sent once the
supervisor reports can-fly; without it within ARM_TIMEOUT_S, the takeoff is cancelled. At the end
of the takeoff time the tracked position must have risen LIFTOFF_FRACTION of hover_height_m, or
the drone is landed (it did not lift off).

In the air:
  no position sent for pose_timeout_s          land (the drone lands on its IMU estimate)
  estimate and tracked position disagree by    land
    more than max_estimate_error_m
  estimate or tracked position more than       KILL: motors stop and the drone drops
    bounds_xy_m sideways from the takeoff
    point, or more than max_height_m above it
  Space, Enter in the terminal, Ctrl+C         KILL
  supervisor reports locked, crashed or        KILL (the drone has already stopped its motors)
    tumbled

From the first arming until exit, the firmware's emergency-stop watchdog is fed every
watchdog_period_s. If this process hangs or the radio drops, the firmware stops the motors by itself. A kill, or the
watchdog running out at exit, locks the drone until it reboots; CrazyflieLink reboots a locked
drone over the radio when a --crazyflie-fly session connects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

GROUNDED, ARMING, TAKEOFF, HOVER, LANDING, KILLED = "grounded", "arming", "takeoff", "hover", "landing", "killed"
AIRBORNE = (ARMING, TAKEOFF, HOVER, LANDING)
TICK_HZ = 50
ARM_SETTLE_S = 0.3        # at least this long between the arming request and the takeoff command
ARM_TIMEOUT_S = 1.5       # cancel the takeoff if the supervisor does not report can-fly by then
STATUS_MAX_AGE_S = 0.5    # supervisor/battery status older than this counts as missing
LIFTOFF_FRACTION = 0.5    # of hover_height_m, risen by the end of the takeoff time
# supervisor.info bits (cflib.crazyflie.supervisor)
BIT_CAN_FLY, BIT_IS_TUMBLED, BIT_IS_LOCKED, BIT_IS_CRASHED = 3, 5, 6, 7
BLOCKING_BITS = ((BIT_IS_LOCKED, "locked"), (BIT_IS_CRASHED, "crashed"), (BIT_IS_TUMBLED, "tumbled"))


def supervisor_problem(status: dict) -> Optional[str]:
    """'locked', 'crashed', 'tumbled' (comma-joined) from a link status, or None."""
    bits = status.get("supervisor")
    if bits is None:
        return None
    problems = [name for bit, name in BLOCKING_BITS if bits & (1 << bit)]
    return ", ".join(problems) or None


def supervisor_can_fly(status: dict) -> bool:
    bits = status.get("supervisor")
    return bits is not None and bool(bits & (1 << BIT_CAN_FLY))
LAND_SETTLE_S = 0.5       # after the landing time, before the motors stop
TAKEOFF_AVERAGE_S = 0.5   # tracked positions averaged for the takeoff point


@dataclass(frozen=True)
class FlightConfig:
    hover_height_m: float
    takeoff_time_s: float
    land_time_s: float
    bounds_xy_m: float
    max_height_m: float
    pose_timeout_s: float
    max_estimate_error_m: float
    max_position_variance_m2: float
    min_battery_v: float
    min_send_hz: float
    watchdog_period_s: float
    max_heading_error_deg: float

    @classmethod
    def from_dict(cls, raw: dict, where: str) -> "FlightConfig":
        known = set(cls.__dataclass_fields__)
        unknown, missing = set(raw) - known, known - set(raw)
        if unknown or missing:
            raise ValueError(f"{where} flight: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        cfg = cls(**{k: float(v) for k, v in raw.items()})
        if any(v <= 0 for v in raw.values()):
            raise ValueError(f"{where} flight: every value must be > 0")
        if cfg.max_height_m <= cfg.hover_height_m:
            raise ValueError(f"{where} flight: max_height_m must exceed hover_height_m")
        if cfg.watchdog_period_s >= 0.5:
            raise ValueError(f"{where} flight: watchdog_period_s must be well under the firmware's 1 s timeout")
        return cfg


class FlightController:
    """link needs: estimate() -> (p, age), status() -> dict with vbat and var, arm(bool),
    takeoff(z, duration), land(z, duration), stop_motors(), emergency_stop(), feed_watchdog()."""

    def __init__(self, link, cfg: FlightConfig, log_event: Callable[[dict], None] = lambda record: None):
        self.link, self.cfg, self._log_event = link, cfg, log_event
        self._lock = threading.RLock()
        self.state = GROUNDED
        self.status = "grounded"
        self.takeoff_point: Optional[np.ndarray] = None
        self._sent: List[Tuple[float, np.ndarray]] = []
        self._state_at = 0.0
        self._watchdog_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="crazyflie-flight", daemon=True)

    # ----- inputs ---------------------------------------------------------------

    def note_sent(self, p: np.ndarray, now: float) -> None:
        """A position was just sent to the drone."""
        with self._lock:
            self._sent = [(t, q) for t, q in self._sent if now - t <= 1.0] + [(now, np.array(p, dtype=float))]

    # ----- requests ---------------------------------------------------------------

    def request_takeoff(self, now: float) -> Optional[str]:
        """Arm and take off; returns the refusal, or None."""
        with self._lock:
            refusal = self.preflight(now)
            if refusal is not None:
                return refusal
            recent = [q for t, q in self._sent if now - t <= TAKEOFF_AVERAGE_S]
            self.takeoff_point = np.mean(recent, axis=0)
            self.link.arm(True)
            self._feed_watchdog(now, force=True)
            self._set(ARMING, now, f"arming; takeoff point {np.round(self.takeoff_point, 3).tolist()}")
            return None

    def request_land(self, reason: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.state == ARMING:
                self.link.arm(False)
                self._set(GROUNDED, now, f"takeoff cancelled: {reason}")
            elif self.state in (TAKEOFF, HOVER):
                self.link.land(float(self.takeoff_point[2]), self.cfg.land_time_s)
                self._set(LANDING, now, f"landing: {reason}")

    def kill(self, reason: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.state == KILLED:
                return
            self.link.emergency_stop()
            self.link.stop_motors()
            self._set(KILLED, now, f"KILLED: {reason}")

    # ----- checks -----------------------------------------------------------------

    def preflight(self, now: float) -> Optional[str]:
        if self.state != GROUNDED:
            return f"not grounded ({self.state})"
        rate = sum(1 for t, _ in self._sent if now - t <= 1.0)
        if rate < self.cfg.min_send_hz:
            return f"positions sent at {rate} Hz, need {self.cfg.min_send_hz:.0f}"
        last_t, last_p = self._sent[-1]
        if now - last_t > self.cfg.pose_timeout_s:
            return f"last position sent {1000 * (now - last_t):.0f} ms ago"
        estimate, age = self.link.estimate()
        if estimate is None or age > 0.2:
            return "no recent state estimate from the drone"
        error = float(np.linalg.norm(estimate - last_p))
        if error > self.cfg.max_estimate_error_m:
            return f"estimate {1000 * error:.0f} mm from the tracked position (reset it with e?)"
        status = self.link.status()
        if not status or status.get("age", 0.0) > STATUS_MAX_AGE_S:
            return "no recent status from the drone"
        problem = supervisor_problem(status)
        if problem is not None:
            return (f"drone reports {problem}: power-cycle it, or restart the session "
                    "(a --crazyflie-fly session reboots a locked drone when it connects)")
        var = status.get("var")
        if var is None or max(var) > self.cfg.max_position_variance_m2:
            return f"estimator variance {None if var is None else max(var):.2g} m^2 too high"
        vbat = status.get("vbat")
        if vbat is None or vbat < self.cfg.min_battery_v:
            return f"battery {vbat} V below {self.cfg.min_battery_v} V"
        return None

    def tick(self, now: float) -> None:
        with self._lock:
            if self._watchdog_at is not None and self.state != KILLED:
                self._feed_watchdog(now)  # from the first arming until exit, so a later flight still has it
            if self.state not in AIRBORNE:
                return
            problem = supervisor_problem(self.link.status())
            if problem is not None:
                self.kill(f"drone reports {problem}", now)
                return
            estimate, _age = self.link.estimate()
            last = self._sent[-1] if self._sent else None
            for name, p in (("estimate", estimate), ("tracked position", None if last is None else last[1])):
                problem = self._out_of_bounds(p)
                if problem is not None:
                    self.kill(f"{name} {problem}", now)
                    return
            if self.state == LANDING:
                if now - self._state_at >= self.cfg.land_time_s + LAND_SETTLE_S:
                    self.link.stop_motors()
                    self.link.arm(False)
                    self._set(GROUNDED, now, "landed")
                return
            if last is None or now - last[0] > self.cfg.pose_timeout_s:
                age = "never" if last is None else f"{1000 * (now - last[0]):.0f} ms ago"
                self.request_land(f"last position sent {age}", now)
                return
            if estimate is not None and now - last[0] < 0.1:
                error = float(np.linalg.norm(estimate - last[1]))
                if error > self.cfg.max_estimate_error_m:
                    self.request_land(f"estimate {1000 * error:.0f} mm from the tracked position", now)
                    return
            if self.state == ARMING:
                self._tick_arming(now)
            elif self.state == TAKEOFF and now - self._state_at >= self.cfg.takeoff_time_s:
                risen = float(last[1][2] - self.takeoff_point[2])
                needed = LIFTOFF_FRACTION * self.cfg.hover_height_m
                if risen < needed:
                    self.request_land(f"did not lift off: rose {1000 * risen:.0f} mm, "
                                      f"needed {1000 * needed:.0f} mm", now)
                else:
                    self._set(HOVER, now, "hovering")

    def _tick_arming(self, now: float) -> None:
        status = self.link.status()
        fresh = bool(status) and status.get("age", 0.0) <= STATUS_MAX_AGE_S
        if fresh and now - self._state_at >= ARM_SETTLE_S and supervisor_can_fly(status):
            target = float(self.takeoff_point[2] + self.cfg.hover_height_m)
            self.link.takeoff(target, self.cfg.takeoff_time_s)
            self._set(TAKEOFF, now, f"taking off to z {target:.3f} m")
        elif now - self._state_at >= ARM_TIMEOUT_S:
            bits = status.get("supervisor") if fresh else None
            self.request_land(f"drone did not report can-fly within {ARM_TIMEOUT_S:.1f} s "
                              f"(supervisor {bits})", now)

    def _out_of_bounds(self, p) -> Optional[str]:
        if p is None or self.takeoff_point is None:
            return None
        sideways = float(np.linalg.norm(p[:2] - self.takeoff_point[:2]))
        if sideways > self.cfg.bounds_xy_m:
            return f"{100 * sideways:.0f} cm sideways from the takeoff point"
        height = float(p[2] - self.takeoff_point[2])
        if height > self.cfg.max_height_m:
            return f"{100 * height:.0f} cm above the takeoff point"
        return None

    # ----- thread -------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Land if airborne (on the IMU estimate: no more positions are coming), then stop."""
        with self._lock:
            airborne = self.state in (TAKEOFF, HOVER)
            if airborne:
                self.request_land("session closing")
            elif self.state == ARMING:
                self.request_land("session closing")
        if airborne:
            print(f"[crazyflie] landing before exit ({self.cfg.land_time_s + LAND_SETTLE_S:.1f} s)")
            deadline = time.monotonic() + self.cfg.land_time_s + LAND_SETTLE_S + 0.5
            while time.monotonic() < deadline and self.state == LANDING:
                time.sleep(0.05)
            if self.state == LANDING:
                self.kill("landing did not finish before exit")
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(1.0 / TICK_HZ):
            try:
                self.tick(time.monotonic())
            except Exception as error:  # a failed check must not leave the drone flying unwatched
                print(f"[crazyflie] flight check failed: {error!r}")
                self.kill(f"flight check failed: {error!r}")

    # ----- internals ----------------------------------------------------------------

    def _feed_watchdog(self, now: float, force: bool = False) -> None:
        if force or self._watchdog_at is None or now - self._watchdog_at >= self.cfg.watchdog_period_s:
            self.link.feed_watchdog()
            self._watchdog_at = now

    def _set(self, state: str, now: float, status: str) -> None:
        previous = self.state
        self.state, self.status, self._state_at = state, status, now
        self._log_event({"event": "flight", "from": previous, "to": state, "status": status,
                         "takeoff_point": None if self.takeoff_point is None else self.takeoff_point.tolist()})
        print(f"[crazyflie] {previous} -> {state}: {status}")
