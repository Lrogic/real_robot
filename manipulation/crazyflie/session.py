"""Stream the tracked Crazyflie's position to it over a Crazyradio (--manipulation crazyflie).

Hooks for pipeline.tracking_loop.track_live. Right after FoundationPose on each frame,
before_recovery sends the tracked position in the manipulator base frame,
p = T_base_camera @ center_pose @ to_origin @ origin_offset, with cflib's send_extpos. The drone's
Kalman filter fuses it with its IMU; attitude is not sent, because the Crazyflie is nearly 4-fold
symmetric and FoundationPose can settle on a yaw 90 or 180 deg off. The drone's own estimate
(stateEstimate.x/y/z) is logged back and drawn next to the tracked point.

A pose is held back when the tracked object has no pose, when its frame is older than
max_frame_age_s, or (with --recovery, unless send_untrusted) when SAM has not confirmed it since
the last mismatch or re-registration (manipulation/policy/trust.py). With --manipulation-dry-run
there is no radio; poses are drawn and logged only. Every frame is logged to
save-path/crazyflie_log.jsonl.

Heading. Only position is sent, so the drone's Kalman filter keeps whatever heading it was
reset to (kalman.initialYaw, 0 = front along base +x until confirmed). A wrong heading turns every
position correction the wrong way. FoundationPose gives the drone's heading only up to a 90 deg
turn (the mesh's arms are on its diagonals, so its x axis points between two arms like the
drone's front), so the front is confirmed by hand: the cyan arrow is the proposed front; h turns
it 90 deg until it points at the M1/M4 end (opposite the blue LEDs), c confirms. That sets
kalman.initialYaw to FoundationPose's heading for that front and resets the filter. Afterwards
the window shows FoundationPose's heading (the 90 deg copy closest to the drone's, green) next to
the drone's own (magenta), and every frame logs both. Every later reset (e) uses the copy
closest to the drone's current heading. The heading is not sent.

Without --crazyflie-fly nothing sends a setpoint and the motors never spin. With it, t takes off
to a hover over the takeoff point and l lands; flight.py has the takeoff gates and the in-flight
checks that land or kill. Takeoff also needs a confirmed heading within
flight.max_heading_error_deg of the drone's.

In the tracking window:
    h        turn the proposed front by 90 deg (after confirming: unconfirm)
    c        confirm the front: set the drone's heading and reset its Kalman filter
    p        pause / resume sending (in the air this makes the drone land)
    e        reset the drone's Kalman filter (on the ground only)
    t        take off and hover (--crazyflie-fly)
    l        land
    Space    KILL: motors stop, the drone drops (also Enter in the terminal, Ctrl+C)
    q / Esc  quit (in the air: land first, q again once grounded)
"""

from __future__ import annotations

import json
import select
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from calibration.transforms import inv_T

from ..common.drawing import draw_label, draw_point_marker, fmt_vec
from ..policy.trust import Trust, TrustTracker
from .flight import AIRBORNE, GROUNDED, KILLED, FlightConfig, FlightController, supervisor_problem

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
REAL_ROBOT_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = "crazyflie_log.jsonl"
TRACKED_COLOR, ESTIMATE_COLOR, TAKEOFF_COLOR = (0, 255, 0), (255, 0, 255), (0, 255, 255)
OK_COLOR, WARN_COLOR, BAD_COLOR = (235, 235, 235), (255, 220, 0), (255, 60, 60)
HEADING_COLOR = (0, 255, 255)
HELP = "h turn / c confirm front | p pause | e reset estimator | q quit"
FLY_HELP = "h/c front | t takeoff | l land | SPACE KILL | p pause | e reset | q quit"
ESTIMATE_VARS = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                 "stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw")
ARROW_M = 0.07
REBOOT_WAIT_S = 6.0      # after the power cycle, before reconnecting (2 s was too short: the connect hung)
CONNECT_TIMEOUT_S = 15.0  # cflib's open_link has none; a connect to a booting drone can hang
STATUS_VARS = ("kalman.varX", "kalman.varY", "kalman.varZ", "pm.vbat", "sys.isFlying", "sys.canfly",
               "supervisor.info")


@dataclass(frozen=True)
class CrazyflieConfig:
    uri: str
    tracked_name: Optional[str]
    origin_offset_m: tuple
    ext_pos_std_dev_m: float
    max_frame_age_s: float
    send_untrusted: bool
    reset_after_sends: int
    log_hz: float
    flight: FlightConfig
    path: Path = DEFAULT_CONFIG

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CrazyflieConfig":
        path = Path(path or DEFAULT_CONFIG).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        known = {f.name for f in fields(cls)} - {"path"}
        unknown, missing = set(raw) - known, known - set(raw)
        if unknown or missing:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        offset = tuple(float(v) for v in raw["origin_offset_m"])
        if len(offset) != 3:
            raise ValueError(f"{path}: origin_offset_m needs 3 values")
        if raw["log_hz"] <= 0 or raw["max_frame_age_s"] <= 0 or raw["ext_pos_std_dev_m"] <= 0:
            raise ValueError(f"{path}: log_hz, max_frame_age_s and ext_pos_std_dev_m must be > 0")
        return cls(**{**raw, "origin_offset_m": offset, "reset_after_sends": int(raw["reset_after_sends"]),
                      "flight": FlightConfig.from_dict(raw["flight"] or {}, str(path))}, path=path)

    def describe(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.__dict__ if isinstance(value, FlightConfig) else (
                str(value) if isinstance(value, Path) else value)
        return out


class CrazyflieLink:
    """cflib connection: external positions and flight commands out; estimate and status back."""

    def __init__(self, cfg: CrazyflieConfig, fly: bool):
        self.cfg, self.fly = cfg, fly
        self._lock = threading.Lock()
        self._estimate: Optional[np.ndarray] = None
        self._attitude: Optional[np.ndarray] = None
        self._estimate_at: Optional[float] = None
        self._status: dict = {}
        self.lost: Optional[str] = None
        self._scf = None
        self._logs = []

    @property
    def vbat(self) -> Optional[float]:
        return self.status().get("vbat")

    def __enter__(self) -> "CrazyflieLink":
        import cflib.crtp

        cflib.crtp.init_drivers()
        self._connect()
        if self.fly:
            self._reboot_if_locked()
        return self

    def _reboot_if_locked(self) -> None:
        """A drone left locked (a kill, or the emergency-stop watchdog running out when the last
        session ended) ignores flight commands until it reboots. On the ground its motors are
        already stopped, so power-cycle its main processor over the radio and reconnect. The power
        cycle runs in its own process (reboot.py): cflib's PowerSwitch opens the radio directly,
        and a new link in this process then hangs."""
        import subprocess

        problem = supervisor_problem(self._wait_for_status())
        if problem is None or "tumbled" in problem:
            return  # a tumbled drone needs turning upright, not a reboot
        print(f"[crazyflie] drone reports {problem}: rebooting it over the radio")
        self._disconnect()
        subprocess.run([sys.executable, "-m", "manipulation.crazyflie.reboot", "--uri", self.cfg.uri],
                       cwd=REAL_ROBOT_DIR, check=True, timeout=30)
        time.sleep(REBOOT_WAIT_S)
        self._connect()
        problem = supervisor_problem(self._wait_for_status())
        print(f"[crazyflie] after reboot the drone reports {problem or 'no lock'}")

    def _wait_for_status(self, timeout_s: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.status()
            if status:
                return status
            time.sleep(0.05)
        print("[crazyflie] WARNING: no status from the drone yet")
        return {}

    def _connect(self) -> None:
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.log import LogConfig
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        with self._lock:
            self._estimate = self._attitude = self._estimate_at = None
            self._status = {}
        self.lost = None
        self._logs = []
        cache = Path.home() / ".cache" / "cflib"
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Connecting to the Crazyflie at {self.cfg.uri} ...")
        self._scf = SyncCrazyflie(self.cfg.uri, cf=Crazyflie(rw_cache=str(cache)))
        self._open_link_with_timeout()
        cf = self._scf.cf
        cf.connection_lost.add_callback(self._on_lost)
        decks = {n for n, v in cf.param.toc.toc.get("deck", {}).items() if cf.param.get_value(f"deck.{n}") != "0"}
        if decks:
            print(f"WARNING: decks attached: {sorted(decks)}; a Flow or Z-ranger deck also feeds the estimator")
        cf.param.set_value("stabilizer.estimator", "2")  # Kalman: the estimator that fuses extpos
        cf.param.set_value("locSrv.extPosStdDev", str(self.cfg.ext_pos_std_dev_m))
        cf.param.set_value("kalman.initialYaw", "0")  # until the front is confirmed (a param survives reconnects)
        if self.fly:
            cf.param.set_value("commander.enHighLevel", "1")

        period_ms = max(10, int(round(1000 / self.cfg.log_hz)))
        estimate = LogConfig(name="estimate", period_in_ms=period_ms)
        for name in ESTIMATE_VARS:
            estimate.add_variable(name, "float")
        status = LogConfig(name="status", period_in_ms=100)
        for name in STATUS_VARS:
            status.add_variable(name)  # type from the firmware's log TOC
        for log in (estimate, status):
            cf.log.add_config(log)
            log.data_received_cb.add_callback(self._on_log)
            log.start()
            self._logs.append(log)
        print(f"Connected to {self.cfg.uri}: Kalman estimator, extPosStdDev {self.cfg.ext_pos_std_dev_m} m"
              + (", high-level commander on" if self.fly else ""))

    def _open_link_with_timeout(self) -> None:
        errors = []

        def open_link():
            try:
                self._scf.open_link()
            except Exception as error:
                errors.append(error)

        opener = threading.Thread(target=open_link, name="crazyflie-connect", daemon=True)
        opener.start()
        opener.join(CONNECT_TIMEOUT_S)
        if opener.is_alive():
            raise RuntimeError(f"no connection to {self.cfg.uri} within {CONNECT_TIMEOUT_S:.0f} s; "
                               "power-cycle the drone by hand (and replug the Crazyradio if that does not help)")
        if errors:
            raise errors[0]

    def __exit__(self, *_exc) -> None:
        self._disconnect()

    def _disconnect(self) -> None:
        if self._scf is None:
            return
        for log in self._logs:
            try:
                log.stop()
            except Exception:  # link may already be gone
                pass
        self._scf.close_link()
        self._scf = None
        print("Crazyflie link closed")

    def _on_lost(self, uri, message) -> None:
        self.lost = f"link lost: {message}"
        print(f"[crazyflie] {self.lost}")

    def _on_log(self, _timestamp, data, logconf) -> None:
        with self._lock:
            if logconf.name == "estimate":
                values = np.array([data[name] for name in ESTIMATE_VARS])
                self._estimate, self._attitude = values[:3], values[3:]
                self._estimate_at = time.monotonic()
            else:
                self._status = {"var": (data["kalman.varX"], data["kalman.varY"], data["kalman.varZ"]),
                                "vbat": data["pm.vbat"], "is_flying": bool(data["sys.isFlying"]),
                                "can_fly": bool(data["sys.canfly"]), "supervisor": int(data["supervisor.info"]),
                                "at": time.monotonic()}

    def estimate(self):
        """Latest (position, age_s), or (None, None)."""
        with self._lock:
            if self._estimate is None:
                return None, None
            return self._estimate.copy(), time.monotonic() - self._estimate_at

    def attitude(self):
        """Latest (roll, pitch, yaw) in deg and its age_s, or (None, None)."""
        with self._lock:
            if self._estimate is None:
                return None, None
            return self._attitude.copy(), time.monotonic() - self._estimate_at

    def set_initial_yaw(self, yaw_rad: float) -> None:
        """Heading the next Kalman reset starts from (kalman.initialYaw, rad; logged yaw is deg)."""
        self._scf.cf.param.set_value("kalman.initialYaw", f"{yaw_rad:.6f}")

    def status(self) -> dict:
        """Latest status with its age in s ("age"), or {}."""
        with self._lock:
            if not self._status:
                return {}
            return {**{k: v for k, v in self._status.items() if k != "at"},
                    "age": time.monotonic() - self._status["at"]}

    def send(self, p: np.ndarray) -> None:
        self._scf.cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

    def reset_estimator(self) -> None:
        cf = self._scf.cf
        cf.param.set_value("kalman.resetEstimation", "1")
        threading.Timer(0.1, cf.param.set_value, ("kalman.resetEstimation", "0")).start()

    # ----- flight (flight.FlightController) ---------------------------------------

    def arm(self, on: bool) -> None:
        self._scf.cf.platform.send_arming_request(on)

    def takeoff(self, z: float, duration_s: float) -> None:
        self._scf.cf.high_level_commander.takeoff(z, duration_s, yaw=None)  # keep the current heading

    def land(self, z: float, duration_s: float) -> None:
        self._scf.cf.high_level_commander.land(z, duration_s, yaw=None)  # keep the current heading

    def stop_motors(self) -> None:
        self._scf.cf.high_level_commander.stop()

    def emergency_stop(self) -> None:
        self._scf.cf.loc.send_emergency_stop()

    def feed_watchdog(self) -> None:
        self._scf.cf.loc.send_emergency_stop_watchdog()


class DryRunLink:
    lost = None
    vbat = None

    def __enter__(self):
        print("Crazyflie dry run: no radio, positions are drawn and logged only")
        return self

    def __exit__(self, *_exc):
        pass

    def estimate(self):
        return None, None

    def attitude(self):
        return None, None

    def set_initial_yaw(self, yaw_rad):
        pass

    def status(self):
        return {}

    def send(self, p):
        pass

    def reset_estimator(self):
        pass


class CrazyflieSession:
    def __init__(self, link, cfg: CrazyflieConfig, T_base_camera: np.ndarray, recovery, save_path: Path,
                 fly: bool = False):
        self.link, self.cfg, self.recovery, self.save_path = link, cfg, recovery, Path(save_path)
        self.X, self.X_inv = T_base_camera, inv_T(T_base_camera)
        self.trust = TrustTracker() if recovery is not None else None
        self.flight = FlightController(link, cfg.flight, self._write) if fly else None
        self.paused = False
        self.sent = 0
        self.held = 0
        self.message = ""
        self._index: Optional[int] = None
        self._rec_index: Optional[int] = None
        self._name: Optional[str] = None
        self._K = None
        self._p: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None          # base <- mesh rotation of the tracked drone
        self._yaw_raw: Optional[float] = None         # heading of the mesh x axis in base xy, deg
        self._proposal: Optional[float] = None        # proposed front before confirming, deg
        self.heading_confirmed = False
        self._heading: Optional[float] = None         # proposal copy, or after confirming the copy closest to the drone
        self._heading_error: Optional[float] = None   # heading minus the drone's yaw, deg
        self._status = "waiting for the first frame"
        self._send_times: List[float] = []
        self._reset_done = cfg.reset_after_sends <= 0
        self._log = open(self.save_path / LOG_FILE, "a")
        self._quit_after_landing = False
        self._stdin_stop = threading.Event()
        self._stdin_thread = threading.Thread(target=self._watch_stdin, name="stdin-kill", daemon=True)
        self._old_sigint = None

    def start(self) -> None:
        self._write({"event": "start", "config": self.cfg.describe(), "T_base_camera": self.X.tolist(),
                     "recovery": self.recovery is not None, "dry_run": isinstance(self.link, DryRunLink),
                     "fly": self.flight is not None})
        print(__doc__)
        if self.recovery is None:
            print("WARNING: no --recovery, so every tracked pose is sent, including a lost track's")
        if self.flight is not None:
            self.flight.start()
            self._stdin_thread.start()
            self._old_sigint = signal.signal(signal.SIGINT, self._on_sigint)
            print("FLIGHT ENABLED. KILL: Space in the window, Enter in this terminal, or Ctrl+C.")

    def close(self) -> None:
        if self._log.closed:
            return
        if self.flight is not None:
            self.flight.close()
            self._stdin_stop.set()
            self._stdin_thread.join(timeout=1.0)
            if self._old_sigint is not None:
                signal.signal(signal.SIGINT, self._old_sigint)
        self._write({"event": "stop", "sent": self.sent, "held": self.held})
        self._log.close()
        print(f"[crazyflie] sent {self.sent} positions, held back {self.held}")

    # ----- kill paths outside the window ------------------------------------

    def _on_sigint(self, _sig, _frame):
        self.flight.kill("Ctrl+C")
        raise KeyboardInterrupt

    def _watch_stdin(self) -> None:
        while not self._stdin_stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            if sys.stdin.readline() == "":
                return
            self.flight.kill("terminal Enter")

    # ----- hooks called by track_live ---------------------------------------

    def on_mouse(self, *_args) -> None:
        pass

    def before_recovery(self, tracker, K: np.ndarray, timing) -> None:
        """Send this frame's position before SAM recovery can block the loop. Trust is what
        after_recovery() left after the previous frame."""
        if self._index is None:
            self._resolve_object(tracker)
        self._K = K
        now = time.monotonic()
        age = now - timing.captured_at if timing.captured_at else None
        self._p = self._position(tracker)
        self._update_heading()
        reason = self._hold_reason(age)
        if reason is None:
            self.link.send(self._p)
            self.sent += 1
            self._send_times = [t for t in self._send_times if now - t < 1.0] + [now]
            if self.flight is not None:
                self.flight.note_sent(self._p, now)
            if not self._reset_done and self.sent >= self.cfg.reset_after_sends:
                self._reset_estimator("auto")
            self._status = "sending"
        else:
            self.held += 1
            self._status = f"holding: {reason}"
        estimate, estimate_age = self.link.estimate()
        record = {"frame_number": timing.frame_number, "frame_age_s": age, "sent": reason is None,
                  "hold_reason": reason, "trust": None if self.trust is None else self.trust.state.value,
                  "p_base": None if self._p is None else self._p.tolist(),
                  "estimate": None if estimate is None else estimate.tolist(),
                  "estimate_age_s": estimate_age,
                  "q_base_object": None if self._R is None else quat_wxyz(self._R),
                  "fp_yaw_raw_deg": self._yaw_raw, "heading_confirmed": self.heading_confirmed,
                  "heading_deg": self._heading, "heading_error_deg": self._heading_error,
                  "drone_rpy_deg": _list(self.link.attitude()[0])}
        if self.flight is not None:
            record["flight"] = self.flight.state
            record["status"] = self.link.status()
        self._write(record)

    def after_recovery(self, tracker, K: np.ndarray, timing) -> None:
        if self.trust is None or self._rec_index is None:
            return
        _state, events = self.trust.observe(self.recovery, self._rec_index, timing.frame_number)
        for event in events:
            self._write({**event, "frame_number": timing.frame_number})
            if event["event"] in ("trust_change", "reregistration", "recovery_disabled"):
                print(f"[crazyflie] {json.dumps(event, default=str)}")

    def on_key(self, key: int) -> bool:
        flight = self.flight
        airborne = flight is not None and flight.state in AIRBORNE
        if key == 32 and flight is not None:
            flight.kill("Space")
        elif key in (ord("q"), 27):
            if not airborne:
                return True
            flight.request_land("quit")
            self._quit_after_landing = True
            self.message = "landing; quitting once grounded"
        elif key == ord("p"):
            self.paused = not self.paused
            self.message = "paused" if self.paused else "resumed"
            self._write({"event": "pause" if self.paused else "resume"})
        elif key == ord("e"):
            if airborne:
                self.message = "estimator reset refused in the air"
            else:
                self._reset_estimator("key")
        elif key in (ord("h"), ord("c")) and airborne:
            self.message = "the front can only be set on the ground"
        elif key == ord("h"):
            if self.heading_confirmed:
                self._unconfirm_heading()
            elif self._proposal is not None:
                self._proposal = wrap_deg(self._proposal + 90.0)
                self._heading = closest_copy(self._yaw_raw, self._proposal) if self._yaw_raw is not None else None
        elif key == ord("c") and not self.heading_confirmed:
            self._confirm_heading()
        elif key == ord("t"):
            if flight is None:
                self.message = "flight needs --crazyflie-fly"
            elif self.heading_refusal() is not None:
                self.message = f"takeoff refused: {self.heading_refusal()}"
                self._write({"event": "takeoff_request", "refusal": self.heading_refusal()})
            else:
                refusal = flight.request_takeoff(time.monotonic())
                self.message = f"takeoff refused: {refusal}" if refusal else ""
                self._write({"event": "takeoff_request", "refusal": refusal})
        elif key == ord("l") and flight is not None:
            flight.request_land("l key")
        if self._quit_after_landing and flight is not None and flight.state in (GROUNDED, KILLED):
            return True
        return False

    def draw(self, rgb: np.ndarray) -> np.ndarray:
        vis = rgb.copy()
        estimate, estimate_age = self.link.estimate()
        flight = self.flight
        if self._K is not None:
            draw_point_marker(vis, self.X_inv, self._p, TRACKED_COLOR, self._K, "FP")
            draw_point_marker(vis, self.X_inv, estimate, ESTIMATE_COLOR, self._K, "CF est")
            if flight is not None and flight.takeoff_point is not None:
                target = flight.takeoff_point + np.array([0.0, 0.0, self.cfg.flight.hover_height_m])
                draw_point_marker(vis, self.X_inv, target, TAKEOFF_COLOR, self._K, "hover",
                                  from_base=flight.takeoff_point, size=5)
            if self._p is not None:
                if not self.heading_confirmed:
                    self._draw_heading(vis, self._heading, HEADING_COLOR, "front? h/c", 3)
                else:
                    self._draw_heading(vis, self._heading, TRACKED_COLOR, "FP front", 2)
                    self._draw_heading(vis, self._drone_yaw(), ESTIMATE_COLOR, "CF front", 1)
        status_color = OK_COLOR if self._status == "sending" else WARN_COLOR
        if self.link.lost:
            status_color = BAD_COLOR
        rate = len(self._send_times)
        vbat = "" if self.link.vbat is None else f"  battery {self.link.vbat:.2f} V"
        lines = [(f"CRAZYFLIE {self._name or '-'}  {self.link.lost or self._status}  "
                  f"~{rate} Hz  sent {self.sent} held {self.held}{vbat}", status_color),
                 (f"FP base     {fmt_vec(self._p)} m", TRACKED_COLOR)]
        if estimate is not None:
            stale = estimate_age > 0.5
            line = f"CF estimate {fmt_vec(estimate)} m" + (f"  ({estimate_age:.1f} s old)" if stale else "")
            if self._p is not None:
                line += f"  diff {1000 * np.linalg.norm(estimate - self._p):.0f} mm"
            lines.append((line, WARN_COLOR if stale else ESTIMATE_COLOR))
        status = self.link.status()
        if status.get("var") is not None:
            std_mm = 1000 * np.sqrt(np.maximum(status["var"], 0))
            problem = supervisor_problem(status)
            lines.append((f"estimate std {fmt_mm(std_mm)} mm  can fly {status.get('can_fly')}  "
                          f"flying {status.get('is_flying')}"
                          + (f"  DRONE {problem.upper()}" if problem else ""), BAD_COLOR if problem else OK_COLOR))
        drone_yaw = self._drone_yaw()
        drone_yaw_text = "-" if drone_yaw is None else f"{drone_yaw:+.0f}"
        if not self.heading_confirmed:
            proposal = "-" if self._heading is None else f"{self._heading:+.0f}"
            lines.append((f"FRONT NOT CONFIRMED: arrow {proposal} deg, drone assumes {drone_yaw_text} deg "
                          "(h turn to the M1/M4 end, c confirm)", WARN_COLOR))
        else:
            heading = "-" if self._heading is None else f"{self._heading:+.0f}"
            error = "-" if self._heading_error is None else f"{self._heading_error:+.1f}"
            bad = self._heading_error is None or abs(self._heading_error) > self.cfg.flight.max_heading_error_deg
            lines.append((f"front: FP {heading} deg, drone {drone_yaw_text} deg, diff {error} deg",
                          WARN_COLOR if bad else OK_COLOR))
        if self.trust is not None:
            trust = self.trust.state
            lines.append((f"trust {trust.value}", BAD_COLOR if trust is Trust.UNTRUSTED else OK_COLOR))
        if flight is not None:
            color = BAD_COLOR if flight.state == KILLED else (TAKEOFF_COLOR if flight.state in AIRBORNE else OK_COLOR)
            lines.append((f"FLIGHT {flight.state.upper()}: {flight.status}", color))
        if self.message:
            lines.append((self.message, WARN_COLOR))
        y = 24
        for text, color in lines:
            draw_label(vis, text, (10, y), color)
            y += 24
        if flight is not None and flight.state == KILLED:
            draw_label(vis, "KILLED - power-cycle the drone before flying again", (10, vis.shape[0] // 2),
                       BAD_COLOR, 0.75, 2)
        draw_label(vis, FLY_HELP if flight is not None else HELP, (10, vis.shape[0] - 12), (200, 200, 200), 0.45)
        return vis

    # ----- internals ----------------------------------------------------------

    def _position(self, tracker) -> Optional[np.ndarray]:
        tracked = tracker.objects[self._index]
        if tracked.center_pose is None:
            self._R = None
            return None
        T_base_object = self.X @ tracked.center_pose @ tracked.to_origin
        self._R = T_base_object[:3, :3]
        return T_base_object[:3, :3] @ np.asarray(self.cfg.origin_offset_m) + T_base_object[:3, 3]

    def _update_heading(self) -> None:
        """FoundationPose's heading for this frame, as the 90 deg copy closest to the proposal
        (before confirming) or to the drone's own yaw (after)."""
        self._yaw_raw = mesh_yaw_deg(self._R)
        self._heading = self._heading_error = None
        if self._yaw_raw is None:
            return
        drone_yaw = self._drone_yaw()
        if not self.heading_confirmed:
            if self._proposal is None:
                self._proposal = self._yaw_raw
            self._heading = closest_copy(self._yaw_raw, self._proposal)
            self._proposal = self._heading  # follow the drone if it is turned before confirming
        elif drone_yaw is not None:
            self._heading = closest_copy(self._yaw_raw, drone_yaw)
        if self._heading is not None and drone_yaw is not None:
            self._heading_error = wrap_deg(self._heading - drone_yaw)

    def _drone_yaw(self) -> Optional[float]:
        attitude, age = self.link.attitude()
        return None if attitude is None or age > 0.5 else float(attitude[2])

    def heading_refusal(self) -> Optional[str]:
        """Why the heading does not allow takeoff, or None."""
        if not self.heading_confirmed:
            return "confirm the drone's front first (h turns the arrow, c confirms)"
        if self._heading_error is None:
            return "no heading from FoundationPose or the drone on this frame"
        if abs(self._heading_error) > self.cfg.flight.max_heading_error_deg:
            return (f"FoundationPose heading {self._heading_error:+.0f} deg from the drone's "
                    f"(limit {self.cfg.flight.max_heading_error_deg:.0f})")
        return None

    def _draw_heading(self, vis, heading_deg: Optional[float], color, label: str, thickness: int) -> None:
        if heading_deg is None or self._p is None:
            return
        import cv2
        from pipeline.drawing import project_point

        h = np.radians(heading_deg)
        tip_base = self._p + ARROW_M * np.array([np.cos(h), np.sin(h), 0.0])
        start = project_point(self.X_inv[:3, :3] @ self._p + self.X_inv[:3, 3], self._K)
        tip = project_point(self.X_inv[:3, :3] @ tip_base + self.X_inv[:3, 3], self._K)
        if start is None or tip is None:
            return
        cv2.arrowedLine(vis, start, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)
        draw_label(vis, label, (tip[0] + 6, tip[1]), color, 0.45)

    def _hold_reason(self, age: Optional[float]) -> Optional[str]:
        if self.paused:
            return "paused"
        if self.link.lost:
            return "link lost"
        if self._p is None or not np.all(np.isfinite(self._p)):
            return "no pose"
        if age is not None and age > self.cfg.max_frame_age_s:
            return f"frame {1000 * age:.0f} ms old"
        if self.trust is not None and not self.cfg.send_untrusted and self.trust.state is Trust.UNTRUSTED:
            return "pose not confirmed by SAM"
        return None

    def _reset_estimator(self, source: str) -> None:
        """Reset the drone's Kalman filter; once the front is confirmed, from the current heading."""
        heading = self._heading if self.heading_confirmed else None
        if self.heading_confirmed and heading is None:
            self.message = "reset refused: no heading on this frame (FoundationPose or the drone)"
            return
        if heading is not None:
            self.link.set_initial_yaw(np.radians(heading))
        self.link.reset_estimator()
        self._reset_done = True
        yaw = "" if heading is None else f", heading {heading:+.1f} deg"
        self.message = f"Kalman filter reset ({source}{yaw})"
        self._write({"event": "reset_estimator", "source": source, "sent": self.sent, "heading_deg": heading})
        print(f"[crazyflie] {self.message}")

    def _confirm_heading(self) -> None:
        if self._heading is None:
            self.message = "no FoundationPose heading on this frame"
            return
        heading = self._heading
        self.heading_confirmed = True
        self._heading = heading  # the reset below uses the confirmed copy, not one from the drone's old yaw
        self._write({"event": "heading_confirmed", "heading_deg": heading, "fp_yaw_raw_deg": self._yaw_raw})
        self._reset_estimator("front confirmed")

    def _unconfirm_heading(self) -> None:
        self.heading_confirmed = False
        self._proposal = self._heading
        self.message = "front unconfirmed: h turns the arrow, c confirms"
        self._write({"event": "heading_unconfirmed"})

    def _resolve_object(self, tracker) -> None:
        names = [t.name for t in tracker.objects]
        if self.cfg.tracked_name is not None:
            if self.cfg.tracked_name not in names:
                raise ValueError(f"tracked_name {self.cfg.tracked_name!r} is not tracked; tracked: {names}")
            name = self.cfg.tracked_name
        else:
            from pipeline.run_folder import load_run

            meshes = {o.name: o.mesh_path for o in load_run(self.save_path).objects}
            matches = [n for n in names if meshes.get(n) is not None and "crazyflie" in meshes[n].name.lower()]
            if len(matches) == 1:
                name = matches[0]
            elif len(names) == 1:
                name = names[0]
            else:
                raise ValueError(f"cannot tell which tracked object is the Crazyflie "
                                 f"({ {n: str(meshes.get(n)) for n in names} }); set tracked_name in {self.cfg.path}")
        self._index, self._name = names.index(name), name
        if self.recovery is not None:
            self._rec_index = [s.name for s in self.recovery.states].index(name)
        self._write({"event": "object", "name": name, "to_origin": tracker.objects[self._index].to_origin.tolist()})
        print(f"[crazyflie] sending the position of {name}")

    def _write(self, record: dict) -> None:
        if not self._log.closed:
            self._log.write(json.dumps({"t": time.time(), **record}, default=str) + "\n")


def fmt_mm(values) -> str:
    return "(" + ", ".join(f"{v:.0f}" for v in values) + ")"


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def mesh_yaw_deg(R: Optional[np.ndarray]) -> Optional[float]:
    """Heading of the mesh x axis projected on base xy, deg; None without a pose or when that
    axis is near vertical. The mesh x axis points between two arms, as the drone's front does."""
    if R is None:
        return None
    x = R[:, 0]
    if np.hypot(x[0], x[1]) < 0.5:  # tilted over 60 deg
        return None
    return float(np.degrees(np.arctan2(x[1], x[0])))


def closest_copy(yaw: float, reference: float) -> float:
    """The heading among yaw + k * 90 deg closest to reference, wrapped to [-180, 180)."""
    k = round(wrap_deg(reference - yaw) / 90.0)
    return wrap_deg(yaw + 90.0 * k)


def quat_wxyz(R: np.ndarray) -> list:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return (q if w >= 0 else -q).tolist()


def _list(values):
    return None if values is None else [float(v) for v in values]


@contextmanager
def crazyflie_session(args, recovery, camera):
    """Connect the Crazyflie (or the dry-run stand-in) and yield a started CrazyflieSession."""
    cfg = CrazyflieConfig.load(args.manipulation_config)
    fly = bool(getattr(args, "crazyflie_fly", False))
    X = camera.T_base_camera
    print(f"Camera {camera.mode}: camera at {np.round(X[:3, 3], 3)} m in the base frame; "
          "positions are sent in the manipulator base frame")
    link = DryRunLink() if args.manipulation_dry_run else CrazyflieLink(cfg, fly)
    with link:
        session = CrazyflieSession(link, cfg, X, recovery, args.save_path, fly)
        session.start()
        try:
            yield session
        finally:
            session.close()
