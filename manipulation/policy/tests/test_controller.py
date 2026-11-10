"""The policy control state machine: keys, the command gate, and the hold / resume rules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy import controller as ctl  # noqa: E402
from manipulation.policy.trust import Trust  # noqa: E402

FRESH = dict(frame_age_s=0.01)


def _controller(mode="full") -> ctl.Controller:
    return ctl.Controller(mode=mode, control_period_s=0.05, slow_time_dilation=4, step_goal_time_s=1.0,
                          max_frame_age_s=0.03,
                          keys={"episode": "s", "ready": "r", "kill": "k", "step": "n"})


def _running(mode="full") -> ctl.Controller:
    c = _controller(mode)
    assert c.press_episode() == (ctl.DO_ARM if mode == "shadow" else ctl.DO_READY)
    c.ready_reached()
    assert c.state == ctl.AWAIT_GO
    assert c.press_go(True, Trust.VERIFIED) is None
    assert c.state == ctl.RUNNING
    return c


def test_timing_per_mode():
    assert _controller("full").goal_time_s == pytest.approx(0.05)
    slow = _controller("slow")
    assert slow.tick_period_s == pytest.approx(0.2) and slow.goal_time_s == pytest.approx(0.2)
    assert _controller("step").goal_time_s == 1.0
    for mode, expected in (("full", 0.1), ("slow", 0.1), ("step", 1.0)):
        c = _controller(mode)
        c.driver_goal_time_s = 0.1
        assert c.goal_time_s == expected
    # The sim IK loop's interpolation time is one policy step, whatever the driver goal time.
    for mode, expected in (("full", 0.05), ("slow", 0.2), ("step", 1.0)):
        c = _controller(mode)
        c.driver_goal_time_s = 0.1
        assert c.step_time_s == pytest.approx(expected)


def _config_copy(tmp_path, edit):
    """jenga/config.yaml with its relative paths made absolute and edit(raw) applied."""
    import yaml

    from manipulation.policy.jenga.config import DEFAULT_CONFIG

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for section, key in (("sim", "scene"), ("sim", "task"), ("policy", "checkpoint"), ("sim_ik", "model")):
        raw[section][key] = str((DEFAULT_CONFIG.parent / raw[section][key]).resolve())
    edit(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_interpolate_goal_config(tmp_path):
    from manipulation.policy.jenga.config import JengaPolicyConfig

    def edit(interpolate, solver="sim"):
        def apply(raw):
            raw["control"]["ik_solver"] = solver
            if interpolate is None:
                raw["sim_ik"].pop("interpolate_goal", None)
            else:
                raw["sim_ik"]["interpolate_goal"] = interpolate
        return apply

    assert JengaPolicyConfig.load(_config_copy(tmp_path, edit(None))).sim_ik.interpolate_goal is False
    cfg = JengaPolicyConfig.load(_config_copy(tmp_path, edit(True)))
    assert cfg.sim_ik.interpolate_goal is True and cfg.describe()["sim_ik"]["interpolate_goal"] is True
    with pytest.raises(ValueError, match="must be true or false"):
        JengaPolicyConfig.load(_config_copy(tmp_path, edit(1)))
    with pytest.raises(ValueError, match="needs control.ik_solver: sim"):
        JengaPolicyConfig.load(_config_copy(tmp_path, edit(True, "driver")))
    assert JengaPolicyConfig.load(_config_copy(tmp_path, edit(False, "driver"))).control.ik_solver == "driver"


def test_episode_key_cycle():
    c = _controller()
    assert c.press_episode() == ctl.DO_READY and c.state == ctl.MOVING
    assert c.press_go(True, Trust.VERIFIED) is not None  # not at the ready pose yet
    c.ready_reached()
    assert c.press_go(False, Trust.VERIFIED) is not None  # episode not active yet
    assert c.press_go(True, Trust.UNTRUSTED) is not None
    assert c.press_go(True, Trust.UNVERIFIED_OK) is None
    assert c.press_episode() == ctl.DO_STOP and c.state == ctl.HOLDING
    assert c.press_episode() == ctl.DO_READY and c.state == ctl.MOVING


def test_running_sends():
    c = _running()
    assert c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)).send
    assert c.tick(ctl.Gate(trust=Trust.UNVERIFIED_OK, in_workspace=True, **FRESH)).send


def test_untrusted_latches_until_ready_key():
    c = _running()
    d = c.tick(ctl.Gate(trust=Trust.UNTRUSTED, in_workspace=True, **FRESH))
    assert not d.send and c.state == ctl.HOLD_TRACKING
    assert not c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)).send
    assert c.state == ctl.HOLD_TRACKING
    assert c.press_go(True, Trust.UNTRUSTED) is not None
    assert c.press_go(True, Trust.VERIFIED) is None
    assert c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)).send


def test_stale_frames_resume_on_their_own():
    c = _running()
    d = c.tick(ctl.Gate(trust=Trust.VERIFIED, frame_age_s=0.05, in_workspace=True))
    assert not d.send and c.state == ctl.HOLD_STALE and "frame age" in d.reason
    assert c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)).send
    assert c.state == ctl.RUNNING


@pytest.mark.parametrize("gate", [
    ctl.Gate(trust=Trust.VERIFIED, in_workspace=False, **FRESH),
    ctl.Gate(trust=Trust.VERIFIED, action_error="no robot state", **FRESH),
])
def test_workspace_and_missing_action_latch(gate):
    c = _running()
    assert not c.tick(gate).send and c.state == ctl.HOLD_TRACKING


def test_driver_error_latches():
    c = _running()
    c.report_send_error("no IK solution")
    assert c.state == ctl.HOLD_TRACKING and "no IK" in c.hold_reason


def test_shadow_never_sends():
    c = _running("shadow")
    d = c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH))
    assert not d.send and d.reason == "shadow mode"
    assert c.press_episode() == ctl.DO_STOP and c.state == ctl.GRAVITY


def test_step_mode_sends_once_per_key():
    c = _running("step")
    gate = ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)
    assert not c.tick(gate).send
    assert c.press_step() is None
    assert c.tick(gate).send
    assert not c.tick(gate).send


def test_kill_is_final():
    c = _running()
    c.kill("kill key")
    assert c.state == ctl.KILLED
    assert c.press_episode() == ctl.DO_NOTHING
    assert c.press_go(True, Trust.VERIFIED) is not None
    assert not c.tick(ctl.Gate(trust=Trust.VERIFIED, in_workspace=True, **FRESH)).send
