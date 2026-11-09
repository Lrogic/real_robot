"""Per-episode timing summaries of the policy log."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy.policy_log import (CONTROL_STATE_TICKS, DEFAULT_SUMMARY_FIELDS, PolicyLog,  # noqa: E402
                                            format_summary, summarize)


def _tick(log, at, build=0.001, age=0.01, **kwargs):
    log.tick({}, at, build, age, False, **kwargs)


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert (s["n"], s["mean"], s["median"], s["worst"]) == (5, 22.0, 3.0, 100.0)
    assert s["p95"] == pytest.approx(80.8)
    assert summarize([]) == {"n": 0}


def test_only_summarized_ticks_count_and_send_fields_need_a_send():
    log = PolicyLog(None)
    _tick(log, 0.0, build=9.0, age=9.0)                                     # not in the summary
    _tick(log, 0.05, build=0.002, age=0.04, in_summary=True)                # a hold: nothing sent
    _tick(log, 0.15, build=0.003, age=0.01, in_summary=True, reached_send=True)
    s = log.end_episode()
    assert s["build_s"]["n"] == 1 and s["build_s"]["worst"] == pytest.approx(0.003)
    assert s["tick_interval_s"]["n"] == 2 and s["tick_interval_s"]["worst"] == pytest.approx(0.1)
    assert s["frame_age_s"]["n"] == 2 and s["frame_age_s"]["worst"] == pytest.approx(0.04)


def test_end_episode_resets_and_none_is_skipped():
    log = PolicyLog(None)
    _tick(log, 0.0, age=None, in_summary=True, reached_send=True)           # first tick: no interval either
    first = log.end_episode()
    assert first["frame_age_s"] == {"n": 0} and first["tick_interval_s"] == {"n": 0} and first["build_s"]["n"] == 1
    assert log.end_episode()["build_s"] == {"n": 0}


def test_configured_fields_come_from_the_record():
    log = PolicyLog(None, ["robot_read_s", "sent", "typo_s"])
    log.tick({"robot_read_s": 0.002, "sent": True}, 0.0, 0.001, 0.01, False, in_summary=True, reached_send=True)
    s = log.end_episode()
    assert list(s) == ["robot_read_s", "sent", "typo_s", CONTROL_STATE_TICKS]
    assert s["robot_read_s"]["n"] == 1 and s["sent"] == {"n": 0}
    assert "typo_s: no samples" in format_summary(s)


def test_control_state_ticks_count_summarized_ticks_per_episode():
    log = PolicyLog(None)
    states = [("await_go", False), ("running", True), ("hold_stale", True), ("hold_stale", True),
              ("hold_tracking", True), ("running", True)]
    for i, (state, in_summary) in enumerate(states):
        log.tick({"control_state": state}, 0.05 * i, 0.001, 0.01, False, in_summary=in_summary)
    s = log.end_episode()
    assert s[CONTROL_STATE_TICKS] == {"running": 2, "hold_stale": 2, "hold_tracking": 1}
    assert "ticks: running 2, hold_stale 2, hold_tracking 1" in format_summary(s)
    assert log.end_episode()[CONTROL_STATE_TICKS] == {"running": 0, "hold_stale": 0, "hold_tracking": 0}


def test_format_uses_ms_for_seconds_fields():
    line = format_summary({"build_s": summarize([0.001]), "step": summarize([3.0])})
    assert "build_s: mean 1.00 / median 1.00 / p95 1.00 / worst 1.00 ms (n=1)" in line
    assert "step: mean 3.00 / median 3.00 / p95 3.00 / worst 3.00 (n=1)" in line


def test_summary_is_json_serializable(tmp_path):
    log = PolicyLog(tmp_path / "log.jsonl")
    _tick(log, 0.0, in_summary=True, reached_send=True)
    log.event("episode_end", summary=log.end_episode())
    log.close()
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert lines[-1]["summary"]["build_s"]["n"] == 1


def test_config_defaults():
    from manipulation.policy.jenga.config import JengaPolicyConfig, LoggingConfig

    assert JengaPolicyConfig.load().logging.summary_fields == DEFAULT_SUMMARY_FIELDS
    assert LoggingConfig.from_raw({}).summary_fields == DEFAULT_SUMMARY_FIELDS
    assert LoggingConfig.from_raw({"summary_fields": []}).summary_fields == ()
    with pytest.raises(ValueError):
        LoggingConfig.from_raw({"summary_fields": "build_s"})
