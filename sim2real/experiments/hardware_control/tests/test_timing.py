"""Deadline scheduling and state-sample bookkeeping."""

import pytest

from timing import Deadlines, StateTracker, interval_stats


def test_deadlines_keep_phase_when_slightly_late():
    d = Deadlines(0.01, 100.0)
    assert d.advance(100.004) == (1, pytest.approx(100.01), 0)
    assert d.advance(100.0185) == (2, pytest.approx(100.02), 0)   # late but within the period: no skip


def test_deadlines_skip_instead_of_catching_up():
    d = Deadlines(0.01, 0.0)
    index, due, skipped = d.advance(0.0435)      # tick 0 overran past deadlines 1, 2, 3 and into slot 4
    assert (index, skipped) == (4, 3)
    assert due == pytest.approx(0.04)
    index, due, skipped = d.advance(0.0436)
    assert (index, skipped) == (5, 0) and due == pytest.approx(0.05)
    assert d.skipped_total == 3


def test_state_tracker_counts_repeats_gaps_and_intervals():
    t = StateTracker()
    assert t.add(10, 1_000, 1.000)
    assert not t.add(10, 1_000, 1.0004)
    assert t.add(11, 2_000, 1.0012)
    assert t.add(14, 5_000, 1.0045)              # ids 12, 13 never seen
    assert not t.add(14, 5_000, 1.005)
    s = t.summary(duration_s=0.5)
    assert (s["fresh_reads"], s["repeated_reads"], s["skipped_message_ids"]) == (3, 2, 2)
    assert s["header_interval_s"]["count"] == 2
    assert s["header_interval_s"]["max"] == pytest.approx(0.003)
    assert s["host_first_seen_interval_s"]["median"] == pytest.approx((0.0012 + 0.0033) / 2)
    assert s["fresh_rate_hz"] == pytest.approx(6.0)


def test_interval_stats_empty():
    assert interval_stats([])["count"] == 0
