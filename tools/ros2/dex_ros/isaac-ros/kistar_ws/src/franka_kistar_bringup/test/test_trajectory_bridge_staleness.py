"""Unit tests for the action-acceptance staleness guard in trajectory_bridge.py.

These tests bypass rclpy node spin-up by constructing a TrajectoryBridge instance
via __new__ and injecting the attributes the freshness path reads. The test target
is the pure decision logic in _goal_callback + _relay_age_sec, which decides whether
to ACCEPT or REJECT an incoming FollowJointTrajectory goal based on the age of the
most recent /joint_states_relay receive timestamp.

Operator escape (max_age_sec=0.0) MUST always ACCEPT, regardless of age.
Rejection log line MUST include both the measured age and configured threshold.
"""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
BRIDGE_PY = os.path.join(
    REPO_ROOT,
    "isaac-ros", "kistar_ws", "src", "franka_kistar_bringup", "scripts",
    "trajectory_bridge.py",
)


def _load_bridge():
    spec = importlib.util.spec_from_file_location("trajectory_bridge_mod", BRIDGE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["trajectory_bridge_mod"] = module
    spec.loader.exec_module(module)
    return module


BRIDGE_MOD = _load_bridge()
TrajectoryBridge = BRIDGE_MOD.TrajectoryBridge

# rclpy is required for GoalResponse symbols
from rclpy.action import GoalResponse  # noqa: E402


class _FakeClock:
    """Mimics rclpy.Clock.now() interface for tests."""

    def __init__(self, ns):
        self._ns = ns

    def now(self):
        return self

    @property
    def nanoseconds(self):
        return self._ns


def _make_bridge(max_age_sec, last_relay_ns, now_ns):
    """Construct a TrajectoryBridge without calling Node.__init__.

    Only the attributes that _goal_callback / _relay_age_sec touch are populated.
    """
    bridge = TrajectoryBridge.__new__(TrajectoryBridge)
    bridge._max_age_sec = max_age_sec
    bridge._last_relay_stamp_ns = last_relay_ns
    bridge.get_clock = lambda: _FakeClock(now_ns)
    bridge.get_logger = lambda: MagicMock()
    return bridge


# ----- Operator escape: max_age_sec == 0.0 ALWAYS ACCEPT -----

@pytest.mark.parametrize(
    "last_ns,now_ns",
    [(None, 1_000_000_000), (0, 10_000_000_000), (500_000_000, 9_000_000_000)],
)
def test_max_age_zero_always_accepts(last_ns, now_ns):
    bridge = _make_bridge(max_age_sec=0.0, last_relay_ns=last_ns, now_ns=now_ns)
    assert bridge._goal_callback(None) == GoalResponse.ACCEPT


# ----- last_relay_stamp_ns is None (relay has never been seen) -> REJECT -----

def test_never_seen_relay_rejects():
    bridge = _make_bridge(max_age_sec=0.5, last_relay_ns=None, now_ns=2_000_000_000)
    assert bridge._goal_callback(None) == GoalResponse.REJECT


# ----- Within threshold -> ACCEPT -----

@pytest.mark.parametrize(
    "max_age,age_sec",
    [(0.5, 0.0), (0.5, 0.1), (0.5, 0.4), (1.0, 0.9), (2.0, 1.5)],
)
def test_within_threshold_accepts(max_age, age_sec):
    now_ns = 5_000_000_000
    last_ns = now_ns - int(age_sec * 1e9)
    bridge = _make_bridge(max_age_sec=max_age, last_relay_ns=last_ns, now_ns=now_ns)
    assert bridge._goal_callback(None) == GoalResponse.ACCEPT


# ----- Above threshold -> REJECT -----

@pytest.mark.parametrize(
    "max_age,age_sec",
    [(0.5, 0.51), (0.5, 1.0), (1.0, 1.01), (1.0, 5.0)],
)
def test_above_threshold_rejects(max_age, age_sec):
    now_ns = 10_000_000_000
    last_ns = now_ns - int(age_sec * 1e9)
    bridge = _make_bridge(max_age_sec=max_age, last_relay_ns=last_ns, now_ns=now_ns)
    assert bridge._goal_callback(None) == GoalResponse.REJECT


# ----- Reject log line includes both age and threshold -----

def test_reject_log_contains_age_and_threshold():
    now_ns = 10_000_000_000
    age_sec = 1.2
    last_ns = now_ns - int(age_sec * 1e9)
    max_age = 0.5
    bridge = _make_bridge(max_age_sec=max_age, last_relay_ns=last_ns, now_ns=now_ns)
    logger = MagicMock()
    bridge.get_logger = lambda: logger

    result = bridge._goal_callback(None)
    assert result == GoalResponse.REJECT
    # warn() should have been called at least once
    assert logger.warn.call_count >= 1
    msg = logger.warn.call_args_list[-1].args[0]
    # Must include both the measured age and the configured threshold
    assert f"{age_sec:.3f}" in msg, f"missing measured age in log: {msg!r}"
    assert f"{max_age:.3f}" in msg, f"missing configured threshold in log: {msg!r}"


def test_never_seen_log_contains_threshold():
    bridge = _make_bridge(max_age_sec=0.5, last_relay_ns=None, now_ns=1_000_000_000)
    logger = MagicMock()
    bridge.get_logger = lambda: logger
    bridge._goal_callback(None)
    msg = logger.warn.call_args_list[-1].args[0]
    assert "0.500" in msg, f"missing configured threshold in never-seen log: {msg!r}"
