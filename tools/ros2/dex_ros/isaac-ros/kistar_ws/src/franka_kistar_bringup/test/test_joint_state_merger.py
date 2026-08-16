"""Unit tests for joint_state_merger.py — bidirectional asymmetry + downstream correctness.

iter-3 Critic critical M3 pin:
    The merger previously only trimmed (position/velocity/effort) when LONGER than
    len(name). For the project-memory documented anomaly on the robot PC ("/joint_states
    has 47 positions for 46 names"), the inverse direction (len(position) < len(name))
    left the published JointState with len(name) > len(position) — a silent data
    corruption surface for robot_state_publisher.

These tests:
    1. Exercise _validate on both asymmetry directions and assert symmetric trim.
    2. Exercise _publish end-to-end with a stub publisher and assert downstream
       JointState has len(name) == len(position) (no mismatch warnings on the
       consumer side).
"""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest
from builtin_interfaces.msg import Time as _RosTime
from sensor_msgs.msg import JointState


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
MERGER_PY = os.path.join(
    REPO_ROOT,
    "isaac-ros", "kistar_ws", "src", "franka_kistar_bringup", "scripts",
    "joint_state_merger.py",
)


def _load_merger():
    spec = importlib.util.spec_from_file_location("joint_state_merger_mod", MERGER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_state_merger_mod"] = module
    spec.loader.exec_module(module)
    return module


MERGER_MOD = _load_merger()
JointStateMerger = MERGER_MOD.JointStateMerger


def _make_merger(remap_rules=None):
    """Construct a JointStateMerger without calling Node.__init__.

    Populate the attributes _validate / _publish read.
    """
    merger = JointStateMerger.__new__(JointStateMerger)
    merger._remap_rules = list(remap_rules) if remap_rules else []
    merger._latest = {}
    merger.get_logger = lambda: MagicMock()
    return merger


def _make_joint_state(names, positions, velocities=None, efforts=None):
    msg = JointState()
    msg.name = list(names)
    msg.position = list(positions)
    msg.velocity = list(velocities) if velocities is not None else []
    msg.effort = list(efforts) if efforts is not None else []
    return msg


# ----- _validate: original direction (positions LONGER than names) -----

def test_validate_47pos_46names_trims_to_46():
    """The original anomaly: 47 positions for 46 names. Trim to 46/46."""
    names = [f"j{i}" for i in range(46)]
    positions = [float(i) for i in range(47)]
    merger = _make_merger()
    out = merger._validate(_make_joint_state(names, positions))
    assert out is not None
    assert len(out.name) == 46
    assert len(out.position) == 46


# ----- _validate: SYMMETRIC direction (names LONGER than positions) -----

def test_validate_46pos_47names_trims_to_46():
    """iter-3 Critic M3: the SYMMETRIC direction MUST also trim, not silently
    leave len(name) > len(position) for downstream corruption.
    """
    names = [f"j{i}" for i in range(47)]
    positions = [float(i) for i in range(46)]
    merger = _make_merger()
    out = merger._validate(_make_joint_state(names, positions))
    assert out is not None
    assert len(out.name) == 46, "names MUST be trimmed in the inverse-asymmetry case"
    assert len(out.position) == 46


# ----- _validate: empty (0/0) -> None with WARN -----

def test_validate_empty_returns_none():
    merger = _make_merger()
    logger = MagicMock()
    merger.get_logger = lambda: logger
    out = merger._validate(_make_joint_state([], []))
    assert out is None
    logger.warn.assert_called()


# ----- _validate: equal-length identity -----

def test_validate_equal_length_identity():
    names = [f"j{i}" for i in range(7)]
    positions = [float(i) for i in range(7)]
    merger = _make_merger()
    out = merger._validate(_make_joint_state(names, positions))
    assert out is not None
    assert len(out.name) == 7
    assert len(out.position) == 7


# ----- _validate: remap is applied before trim -----

def test_validate_remap_then_trim():
    names = ["fr3_l_joint1", "fr3_l_joint2"]
    positions = [0.0, 0.0]
    merger = _make_merger(remap_rules=[("fr3_l_", "left_fr3_")])
    out = merger._validate(_make_joint_state(names, positions))
    assert out is not None
    assert out.name == ["left_fr3_joint1", "left_fr3_joint2"]


# ----- _publish: downstream correctness with anomalous input -----

def test_publish_emits_equal_length_joint_state_with_anomaly():
    """Downstream-correctness assertion: feed the project-memory anomaly through
    _validate -> _publish and assert the published JointState has equal name and
    position lengths (a stub robot_state_publisher-like consumer would see no
    name/position mismatch warning).
    """
    merger = _make_merger()
    # Stub clock + publisher (clock must return a real Time message — std_msgs.Header
    # sets stamp via a type-checking setter).
    fake_clock = MagicMock()
    fake_clock.now.return_value.to_msg.return_value = _RosTime(sec=0, nanosec=0)
    merger.get_clock = lambda: fake_clock

    captured = []
    merger._pub = MagicMock()
    merger._pub.publish.side_effect = lambda msg: captured.append(msg)

    # Anomalous input (47 positions / 46 names) under one input topic
    names = [f"j{i}" for i in range(46)]
    positions = [float(i) for i in range(47)]
    merger._latest["/joint_states_l"] = _make_joint_state(names, positions)
    merger._publish()

    assert len(captured) == 1
    out = captured[0]
    # Downstream-correctness assertion: lengths match (no robot_state_publisher
    # mismatch warning would fire on the consumer side).
    assert len(out.name) == len(out.position), (
        f"Published JointState has length mismatch (downstream corruption surface): "
        f"len(name)={len(out.name)} len(position)={len(out.position)}"
    )


def test_publish_emits_equal_length_with_inverse_asymmetry():
    """Symmetric coverage: 46 positions / 47 names through _publish."""
    merger = _make_merger()
    fake_clock = MagicMock()
    fake_clock.now.return_value.to_msg.return_value = _RosTime(sec=0, nanosec=0)
    merger.get_clock = lambda: fake_clock

    captured = []
    merger._pub = MagicMock()
    merger._pub.publish.side_effect = lambda msg: captured.append(msg)

    names = [f"j{i}" for i in range(47)]
    positions = [float(i) for i in range(46)]
    merger._latest["/joint_states_r"] = _make_joint_state(names, positions)
    merger._publish()

    assert len(captured) == 1
    out = captured[0]
    assert len(out.name) == len(out.position) == 46


def test_publish_merges_two_inputs_with_anomaly_each():
    """End-to-end: two upstream topics each have their own asymmetry. The merged
    output must still have equal name+position lengths.
    """
    merger = _make_merger()
    fake_clock = MagicMock()
    fake_clock.now.return_value.to_msg.return_value = _RosTime(sec=0, nanosec=0)
    merger.get_clock = lambda: fake_clock
    captured = []
    merger._pub = MagicMock()
    merger._pub.publish.side_effect = lambda msg: captured.append(msg)

    # Left: 46 names / 47 positions (original anomaly)
    merger._latest["/joint_states_l"] = _make_joint_state(
        [f"l{i}" for i in range(46)], [float(i) for i in range(47)]
    )
    # Right: 47 names / 46 positions (inverse anomaly)
    merger._latest["/joint_states_r"] = _make_joint_state(
        [f"r{i}" for i in range(47)], [float(i) for i in range(46)]
    )
    merger._publish()
    out = captured[0]
    assert len(out.name) == len(out.position)
    # Each side contributed 46 valid (trimmed) entries
    assert len(out.name) == 92
