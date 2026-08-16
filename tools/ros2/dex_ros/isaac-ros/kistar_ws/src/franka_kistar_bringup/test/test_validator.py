"""Unit tests for _validate_args inside dual_fr3_kistar_planning_pc.launch.py.

The validator is a read-only OpaqueFunction body that reads three LaunchConfigurations
and returns a list of actions that either logs received args (happy path) or appends
LogInfo + EmitEvent(Shutdown) on failure (validation path).

These tests build a real LaunchContext, inject test values into its
launch_configurations dict, call _validate_args(context), then assert on whether the
returned actions include an EmitEvent(Shutdown).

Hardened against the iter-3 Critic C2 fix: empty-string canonical MUST trigger
Shutdown (it does NOT silently fall back to 'fake').
"""
import importlib.util
import os
import sys

import pytest
from launch import LaunchContext
from launch.actions import EmitEvent


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
LAUNCH_PY = os.path.join(
    REPO_ROOT,
    "isaac-ros", "kistar_ws", "src", "franka_kistar_bringup", "launch", "deprecated",
    "dual_fr3_kistar_planning_pc.launch.py",
)


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("dual_fr3_planning_pc_launch", LAUNCH_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dual_fr3_planning_pc_launch"] = module
    spec.loader.exec_module(module)
    return module


LAUNCH_MOD = _load_launch_module()


def _make_context(canonical: str, plural: str = "__UNSET__", robot_ip: str = ""):
    ctx = LaunchContext()
    ctx.launch_configurations["joint_state_mode"] = canonical
    ctx.launch_configurations["joint_states_mode"] = plural
    ctx.launch_configurations["robot_ip"] = robot_ip
    return ctx


def _has_shutdown(actions):
    return any(isinstance(a, EmitEvent) for a in actions)


# ----- Happy path: valid canonical + unset plural -----

@pytest.mark.parametrize("canonical", ["fake", "direct", "merger"])
def test_valid_canonical_no_shutdown(canonical):
    ctx = _make_context(canonical=canonical)
    actions = LAUNCH_MOD._validate_args(ctx)
    assert not _has_shutdown(actions), (
        f"Valid canonical='{canonical}' must NOT emit Shutdown. Got actions: {actions}"
    )


# ----- iter-3 Critic C2: empty string MUST be rejected -----

def test_empty_string_canonical_rejected():
    """iter-3 Critic critical C2 pin: empty string is NOT silently 'fake'.

    In ROS 2 launch, joint_state_mode:='' overrides the default; all three
    IfCondition predicates evaluate False; the launch would silently hang with
    no joint-state source. The validator MUST emit Shutdown.
    """
    ctx = _make_context(canonical="")
    actions = LAUNCH_MOD._validate_args(ctx)
    assert _has_shutdown(actions), (
        "Empty-string canonical MUST emit Shutdown (silent-hang surface). "
        f"Got actions: {actions}"
    )


# ----- Bogus canonical values -----

@pytest.mark.parametrize("bad", ["fakedirect", "DIRECT", "Direct", " fake ", "real", "Fake"])
def test_invalid_canonical_emits_shutdown(bad):
    ctx = _make_context(canonical=bad)
    actions = LAUNCH_MOD._validate_args(ctx)
    assert _has_shutdown(actions), (
        f"Invalid canonical='{bad}' must emit Shutdown. Got actions: {actions}"
    )


# ----- Plural-was-passed: ANY value other than __UNSET__ triggers Shutdown -----

@pytest.mark.parametrize("plural", ["fake", "direct", "merger", "", "anything"])
def test_plural_passed_emits_shutdown(plural):
    ctx = _make_context(canonical="direct", plural=plural)
    actions = LAUNCH_MOD._validate_args(ctx)
    assert _has_shutdown(actions), (
        f"Plural was passed (='{plural}') — validator MUST emit Shutdown. "
        f"Got actions: {actions}"
    )


def test_plural_unset_does_not_emit_shutdown():
    ctx = _make_context(canonical="direct", plural="__UNSET__")
    actions = LAUNCH_MOD._validate_args(ctx)
    assert not _has_shutdown(actions)


# ----- Both spellings simultaneously: still rejected -----

def test_both_spellings_emit_shutdown():
    ctx = _make_context(canonical="direct", plural="fake")
    actions = LAUNCH_MOD._validate_args(ctx)
    assert _has_shutdown(actions)


# ----- robot_ip is informational only: never causes Shutdown by itself -----

@pytest.mark.parametrize("ip", ["", "192.168.0.100", "0.0.0.0", "not.an.ip"])
def test_robot_ip_is_informational(ip):
    ctx = _make_context(canonical="direct", robot_ip=ip)
    actions = LAUNCH_MOD._validate_args(ctx)
    assert not _has_shutdown(actions), (
        f"robot_ip='{ip}' is informational only and must NOT cause Shutdown."
    )
