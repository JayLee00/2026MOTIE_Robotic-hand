#!/usr/bin/env python3
"""measure_launch_time.py — AC4 readiness probe for dual_fr3_kistar_planning_pc_v2.launch.py.

Subscribes to /tf and /joint_states_relay, and creates an ActionClient on /move_action
whose wait_for_server first-success time is the primary "move_group ready" signal.

`headless_ready_max = max(t_tf_first, t_joint_states_relay_first, t_move_action_ready)`

is the AC4 metric. /get_planning_scene service availability and /move_group/status first
message are reported as DIAGNOSTICS — they are NOT included in `headless_ready_max`.
This substitution from the spec's literal /move_group/status to /move_action is
documented in:
  - this module docstring
  - the PR-1 commit message
  - docs/run/run_real_moveit.md "Notes — joint state safety" block

Usage:
    # Standard measurement run (start in another terminal in parallel with `ros2 launch`):
    python3 measure_launch_time.py --timeout 30

    # Print the recommended NON-DESTRUCTIVE cold-launch protocol; does not execute it:
    python3 measure_launch_time.py --cold

    # Print the DESTRUCTIVE cache-drop protocol with safety warning; does not execute it:
    python3 measure_launch_time.py --cold-aggressive

Exit codes:
    0  all three primary signals observed within --timeout
    1  one or more primary signals not observed; partial table printed
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

try:
    from rclpy.action import ActionClient
    from moveit_msgs.action import MoveGroup
    _HAVE_MOVE_GROUP_ACTION = True
except Exception:  # pragma: no cover - environment without moveit_msgs
    _HAVE_MOVE_GROUP_ACTION = False


COLD_PROTOCOL_SAFE = """
# Recommended NON-DESTRUCTIVE cold-launch protocol:
#   Run BEFORE each cold-launch run; do NOT execute drop_caches by default.
pkill -f 'ros2 daemon' || true
ros2 daemon stop || true
sleep 5
# Page-cache effects are noise-bounded across 5-run medians; do not assume
# true cold-cache without the aggressive sequence below.
""".strip()

COLD_PROTOCOL_AGGRESSIVE = """
# DESTRUCTIVE cold-cache protocol — affects ALL users of this machine.
#   The script does NOT execute these. Run by hand only if you understand the impact.
#   Page cache is dropped for every process on the host.
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
pkill -f 'ros2 daemon' || true
ros2 daemon stop || true
sleep 5
""".strip()


class LaunchTimeMeter(Node):
    def __init__(self, timeout: float):
        super().__init__("measure_launch_time")
        self._t0 = time.monotonic()
        self._timeout = timeout
        self._t_tf_first = None
        self._t_joint_states_relay_first = None
        self._t_move_action_ready = None
        # Diagnostic-only:
        self._t_move_group_status_diag = None
        self._t_get_planning_scene_diag = None

        self.create_subscription(TFMessage, "/tf", self._on_tf, 10)
        self.create_subscription(JointState, "/joint_states_relay", self._on_relay, 10)
        # /move_group/status is a publication from move_group during planning — diagnostic only.
        # We accept either rclpy.qos default; a missed first message is fine since it is NOT in
        # headless_ready_max.
        try:
            from action_msgs.msg import GoalStatusArray
            self.create_subscription(
                GoalStatusArray, "/move_action/_action/status", self._on_status, 10
            )
        except Exception:
            pass

        # ActionClient for /move_action — wait_for_server is the PRIMARY readiness signal.
        if _HAVE_MOVE_GROUP_ACTION:
            self._action_client = ActionClient(self, MoveGroup, "/move_action")
        else:
            self._action_client = None

    # ----- Callbacks -----

    def _now_rel(self):
        return time.monotonic() - self._t0

    def _on_tf(self, _msg):
        if self._t_tf_first is None:
            self._t_tf_first = self._now_rel()

    def _on_relay(self, _msg):
        if self._t_joint_states_relay_first is None:
            self._t_joint_states_relay_first = self._now_rel()

    def _on_status(self, _msg):
        if self._t_move_group_status_diag is None:
            self._t_move_group_status_diag = self._now_rel()

    # ----- Main loop -----

    def poll_diagnostics(self):
        # /get_planning_scene service availability is diagnostic only.
        if self._t_get_planning_scene_diag is None:
            try:
                names_types = self.get_service_names_and_types()
                if any(n == "/get_planning_scene" for n, _t in names_types):
                    self._t_get_planning_scene_diag = self._now_rel()
            except Exception:
                pass

    def poll_action_server(self):
        if self._action_client is None or self._t_move_action_ready is not None:
            return
        if self._action_client.server_is_ready():
            self._t_move_action_ready = self._now_rel()

    def primary_all_done(self):
        return (
            self._t_tf_first is not None
            and self._t_joint_states_relay_first is not None
            and self._t_move_action_ready is not None
        )

    def headless_ready_max(self):
        vals = [
            self._t_tf_first,
            self._t_joint_states_relay_first,
            self._t_move_action_ready,
        ]
        if any(v is None for v in vals):
            return None
        return max(vals)

    def report(self):
        def fmt(v):
            return f"{v:7.3f}" if v is not None else "    n/a"
        max_v = self.headless_ready_max()
        print("=" * 60)
        print("measure_launch_time.py — readiness table (seconds since probe start)")
        print(f"  t_tf_first                 = {fmt(self._t_tf_first)}")
        print(f"  t_joint_states_relay_first = {fmt(self._t_joint_states_relay_first)}")
        print(f"  t_move_action_ready        = {fmt(self._t_move_action_ready)}  [PRIMARY]")
        print(f"  t_move_group_status_diag   = {fmt(self._t_move_group_status_diag)}  [diagnostic only]")
        print(f"  t_get_planning_scene_diag  = {fmt(self._t_get_planning_scene_diag)}  [diagnostic only]")
        print(f"  headless_ready_max         = {fmt(max_v)}")
        print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Maximum seconds to wait for all primary signals (default 30)")
    p.add_argument("--cold", action="store_true",
                   help="Print the recommended NON-DESTRUCTIVE cold-launch protocol and exit")
    p.add_argument("--cold-aggressive", action="store_true",
                   help="Print the DESTRUCTIVE drop_caches protocol with safety warning; does NOT execute")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cold:
        print(COLD_PROTOCOL_SAFE)
        return 0
    if getattr(args, "cold_aggressive", False):
        print("!! DESTRUCTIVE — drops the kernel page cache for ALL users on this host !!")
        print(COLD_PROTOCOL_AGGRESSIVE)
        return 0

    rclpy.init()
    node = LaunchTimeMeter(timeout=args.timeout)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.poll_action_server()
            node.poll_diagnostics()
            if node.primary_all_done():
                break
    finally:
        node.report()
        node.destroy_node()
        rclpy.shutdown()

    return 0 if node.primary_all_done() else 1


if __name__ == "__main__":
    sys.exit(main())
