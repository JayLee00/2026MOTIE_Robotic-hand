#!/usr/bin/env python3
"""Joint-states observer — OBSERVABILITY ONLY (NOT a protection mechanism).

DDS multi-subscriber semantics mean this node CANNOT block, drop, or "deny" any other
subscriber on /joint_states. The actual protection against the robot PC's stray
fake /joint_states reaching the planning PC's robot_state_publisher / move_group is
the DESTINATION REWRITE to /joint_states_relay in dual_fr3_kistar_planning_pc.launch.py
(see the comment at the robot_rsp / move_group remappings). This node only surfaces a
WARN-level DiagnosticStatus so operators see in the field when something is publishing
to the bare /joint_states topic.

Behavior:
- Subscribes to /joint_states (sensor_msgs/JointState) so DDS surfaces the publisher to us.
- Every 1.0s, queries get_publishers_info_by_topic("/joint_states"):
  - If publisher_count > 0, publishes diagnostic_msgs/DiagnosticStatus with level=WARN.
  - If publisher_count == 0, no DiagnosticStatus is published (silent good state).
- The "mode" parameter is passed for diagnostic message context only; the launch file
  is responsible for only spawning this node when mode in {direct, merger}.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue


PUBLISH_PERIOD_SEC = 1.0
TOPIC = "/joint_states"


class JointStatesObserver(Node):
    def __init__(self):
        super().__init__("joint_states_observer")
        self.declare_parameter("mode", "direct")
        self.declare_parameter("diagnostic_topic", "/diagnostics")
        mode = self.get_parameter("mode").get_parameter_value().string_value
        diag_topic = self.get_parameter("diagnostic_topic").get_parameter_value().string_value

        self._mode = mode
        self._last_msg_count = 0

        # Subscribe so DDS exposes the publisher to us; we do not act on payloads.
        self._sub = self.create_subscription(
            JointState,
            TOPIC,
            self._on_joint_state,
            10,
        )
        self._pub = self.create_publisher(DiagnosticArray, diag_topic, 10)
        self._timer = self.create_timer(PUBLISH_PERIOD_SEC, self._tick)

        self.get_logger().info(
            f"joint_states_observer started: mode='{mode}'. "
            f"OBSERVABILITY ONLY — cannot block subscribers. Real protection is the "
            f"destination remap in the launch file."
        )

    def _on_joint_state(self, _msg):
        self._last_msg_count += 1

    def _publisher_count(self):
        try:
            return len(self.get_publishers_info_by_topic(TOPIC))
        except Exception:
            return 0

    def _tick(self):
        count = self._publisher_count()
        if count == 0:
            return  # quiet healthy state — no DiagnosticStatus emitted
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.WARN
        status.name = "joint_states_observer"
        status.message = (
            f"/joint_states has {count} publisher(s) while mode='{self._mode}'. "
            "Planning PC consumers are remapped to /joint_states_relay so this is "
            "OBSERVABILITY-ONLY; it indicates a stray publisher (often the robot PC's "
            "fake joint_state_publisher) is on the wire."
        )
        status.hardware_id = "planning_pc"
        status.values = [
            KeyValue(key="publisher_count", value=str(count)),
            KeyValue(key="mode", value=self._mode),
            KeyValue(key="observed_msgs_since_start", value=str(self._last_msg_count)),
        ]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self._pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatesObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
