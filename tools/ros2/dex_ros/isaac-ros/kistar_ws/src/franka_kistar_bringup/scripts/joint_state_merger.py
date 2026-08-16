#!/usr/bin/env python3
"""Merge/relay joint states with message validation and joint name remapping.

Modes (configured via launch):
  - Merge: subscribe to multiple input topics, merge into one output
  - Relay: subscribe to a single input topic, validate and republish

Supports joint name remapping (e.g. robot convention → URDF convention).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateMerger(Node):
    def __init__(self):
        super().__init__("joint_state_merger")

        self.declare_parameter("input_topics", ["/left/joint_states", "/right/joint_states"])
        self.declare_parameter("output_topic", "/joint_states_relay")
        self.declare_parameter("publish_rate", 50.0)
        # Joint name remapping: comma-separated "src:dst" pairs
        # e.g. "fr3_l_:left_fr3_,fr3_r_:right_fr3_,l_:left_,r_:right_"
        self.declare_parameter("joint_name_remap", "")
        # Position unit conversion: comma-separated "substr:factor" pairs matched
        # against the REMAPPED joint name (first match wins). Needed because the
        # robot PC publishes hand joints in raw encoder counts (1 tick = pi/8192
        # rad; see Dual_Arm_Hand_Ctrl inc/shm.h "rad 변환은 소비자 책임") while
        # robot_state_publisher expects radians — unconverted, encoder noise of a
        # few ticks renders as a few RADIANS of finger motion in RViz.
        # e.g. "index_:3.834951969714103e-4,thumb_:3.834951969714103e-4,..."
        self.declare_parameter("joint_position_scale", "")

        input_topics = self.get_parameter("input_topics").get_parameter_value().string_array_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        remap_str = self.get_parameter("joint_name_remap").get_parameter_value().string_value
        scale_str = self.get_parameter("joint_position_scale").get_parameter_value().string_value

        # Parse remap rules: longest prefix first to avoid partial matches
        self._remap_rules = []
        if remap_str:
            for rule in remap_str.split(","):
                src, dst = rule.strip().split(":")
                self._remap_rules.append((src, dst))
        self._remap_rules.sort(key=lambda r: -len(r[0]))

        if self._remap_rules:
            self.get_logger().info(f"Joint name remap rules: {self._remap_rules}")

        # Parse scale rules; matched by substring containment on the remapped name.
        self._scale_rules = []
        if scale_str:
            for rule in scale_str.split(","):
                substr, factor = rule.strip().split(":")
                self._scale_rules.append((substr, float(factor)))
        if self._scale_rules:
            self.get_logger().info(f"Joint position scale rules: {self._scale_rules}")

        self._latest = {}
        self._subs = []
        for topic in input_topics:
            sub = self.create_subscription(
                JointState, topic,
                lambda msg, t=topic: self._on_joint_state(t, msg),
                10,
            )
            self._subs.append(sub)
            self.get_logger().info(f"Subscribing to: {topic}")

        self._pub = self.create_publisher(JointState, output_topic, 10)
        self._timer = self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f"Publishing on: {output_topic} at {rate} Hz")

    def _remap_name(self, name):
        for src, dst in self._remap_rules:
            if name.startswith(src):
                return dst + name[len(src):]
        return name

    def _validate(self, msg):
        """Fix mismatched array sizes (BIDIRECTIONAL) and remap joint names.

        Bug pinned by the project-memory note: robot PC has historically published
        a /joint_states with 47 positions for 46 names. The previous implementation
        only trimmed when len(position) > len(name); the inverse direction left a
        downstream JointState with len(name) > len(position) — silent data corruption
        in robot_state_publisher.

        Canonical length n = min(len(name), len(position) or len(name)). All arrays
        (name, position, velocity, effort) are trimmed symmetrically to n. velocity
        and effort drop entirely when present but mismatched, since their semantics
        are positional w.r.t. name.
        """
        n_name = len(msg.name)
        n_pos = len(msg.position) if msg.position else n_name
        n = min(n_name, n_pos)
        if n == 0:
            self.get_logger().warn(
                "empty joint state from upstream (n_name=%d, n_pos=%d); dropping" % (n_name, n_pos)
            )
            return None
        # Remap names BEFORE trim so the remap operates on the full list (a remap
        # that depends on a name suffix should not be silently truncated by trim).
        if self._remap_rules:
            msg.name = [self._remap_name(nm) for nm in msg.name]
        # Bidirectional trim: both name>position and position>name are handled.
        if len(msg.name) > n:
            msg.name = list(msg.name[:n])
        if len(msg.position) > n:
            msg.position = list(msg.position[:n])
        if msg.velocity:
            if len(msg.velocity) > n:
                msg.velocity = list(msg.velocity[:n])
            elif len(msg.velocity) < n:
                # Mismatched-shorter velocity is unusable positionally; drop it.
                msg.velocity = []
        if msg.effort:
            if len(msg.effort) > n:
                msg.effort = list(msg.effort[:n])
            elif len(msg.effort) < n:
                msg.effort = []
        # Unit conversion (e.g. hand encoder ticks → rad), matched on the
        # remapped name. Runs ONCE per incoming message (_on_joint_state), never
        # per publish tick — scaling is not idempotent, re-applying it every
        # tick would collapse the values.
        if self._scale_rules:
            position = list(msg.position)
            # min() guard: a malformed upstream message can carry fewer
            # positions than names (the robot PC has history of mismatched
            # array sizes) — indexing past position would crash the relay.
            for i in range(min(len(msg.name), len(position))):
                for substr, factor in self._scale_rules:
                    if substr in msg.name[i]:
                        position[i] *= factor
                        break
            msg.position = position
        return msg

    def _publish(self):
        if not self._latest:
            return

        merged = JointState()
        merged.header.stamp = self.get_clock().now().to_msg()

        for validated in self._latest.values():
            merged.name.extend(validated.name)
            merged.position.extend(validated.position)
            merged.velocity.extend(validated.velocity)
            merged.effort.extend(validated.effort)

        self._pub.publish(merged)

    def _on_joint_state(self, topic, msg):
        # Validate/remap/scale once per incoming message. The previous design
        # re-ran _validate on the cached message every publish tick; that was
        # only safe because prefix remaps happen not to re-match their own
        # output — position scaling has no such accident, so processing moved
        # here (also cheaper: work scales with input rate, not publish rate).
        validated = self._validate(msg)
        if validated is not None:
            self._latest[topic] = validated


def main(args=None):
    rclpy.init(args=args)
    node = JointStateMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
