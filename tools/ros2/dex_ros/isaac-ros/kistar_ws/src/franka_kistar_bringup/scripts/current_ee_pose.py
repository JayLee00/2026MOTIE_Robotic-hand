#!/usr/bin/env python3
"""
Current EE Pose Reader

Prints the current end-effector pose (x y z qx qy qz qw) of one arm, computed
from the live TF tree (robot_state_publisher <- /joint_states_relay in direct
mode, i.e. the REAL robot posture). Both reference frames are printed:

  [base]  {side}_fr3_link0 -> ee_link   — paste directly into pose_commander
          (reference_frame:={side}_fr3_link0)
  [world] world -> ee_link              — world-frame pose (camera/table space).
          NOTE: the robot PC's /franka/{side}/ee_target_world uses ITS OWN
          world (T_base_world, identity until calibrated) — with identity
          calibration that equals the robot's base frame, not this world.

Requires dual_fr3_kistar_planning_pc_v2.launch.py to be running (TF source).

Usage:
  ros2 run franka_kistar_bringup current_ee_pose.py --ros-args -p side:=right
  ros2 run franka_kistar_bringup current_ee_pose.py --ros-args -p side:=left -p continuous:=true

Author: Chanyoung Ahn
Date: 2026
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class CurrentEePose(Node):
    def __init__(self):
        super().__init__('current_ee_pose')

        self.declare_parameter('side', 'right')
        # Empty string means "derive from side".
        self.declare_parameter('ee_link', '')
        self.declare_parameter('base_frame', '')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('continuous', False)
        self.declare_parameter('rate_hz', 2.0)
        # How long to keep retrying TF lookups before giving up (one-shot mode).
        self.declare_parameter('timeout_sec', 10.0)

        self.side = self.get_parameter('side').value
        if self.side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got: {self.side!r}")
        self.ee_link = self.get_parameter('ee_link').value or f'{self.side}_fr3_link8'
        self.base_frame = self.get_parameter('base_frame').value or f'{self.side}_fr3_link0'
        self.world_frame = self.get_parameter('world_frame').value
        self.continuous = bool(self.get_parameter('continuous').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._elapsed = 0.0
        self._printed_header = False
        # Set by _tick when finished; main() spins until this flips (calling
        # rclpy.shutdown() from inside a timer callback is unreliable).
        self.done = False

        period = 1.0 / max(0.1, self.rate_hz)
        self._timer = self.create_timer(period, self._tick)

    def _lookup(self, ref_frame):
        """Return 'x y z qx qy qz qw' for ref_frame -> ee_link, or None."""
        try:
            tf = self._tf_buffer.lookup_transform(ref_frame, self.ee_link, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return f'{t.x:.4f} {t.y:.4f} {t.z:.4f} {q.x:.4f} {q.y:.4f} {q.z:.4f} {q.w:.4f}'

    def _tick(self):
        base_pose = self._lookup(self.base_frame)
        world_pose = self._lookup(self.world_frame)

        if base_pose is None and world_pose is None:
            self._elapsed += 1.0 / max(0.1, self.rate_hz)
            if self._elapsed >= self.timeout_sec:
                self.get_logger().error(
                    f'No TF for {self.ee_link} within {self.timeout_sec}s — '
                    'is dual_fr3_kistar_planning_pc_v2.launch.py running '
                    '(and, in direct mode, is the robot streaming joint states)?'
                )
                self.done = True
            return

        if not self._printed_header:
            print(f'# ee_link: {self.ee_link}   (format: x y z qx qy qz qw)', flush=True)
            self._printed_header = True

        if base_pose is not None:
            print(f'[base  {self.base_frame}] {base_pose}', flush=True)
        else:
            print(f'[base  {self.base_frame}] (TF unavailable)', flush=True)
        if world_pose is not None:
            print(f'[world {self.world_frame}] {world_pose}', flush=True)
        else:
            print(f'[world {self.world_frame}] (TF unavailable)', flush=True)

        if not self.continuous:
            self.done = True
        else:
            print('-' * 68, flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = CurrentEePose()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
