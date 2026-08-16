#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

import tf2_ros


class PlanningSceneStaticBoxes(Node):
    def __init__(self):
        super().__init__("planning_scene_static_boxes")

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("table_frame", "table_link")
        self.declare_parameter("ttable_frame", "ttable_link")
        self.declare_parameter("sw_frame", "sw_wall_frame")
        self.declare_parameter("profile_frame", "profile_frame")

        self.declare_parameter("table_size", [1.0, 0.8, 0.05])
        self.declare_parameter("ttable_size", [0.6, 0.6, 0.05])
        self.declare_parameter("sw_wall_size", [0.1, 2.0, 2.0])

        # Side camera collision box at side_camera_link, world-axis aligned.
        # side_camera_z_offset shifts the box centre vertically relative to the
        # camera frame (negative = lower). Default -0.30 m moves the box down so
        # its body wraps the camera pole below the lens.
        self.declare_parameter("side_camera_frame", "side_camera_link")
        self.declare_parameter("side_camera_size", [0.2, 0.2, 0.7])
        self.declare_parameter("side_camera_z_offset", -0.30)

        self.declare_parameter("timeout_sec", 20.0)

        self.world_frame = self.get_parameter("world_frame").value
        self.table_frame = self.get_parameter("table_frame").value
        self.ttable_frame = self.get_parameter("ttable_frame").value
        self.table_size = self.get_parameter("table_size").value
        self.ttable_size = self.get_parameter("ttable_size").value

        self.sw_wall_frame = self.get_parameter("sw_frame").value
        self.sw_wall_size = self.get_parameter("sw_wall_size").value
        self.profile_frame = self.get_parameter("profile_frame").value

        self.side_camera_frame = self.get_parameter("side_camera_frame").value
        self.side_camera_size = self.get_parameter("side_camera_size").value
        self.side_camera_z_offset = float(self.get_parameter("side_camera_z_offset").value)

        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.timer = self.create_timer(0.5, self._tick)
        self.start_time = self.get_clock().now()

    def _lookup_pose(self, target_frame: str) -> Pose:
        tf = self.tf_buffer.lookup_transform(
            self.world_frame, target_frame, rclpy.time.Time()
        )
        p = Pose()
        p.position.x = tf.transform.translation.x
        p.position.y = tf.transform.translation.y
        p.position.z = tf.transform.translation.z
        p.orientation = tf.transform.rotation
        return p

    def _make_box(self, object_id: str, pose: Pose, size_xyz) -> CollisionObject:
        co = CollisionObject()
        co.id = object_id
        co.header.frame_id = self.world_frame

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        co.primitives = [prim]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        return co

    def _tick(self):
        # timeout
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed > self.timeout_sec:
            self.get_logger().error("Timeout waiting for TF/service. Giving up.")
            rclpy.shutdown()
            return

        if not self.client.wait_for_service(timeout_sec=0.5):
            return

        try:
            table_pose = self._lookup_pose(self.table_frame)
            ttable_pose = self._lookup_pose(self.ttable_frame)
            sw_wall_pose = self._lookup_pose(self.sw_wall_frame)
            profile_pose = self._lookup_pose(self.profile_frame)
            side_camera_pose = self._lookup_pose(self.side_camera_frame)
            # Force world-axis alignment so the box is not rotated by the camera's pitch/yaw.
            side_camera_pose.orientation.x = 0.0
            side_camera_pose.orientation.y = 0.0
            side_camera_pose.orientation.z = 0.0
            side_camera_pose.orientation.w = 1.0
            # Drop the box centre below the camera (default -0.30 m so the box wraps the pole).
            side_camera_pose.position.z += self.side_camera_z_offset

            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects.append(
                self._make_box("table", table_pose, self.table_size)
            )
            scene.world.collision_objects.append(
                self._make_box("ttable", ttable_pose, self.ttable_size)
            )

            # Table Camera
            scene.world.collision_objects.append(
                self._make_box("camera", profile_pose, [0.25, 0.1, 2.5])
            )
            # scene.world.collision_objects.append(self._make_box("camera", [0.05, 0.032, 0.], [0.22, 0.15, 2.0]))

            # SW Wall
            scene.world.collision_objects.append(
                self._make_box("sw_wall", sw_wall_pose, self.sw_wall_size)
            )

            # Side camera (right-arm side, 40x40x60 cm). Axis-aligned at side_camera_link.
            scene.world.collision_objects.append(
                self._make_box("side_camera", side_camera_pose, self.side_camera_size)
            )

            req = ApplyPlanningScene.Request()
            req.scene = scene
            fut = self.client.call_async(req)

            # NOTE: do NOT call rclpy.spin_until_future_complete() inside a timer
            # callback under humble — it deadlocks/triggers "sequence size exceeds
            # remaining buffer" because the executor is already spinning us.
            # Instead, attach a done-callback and let the timer keep ticking.
            fut.add_done_callback(self._on_apply_done)
            # Cancel further ticks; the done-callback will shutdown the node.
            self.timer.cancel()

        except Exception as e:
            # TF 아직 안 올라왔을 수 있음 -> 다음 tick에서 재시도
            self.get_logger().warn(f"Waiting... ({e})")

    def _on_apply_done(self, fut):
        try:
            res = fut.result()
            if res and res.success:
                self.get_logger().info(
                    "Applied planning scene: added table/ttable/camera/sw_wall/side_camera collision boxes."
                )
            else:
                self.get_logger().error("Failed to apply planning scene.")
        except Exception as e:
            self.get_logger().error(f"ApplyPlanningScene call raised: {e}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = PlanningSceneStaticBoxes()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
