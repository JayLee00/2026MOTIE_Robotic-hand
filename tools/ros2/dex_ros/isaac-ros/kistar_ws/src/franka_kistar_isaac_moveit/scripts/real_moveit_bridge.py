#!/usr/bin/env python3
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

from kistar_hand_ros2.msg import FrankaArmState, FrankaArmTarget


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _as_list(val, n: int) -> List[float]:
    if isinstance(val, (int, float)):
        return [float(val)] * n
    if isinstance(val, list) and len(val) == n:
        return [float(x) for x in val]
    # fallback
    return [float(val)] * n


def global_scale_for_limits(vec: List[float], limits: List[float]) -> float:
    """Return scale in (0,1] so that |scale*vec[i]| <= limits[i] for all i."""
    scale = 1.0
    for v, lim in zip(vec, limits):
        av = abs(v)
        if av > 1e-12:
            scale = min(scale, lim / av)
    return scale


class RealMoveItBridge(Node):
    """MoveIt -> real Franka arm bridge (right arm by default)."""

    def __init__(self):
        super().__init__("real_moveit_bridge")

        # ------------------- ROS params -------------------
        self.declare_parameter("arm_side", "right")
        self.declare_parameter("arm_state_topic", "")
        self.declare_parameter("arm_target_topic", "")
        self.declare_parameter("joint_states_topic", "joint_states")
        self.declare_parameter(
            "traj_action_name", "/fr3_arm_controller/follow_joint_trajectory"
        )
        self.declare_parameter("command_rate_hz", 100.0)
        self.declare_parameter(
            "joint_names",
            [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
        )
        self.declare_parameter("min_dt", 0.002)

        # ------------------- Tracker params -------------------
        self.declare_parameter("use_jerk_limited_tracker", True)

        # PD on measured (q, dq_est) to generate desired acceleration
        # self.declare_parameter("kp", 45.0)              # rad -> rad/s^2
        self.declare_parameter("kp", 10.0)  # rad -> rad/s^2
        self.declare_parameter("kd", 5.0)  # (rad/s error) -> rad/s^2

        self.declare_parameter("deadband_rad", 0.0002)
        self.declare_parameter("deadband_vel", 0.01)
        self.declare_parameter("settle_time", 0.15)  # extra time to settle (sec)

        # # limits (scalar or per-joint list)
        self.declare_parameter("max_accel", [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        self.declare_parameter("use_vel_limit", True)
        self.declare_parameter(
            "max_vel", [2.0, 2.0, 2.0, 2.0, 3.0, 2.5, 3.0]
        )  # 대충 70~80%로 시작
        self.declare_parameter(
            "max_jerk", 12.0
        )  # 15 넘기면 너는 흔들린다 했으니 10~12 추천

        # # optional velocity limit (helps prevent aggressive catch-up)
        # self.declare_parameter("use_vel_limit", True)
        # self.declare_parameter("max_vel", 2.0)          # rad/s (scalar)

        # seed internal state from measured at goal start
        self.declare_parameter("seed_from_state", True)

        # anti-windup / resync (if internal state drifts too far from measured)
        self.declare_parameter(
            "resync_error_rad", 0.35
        )  # if |q_cmd-q_meas| exceeds -> snap back
        self.declare_parameter(
            "resync_dq_to_meas", True
        )  # when resync, also zero dq_cmd

        # dq estimation from positions (since FrankaArmState has no velocity)
        self.declare_parameter("dq_lpf_alpha", 0.93)  # 0.85~0.95 typical
        # self.declare_parameter("dq_lpf_alpha", 0.96)     # 0.85~0.95 typical

        # dummy hand joint_states publish (to satisfy MoveIt complete state)
        self.declare_parameter("publish_dummy_hand_joints", False)
        self.declare_parameter(
            "dummy_hand_joint_names",
            [
                "index_joint_0",
                "index_joint_1",
                "index_joint_2",
                "index_joint_3",
                "middle_joint_0",
                "middle_joint_1",
                "middle_joint_2",
                "middle_joint_3",
                "ring_joint_0",
                "ring_joint_1",
                "ring_joint_2",
                "ring_joint_3",
                "thumb_joint_0",
                "thumb_joint_1",
                "thumb_joint_2",
                "thumb_joint_3",
            ],
        )
        self.declare_parameter("dummy_hand_joint_value", 0.0)

        # ------------------- Parse params -------------------
        arm_side = self.get_parameter("arm_side").get_parameter_value().string_value
        if arm_side not in ("left", "right"):
            self.get_logger().warn(
                f"Invalid arm_side '{arm_side}', falling back to 'right'."
            )
            arm_side = "right"
        self.arm_id = 0 if arm_side == "right" else 1

        arm_state_topic = (
            self.get_parameter("arm_state_topic").get_parameter_value().string_value
        )
        arm_target_topic = (
            self.get_parameter("arm_target_topic").get_parameter_value().string_value
        )
        joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        traj_action_name = (
            self.get_parameter("traj_action_name").get_parameter_value().string_value
        )

        if not arm_state_topic:
            arm_state_topic = f"/franka/arm_state/{arm_side}"
        if not arm_target_topic:
            arm_target_topic = f"/franka/arm_target/{arm_side}"

        self.joint_names = list(
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        if not self.joint_names:
            self.joint_names = [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ]
        self.nj = len(self.joint_names)

        self.command_rate_hz = float(
            self.get_parameter("command_rate_hz").get_parameter_value().double_value
        )
        self.command_rate_hz = max(self.command_rate_hz, 1.0)
        self.min_dt = max(
            float(self.get_parameter("min_dt").get_parameter_value().double_value),
            0.001,
        )

        self.use_tracker = bool(
            self.get_parameter("use_jerk_limited_tracker")
            .get_parameter_value()
            .bool_value
        )
        self.kp = float(self.get_parameter("kp").get_parameter_value().double_value)
        self.kd = float(self.get_parameter("kd").get_parameter_value().double_value)

        self.deadband = float(
            self.get_parameter("deadband_rad").get_parameter_value().double_value
        )
        self.deadband_vel = float(
            self.get_parameter("deadband_vel").get_parameter_value().double_value
        )
        self.settle_time = float(
            self.get_parameter("settle_time").get_parameter_value().double_value
        )

        a_raw = self.get_parameter("max_accel").value  # float 또는 list
        j_raw = self.get_parameter("max_jerk").value
        v_raw = self.get_parameter("max_vel").value

        self.a_max = _as_list(a_raw, self.nj)
        self.j_max = _as_list(j_raw, self.nj)
        self.v_max = _as_list(v_raw, self.nj)

        # self.a_max = _as_list(float(self.get_parameter("max_accel").get_parameter_value().double_value), self.nj)
        # self.j_max = _as_list(float(self.get_parameter("max_jerk").get_parameter_value().double_value), self.nj)

        self.use_vel_limit = bool(
            self.get_parameter("use_vel_limit").get_parameter_value().bool_value
        )
        # self.v_max = _as_list(float(self.get_parameter("max_vel").get_parameter_value().double_value), self.nj)

        self.seed_from_state = bool(
            self.get_parameter("seed_from_state").get_parameter_value().bool_value
        )
        self.resync_err = float(
            self.get_parameter("resync_error_rad").get_parameter_value().double_value
        )
        self.resync_dq = bool(
            self.get_parameter("resync_dq_to_meas").get_parameter_value().bool_value
        )

        self.dq_alpha = float(
            self.get_parameter("dq_lpf_alpha").get_parameter_value().double_value
        )
        self.dq_alpha = clamp(self.dq_alpha, 0.0, 0.999)

        self.publish_dummy_hand = bool(
            self.get_parameter("publish_dummy_hand_joints")
            .get_parameter_value()
            .bool_value
        )
        self.dummy_names = list(
            self.get_parameter("dummy_hand_joint_names")
            .get_parameter_value()
            .string_array_value
        )
        self.dummy_value = float(
            self.get_parameter("dummy_hand_joint_value")
            .get_parameter_value()
            .double_value
        )

        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        # ------------------- Measured state (q, dq_est) -------------------
        self._last_q_meas: Optional[List[float]] = None
        self._last_q_meas_prev: Optional[List[float]] = None
        self._last_state_t: Optional[float] = None
        self._dq_est: Optional[List[float]] = None  # LPF estimated velocities

        # ------------------- ROS interfaces -------------------
        self.cb_group = ReentrantCallbackGroup()
        self.js_pub = self.create_publisher(JointState, joint_states_topic, 10)

        self.state_sub = self.create_subscription(
            FrankaArmState,
            arm_state_topic,
            self._arm_state_cb,
            qos_profile_sensor_data,
            callback_group=self.cb_group,
        )
        self.cmd_pub = self.create_publisher(FrankaArmTarget, arm_target_topic, 10)

        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            traj_action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            f"RealMoveItBridge started (arm_side={arm_side}, state={arm_state_topic}, target={arm_target_topic}, "
            f"rate={self.command_rate_hz}Hz, tracker={self.use_tracker}, dq_est_alpha={self.dq_alpha})."
        )

    # ------------------- Callbacks -------------------
    def _arm_state_cb(self, msg: FrankaArmState):
        q = list(msg.joint_positions)
        if len(q) != self.nj:
            self.get_logger().warn(f"ArmState length mismatch: {len(q)} vs {self.nj}")
            return

        now = time.monotonic()
        if self._last_state_t is not None and self._last_q_meas is not None:
            dt = now - self._last_state_t
            if dt > 1e-4:
                dq_raw = [(q[i] - self._last_q_meas[i]) / dt for i in range(self.nj)]
                if self._dq_est is None:
                    self._dq_est = dq_raw
                else:
                    a = self.dq_alpha
                    self._dq_est = [
                        a * self._dq_est[i] + (1.0 - a) * dq_raw[i]
                        for i in range(self.nj)
                    ]

        self._last_q_meas_prev = self._last_q_meas
        self._last_q_meas = q
        self._last_state_t = now

        # publish joint_states for MoveIt
        js = JointState()
        js.header.stamp = self.system_clock.now().to_msg()

        if self.publish_dummy_hand:
            js.name = self.joint_names + self.dummy_names
            js.position = q + [self.dummy_value] * len(self.dummy_names)
        else:
            js.name = self.joint_names
            js.position = q

        # torque -> effort (optional)
        if hasattr(msg, "joint_torques") and len(msg.joint_torques) == self.nj:
            eff = list(msg.joint_torques)
            if self.publish_dummy_hand:
                eff += [0.0] * len(self.dummy_names)
            js.effort = eff

        self.js_pub.publish(js)

    def goal_callback(self, goal_request):
        traj = goal_request.trajectory
        self.get_logger().info(
            f"[Action] Goal received: {len(traj.points)} points, joints={list(traj.joint_names)}"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("[Action] Cancel requested")
        return CancelResponse.ACCEPT

    # ------------------- Trajectory utils -------------------
    def _prep_trajectory(
        self, traj
    ) -> Tuple[List[float], List[List[float]], List[Optional[List[float]]]]:
        n = len(traj.points)
        goal_names = list(traj.joint_names)

        order_idx = None
        if goal_names and self.joint_names and goal_names != self.joint_names:
            try:
                order_idx = [goal_names.index(name) for name in self.joint_names]
                self.get_logger().warn(
                    "Joint order differs; reordering to match expected joint_names."
                )
            except ValueError:
                self.get_logger().warn(
                    "Joint name mismatch; using goal order without reordering."
                )
                order_idx = None

        times: List[float] = []
        qs: List[List[float]] = []
        qds: List[Optional[List[float]]] = []
        last_t = 0.0

        for pt in traj.points:
            q = list(pt.positions)
            if order_idx is not None:
                q = [q[idx] for idx in order_idx]
            if len(q) != self.nj:
                raise RuntimeError("Position length mismatch")

            v = None
            if len(pt.velocities) == self.nj:
                v = list(pt.velocities)
                if order_idx is not None:
                    v = [v[idx] for idx in order_idx]

            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if t <= last_t:
                t = last_t + self.min_dt

            times.append(t)
            qs.append(q)
            qds.append(v)
            last_t = t

        if n >= 2 and times[-1] <= 0.0:
            times[-1] = self.min_dt

        return times, qs, qds

    def _ref_at(self, t: float, times, qs, qds) -> Tuple[List[float], List[float]]:
        idx = 0
        while idx < len(times) - 2 and t > times[idx + 1]:
            idx += 1

        t0, t1 = times[idx], times[idx + 1]
        q0, q1 = qs[idx], qs[idx + 1]
        v0, v1 = qds[idx], qds[idx + 1]

        dt = max(t1 - t0, 1e-6)
        s = clamp((t - t0) / dt, 0.0, 1.0)

        # If MoveIt provides velocities, use cubic Hermite for smooth reference.
        if v0 is not None and v1 is not None:
            s2 = s * s
            s3 = s2 * s

            h00 = 2.0 * s3 - 3.0 * s2 + 1.0
            h10 = s3 - 2.0 * s2 + s
            h01 = -2.0 * s3 + 3.0 * s2
            h11 = s3 - s2

            q_ref = [
                h00 * q0[j] + h10 * dt * v0[j] + h01 * q1[j] + h11 * dt * v1[j]
                for j in range(self.nj)
            ]

            # derivative of Hermite curve
            dh00 = (6.0 * s2 - 6.0 * s) / dt
            dh10 = 3.0 * s2 - 4.0 * s + 1.0
            dh01 = (-6.0 * s2 + 6.0 * s) / dt
            dh11 = 3.0 * s2 - 2.0 * s

            v_ref = [
                dh00 * q0[j] + dh10 * v0[j] + dh01 * q1[j] + dh11 * v1[j]
                for j in range(self.nj)
            ]
        else:
            # linear reference
            q_ref = [q0[j] + s * (q1[j] - q0[j]) for j in range(self.nj)]
            v_ref = [(q1[j] - q0[j]) / dt for j in range(self.nj)]

        return q_ref, v_ref

    # ------------------- Main action execution -------------------
    def execute_callback(self, goal_handle):
        traj = goal_handle.request.trajectory
        n_points = len(traj.points)
        if n_points == 0:
            self.get_logger().warn("[Action] Empty trajectory")
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        try:
            times, qs, qds = self._prep_trajectory(traj)
        except RuntimeError:
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            goal_handle.abort()
            return result

        duration = times[-1]
        dt = 1.0 / self.command_rate_hz
        end_extra = max(0.0, self.settle_time)

        # ----- seed internal command state -----
        q_meas0 = self._last_q_meas if self._last_q_meas is not None else qs[0]
        dq_meas0 = self._dq_est if self._dq_est is not None else [0.0] * self.nj

        if self.seed_from_state:
            q_cmd = list(q_meas0)
        else:
            q_cmd = list(qs[0])

        dq_cmd = list(dq_meas0) if self.seed_from_state else [0.0] * self.nj
        a_cmd_prev = [0.0] * self.nj  # previous commanded accel (for jerk limiting)

        self.get_logger().info(
            f"[Action] Executing trajectory ({n_points} points, duration={duration:.3f}s, "
            f"rate={self.command_rate_hz:.1f}Hz, tracker={self.use_tracker})"
        )

        start_time = time.monotonic()
        next_tick = start_time
        end_time = start_time + duration + end_extra

        while True:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("[Action] Goal canceled")
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            now = time.monotonic()
            if now >= end_time:
                break

            t = now - start_time

            if t <= duration:
                q_ref, v_ref = self._ref_at(t, times, qs, qds)
            else:
                # settle: hold final
                q_ref = list(qs[-1])
                v_ref = [0.0] * self.nj

            # measured feedback (best effort)
            q_meas = self._last_q_meas if self._last_q_meas is not None else q_cmd
            dq_meas = self._dq_est if self._dq_est is not None else dq_cmd

            if not self.use_tracker:
                out = q_ref
            else:
                # --- PD in acceleration space, using measured q/dq ---
                a_des = [0.0] * self.nj
                for j in range(self.nj):
                    e = q_ref[j] - q_meas[j]

                    # deadband near target to avoid micro-oscillation
                    if (
                        abs(e) < self.deadband
                        and abs(v_ref[j]) < self.deadband_vel
                        and abs(dq_meas[j]) < self.deadband_vel
                    ):
                        a = 0.0
                    else:
                        a = self.kp * e + self.kd * (v_ref[j] - dq_meas[j])
                    a_des[j] = a

                # --- jerk limiting with GLOBAL scaling (keeps joint-space direction) ---
                da = [(a_des[j] - a_cmd_prev[j]) for j in range(self.nj)]
                j_req = [da[j] / dt for j in range(self.nj)]
                scale_j = global_scale_for_limits(j_req, self.j_max)
                da = [scale_j * x for x in da]

                a_cmd = [a_cmd_prev[j] + da[j] for j in range(self.nj)]

                # --- accel limiting with GLOBAL scaling ---
                scale_a = global_scale_for_limits(a_cmd, self.a_max)
                a_cmd = [scale_a * x for x in a_cmd]

                # integrate
                for j in range(self.nj):
                    dq_cmd[j] += a_cmd[j] * dt
                    if self.use_vel_limit:
                        dq_cmd[j] = clamp(dq_cmd[j], -self.v_max[j], self.v_max[j])
                    q_cmd[j] += dq_cmd[j] * dt

                a_cmd_prev = a_cmd

                # anti-windup: if internal state drifts too far from measured, resync
                max_err = max(abs(q_cmd[j] - q_meas[j]) for j in range(self.nj))
                if max_err > self.resync_err:
                    q_cmd = list(q_meas)
                    if self.resync_dq:
                        dq_cmd = list(dq_meas)
                    a_cmd_prev = [0.0] * self.nj

                out = q_cmd

            # publish target
            cmd = FrankaArmTarget()
            cmd.joint_targets = out
            cmd.arm_id = self.arm_id
            self.cmd_pub.publish(cmd)

            # rate control
            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                # behind schedule
                next_tick = time.monotonic()

        # final setpoint (publish a few ticks helps some low-level controllers)
        final_q = list(qs[-1])
        for _ in range(max(1, int(0.10 * self.command_rate_hz))):  # 100ms hold
            cmd = FrankaArmTarget()
            cmd.joint_targets = final_q
            cmd.arm_id = self.arm_id
            self.cmd_pub.publish(cmd)
            time.sleep(1.0 / self.command_rate_hz)

        self.get_logger().info("[Action] Trajectory done")
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = RealMoveItBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
