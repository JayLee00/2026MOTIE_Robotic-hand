"""Real ROS2 backend for PlacePipeline (runs in the dex_ros Humble container,
ROS_DOMAIN_ID=9). Implements the same Backend interface as backends/mock.py:

    move_to_joints / start_move_to_joints / wait_move   -> MoveGroup joint goal
    move_to_ee_pose                                      -> MoveGroup pose goal (right_fr3_link8)
    capture_rgbd                                         -> front_cam color + aligned depth + K (ROS topics)
    tf(frame)                                            -> world<-frame (4x4) via tf2
    hand_q                                               -> /joint_states_relay right hand joints (rad)

⚠ Requires the live stack (cannot run in the cu128 conda env — no rclpy). Launch
the digital-twin first:  dual_fr3_kistar_planning_pc_v2.launch.py joint_state_mode:=direct.
The front RealSense now runs on a SEPARATE PC (산업부 PC) and streams over ROS2 domain 9 —
this backend SUBSCRIBES to /front_cam/front/... (no host capture service). That camera node
MUST publish depth ALIGNED to color (realsense align_depth.enable:=true) so a color-pixel
mask indexes the depth 1:1. camera->world uses the offline extrinsic (camera fixed to base).
"""
import threading
import time
from collections import deque

import numpy as np

GROUP = "right_arm"
LEFT_GROUP = "left_arm"
EE_LINK = "right_fr3_link8"
BASE_LINK = "right_fr3_link0"
ARM_JOINTS = [f"right_fr3_joint{i}" for i in range(1, 8)]
LEFT_ARM_JOINTS = [f"left_fr3_joint{i}" for i in range(1, 8)]

# Palm / hand-root frame. NOTE: the v2 URDF (dual_fr3_kistar_v2 — the launch the
# orchestrator runs) has NO "right_palm" link; that name came from the v1 kistar_hand
# macro. The palm/hand-base frame is "right_kistar_hand_base" (palm normal = +x, verified
# from the URDF finger geometry: MCP bases at x~-0.011, fingertips curl to x~+0.013).
# Both the palm-normal lookup (§9-B) and the DRO hand-root map here. Camera uses the
# offline extrinsic, not TF.
PALM_FRAME = "right_kistar_hand_base"
FRAME_ALIAS = {"hand_root": PALM_FRAME, "right_palm": PALM_FRAME}
# Fingertip tip links (v2 URDF) — their world positions (live FK via TF) are on/near the
# grasped object's surface (incl. the camera-occluded back), used as contact points for the
# primitive-fit completion. (Only the CONTACT points, not the full hand cloud.)
FINGERTIP_FRAMES = ["right_thumb_3_tip", "right_index_3_tip",
                    "right_middle_3_tip", "right_ring_3_tip"]

# Front RealSense topics, published by the 산업부 PC over ROS2 domain 9. We subscribe to the
# COMPRESSED image transports (jpeg color ~58 KB, compressedDepth PNG ~92 KB), NOT the raw
# 921 KB/614 KB images: raw frames fragment into many UDP packets and, with both large streams
# up, Fast-DDS drops the aligned-depth fragments so depth collapses to ~1 Hz while color stays
# ~12 Hz (the link is Gigabit — this is UDP fragmentation, not bandwidth). The tiny compressed
# messages fit in a few packets → both stream at a steady 15 Hz. Depth MUST be aligned to color
# (realsense align_depth.enable:=true) so a color-pixel mask indexes it 1:1.
TOPIC_RGB = "/front_cam/front/color/image_raw/compressed"
TOPIC_DEPTH = "/front_cam/front/aligned_depth_to_color/image_raw/compressedDepth"
TOPIC_CAMINFO = "/front_cam/front/color/camera_info"
# Live KISTAR hand joints come from the merged /joint_states_relay (RADIANS, after
# joint_state_merger's counts->rad fix + r_->right_ remap); extracted by NAME.
TOPIC_HAND = "/joint_states_relay"
HAND_JOINTS = [f"right_{f}_joint_{j}"
               for f in ("thumb", "index", "middle", "ring") for j in range(4)]

# --- Release/Retract I/O (2nd goal). PaXini fingertip force (SHM->ROS bridge,
# shm_state_publisher) + the hand-command surface (hand_target_receiver). All on the
# Control-PC runtime's control_pc.launch.py, domain 9. Units: /paxini/right/ft = flat 12
# floats = [thumb,index,middle,ring] x [Fx,Fy,Fz] N (tared to ~0 at no-contact). Hand
# commands: cmd_mode Int32 (0=Voltage, 1=Position), cmd_servo Bool, q_target Float32[16]
# (PWM duty +-2100 in Voltage). See DUAL_ARM_HAND_CTRL.md safety rule 3: NO mode hot-switch
# (servo OFF -> set mode -> safe target -> servo ON).
TOPIC_PAXINI_FT = "/paxini/right/ft"
TOPIC_HAND_QTAR = "/hand/right/q_target"
TOPIC_HAND_MODE = "/hand/right/cmd_mode"
TOPIC_HAND_SERVO = "/hand/right/cmd_servo"
HAND_MODE_VOLTAGE = 0
HAND_MODE_POSITION = 1
# 2026-08-16 모드 전환 프로토콜 (사용자 지정): 모드 변경은 반드시 servo OFF 창 안에서,
# 명령 하나당 한 틱(0.05s)씩 끊어 보낸다 — 핸드 펌웨어가 명령 전환을 소화할 시간을 준다.
#   진입: [mode1/servoON/counts 유지] → servoOFF → mode0 → duty목표 → servoON
#   복귀: 역순 — duty0 → servoOFF → mode1 (→ servo 는 설계상 OFF 유지, 다음 단계가 재무장)
HAND_SWITCH_TICK_MS = 50

# Runaway guard for the Voltage-mode release path. Every 16-DoF duty vector published to the
# hand is sanitised through _safe_duty16(): non-finite -> 0, and each value hard-clamped to
# ±RELEASE_DUTY_ABS_MAX. The firmware range is ±2100; we cap FAR below it because release is
# meant to be gentle, so no mistuned constant / stray value can ever drive the hand hard.
RELEASE_DUTY_ABS_MAX = 500.0
# PaXini staleness guard: paxini_ft() returns None (not a frozen last value) once the newest
# sample is older than this, so a dead/stalled writer can't feed a constant force to the
# release/retract triggers (which would silently prevent — or falsely fire — a trigger).
PAXINI_MAX_AGE = 0.3        # s

# --- Grip handover (Stiffness(3) -> Place(4)) ---------------------------------------------
# The object is already gripped when Place's turn starts: Stiffness was streaming a Position
# target (encoder counts) to hold it. Control transfers to us, so WE must keep that same target
# alive for the whole place stage — otherwise the only thing holding the grip is whatever the
# receiver latched, and under require_control:=true a non-owner's stream is ignored anyway.
# So: cache the last EXTERNAL q_target seen on the bus, and republish exactly that vector at
# HAND_HOLD_RATE_HZ from Place's start until the release path switches the hand to Voltage.
HAND_HOLD_RATE_HZ = 20.0
# Reject a handover value older than this (s): a stale target from a long-dead stage would be
# the WRONG grip. Better to hold nothing (receiver keeps its latched target) than to command a
# wrong 16-DoF position.
HAND_HOLD_MAX_AGE = 30.0


def _safe_duty16(v):
    """Sanitise a 16-value duty vector for publishing: coerce to float, non-finite -> 0, and
    hard-clamp to ±RELEASE_DUTY_ABS_MAX. Raises ValueError on the wrong length so a malformed
    command is never sent (the runtime would otherwise reject size != 16 anyway)."""
    import math
    v = list(v)
    if len(v) != 16:
        raise ValueError(f"duty vector must be 16 values, got {len(v)}")
    out = []
    for x in v:
        try:
            x = float(x)
        except (TypeError, ValueError):
            x = 0.0
        if not math.isfinite(x):
            x = 0.0
        out.append(max(-RELEASE_DUTY_ABS_MAX, min(RELEASE_DUTY_ABS_MAX, x)))
    return out


def _safe_counts16(v):
    """Validate a 16-value POSITION target (encoder counts) for republishing.

    Deliberately does NOT clamp: counts run to the hundreds/thousands, so applying the Voltage
    duty clamp (_safe_duty16, ±500) here would silently re-command the hand to a wrong, much
    more open pose and drop the object. Returns None if the vector is unusable — the caller
    must then hold nothing rather than command a fabricated target.
    """
    import math
    try:
        v = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if len(v) != 16 or not all(math.isfinite(x) for x in v):
        return None
    return v


def _cimg_to_rgb(m):
    """sensor_msgs/CompressedImage (jpeg) -> (H,W,3) uint8 RGB."""
    import cv2
    a = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)   # decodes to BGR
    if a is None:
        raise ValueError(f"jpeg decode failed (format {m.format!r})")
    return np.ascontiguousarray(a[:, :, ::-1])                            # BGR -> RGB


def _cimg_to_depth_m(m):
    """sensor_msgs/CompressedImage (compressedDepth) -> (H,W) float32 METERS. Payload = a
    12-byte ConfigHeader (format enum + 2 quant floats) then a 16-bit PNG: 16UC1 holds raw mm,
    32FC1 is inverse-quantised depth = A/(png - B) (0 where invalid)."""
    import cv2
    import struct
    png = cv2.imdecode(np.frombuffer(m.data[12:], np.uint8), cv2.IMREAD_UNCHANGED)  # uint16
    if png is None:
        raise ValueError(f"compressedDepth PNG decode failed (format {m.format!r})")
    enc = m.format.split(";")[0].strip().lower()
    if enc in ("16uc1", "mono16"):
        return png.astype(np.float32) / 1000.0                            # mm -> m
    if enc == "32fc1":
        _, qa, qb = struct.unpack("<iff", bytes(m.data[:12]))
        d = np.zeros(png.shape, np.float32)
        v = png > 0
        d[v] = qa / (png[v].astype(np.float32) - qb)                      # inverse-quantise -> m
        return d
    raise ValueError(f"unexpected compressedDepth format: {m.format!r}")


def _quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n < 1e-12 else 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    return np.array([
        [1 - (y * ys + z * zs), x * ys - w * zs, x * zs + w * ys],
        [x * ys + w * zs, 1 - (x * xs + z * zs), y * zs - w * xs],
        [x * zs - w * ys, y * zs + w * xs, 1 - (x * xs + y * ys)],
    ])


def _mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = [0, 0, 0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    return float(x), float(y), float(z), float(w)


def retime_decel(positions, duration, v_limit=1.0):
    """Retime a planned joint path with a DECELERATING (ease-out) profile.

    positions: list of joint-position lists (M x D) — the planned descent path (timing
    discarded). Returns (times, velocities): the path is arc-length parameterised and the
    ease-out position profile s(tau)=1-(1-tau)^2 is inverted so t(u)=T*(1-sqrt(1-u)), i.e.
    velocity is high at the start and decays to 0 at the target (soft landing). Because
    velocity ~ arc_length/T, a longer or shorter descent still finishes in ~T seconds
    ("value-dependent deceleration, similar arrival time"). Time is stretched only if the
    peak joint velocity would exceed v_limit (rad/s), keeping the motion within limits.
    """
    P = np.asarray(positions, float)
    M, D = (P.shape if P.ndim == 2 else (len(P), 0))
    if M < 2:
        return [0.0] * M, [[0.0] * D for _ in range(M)]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])
    if total < 1e-9:                                   # no motion
        return list(np.linspace(0.0, duration, M)), [[0.0] * D for _ in range(M)]
    u = L / total
    t = duration * (1.0 - np.sqrt(1.0 - u))            # ease-out inverse -> decelerating
    V = np.zeros_like(P)
    for m in range(1, M - 1):
        dt = t[m + 1] - t[m - 1]
        if dt > 1e-9:
            V[m] = (P[m + 1] - P[m - 1]) / dt
    vmax = float(np.abs(V).max())
    if vmax > v_limit:                                 # stretch time to respect joint limits
        f = vmax / v_limit
        t = t * f
        V = V / f
    return t.tolist(), V.tolist()


def retime_uniform(positions, total_time, v_limit=1.0):
    """Retime a Cartesian-straight path at a CONSTANT speed (no ease-out) over ~total_time s
    — used for the slow release ascent and the normal-speed retract. Joint arc-length is
    parameterised linearly in time (the Cartesian path is near-uniform in 5 mm steps, so
    constant joint-arclength speed ~= constant Cartesian speed). Time is stretched only if
    the peak joint velocity would exceed v_limit (rad/s)."""
    P = np.asarray(positions, float)
    M, D = (P.shape if P.ndim == 2 else (len(P), 0))
    if M < 2:
        return [0.0] * M, [[0.0] * D for _ in range(M)]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])
    T = max(float(total_time), 1e-3)
    t = np.linspace(0.0, T, M) if total < 1e-9 else T * (L / total)   # constant speed
    V = np.zeros_like(P)
    for m in range(1, M - 1):
        dt = t[m + 1] - t[m - 1]
        if dt > 1e-9:
            V[m] = (P[m + 1] - P[m - 1]) / dt
    vmax = float(np.abs(V).max())
    if vmax > v_limit:
        f = vmax / v_limit
        t = t * f
        V = V / f
    return t.tolist(), V.tolist()


# Cartesian waypoint spacing for straight-line moves (compute_cartesian_path max_step). The
# planned path spaces EE waypoints <= this, so (n_points - 1) * CART_MAX_STEP is the path's
# Cartesian LENGTH. The release/retract moves are timed from THAT length (path planned from
# move_group's fresh state) instead of a live tf read — a tf read right after a monitored cancel
# can be stale, collapse the duration toward the 0.5 s floor, and make the move spike to full speed.
CART_MAX_STEP = 0.005


class RosBackend:
    # Pose-goal tolerances. Kept loose enough that MoveIt can actually plan/IK the
    # goal (RViz's interactive Plan uses similarly loose defaults) — a 5mm/0.57deg
    # goal often returns NO_IK_SOLUTION/PLANNING_FAILED even for a reachable pose.
    POS_TOL = 0.01     # m, position sphere radius
    ORI_TOL = 0.10     # rad (~5.7 deg) per axis

    def __init__(self, node=None, vel_scale=0.1, plan_time=10.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from sensor_msgs.msg import JointState, CompressedImage, CameraInfo
        from std_msgs.msg import Float32MultiArray, Bool, Int32
        from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                               QoSReliabilityPolicy, QoSHistoryPolicy)
        from rclpy.executors import SingleThreadedExecutor
        import tf2_ros

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = node or Node("place_pipeline_backend")
        self.vel, self.plan_time = vel_scale, plan_time
        self.exec_timeout = 300.0     # wait a WHOLE plan+execute out; a premature return would
                                      # let the next goal preempt the still-running one (branching)
        self._mg = ActionClient(self.node, MoveGroup, "/move_action")
        self._et = ActionClient(self.node, ExecuteTrajectory, "/execute_trajectory")
        self._cart = None             # /compute_cartesian_path client (lazy)
        self._tfbuf = tf2_ros.Buffer()
        self._tfl = tf2_ros.TransformListener(self._tfbuf, self.node)
        self._lock = threading.Lock()
        self._joint_pos = {}                                     # name -> position (accumulated)
        self._rgb = self._depth = self._K = None                 # latest front_cam frame (diag)
        # Short ring buffers of recent color/depth frames as (recv_t, hdr_t, array): recv_t is
        # OUR receipt clock (freshness — skew-free vs the arm), hdr_t is the CAMERA header stamp
        # (color<->depth alignment — same clock, so their diff is skew-free). See capture_rgbd.
        # Ring buffers hold the RAW CompressedImage msgs (recv_t, hdr_t, msg) — decode is
        # DEFERRED to capture_rgbd (see there): the callback stays trivially cheap so the
        # single-threaded executor never falls behind at high frame rates (30 Hz+), and a
        # corrupt/undecodable frame is skipped at capture time instead of throwing inside the
        # callback and aborting the spin. maxlen sized in TIME (2 s @ 30 Hz) so a higher rate
        # doesn't shrink the freshness window.
        self._rgb_buf = deque(maxlen=60)
        self._depth_buf = deque(maxlen=60)
        self._decode_skips = {"color": 0, "depth": 0}            # undecodable frames skipped in capture
        self._paxini_buf = deque(maxlen=200)                     # (recv_t, (4,3) ft [N]) — release/retract
        self._markers = {}                                       # ns -> Marker (staged RViz viz)
        self._marker_pub = None
        self._stb = None
        # Front RealSense streams over domain 9 from the 산업부 PC. Subscribe to the COMPRESSED
        # color/depth transports (see TOPIC_* — avoids the raw-frame UDP-fragment drop that
        # starved depth) with sensor QoS. camera->world uses the offline extrinsic. Live hand
        # joints accumulate by name from /joint_states_relay.
        self.node.create_subscription(JointState, TOPIC_HAND, self._on_joints, 10)
        # PaXini fingertip force (best_effort, ~90 Hz) for the release/retract triggers.
        self.node.create_subscription(Float32MultiArray, TOPIC_PAXINI_FT, self._on_paxini,
                                      qos_profile_sensor_data)
        # Camera subscriptions live on a SEPARATE node spun by a dedicated BACKGROUND THREAD.
        # The main thread stops spinning for SECONDS during blocking model HTTP calls
        # (Molmo/SAM/IGR/AnyPlace) and long MoveIt waits; with the camera on the single main
        # executor, the DDS reader goes unserviced during those gaps and intermittently blacks
        # out for ~15 s (the whole capture timeout) → capture failed. A dedicated executor keeps
        # the reader serviced continuously, so frames always flow into the ring buffers no matter
        # what the main thread is doing, and capture_rgbd just READS them (no spin). Robust at any
        # frame rate (30 Hz+). See capture_rgbd.
        self._cam_node = Node("place_pipeline_camera")
        self._cam_node.create_subscription(CompressedImage, TOPIC_RGB, self._on_rgb, qos_profile_sensor_data)
        self._cam_node.create_subscription(CompressedImage, TOPIC_DEPTH, self._on_depth, qos_profile_sensor_data)
        self._cam_node.create_subscription(CameraInfo, TOPIC_CAMINFO, self._on_caminfo, qos_profile_sensor_data)
        # Grip handover: watch the hand command bus for the target the PREVIOUS stage (Stiffness)
        # is streaming. This rides the camera node's dedicated executor ON PURPOSE — the main node
        # is not spun while we wait for our turn or during long model/MoveIt calls, so a
        # subscription there would miss the handover value exactly when we need it.
        self._cam_node.create_subscription(Float32MultiArray, TOPIC_HAND_QTAR, self._on_hand_qtar,
                                           qos_profile_sensor_data)
        self._cam_exec = SingleThreadedExecutor()
        self._cam_exec.add_node(self._cam_node)
        self._cam_thread = threading.Thread(target=self._cam_exec.spin, daemon=True)
        self._cam_thread.start()
        # Hand-command publishers (QoS matched to hand_target_receiver): q_target BEST_EFFORT
        # depth 1, cmd_mode/cmd_servo RELIABLE depth 1.
        _be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        _rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                          history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self._hand_qtar_pub = self.node.create_publisher(Float32MultiArray, TOPIC_HAND_QTAR, _be)
        self._hand_mode_pub = self.node.create_publisher(Int32, TOPIC_HAND_MODE, _rel)
        self._hand_servo_pub = self.node.create_publisher(Bool, TOPIC_HAND_SERVO, _rel)
        self._hand_in_voltage = False          # True while the release path holds the hand in Voltage
        # Grip handover state (see HAND_HOLD_RATE_HZ). _hand_qtar_ext = (monotonic, counts16) of
        # the last EXTERNAL target seen; the hold thread republishes the snapshot we took at
        # Place's start.
        self._hand_qtar_ext = None
        self._hand_hold_vec = None
        self._hand_hold_stop = threading.Event()
        self._hand_hold_thread = None
        self._arm_hand_safety_net()            # restore Position+servo-OFF even on Ctrl+C / crash / kill
        self._mg.wait_for_server(timeout_sec=10.0)

    # ---- subscriptions -------------------------------------------------------
    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def _on_joints(self, m):
        with self._lock:
            self._joint_pos.update(zip(m.name, m.position))      # accumulate across messages

    def _on_rgb(self, m):
        # runs in the camera thread; store RAW (decode deferred). recv-time = monotonic wall
        # clock (thread-safe, independent of the ROS/main-node clock).
        with self._lock:
            self._rgb = m
            self._rgb_buf.append((time.monotonic(), self._stamp(m), m))

    def _on_depth(self, m):
        with self._lock:
            self._depth = m
            self._depth_buf.append((time.monotonic(), self._stamp(m), m))

    def _on_caminfo(self, m):
        with self._lock:
            self._K = np.asarray(m.k, float).reshape(3, 3)

    def _on_hand_qtar(self, m):
        """Cache the last EXTERNAL hand target (Position counts) for the grip handover.

        Ignored while WE drive the topic — the hold republish and the Voltage-mode duties would
        otherwise overwrite the handover value with our own echo (and duties are NOT a position).
        Runs on the camera executor thread, so it keeps working while the main thread blocks.
        """
        if self._hand_hold_thread is not None or self._hand_in_voltage:
            return
        v = _safe_counts16(m.data)
        if v is None:
            return
        with self._lock:
            self._hand_qtar_ext = (time.monotonic(), v)

    def _on_paxini(self, m):
        a = np.asarray(m.data, float)
        if a.size >= 12:                                         # 4 fingers x [Fx,Fy,Fz]
            arr = a[:12].reshape(4, 3)
            with self._lock:
                self._paxini_buf.append((self._now(), arr))

    # ---- Backend interface ---------------------------------------------------
    def capture_rgbd(self, settle=0.5, timeout=30.0):
        """FRESH RGB-D at the CURRENT (settled) pose.

        The arm is STATIONARY at the pose, so color and depth need NOT be the same instant — only
        both captured at THIS pose. So we accept only frames RECEIVED after now+settle (i.e. after
        the arm arrived and settled) and return the newest such color + newest such depth. Freshness
        uses OUR receipt clock, robust to camera-PC clock skew and to bursty header stamps.

        The camera runs on its own background-spun node (see __init__), so frames flow into the
        ring buffers continuously — even while the main thread is blocked in a model HTTP call or a
        MoveIt wait. capture_rgbd therefore just READS the buffers (no ROS spin), polling until a
        fresh pair appears; recv-time is a monotonic wall clock (matches the camera callbacks).

        The ring buffers hold RAW CompressedImage msgs; decode happens HERE (deferred out of the
        callback) so a corrupt/partial frame that fails to decode is SKIPPED (newest→older) instead
        of throwing and aborting — more likely to matter at 30 Hz+ (more frames, more chances of a
        bad one). We decode only the newest fresh frame that actually decodes, per stream."""
        t_ref = time.monotonic() + settle
        picked = {}

        def _newest_decodable(snapshot, decode, kind):
            for rt, _, m in sorted(snapshot, key=lambda x: -x[0]):   # newest first
                if rt < t_ref:                                       # older than settle -> not fresh
                    break
                try:
                    return decode(m)
                except Exception as e:                               # noqa: BLE001 corrupt frame -> skip
                    self._decode_skips[kind] += 1
                    self._cam_node.get_logger().warn(
                        f"{kind} frame decode skipped ({type(e).__name__}: {e}) -> older fresh frame")
            return None

        def ready():
            with self._lock:                                         # snapshot cheaply, decode outside lock
                if self._K is None:
                    return False
                rgb_snap, dep_snap, K = list(self._rgb_buf), list(self._depth_buf), self._K.copy()
            ra = _newest_decodable(rgb_snap, _cimg_to_rgb, "color")
            if ra is None:
                return False
            da = _newest_decodable(dep_snap, _cimg_to_depth_m, "depth")
            if da is None:
                return False
            picked["rgb"], picked["depth"], picked["K"] = ra, da, K
            return True

        deadline = time.monotonic() + timeout                       # camera thread fills buffers -> just read
        while not ready():
            if time.monotonic() > deadline:
                raise TimeoutError(self._camera_diag())             # say WHY, like tf() does
            time.sleep(0.01)
        return picked["rgb"].copy(), picked["depth"].astype(np.float32), picked["K"]

    def camera_ready(self, timeout=8.0):
        """PRE-FLIGHT (call before the arm moves): confirm all 3 camera streams have a
        live PUBLISHER and that one full color+aligned_depth+K frame actually arrives and
        decodes. Raises TimeoutError(_camera_diag()) otherwise. This fails FAST + clearly
        (e.g. 'publishers[...depth...]=0 -> enable align_depth') instead of moving to
        parent_pose and only then dying at capture_rgbd."""
        # camera is on the background node -> just poll its publisher view (no main-thread spin).
        deadline = time.monotonic() + timeout
        while not all(self._cam_node.count_publishers(t) > 0
                      for t in (TOPIC_RGB, TOPIC_DEPTH, TOPIC_CAMINFO)):
            if time.monotonic() > deadline:
                raise TimeoutError(self._camera_diag())
            time.sleep(0.05)
        self.capture_rgbd()                                    # a full frame must really arrive

    def _camera_diag(self):
        """Explain a front_cam failure: which of the 3 streams arrived vs how many
        PUBLISHERS each topic has, plus the DDS scope. count_publishers (not our own
        subscription) tells camera-side from code-side: publishers=0 -> that stream isn't
        being published (align_depth off / camera down); all 0 -> camera PC (192.168.0.38)
        unreachable (ROS_STATIC_PEERS / link)."""
        import os
        with self._lock:
            got = {"color": self._rgb is not None, "depth": self._depth is not None,
                   "K": self._K is not None}
            now = time.monotonic()                             # buffer recv-t is monotonic (camera thread)
            rgb_age = min((now - rt for rt, _, _ in self._rgb_buf), default=None)
            dep_age = min((now - rt for rt, _, _ in self._depth_buf), default=None)
            sync = min((abs(rh - dh) for _, rh, _ in self._rgb_buf for _, dh, _ in self._depth_buf),
                       default=None)
        pubs = {t: self._cam_node.count_publishers(t)
                for t in (TOPIC_RGB, TOPIC_DEPTH, TOPIC_CAMINFO)}
        fresh = (f"newest color {rgb_age:.2f}s ago, depth {dep_age:.2f}s ago, "
                 f"best color<->depth stamp gap {sync * 1000:.0f}ms"
                 if None not in (rgb_age, dep_age, sync) else "no frames buffered")
        return ("front_cam: no fresh time-aligned color+aligned_depth+camera_info frame.\n"
                f"  received  : {got}\n"
                f"  publishers: {pubs}\n"
                f"  buffers   : {fresh}\n"
                f"  decode-skips: {self._decode_skips} (undecodable frames skipped in capture)\n"
                f"  DDS: DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')} "
                f"RANGE={os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')} "
                f"STATIC_PEERS={os.environ.get('ROS_STATIC_PEERS')}\n"
                "  publishers=0 -> that stream isn't published (align_depth off / camera down);\n"
                "  ALL 0 -> camera PC unreachable (ROS_STATIC_PEERS, docker/run.sh). Publishers OK\n"
                "  but 'newest depth' many seconds old -> the aligned-depth stream stalled/bursty\n"
                "  (link contention with color); raise capture timeout or reduce stream bandwidth.\n"
                "  decode-skips high with fresh frames present -> corrupt/unexpected-format frames\n"
                "  (check the aligned_depth encoding: 16UC1/32FC1 compressedDepth; color jpeg).")

    def hand_q(self):
        """Live KISTAR hand joints (16, radians) in FK order [thumb,index,middle,ring]x[0..3]."""
        self._spin_until(lambda: all(n in self._joint_pos for n in HAND_JOINTS),
                         "hand joints (/joint_states_relay)", timeout=5.0)
        with self._lock:
            return np.array([self._joint_pos[n] for n in HAND_JOINTS], float)

    def fingertip_points(self):
        """World positions of the 4 fingertip links (live FK via TF) = object contact points.
        Returns (M,3) or None; best-effort per finger (short timeout)."""
        from rclpy.time import Time
        pts = []
        for f in FINGERTIP_FRAMES:
            try:
                self._spin_until(lambda f=f: self._tfbuf.can_transform("world", f, Time()),
                                 f"tf {f}", timeout=2.0)
                t = self._tfbuf.lookup_transform("world", f, Time()).transform.translation
                pts.append([t.x, t.y, t.z])
            except Exception:                                    # noqa: BLE001
                pass
        return np.asarray(pts, float) if pts else None

    # ---- PaXini fingertip force + hand command (release/retract) --------------
    def paxini_ft(self):
        """Latest PaXini fingertip force, (4,3) [thumb,index,middle,ring]x[Fx,Fy,Fz] N, or
        None if nothing buffered OR the newest sample is stale (> PAXINI_MAX_AGE old — writer
        dead/stalled). Returning None makes the release/retract triggers ignore frozen data
        rather than act on it. Processes any pending messages first (non-blocking)."""
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        with self._lock:
            if not self._paxini_buf:
                return None
            recv_t, arr = self._paxini_buf[-1]
        if self._now() - recv_t > PAXINI_MAX_AGE:
            return None
        return arr.copy()

    def collect_paxini(self, duration=0.4, min_samples=8):
        """Collect a baseline window of /paxini/ft while the grasp is held at the waypoint.
        Returns (N,4,3) [N] of distinct recent samples, or None if the stream is silent."""
        clk = self.node.get_clock()
        t0 = clk.now()
        got, seen = [], set()
        while True:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
            with self._lock:
                if self._paxini_buf:
                    rt, arr = self._paxini_buf[-1]
                    if rt not in seen:
                        seen.add(rt)
                        got.append(arr)
            el = (clk.now() - t0).nanoseconds * 1e-9
            if (el >= duration and len(got) >= min_samples) or el > duration + 2.0:
                break
        return np.asarray(got, float) if got else None

    def _spin_ms(self, ms):
        clk = self.node.get_clock()
        t0 = clk.now()
        while (clk.now() - t0).nanoseconds * 1e-9 < ms / 1000.0:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)

    def hand_grip_handover(self):
        """Return the last EXTERNAL hand target (Position counts) seen on the bus, or None.

        At Place's turn this is what Stiffness(3) was streaming to hold the grasped object.
        None means we never saw one (or it is too old) — the caller must NOT fabricate a target.
        """
        with self._lock:
            snap = self._hand_qtar_ext
        if snap is None:
            return None
        age, vec = time.monotonic() - snap[0], snap[1]
        return None if age > HAND_HOLD_MAX_AGE else vec

    def hand_hold_start(self):
        """Take over the grip: republish the handover target at HAND_HOLD_RATE_HZ until stopped.

        Called at the START of Place's turn. The previous stage stops streaming when it releases
        control, and under require_control:=true only the OWNER's targets are honoured — so the
        new owner has to keep the same Position target alive or nothing is actively holding the
        object while the arm moves.

        Runs on its own thread: rclpy publish is thread-safe and needs no executor, so the hold
        survives the long model HTTP calls / MoveIt waits during which the main node is not spun.

        Returns the vector being held, or None if there was nothing valid to take over (logged by
        the caller; the receiver then just keeps whatever it latched — safer than a guess).
        """
        from std_msgs.msg import Float32MultiArray

        if self._hand_hold_thread is not None:
            return self._hand_hold_vec                # already holding (idempotent)
        vec = self.hand_grip_handover()
        if vec is None:
            return None

        self._hand_hold_vec = vec
        self._hand_hold_stop.clear()
        msg = Float32MultiArray()
        msg.data = list(vec)
        period = 1.0 / HAND_HOLD_RATE_HZ

        def _loop():
            # q_target is BEST_EFFORT (no retransmit) — a steady stream is also what makes a
            # dropped datagram harmless.
            while not self._hand_hold_stop.wait(period):
                if self._hand_in_voltage:             # release path took over -> counts are no
                    break                             # longer a valid command; stop immediately
                try:
                    self._hand_qtar_pub.publish(msg)
                except Exception:                     # noqa: BLE001 — teardown races
                    break

        self._hand_hold_thread = threading.Thread(target=_loop, name="hand-grip-hold", daemon=True)
        self._hand_hold_thread.start()
        return vec

    def hand_hold_stop(self, timeout=1.0):
        """Stop the grip hold. MUST run before any Position->Voltage switch: the held vector is
        encoder counts, and counts published in Voltage mode are raw duty -> runaway."""
        t = self._hand_hold_thread
        self._hand_hold_thread = None                 # re-enable _on_hand_qtar caching
        if t is None:
            return
        self._hand_hold_stop.set()
        t.join(timeout=timeout)

    def hand_release_sequence(self, duty16, settle=0.3):
        """Switch the hand Position->Voltage SAFELY (DUAL_ARM_HAND_CTRL.md rule 3: no mode
        hot-switch — servo OFF -> set mode -> seed safe target -> servo ON) and then apply the
        weak-open duty. duty16 = 16 PWM duties (+-2100); republished once for best-effort."""
        from std_msgs.msg import Float32MultiArray, Bool, Int32
        self.hand_hold_stop()                         # stop streaming COUNTS before Voltage mode
        self._hand_in_voltage = True                  # arm the exit safety net (restore on abnormal exit)

        def servo(on):
            mm = Bool(); mm.data = bool(on); self._hand_servo_pub.publish(mm)

        def mode(v):
            mm = Int32(); mm.data = int(v); self._hand_mode_pub.publish(mm)

        def duty(v):
            mm = Float32MultiArray(); mm.data = _safe_duty16(v)   # sanitise + hard-clamp (runaway guard)
            self._hand_qtar_pub.publish(mm)

        # 2026-08-16 전환 프로토콜(사용자 지정, HAND_SWITCH_TICK_MS 참조):
        #   [mode1/servoON/counts 유지] → servoOFF → mode0 → duty(pwm 목표) → servoON
        # 명령 하나당 한 틱(0.05s). 모드 변경은 servo OFF 창 안에서만 일어난다.
        # servo ON 순간 래치돼 있는 target 이 (스테일 counts 가 아니라) 클램프된
        # 목표 duty 임을 off-창 안의 duty 시딩 ×2 가 보장한다 (best_effort 유실 대비).
        safe = _safe_duty16(duty16)                   # validate/clamp ONCE up front (raises on bad length)
        tick = HAND_SWITCH_TICK_MS
        servo(False); self._spin_ms(tick)             # ① servo OFF — 전환 창 열기
        mode(HAND_MODE_VOLTAGE); self._spin_ms(tick)  # ② 0 = Voltage (off 창 안)
        duty(safe); self._spin_ms(tick)               # ③ pwm 목표값 시딩 (off 창 안, ±500 클램프)
        duty(safe); self._spin_ms(tick)               #    재발행 (q_target 은 best_effort)
        servo(True); self._spin_ms(int(settle * 1000))  # ④ servo ON — 목표 duty 로 약하게 벌림
        duty(safe); self._spin_ms(tick)               # 재확인 발행 (유실 대비)

    def hand_relax(self):
        """Zero all duties (fingers limp, no drive) — left state after retract."""
        from std_msgs.msg import Float32MultiArray
        mm = Float32MultiArray(); mm.data = _safe_duty16([0.0] * 16)
        self._hand_qtar_pub.publish(mm); self._spin_ms(60)
        self._hand_qtar_pub.publish(mm); self._spin_ms(30)

    def hand_safe_shutdown(self):
        """End-of-run SAFE hand state: zero the Voltage drive, then return the hand to POSITION mode
        with servo OFF. The release path switches the hand to VOLTAGE; if a run ENDS there, the NEXT
        run's POSITION targets (encoder counts, hundreds–thousands) are reinterpreted as raw duty and
        the hand RUNS AWAY. Servo is turned OFF *first* so neither the mode switch nor any latched
        target can drive the hand during the transition, then mode -> Position; servo is left OFF."""
        from std_msgs.msg import Float32MultiArray, Bool, Int32

        self.hand_hold_stop()                         # never let the grip hold outlive the run

        def servo(on):
            mm = Bool(); mm.data = bool(on); self._hand_servo_pub.publish(mm)

        def mode(v):
            mm = Int32(); mm.data = int(v); self._hand_mode_pub.publish(mm)

        def duty(v):
            mm = Float32MultiArray(); mm.data = _safe_duty16(v)
            self._hand_qtar_pub.publish(mm)

        # 2026-08-16 복귀(역순) 프로토콜: duty0 → servoOFF → mode1, 한 틱(0.05s)씩.
        # 모드 변경은 servo OFF 창 안에서만. servo 는 설계상 OFF 로 남긴다
        # (다음 체인의 파지 단계가 재무장 — 여기서 켜면 스테일 counts 로 구동할 위험).
        tick = HAND_SWITCH_TICK_MS
        duty([0.0] * 16); self._spin_ms(tick)         # ① Voltage 드라이브 0 → limp
        duty([0.0] * 16); self._spin_ms(tick)         #    재발행 (best_effort)
        servo(False); self._spin_ms(tick)             # ② servo OFF — 전환 창 열기
        mode(HAND_MODE_POSITION); self._spin_ms(tick) # ③ 1 = Position (off 창 안)
        servo(False); self._spin_ms(tick)             # ④ servo OFF 확인 재발행 — OFF 로 종료
        self._hand_in_voltage = False                 # restored -> exit safety net becomes a no-op

    def _arm_hand_safety_net(self):
        """Guarantee the hand is left SAFE (Position mode + servo OFF) even on an ABNORMAL exit —
        Ctrl+C, an unhandled exception, or SIGTERM — but ONLY if WE switched it to Voltage (the
        release path). Without this, a run killed mid-release leaves the hand in Voltage mode and the
        NEXT run's position targets are read as raw duty -> runaway. A live grasp we never touched
        (still Position) is left ALONE, so an early crash doesn't drop a held object.

        The SIGINT/SIGTERM handlers restore immediately (context still valid, so the publishes go
        out), then re-raise; atexit is the backstop for a plain exception/return."""
        import atexit
        import signal

        def restore(*_a):
            if getattr(self, "_hand_in_voltage", False):
                try:
                    self.hand_safe_shutdown()          # publishes servo-OFF + Position; clears the flag
                except Exception:                      # noqa: BLE001 — best effort during teardown
                    pass

        atexit.register(restore)                       # plain exit / unhandled exception / KeyboardInterrupt

        def _sig(signum, frame):
            restore()                                  # do it NOW, while rclpy + the publishers are alive
            raise KeyboardInterrupt if signum == signal.SIGINT else SystemExit(128 + signum)
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)                 # only settable from the main thread
            except (ValueError, OSError):
                pass

    def tf(self, frame):
        from rclpy.time import Time
        if frame == "camera":                                    # fixed to base -> offline extrinsic
            from vision_pipeline.core.extrinsic import world_from_cam
            return world_from_cam()
        target = FRAME_ALIAS.get(frame, frame)
        try:
            self._spin_until(lambda: self._tfbuf.can_transform("world", target, Time()),
                             f"tf world<-{target}")
        except TimeoutError:                                      # show what the container DID receive
            raise TimeoutError(
                f"tf world<-{target} unavailable after spin. Frames this container sees:\n"
                + self._tfbuf.all_frames_as_string())
        t = self._tfbuf.lookup_transform("world", target, Time())
        tr, q = t.transform.translation, t.transform.rotation
        T = np.eye(4)
        T[:3, :3] = _quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [tr.x, tr.y, tr.z]
        return T

    def move_to_joints(self, joints):
        return self.wait_move(self.start_move_to_joints(joints))

    def start_move_to_joints(self, joints):
        from moveit_msgs.msg import Constraints, JointConstraint
        cons = Constraints()
        for jn, val in joints.items():
            cons.joint_constraints.append(JointConstraint(
                joint_name=jn, position=float(val),
                tolerance_above=1e-3, tolerance_below=1e-3, weight=1.0))
        return self._send_goal(cons)

    def _ee_constraints(self, T_world_ee):
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        x, y, z, w = _mat_to_quat(T_world_ee[:3, :3])
        cons = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = EE_LINK
        bv = BoundingVolume()
        sp = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[self.POS_TOL])
        bv.primitives.append(sp)
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, T_world_ee[:3, 3])
        p.orientation.w = 1.0
        bv.primitive_poses.append(p)
        pc.constraint_region = bv
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = "world"
        oc.link_name = EE_LINK
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = x, y, z, w
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = self.ORI_TOL
        oc.weight = 1.0
        cons.position_constraints.append(pc)
        cons.orientation_constraints.append(oc)
        return cons

    def move_to_ee_pose(self, T_world_ee):
        token = self._send_goal(self._ee_constraints(T_world_ee))
        if token is None:
            return -1                                            # goal rejected -> PLANNING_FAILED
        self._spin_future(token, timeout=self.exec_timeout)      # wait the whole plan+execute
        res = token.result()
        return int(res.result.error_code.val) if res else 99999  # MoveIt error code (1 = SUCCESS)

    def start_move_left_arm(self, values):
        """Fire the LEFT arm (left_arm group) to a 7-joint pose asynchronously — returns a
        token immediately (does NOT wait for arrival) so the pipeline keeps running."""
        from moveit_msgs.msg import Constraints, JointConstraint
        cons = Constraints()
        for jn, val in zip(LEFT_ARM_JOINTS, values):
            cons.joint_constraints.append(JointConstraint(
                joint_name=jn, position=float(val),
                tolerance_above=1e-3, tolerance_below=1e-3, weight=1.0))
        return self._send_goal(cons, group=LEFT_GROUP)

    def move_right_arm(self, values):
        """Move the RIGHT arm (right_arm group) to a 7-joint pose, BLOCKING until arrival."""
        return self.move_to_joints({jn: v for jn, v in zip(ARM_JOINTS, values)})

    def descend_decel(self, T_world_ee, duration=4.0):
        """STRAIGHT-LINE, COLLISION-CHECKED Cartesian descent to T_world_ee: from the waypoint
        (already at the T_act orientation) the EE translates straight down with its orientation
        held FIXED, retimed with a decelerating ease-out (~duration s). Uses a CARTESIAN path
        (/compute_cartesian_path, avoid_collisions=True). If a collision-free straight-line path
        can't be found (fraction<0.9) or execution fails, returns a FAILURE code so the caller
        tries another candidate — it does NOT fall back to a curving OMPL pose goal (that would
        both re-plan the orientation and could move through collisions)."""
        from builtin_interfaces.msg import Duration as DurationMsg
        traj = self._cartesian_path(T_world_ee)
        if traj is None or not traj.joint_trajectory.points:
            self.node.get_logger().warn("descend: no collision-free straight-line path -> fail (try next candidate)")
            return -1                                            # PLANNING_FAILED-ish -> caller retries
        pts = [list(p.positions) for p in traj.joint_trajectory.points]
        times, vels = retime_decel(pts, duration)
        for p, tt, vv in zip(traj.joint_trajectory.points, times, vels):
            sec = int(tt)
            p.time_from_start = DurationMsg(sec=sec, nanosec=int(round((tt - sec) * 1e9)))
            p.velocities = list(vv)
            p.accelerations = []
        return self._execute_traj(traj)                          # 1=SUCCESS, else failure (caller retries)

    def _apply_timing(self, traj, times, vels):
        from builtin_interfaces.msg import Duration as DurationMsg
        for p, tt, vv in zip(traj.joint_trajectory.points, times, vels):
            sec = int(tt)
            p.time_from_start = DurationMsg(sec=sec, nanosec=int(round((tt - sec) * 1e9)))
            p.velocities = list(vv)
            p.accelerations = []

    def descend_decel_monitored(self, T_world_ee, should_stop, duration=4.0):
        """Collision-checked decelerating descent to T_world_ee (as descend_decel) but with a
        live abort: `should_stop()` is polled while the trajectory executes and, when it fires
        (PaXini disturbance = the object/finger touched something), the in-flight MoveIt
        trajectory is CANCELLED so the arm stops where it is. Returns:
          'reached'     — arrived at T_act with no disturbance (Case 2)
          'disturbance' — should_stop fired mid-descent, cancelled here (Case 1)
          'failed'      — no collision-free straight-line path / execution error (caller retries)."""
        traj = self._cartesian_path(T_world_ee)
        if traj is None or not traj.joint_trajectory.points:
            self.node.get_logger().warn("descend(monitored): no collision-free straight-line path -> fail")
            return "failed"
        pts = [list(p.positions) for p in traj.joint_trajectory.points]
        self._apply_timing(traj, *retime_decel(pts, duration))
        r = self._execute_traj_monitored(traj, should_stop)
        return {"reached": "reached", "stopped": "disturbance", "failed": "failed"}[r]

    def _cart_time(self, pts, speed):
        """Duration (s) to run a Cartesian move at ~`speed` (m/s), derived from the PLANNED PATH's
        own length — NOT from a live tf read. `_cartesian_path` is planned from move_group's fresh
        state and spaces waypoints <= CART_MAX_STEP apart, so (n-1)*CART_MAX_STEP is the path's
        Cartesian length. This tracks the true distance (so `speed` is honoured as-is) and can never
        collapse to ~0 — unlike a tf(EE_LINK) read right after a monitored cancel, which could be
        stale, drive the duration to the 0.5 s floor, and make the real path execute at full speed."""
        path_len = max(0.0, (len(pts) - 1) * CART_MAX_STEP)
        return max(0.5, path_len / max(speed, 1e-3))

    def ascend_slow_monitored(self, T_up, should_stop, speed=0.02):
        """Slow, collision-checked straight-line ascent to T_up (the collision-free waypoint),
        watching for no-load. `should_stop()` = PaXini no-load (object left the hand). Returns
        'noload' (fired -> retract now), 'reached' (hit the waypoint still loaded), or 'failed'."""
        traj = self._cartesian_path(T_up)
        if traj is None or not traj.joint_trajectory.points:
            self.node.get_logger().warn("ascend: no collision-free straight-line path -> fail")
            return "failed"
        pts = [list(p.positions) for p in traj.joint_trajectory.points]
        self._apply_timing(traj, *retime_uniform(pts, self._cart_time(pts, speed)))
        r = self._execute_traj_monitored(traj, should_stop)
        return {"reached": "reached", "stopped": "noload", "failed": "failed"}[r]

    def move_ee_linear(self, T_target, speed=0.08):
        """Normal-speed collision-checked straight-line Cartesian move to T_target (the retract
        lift back up to the waypoint). Blocking. Returns True on success."""
        traj = self._cartesian_path(T_target)
        if traj is None or not traj.joint_trajectory.points:
            self.node.get_logger().warn("move_ee_linear: no collision-free straight-line path")
            return False
        pts = [list(p.positions) for p in traj.joint_trajectory.points]
        self._apply_timing(traj, *retime_uniform(pts, self._cart_time(pts, speed)))
        return self._execute_traj(traj) == 1

    def _execute_traj_monitored(self, robot_traj, should_stop, poll_hz=50.0):
        """Execute a RobotTrajectory while polling should_stop(); cancel the goal if it fires.
        Returns 'reached' (natural success), 'stopped' (should_stop fired -> cancelled), or
        'failed' (no execute server / goal rejected / aborted)."""
        from moveit_msgs.action import ExecuteTrajectory
        if not self._et.wait_for_server(timeout_sec=5.0):
            return "failed"
        fut = self._et.send_goal_async(ExecuteTrajectory.Goal(trajectory=robot_traj))
        self._spin_future(fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return "failed"
        rfut = gh.get_result_async()
        clk = self.node.get_clock()
        t0 = clk.now()
        dt = 1.0 / poll_hz
        while not rfut.done():
            self.rclpy.spin_once(self.node, timeout_sec=dt)
            try:
                if should_stop(self.paxini_ft()):
                    cfut = gh.cancel_goal_async()
                    self._spin_future(cfut, timeout=5.0)         # request cancel
                    self._spin_future(rfut, timeout=5.0)         # let the result settle
                    return "stopped"
            except Exception as e:                               # noqa: BLE001 — a monitor error must not run on
                self.node.get_logger().warn(f"stop-monitor error: {e}")
            if (clk.now() - t0).nanoseconds * 1e-9 > self.exec_timeout:
                return "failed"
        res = rfut.result()
        return "reached" if (res and res.result.error_code.val == 1) else "failed"

    def _cartesian_path(self, T_world_ee, max_step=CART_MAX_STEP):
        """Straight-line Cartesian path from the CURRENT EE pose to T_world_ee (linear
        translation, orientation interpolated — so constant when start==end orientation).
        Returns a RobotTrajectory, or None if the path fraction is < 0.9. COLLISION-CHECKED
        (avoid_collisions=True): the arm/hand must not hit the table/tray; the grasped object
        is not in the planning scene so the object entering a tray hole isn't itself blocked.
        Start state = current (empty start_state)."""
        from moveit_msgs.srv import GetCartesianPath
        from geometry_msgs.msg import Pose
        if self._cart is None:
            self._cart = self.node.create_client(GetCartesianPath, "/compute_cartesian_path")
        if not self._cart.wait_for_service(timeout_sec=5.0):
            return None
        x, y, z, w = _mat_to_quat(T_world_ee[:3, :3])
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, T_world_ee[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = x, y, z, w
        req = GetCartesianPath.Request()
        req.header.frame_id = "world"
        req.group_name = GROUP
        req.link_name = EE_LINK
        req.max_step = max_step                                  # 5mm Cartesian resolution
        req.jump_threshold = 0.0                                 # no joint-jump check (short descent)
        req.avoid_collisions = True                              # arm/hand must not hit table/tray (§9-H)
        req.waypoints = [p]
        fut = self._cart.call_async(req)
        self._spin_future(fut, timeout=self.exec_timeout)
        res = fut.result()
        if res is None or res.fraction < 0.9:
            self.node.get_logger().warn(
                f"descend: cartesian fraction={getattr(res, 'fraction', None)} (<0.9)")
            return None
        return res.solution

    def _execute_traj(self, robot_traj):
        from moveit_msgs.action import ExecuteTrajectory
        if not self._et.wait_for_server(timeout_sec=5.0):
            return 99999                                         # FAILURE (no execute server)
        fut = self._et.send_goal_async(ExecuteTrajectory.Goal(trajectory=robot_traj))
        self._spin_future(fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return -1
        rfut = gh.get_result_async()
        self._spin_future(rfut, timeout=self.exec_timeout)
        res = rfut.result()
        return int(res.result.error_code.val) if res else 99999

    # ---- MoveGroup plumbing --------------------------------------------------
    def _send_goal(self, constraints, group=GROUP, plan_only=False):
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, PlanningOptions
        req = MotionPlanRequest()
        req.group_name = group
        req.goal_constraints.append(constraints)
        req.num_planning_attempts = 10
        req.allowed_planning_time = self.plan_time
        req.max_velocity_scaling_factor = self.vel
        req.max_acceleration_scaling_factor = self.vel
        req.workspace_parameters = WorkspaceParameters()
        goal = MoveGroup.Goal(request=req, planning_options=PlanningOptions(plan_only=plan_only))
        fut = self._mg.send_goal_async(goal)
        self._spin_future(fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return None
        rfut = gh.get_result_async()
        return rfut

    def wait_move(self, token):
        if token is None:                                    # goal was REJECTED by MoveGroup
            self.node.get_logger().warn(
                "move goal rejected (no goal handle). If you also see 'Ignoring unexpected goal "
                "response ... more than one action server for /move_action', TWO move_group nodes "
                "are running — kill the duplicate digital-twin (see moveit_ready()).")
            return False
        self._spin_future(token, timeout=self.exec_timeout)  # whole plan+execute (no early return)
        res = token.result()
        ok = bool(res) and res.result.error_code.val == 1
        if not ok:                                           # surface WHY (was silently a bare False)
            code = res.result.error_code.val if res else None
            self.node.get_logger().warn(f"move failed: MoveIt error_code={code} (1=SUCCESS, "
                                        "-1=PLANNING_FAILED, -10=START_STATE_IN_COLLISION, "
                                        "-12=GOAL_IN_COLLISION, -31=NO_IK_SOLUTION)")
        return ok

    def moveit_ready(self, settle=2.0):
        """Preflight: fail LOUDLY unless EXACTLY ONE move_group / /move_action server is up.

        TWO move_group nodes (a duplicate or stale digital-twin launch) make rclpy's MoveGroup
        ActionClient discard goal responses ('Ignoring unexpected goal response. There may be
        more than one action server for /move_action'), so _send_goal returns None and EVERY arm
        move silently fails — the run dies cryptically at the first move (parent_pose). Detect it
        here instead. The main move_group node is named exactly 'move_group' (its internal
        'move_group_private_*' companion is ignored), so the instance count = #'move_group' nodes."""
        clk = self.node.get_clock()
        t0 = clk.now()
        while (clk.now() - t0).nanoseconds * 1e-9 < settle:   # let node-graph discovery settle
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        n = sum(1 for name, _ in self.node.get_node_names_and_namespaces() if name == "move_group")
        if n == 0:
            raise RuntimeError(
                "no move_group node on domain 9 — launch the digital twin: "
                "dual_fr3_kistar_planning_pc_v2.launch.py joint_state_mode:=direct")
        if n > 1:
            raise RuntimeError(
                f"{n} move_group nodes are running (duplicate/stale digital-twin launches) -> "
                "duplicate /move_action servers. rclpy's MoveGroup ActionClient then REJECTS every "
                "goal ('Ignoring unexpected goal response'), so ALL arm moves fail (the run dies at "
                "parent_pose). Kill the extra digital-twin so exactly ONE move_group runs, then "
                "re-run.  Check: ros2 node list | grep -xc /move_group  (must be 1)")
        return True

    # ---- spin helpers --------------------------------------------------------
    def _spin_future(self, fut, timeout=60.0):
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)

    def _spin_until(self, pred, what, timeout=15.0):
        clk = self.node.get_clock()
        t0 = clk.now()
        while not pred():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if (clk.now() - t0).nanoseconds * 1e-9 > timeout:
                raise TimeoutError(f"timed out waiting for {what}")

    # ---- RViz live staged viz -----------------------------------------------
    # The orchestrator (via Monitor) calls these AS each stage completes, so the clouds
    # appear/dim in RViz in real time. All markers ride ONE topic /place_debug/markers
    # (the fr3_kistar.rviz MarkerArray display shows them automatically — no manual Add).
    def _marker_pubr(self):
        if self._marker_pub is None:
            from visualization_msgs.msg import MarkerArray
            self._marker_pub = self.node.create_publisher(MarkerArray, "/place_debug/markers", 5)
        return self._marker_pub

    def viz_points(self, ns, points, color, alpha=1.0, size=0.005, mid=0):
        """Add/replace a POINTS marker (namespace `ns`) in world frame."""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        pts = np.asarray(points, float)
        if len(pts) > 25000:
            pts = pts[np.linspace(0, len(pts) - 1, 25000).astype(int)]
        m = Marker()
        m.header.frame_id = "world"
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = m.scale.y = float(size)
        m.pose.orientation.w = 1.0
        m.frame_locked = True
        m.color = ColorRGBA(r=color[0] / 255., g=color[1] / 255., b=color[2] / 255., a=float(alpha))
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self._markers[ns] = m
        self._publish_markers()

    def viz_alpha(self, ns, alpha):
        """Change the transparency of an already-shown cloud (e.g. dim a prior stage)."""
        m = self._markers.get(ns)
        if m is not None:
            m.color.a = float(alpha)
            self._publish_markers()

    def viz_ee(self, T):
        """Show the EE goal: bright XYZ=RGB axis triad + latched TF `place_ee_target`."""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point, TransformStamped
        from std_msgs.msg import ColorRGBA
        from tf2_ros import StaticTransformBroadcaster
        T = np.asarray(T, float)
        o = T[:3, 3]
        for i, col in enumerate([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]):
            m = Marker()
            m.header.frame_id = "world"
            m.ns = "place_ee_axes"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = 0.008, 0.016, 0.0
            d = T[:3, i] * 0.10
            m.points = [Point(x=float(o[0]), y=float(o[1]), z=float(o[2])),
                        Point(x=float(o[0] + d[0]), y=float(o[1] + d[1]), z=float(o[2] + d[2]))]
            m.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
            m.frame_locked = True
            self._markers[f"place_ee_axes_{i}"] = m
        if self._stb is None:
            self._stb = StaticTransformBroadcaster(self.node)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "place_ee_target"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, o)
        qx, qy, qz, qw = _mat_to_quat(T[:3, :3])
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = qx, qy, qz, qw
        self._stb.sendTransform(t)
        self._publish_markers()

    def _publish_markers(self):
        from visualization_msgs.msg import MarkerArray
        arr = MarkerArray()
        stamp = self.node.get_clock().now().to_msg()
        for m in self._markers.values():
            m.header.stamp = stamp
            arr.markers.append(m)
        self._marker_pubr().publish(arr)
        for _ in range(3):
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def viz_hold(self, seconds=300.0, log=print):
        """Keep the node alive (re-publishing markers @2Hz) so RViz keeps showing the
        result after the pipeline finishes. Ctrl+C stops early."""
        log(f"[rviz] result live on /place_debug/markers + TF place_ee_target; "
            f"holding {seconds:.0f}s (Ctrl+C to stop)")
        t0 = self.node.get_clock().now()
        try:
            while (self.node.get_clock().now() - t0).nanoseconds * 1e-9 < seconds:
                self._publish_markers()
                self.rclpy.spin_once(self.node, timeout_sec=0.5)
        except KeyboardInterrupt:
            pass
