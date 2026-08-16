#!/usr/bin/env python
"""Live fruit-bbox GUI on the ROS2 camera topics (streaming SAM2 video-predictor).

Subscribes the RealSense ROS2 stream under ``/front_cam/front`` (see
``run_real_robust.py`` for the topic map):

    /front_cam/front/color/image_raw                    (rgb8)   [required]
    /front_cam/front/color/camera_info                  (K)      [required]
    /front_cam/front/aligned_depth_to_color/image_raw   (16UC1)  [3D fit]

color -> SAM2 mask -> Z-depth deprojection of that mask only ->
axis-fixed sphere fit -> 3D bbox, smoothed by a constant-velocity position KF and
drawn live in a cv2 window. The bounding box (centre AND size) is derived ONLY
from the SAM2 mask -- no colour heuristic.

Segmentation uses the STATEFUL SAM2 video predictor (:class:`Sam2StreamTracker`):
on (re)seed it encodes the current frame and prompts at the seed point; every
subsequent frame is tracked via the predictor's memory bank (temporally coherent
masks, no per-frame re-prompt). If CUDA / the checkpoint are unavailable so the
stream tracker can't build, it falls back to the per-frame SAM2 *image* predictor
(:class:`RealSenseSegmenter.segment_at`) and the overlay says so.

Run (from the repo root, with the GUI display exported; the ``ros`` conda env
has rclpy + torch + sam2 together):

    DISPLAY=:1 conda run -n ros --no-capture-output python live_bbox_gui.py

The aligned-depth topic only exists while ``align_depth.enable`` is true:

    ros2 param set /front_cam/front align_depth.enable true

Prompting: (re)seed at the image centre initially; LEFT-CLICK the fruit to reseed
on it; press 'r' to reset and reseed at the current prompt point. Between reseeds
the video predictor carries the object via memory -- no reprompt each frame.

Keys:  q / ESC = quit   r = reset tracker   left-click = seed SAM2 on the fruit

SAM2 is the slow stage, so expect a lower frame rate than a colour threshold; the
first frame also pays the model-load cost.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from in_hand_tracker.io.realsense_source import (
    deproject_zdepth,
    project_points,
)
from run_real_robust import (           # shared ROS2 camera node + helpers
    _DEFAULT_NAMESPACE,
    _build_node,
    _spin_until,
)
from in_hand_tracker.perception.ellipsoid_fit import fit_ellipsoid_center
from in_hand_tracker.perception.realsense_segmenter import (
    RealSenseSegmenter,
    RealSenseSegmenterConfig,
)
from in_hand_tracker.perception.sam2_stream import Sam2StreamTracker
from in_hand_tracker.perception.apriltag_detector import AprilTagDetector, GatingConfig
from in_hand_tracker.estimation.position_kf import PositionKF
from in_hand_tracker.estimation.orientation_mekf import OrientationMEKF
from scipy.spatial.transform import Rotation

EST_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "in_hand_tracker", "config", "estimator.yaml")

# Edges of the 8-corner cube (corner order: dx outer, dy mid, dz inner -> idx
# i*4+j*2+k). Two corners share an edge iff they differ in exactly one axis bit.
CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
              if bin(a ^ b).count("1") == 1]


def find_orange_px(bgr):
    """(u, v) centroid of the largest orange HSV blob, or None. Headless seeding
    helper so the demo can lock onto the fruit without a mouse click."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (5, 90, 90), (25, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < 300:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(cent[i][0]), int(cent[i][1]))


class Sam3Client:
    """Thin TCP client to the SAM3 grounding microservice (scripts/sam3_server.py,
    running in the grasp_fruit env). One request = one frame + a text query; the
    reply is a list of detections. Used ONLY on (re)seed, so the ~150 ms round-trip
    does not touch the per-frame SAM2 tracking rate. Any socket error -> [] so the
    caller falls back to the ROI+depth seed."""

    def __init__(self, host="127.0.0.1", port=55003, timeout=8.0):
        self.host = host; self.port = int(port); self.timeout = float(timeout)
        self.sock = None

    def _connect(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            ch = self.sock.recv(n - len(buf))
            if not ch:
                return None
            buf.extend(ch)
        return bytes(buf)

    def query(self, bgr, text):
        """Return [{box:[x0,y0,x1,y1], score, mask(HxW bool)}, ...] or []."""
        import struct, pickle
        try:
            if self.sock is None:
                self._connect()
            H, W = bgr.shape[:2]
            req = {"query": str(text), "shape": [H, W, 3],
                   "data": np.ascontiguousarray(bgr, dtype=np.uint8).tobytes()}
            p = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
            self.sock.sendall(struct.pack(">I", len(p)) + p)
            hdr = self._recv(4)
            if hdr is None:
                raise IOError("server closed")
            (n,) = struct.unpack(">I", hdr)
            rep = pickle.loads(self._recv(n))
        except Exception:
            self.sock = None                     # force reconnect next call
            return []
        if not rep.get("ok"):
            return []
        dets = []
        for d in rep.get("dets", []):
            mh, mw = d["mask_shape"]
            m = np.unpackbits(np.frombuffer(d["mask_packed"], np.uint8))[: mh * mw]
            dets.append({"box": d["box"], "score": d["score"],
                         "mask": m.reshape(mh, mw).astype(bool)})
        return dets


def pick_inhand_det(dets, roi, depth_m, zband, score_floor=0.3):
    """From pooled SAM3 detections pick the in-hand fruit: box-centre inside the
    grasp ROI (and at grasp depth if available), score above a floor, then the
    LARGEST such box. SAM3 returns the globally best instance (often a tray fruit),
    and 'fruit'-type queries also throw small spurious boxes -- picking the biggest
    in-ROI detection reliably lands on the fruit that fills the grasp."""
    if not dets or roi is None:
        return None
    u0, v0, u1, v1 = roi
    z0, z1 = zband
    best = None; best_area = -1.0
    for d in dets:
        if float(d.get("score", 0.0)) < score_floor:
            continue
        x0, y0, x1, y1 = d["box"]
        cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
        if not (u0 <= cx <= u1 and v0 <= cy <= v1):
            continue
        if depth_m is not None:
            ci = int(np.clip(cy, 0, depth_m.shape[0] - 1))
            cj = int(np.clip(cx, 0, depth_m.shape[1] - 1))
            zc = float(depth_m[ci, cj])
            if zc > 0 and not (z0 <= zc <= z1):
                continue
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > best_area:
            best_area = area; best = d
    return best


def roi_from_frac(W, H, cxf, cyf, wf, hf):
    """Pixel ROI box (u0, v0, u1, v1) from centre/size fractions of the image."""
    w = wf * W; h = hf * H
    u0 = int(np.clip(cxf * W - w / 2.0, 0, W - 1))
    u1 = int(np.clip(cxf * W + w / 2.0, 0, W - 1))
    v0 = int(np.clip(cyf * H - h / 2.0, 0, H - 1))
    v1 = int(np.clip(cyf * H + h / 2.0, 0, H - 1))
    return (u0, v0, u1, v1)


def inhand_region(depth_m, roi, zband):
    """Bool mask of pixels INSIDE the ROI box AND within the grasp depth band.

    The robot hand is fixed, so the in-hand fruit always lives in this fixed image
    box at the known grasp depth; everything else (tray fruit at other image
    positions, background/arm at other depths) is excluded. This is the region the
    tracker is allowed to seed/track in."""
    H, W = depth_m.shape
    m = np.zeros((H, W), dtype=bool)
    u0, v0, u1, v1 = roi
    m[v0:v1, u0:u1] = True
    z0, z1 = zband
    return m & (depth_m > z0) & (depth_m < z1)


def inhand_seed_px(depth_m, roi, zband):
    """Auto seed point (u, v) for the in-hand fruit: centroid of the ROI-and-depth
    region, biased to the CLOSEST surface (the fruit is the nearest thing in the
    grasp). Returns None if the region is empty (nothing graspable in view)."""
    reg = inhand_region(depth_m, roi, zband)
    n = int(reg.sum())
    if n < 30:
        return None
    z = depth_m[reg]
    ys, xs = np.nonzero(reg)
    # keep the nearest 60% of the region (drops a bit of background/finger bleed)
    thr = np.percentile(z, 60.0)
    keep = z <= thr
    if keep.sum() >= 20:
        xs, ys = xs[keep], ys[keep]
    return (int(np.median(xs)), int(np.median(ys)))


def obb3d_from_mask_rect(rect_px, medz, K):
    """3D enclosing bounding box (CAMERA frame) from the mask's oriented 2D rect.

    This is the markerless source of the published box POINTS (goal per the user):
    the two in-plane axes + extents come straight from the SAM2 silhouette rect so
    the box hugs the segmentation and elongates with the fruit; the depth
    (camera-forward) half-extent is set to the SHORTER in-plane half (prolate /
    round cross-section assumption) and the box is pushed BACK by that amount so it
    encloses the unseen far side of the fruit (the front face sits at the visible
    surface depth ``medz``).

    Args:
        rect_px: (4, 2) pixel corners of the silhouette's oriented rectangle
            (``cv2.boxPoints`` of ``cv2.minAreaRect``), in order.
        medz: front-surface depth over the mask (median, meters).
        K: (3, 3) intrinsics.

    Returns:
        ``(corners8 (8,3), center (3,), half (3,), R (3,3))`` in the camera frame,
        or ``None`` on a degenerate rect. ``corners8`` ordering matches
        ``CUBE_EDGES`` (sx outer, sy mid, sz inner over (-1, +1)). ``R`` columns are
        the box axes (right-handed, det = +1) so a quaternion is well-defined.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    rect = np.asarray(rect_px, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(medz) or medz <= 0:
        return None

    def deproj(uv, z):
        return np.array([(uv[0] - cx) / fx * z, (uv[1] - cy) / fy * z, z])

    # in-plane metric half-extents at the front depth (approx) -> depth half-extent
    p0 = deproj(rect[0], medz); p1 = deproj(rect[1], medz); p3 = deproj(rect[3], medz)
    hw = 0.5 * float(np.linalg.norm(p1 - p0))
    hh = 0.5 * float(np.linalg.norm(p3 - p0))
    hd = min(hw, hh)
    if not np.isfinite(hd) or hd <= 1e-4:
        return None

    zc = medz + hd                                  # box centre depth (behind surface)
    r = np.array([deproj(uv, zc) for uv in rect])   # mid-plane rect at z = zc
    center = r.mean(axis=0)
    e1 = r[1] - r[0]; e2 = r[3] - r[0]
    n1 = float(np.linalg.norm(e1)); n2 = float(np.linalg.norm(e2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    e1 = e1 / n1; e2 = e2 / n2; e3 = np.array([0.0, 0.0, 1.0])
    hw, hh = 0.5 * n1, 0.5 * n2
    R = np.stack([e1, e2, e3], axis=1)
    if np.linalg.det(R) < 0:                        # enforce right-handed (det +1)
        e2 = -e2
        R = np.stack([e1, e2, e3], axis=1)
    half = np.array([hw, hh, hd])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center + (signs * half) @ R.T
    return corners, center, half, R


def start_ros2_camera(color_topic, info_topic, depth_topic, wait_s=20.0):
    """Build the shared ROS2 camera node and wait for color + K (+ depth)."""
    rclpy, node = _build_node(color_topic, info_topic, depth_topic, use_depth=True)
    print(f"[live] waiting for first color frame on {color_topic} ...")
    if not _spin_until(rclpy, node, node.has_color, wait_s):
        raise RuntimeError(
            f"no image on {color_topic} within {wait_s:.0f}s "
            "(is the camera driver running?)"
        )
    if not _spin_until(rclpy, node, lambda: node.K is not None, 5.0):
        raise RuntimeError(f"no CameraInfo on {info_topic} (K is required here)")
    _spin_until(rclpy, node, lambda: node._latest_depth_m is not None, 3.0)
    if node._latest_depth_m is None:
        print(f"[live] WARN: no depth on {depth_topic}; 3D fit/KF will stay idle.\n"
              "       enable it:  ros2 param set /front_cam/front align_depth.enable true")
    print("[live] camera topics up")
    return rclpy, node


def main():
    ap = argparse.ArgumentParser(description="Live SAM2 fruit-bbox GUI (ROS2 topics)")
    ap.add_argument("--namespace", default=_DEFAULT_NAMESPACE,
                    help="camera namespace (default: %(default)s)")
    ap.add_argument("--color-topic", default=None,
                    help="override; default <ns>/color/image_raw")
    ap.add_argument("--info-topic", default=None,
                    help="override; default <ns>/color/camera_info")
    ap.add_argument("--depth-topic", default=None,
                    help="override; default <ns>/aligned_depth_to_color/image_raw")
    ap.add_argument("--wait", type=float, default=20.0,
                    help="seconds to wait for the first frame")
    ap.add_argument("--min-mask-px", type=int, default=400, help="min SAM2 mask pixels to trust")
    ap.add_argument("--tag-size", type=float, default=0.02, help="AprilTag edge length (m)")
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--tag-id", type=int, default=10, help="reference tag id driving orientation")
    # Robust MULTI-VIEW (a,c) calibration (not a one-shot from the initial view):
    # sample the silhouette semi-axes only from clean SOLID full silhouettes; lock
    # once stable across views (encourages rotating the fruit -> doc 3.1 "2시점").
    ap.add_argument("--cal-min", type=int, default=8,
                    help="min clean-silhouette samples before the (a,c) size may lock")
    ap.add_argument("--cal-cv", type=float, default=0.08,
                    help="lock once the recent samples' coefficient-of-variation <= this")
    ap.add_argument("--cal-max", type=int, default=50,
                    help="hard cap: lock to the running median after this many samples")
    ap.add_argument("--ca-threshold", type=float, default=1.2,
                    help="c/a ratio at/above which the fruit is treated as elongated (else sphere)")
    ap.add_argument("--log", default="live_bbox_log.jsonl",
                    help="JSONL file: press 's' to append the current bbox/COM/orientation record")
    # AprilTag gating (relaxed vs library defaults so it triggers more readily).
    ap.add_argument("--tag-loose", action="store_true",
                    help="use ANY detected id-tag pose for orientation, ignoring the gates")
    ap.add_argument("--tag-min-margin", type=float, default=12.0)
    ap.add_argument("--tag-max-angle", type=float, default=85.0)
    ap.add_argument("--tag-max-reproj", type=float, default=10.0)
    ap.add_argument("--tag-min-size", type=float, default=8.0)
    ap.add_argument("--tag-upscale", type=float, default=2.0,
                    help="upsample gray before tag detect; a ~25px tag on the "
                         "fruit needs 2x to decode reliably (1.0 = off)")
    # Position-KF tuning. q is the white-acceleration spectral density; r the
    # measurement variance. The fit centre is accurate here (fitRMS ~2 mm,
    # sub-mm jitter), so we trust it: a HIGHER q + LOWER r raise the Kalman gain
    # so the box FOLLOWS in-hand motion instead of lagging behind it. (The old
    # q=0.3 / r=2e-3 over-damped for a stationary fruit and could not keep up
    # with a moving hand.)
    ap.add_argument("--kf-q", type=float, default=6.0,
                    help="KF process noise ((m/s^2)^2); higher = follows motion faster")
    ap.add_argument("--kf-r", type=float, default=4e-4,
                    help="KF measurement variance (m^2); lower = trusts the fit / less lag")
    # AprilTag detect is the #1 per-frame cost (~35 ms at upscale 2.0, vs ~26 ms
    # for SAM2). The orientation MEKF PREDICTS between updates, so we can detect
    # the tag only every N frames and interpolate attitude -> big fps win with no
    # loss of the (slowly-changing) in-hand orientation. N=1 restores per-frame.
    ap.add_argument("--tag-every-n", type=int, default=2,
                    help="run AprilTag detection every N frames (MEKF predicts between); "
                         "1 = every frame")
    ap.add_argument("--auto-seed-orange", action="store_true",
                    help="on first seed, auto-target the largest orange HSV blob "
                         "instead of the image centre (headless demo without a click)")
    ap.add_argument("--frame-id", default="front_cam_front_color_optical_frame",
                    help="fallback TF frame for the published bbox Marker when the "
                         "camera stream does not carry a header frame_id")
    # In-hand constraint: the robot hand is FIXED, so restrict segmentation to a
    # fixed image ROI at the grasp AND the grasp depth band. This auto-seeds on the
    # in-hand fruit (no click), gates the mask to that region, and re-seeds if the
    # tracker drifts off it -- so ONLY the in-hand fruit is tracked.
    ap.add_argument("--no-inhand", action="store_true",
                    help="disable the fixed ROI+depth in-hand constraint (free seeding)")
    ap.add_argument("--roi-frac", default="0.5,0.45,0.55,0.7",
                    help="ROI as cx,cy,w,h fractions of the image (grasp region)")
    ap.add_argument("--depth-band", default="0.15,0.45",
                    help="grasp depth band zmin,zmax (m) for the in-hand fruit")
    # SAM3 grounding for ROBUST (re)seeding: at seed time, ask the SAM3 service for
    # the fruit by name and seed SAM2 with its box (fixes "seeds the wrong object"
    # + gives a cleaner initial mask). SAM3 runs only on (re)seed, so tracking stays
    # at the SAM2 rate. Requires scripts/sam3_server.py running in the grasp_fruit
    # env. Falls back to the ROI+depth point seed if the service is unreachable.
    ap.add_argument("--no-sam3", action="store_true",
                    help="disable SAM3-grounded seeding (use ROI+depth point seed only)")
    ap.add_argument("--fruit-query", default="orange,lemon",
                    help="SAM3 text query(ies) for the in-hand fruit, comma-separated "
                         "(pooled; e.g. 'orange,lemon'). MUST name the actual fruit -- "
                         "'orange' will not detect a lemon.")
    ap.add_argument("--sam3-host", default="127.0.0.1")
    ap.add_argument("--sam3-port", type=int, default=55003)
    args = ap.parse_args()

    print("[live] loading SAM2 (first call pays the model-load cost) ...")
    seg_cfg = RealSenseSegmenterConfig.from_yaml(EST_YAML)
    # Primary: stateful streaming SAM2 video predictor. Fallback: per-frame image
    # predictor (RealSenseSegmenter) if CUDA / ckpt are unavailable.
    stream = None
    seg = None
    try:
        stream = Sam2StreamTracker(
            seg_cfg.sam2_ckpt, seg_cfg.sam2_config, device=seg_cfg.device
        )
        print("[live] streaming SAM2 video predictor ready (seg=sam2-stream)")
    except Exception as e:  # noqa: BLE001
        print(f"[live] stream tracker unavailable ({type(e).__name__}: {str(e)[:80]}); "
              "falling back to per-frame image predictor")
        seg = RealSenseSegmenter(seg_cfg)

    ns = args.namespace.rstrip("/")
    color_topic = args.color_topic or f"{ns}/color/image_raw"
    info_topic = args.info_topic or f"{ns}/color/camera_info"
    depth_topic = args.depth_topic or f"{ns}/aligned_depth_to_color/image_raw"
    rclpy, cam = start_ros2_camera(color_topic, info_topic, depth_topic, args.wait)
    K = cam.K
    fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])

    # Publishers for the GOAL deliverable: the enclosing 3D bbox corner POINTS
    # (camera frame). /inhand/bbox_corners = raw 8x3 float array for downstream
    # numeric use; /inhand/bbox = a LINE_LIST Marker wireframe for RViz.
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    pub_corners = cam.create_publisher(Float32MultiArray, "/inhand/bbox_corners", 10)
    pub_marker = cam.create_publisher(MarkerArray, "/inhand/bbox", 10)

    def publish_bbox(corners, center, half, frame_id, stamp):
        """Publish the 8 corner points (8x3, camera frame) + an RViz wireframe."""
        fid = frame_id or args.frame_id
        # 1) raw corner points: Float32MultiArray with an (8,3) layout.
        arr = Float32MultiArray()
        arr.layout.dim = [
            MultiArrayDimension(label="corner", size=8, stride=24),
            MultiArrayDimension(label="xyz", size=3, stride=3),
        ]
        arr.data = [float(v) for v in np.asarray(corners, dtype=np.float64).reshape(-1)]
        pub_corners.publish(arr)
        # 2) wireframe marker (12 edges via CUBE_EDGES).
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns = "inhand_bbox"; m.id = 0
        m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.scale.x = 0.003
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.1, 1.0
        m.pose.orientation.w = 1.0
        for ia, ib in CUBE_EDGES:
            for idx in (ia, ib):
                p = Point()
                p.x, p.y, p.z = (float(corners[idx][0]), float(corners[idx][1]),
                                 float(corners[idx][2]))
                m.points.append(p)
        ma = MarkerArray(); ma.markers.append(m)
        pub_marker.publish(ma)

    # AprilTag (tag --tag-id) -> orientation MEKF (C3/C4). Seen -> update attitude;
    # hidden -> MEKF predicts (orientation persists through occlusion).
    try:
        gating = GatingConfig(
            min_decision_margin=args.tag_min_margin,
            max_view_angle_deg=args.tag_max_angle,
            max_reproj_error_px=args.tag_max_reproj,
            min_tag_size_px=args.tag_min_size,
        )
        tag_det = AprilTagDetector(families=args.tag_family, tag_size=args.tag_size,
                                   gating=gating, upscale=args.tag_upscale)
    except Exception as e:  # noqa: BLE001
        print("[live] AprilTag detector unavailable:", type(e).__name__, str(e)[:80])
        tag_det = None
    mekf = OrientationMEKF()
    ori_init = False
    TAG_MEAS_COV = (np.deg2rad(5.0) ** 2) * np.eye(3)

    kf = PositionKF(q=args.kf_q, r=args.kf_r)   # C4 CV position filter -> smooth + ride occlusion
    # Outlier gate on the measured centre: reject a jump larger than what a
    # plausible in-hand speed could produce over the frame interval. It is
    # dt-AWARE (base + max_speed*dt) so a fast hand at a low frame rate is not
    # mistaken for an outlier and frozen out -- the old fixed 5 cm gate rejected
    # valid measurements whenever the fruit moved faster than ~0.6 m/s at 13 fps.
    JUMP_BASE = 0.05                       # base tolerance (m)
    JUMP_SPEED = 1.5                       # max plausible in-hand speed (m/s)
    MIN_INLIERS = 60
    # Shape model (auto-branch): estimate equatorial a + polar c from the mask
    # silhouette ellipse over multiple views; c/a >= --ca-threshold -> elongated
    # (oriented ellipsoid box via the tag), else spheroid (sphere). Locked when stable.
    a_cal = 0.038
    c_cal = 0.038
    r_fit = 0.038                          # smooth radius for the STABLE sphere centre fit
    r_hist = deque(maxlen=15)              # median window: rejects mask-flicker spikes
    shape_class = "spheroid"
    shape_locked = False
    a_samples = []
    c_samples = []
    box_half = np.array([0.038, 0.038, 0.038])   # current box half-extents (m), for drawing
    box_R = None                                  # box orientation (None = axis-aligned)
    last_t = None
    frame_idx = 0
    last_log_t = -1e9                             # for the brief on-screen flash
    recording = False                             # 's' toggles: 1st press starts, 2nd stops+saves
    rec_buffer = []                               # stacked per-frame records of the session
    session_idx = 0
    recent_centers = deque(maxlen=20)      # for the centre-jitter (3D repeatability) metric
    sil_aspect_hist = deque(maxlen=15)     # smoothed silhouette aspect (round vs elongated)

    # In-hand ROI + depth constraint (fixed hand -> fixed grasp region).
    inhand_on = not args.no_inhand
    _rf = [float(x) for x in args.roi_frac.split(",")]
    zband = tuple(float(x) for x in args.depth_band.split(","))
    roi_box = None                         # (u0,v0,u1,v1), computed once W,H known
    lost_frames = 0                        # consecutive frames with a collapsed mask
    last_sam3_frame = -10 ** 9             # rate-limit SAM3 (only ~every N frames)
    SAM3_MIN_INTERVAL = 30                 # min frames between SAM3 calls (~2 s)
    LOST_THRESH = 8                        # sustained-loss frames before re-acquire
    fruit_queries = [q.strip() for q in args.fruit_query.split(",") if q.strip()]
    sam3 = None if args.no_sam3 else Sam3Client(args.sam3_host, args.sam3_port)
    if sam3 is not None:
        print(f"[live] SAM3 seeding ON (query={args.fruit_query!r}, "
              f"{args.sam3_host}:{args.sam3_port}); start scripts/sam3_server.py in "
              f"the grasp_fruit env. Falls back to ROI+depth if unreachable.")
    # Tag-visual persistence: with --tag-every-n the tag is DETECTED only every N
    # frames, but the on-screen tag cues (axes brightness, magenta marker, status
    # text) must NOT toggle on the skipped frames -- that per-frame blink is pure
    # rendering, not a real state change. We track the last successful tag frame /
    # position and drive the visuals off "recently seen" (freshness), not "seen on
    # this exact frame".
    tag_last_frame = -10 ** 9              # frame_i of the last APPLIED (accepted) tag
    tag_last_center_px = None              # last magenta marker position (persisted)
    tag_last_obs = None                    # last DETECTION (any gate) for the readout
    tag_seen_frame = -10 ** 9              # frame_i of the last detection (any gate)
    predict_run = 0                        # consecutive predict-only frames (colour hysteresis)

    # Mouse seeding: click the fruit to (re)acquire SAM2 there.
    ui = {"seed_px": None, "reset": False}
    seeded = False                         # whether the stream tracker is seeded
    seed_method = "-"                      # how the current track was seeded (sam3/roi)
    tag_err_printed = False                # one-shot warning for tag-detect failures

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["seed_px"] = (int(x), int(y))
            ui["reset"] = True

    win = "in_hand_tracker live (SAM2)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    cv2.setMouseCallback(win, on_mouse)

    # Draw-level smoother: the KF output still carries ~1.5 mm of residual
    # noise (~3-4 px at working distance), which reads as a trembling box on a
    # perfectly stationary fruit. A deadband freezes sub-threshold changes
    # completely; real motion (above it) is followed with a light EMA.
    draw_state = {}

    def smooth_draw(key, new, dead=0.0006, alpha=0.6, snap=0.012):
        """Draw-level polish that must NOT hide real motion. Three regimes:
        below ``dead`` -> hold (kills sub-mm jitter on a still fruit); above
        ``snap`` -> jump straight to the new value (fast in-hand motion follows
        with zero EMA lag); in between -> a light EMA. The old single-EMA +
        large deadband froze slow motion and lagged fast motion."""
        new = np.asarray(new, dtype=np.float64)
        cur = draw_state.get(key)
        if cur is None or not np.all(np.isfinite(cur)):
            draw_state[key] = new
            return draw_state[key]
        d = float(np.linalg.norm(new - cur))
        if d >= snap:
            draw_state[key] = new
        elif d >= dead:
            draw_state[key] = cur + alpha * (new - cur)
        return draw_state[key]

    fps = 0.0
    frame_i = -1
    try:
        while True:
            frame_i += 1
            # Block for a genuinely NEW color frame from the topic.
            target = cam.color_count() + 1
            if not _spin_until(rclpy, cam, lambda: cam.color_count() >= target, 5.0):
                print("[live] shutting down." if not rclpy.ok() else
                      "[live] timed out waiting for a new image; is the driver up?")
                break
            bgr, depth_m, _ = cam.grab()
            if depth_m is None or depth_m.shape != bgr.shape[:2]:
                depth_m = np.zeros(bgr.shape[:2], dtype=np.float32)  # 3D fit idles
            H, W = depth_m.shape
            vis = bgr.copy()

            now = time.time()
            dt = (now - last_t) if last_t is not None else (1.0 / 30.0)
            last_t = now
            frame_idx += 1

            if ui["reset"]:
                kf = PositionKF(q=args.kf_q, r=args.kf_r)
                mekf = OrientationMEKF(); ori_init = False
                shape_locked = False; a_samples = []; c_samples = []
                shape_class = "spheroid"; r_fit = 0.038; r_hist.clear()
                draw_state.clear()
                tag_last_frame = -10 ** 9; tag_last_center_px = None; predict_run = 0
                tag_last_obs = None; tag_seen_frame = -10 ** 9
                seeded = False              # force a stream re-seed this frame
                ui["reset"] = False

            # Fixed grasp ROI (compute once the frame size is known).
            if inhand_on and roi_box is None:
                roi_box = roi_from_frac(W, H, _rf[0], _rf[1], _rf[2], _rf[3])
            roi_center = (((roi_box[0] + roi_box[2]) // 2, (roi_box[1] + roi_box[3]) // 2)
                          if roi_box is not None else (W // 2, H // 2))
            if inhand_on and roi_box is not None:
                cv2.rectangle(vis, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]),
                              (180, 180, 180), 1)   # grasp ROI (fixed hand region)

            # --- choose the SAM2 seed point (used only when (re)seeding) ---
            if ui["seed_px"] is not None:
                px = ui["seed_px"]                              # manual click override
            elif inhand_on:
                # Auto-seed strictly on the in-hand fruit (ROI + grasp depth).
                px = inhand_seed_px(depth_m, roi_box, zband) or roi_center
            elif kf.initialized:
                cuv = np.asarray(project_points(kf.center()[None], K)).reshape(-1, 2)[0]
                if np.all(np.isfinite(cuv)):
                    px = (int(np.clip(cuv[0], 0, W - 1)), int(np.clip(cuv[1], 0, H - 1)))
                else:
                    px = (W // 2, H // 2)
            elif args.auto_seed_orange and not seeded:
                px = find_orange_px(bgr) or (W // 2, H // 2)
            else:
                px = (W // 2, H // 2)

            # --- SAM3-grounded (re)seed box (only when re-seeding; ~150 ms) ---
            # Ask SAM3 for the fruit by name and keep the detection whose box is in
            # the grasp ROI -> robustly the IN-HAND fruit (not a tray one). Seeds
            # SAM2 with that box for a clean initial mask. Falls back to the point.
            sam3_seed_box = None
            if (not seeded and sam3 is not None and ui["seed_px"] is None
                    and (frame_i - last_sam3_frame) >= SAM3_MIN_INTERVAL):
                last_sam3_frame = frame_i           # rate-limit: at most ~every N frames
                dets = []
                for q in fruit_queries:             # pool all query terms
                    dets += sam3.query(bgr, q)
                pick = pick_inhand_det(dets, roi_box, depth_m, zband)
                if pick is not None:
                    sam3_seed_box = [float(v) for v in pick["box"]]
                    seed_method = "sam3"
                else:
                    seed_method = "roi" if not dets else "sam3-miss"

            # --- SAM2 segmentation (mask is the ONLY source of the bbox) ---
            if stream is not None:
                method = "sam2-stream"
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if not seeded:
                    # (re)seed: encode this frame + prompt (SAM3 box if available).
                    stream.reset()
                    stream.load_first_frame(rgb)
                    if sam3_seed_box is not None:
                        mask = np.asarray(stream.add_prompt(box=sam3_seed_box), dtype=bool)
                    else:
                        mask = np.asarray(stream.add_prompt(points=[px]), dtype=bool)
                    seeded = True
                else:
                    # propagate via the video predictor's memory (no reprompt).
                    mask = np.asarray(stream.track(rgb), dtype=bool)
            else:
                res = seg.segment_at(bgr, [px])
                mask = np.asarray(res.mask, dtype=bool)
                method = res.method

            # --- In-hand gate: keep ONLY mask pixels in the grasp ROI + depth band,
            # so a tray fruit / arm / background the tracker may bleed onto is
            # dropped. If the gated mask collapses (tracker drifted off the in-hand
            # fruit), force a re-seed from the ROI next frame. ---
            if inhand_on and roi_box is not None:
                reg = inhand_region(depth_m, roi_box, zband)
                mask = mask & reg
                # Re-acquire only on SUSTAINED loss (not a single dropped frame), so
                # a transient occlusion does not thrash the (expensive) re-seed. The
                # SAM3 call on re-seed is separately rate-limited above.
                if int(mask.sum()) < args.min_mask_px:
                    lost_frames += 1
                else:
                    lost_frames = 0
                if lost_frames >= LOST_THRESH:
                    seeded = False
                    lost_frames = 0

            # --- AprilTag orientation: tag --tag-id drives the MEKF (C3/C4) ---
            mekf.predict(dt=dt)
            tag_obs = None            # id-tag detection (ANY gate status, for diagnostics)
            tag_used = False          # whether its orientation was applied this frame
            # Detect only every N frames: the tag detect (~35 ms at upscale 2.0)
            # is the biggest per-frame cost, and in-hand orientation changes slowly
            # -- the MEKF predict() above carries the attitude on the skipped
            # frames. (mekf.predict already ran this frame regardless.)
            run_tag = (tag_det is not None
                       and frame_i % max(1, args.tag_every_n) == 0)
            if run_tag:
                try:
                    for o in tag_det.detect(bgr, (fx, fy, cx, cy)):
                        if int(o.tag_id) == args.tag_id:
                            tag_obs = o
                            break
                except Exception as e:  # noqa: BLE001
                    tag_obs = None
                    # Do NOT swallow this silently: a missing pupil_apriltags
                    # used to fail here every frame while the overlay just said
                    # "not detected".
                    if not tag_err_printed:
                        print(f"[live] tag detect FAILING: {type(e).__name__}: {e}")
                        tag_err_printed = True
            if tag_obs is not None:
                # Persist the last detection (ANY gate) so the on-screen readout is
                # steady across --tag-every-n skipped frames instead of toggling.
                tag_last_obs = tag_obs
                tag_seen_frame = frame_i
            if tag_obs is not None and (tag_obs.accepted or (args.tag_loose and tag_obs.R is not None)):
                mekf.update(tag_obs.quat_wxyz(), TAG_MEAS_COV)
                ori_init = True
                tag_used = True
                tag_last_frame = frame_i
                if tag_obs.center_px is not None:
                    tag_last_center_px = np.asarray(tag_obs.center_px)
            # "Recently seen" over the detection cadence -> stable visuals across the
            # skipped frames (no per-frame blink). Only goes stale after we actually
            # miss the tag for longer than one detect interval.
            tag_fresh = ori_init and (frame_i - tag_last_frame) <= max(1, args.tag_every_n)
            if ori_init:
                qw = mekf.quat()
                R_wo = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            else:
                R_wo = None

            measured_C = None
            n_in = 0
            fit_resid_mm = None            # 3D surface-fit RMS error (GT-free accuracy proxy)
            inlier_ratio = None
            n_used = 0
            mask_rect_px = None            # silhouette oriented-rect corners (4,2)
            obb3d = None                   # published 3D box (corners, center, half, R)
            if mask.sum() >= args.min_mask_px:
                # mask outline (cyan) so the SAM2 segmentation is visible.
                mm = (mask.astype(np.uint8)) * 255
                cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1)

                # Silhouette-derived ORIENTED 2D bounding box (orange). Taken straight
                # from the SAM2 mask via minAreaRect, so it HUGS the segmentation and
                # rotates/elongates with the fruit -- no shape lock, tag, or depth
                # needed. This is what makes an elongated fruit read as an elongated
                # box instead of the (shape-locked) sphere circle.
                if cnts:
                    cmax0 = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(cmax0) >= args.min_mask_px:
                        (_bx, _by), (bw, bh), bang = cv2.minAreaRect(cmax0)
                        sil_aspect_hist.append(max(bw, bh) / max(1.0, min(bw, bh)))
                        # Elongated (smoothed aspect >= threshold) -> ORIENTED rect that
                        # tracks the fruit's long axis; round -> AXIS-ALIGNED rect (the
                        # min-area rect of a near-circle has an arbitrary tilt, which
                        # looks wrong, so a straight box reads cleaner there). The SAME
                        # corners feed the 2D overlay AND the published 3D box, so what
                        # is drawn matches what is published.
                        if float(np.median(sil_aspect_hist)) >= args.ca_threshold:
                            mask_rect_px = cv2.boxPoints(((_bx, _by), (bw, bh), bang))
                        else:
                            rx, ry, rw, rh = cv2.boundingRect(cmax0)
                            mask_rect_px = np.array(
                                [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]],
                                dtype=np.float64,
                            )
                        # NB: the box is DRAWN once in the viz section below (as the 3D
                        # box when depth is available, else this 2D rect) -- drawing it
                        # here too would double up the overlay.

                valid = mask & (depth_m > 0.05) & (depth_m < 2.0)
                if valid.sum() >= 30:
                    medz = float(np.median(depth_m[valid]))
                    # GOAL: enclosing 3D bbox corner POINTS (camera frame) from the
                    # silhouette rect + depth, published to ROS2 every frame.
                    if mask_rect_px is not None:
                        obb3d = obb3d_from_mask_rect(mask_rect_px, medz, K)
                        if obb3d is not None:
                            _c8, _cc, _hh3, _RR = obb3d
                            publish_bbox(_c8, _cc, _hh3, cam._color_frame_id,
                                         cam.get_clock().now().to_msg())
                    # Smooth equivalent-circle radius -> STABLE sphere centre fit
                    # (well-conditioned; restores the pre-ellipsoid smoothness).
                    # medz is the FRONT-surface depth, but the mask silhouette is set
                    # by the object CENTRE depth (medz + R). Using medz alone under-
                    # sizes the sphere by ~R/z (~8% at 30 cm), so it reads as sitting
                    # just inside the mask. Solve R = r_px*(medz + R)/fx in closed form
                    # (den = 1 - r_px/fx) so the drawn silhouette OVERLAYS the SAM2 mask.
                    mask_r_px = float(np.sqrt(mask.sum() / np.pi))
                    den = max(0.2, 1.0 - mask_r_px / fx)
                    r_est = float(np.clip(mask_r_px * medz / fx / den, 0.012, 0.09))
                    # Silhouette ELLIPSE semi-axes -> SHAPE decision + box dims ONLY.
                    a_est = c_est = None
                    solid_ok = False
                    if cnts:
                        cmax = max(cnts, key=cv2.contourArea)
                        carea = float(cv2.contourArea(cmax))
                        hull = cv2.convexHull(cmax)
                        harea = float(cv2.contourArea(hull)) if hull is not None else carea
                        solid_ok = (carea >= 1.5 * args.min_mask_px
                                    and carea / max(1.0, harea) >= 0.88)
                        if len(cmax) >= 5:
                            (_, (ax1, ax2), _ang) = cv2.fitEllipse(cmax)
                            major_px = max(ax1, ax2); minor_px = min(ax1, ax2)
                            a_est = float(np.clip(0.5 * minor_px * medz / fx, 0.012, 0.08))
                            c_est = float(np.clip(0.5 * major_px * medz / fx, 0.012, 0.10))
                    # Calibrate while not locked; DECIDE the shape ONCE at lock (no
                    # per-frame spheroid<->elongated flapping -> no fit-param jitter).
                    if not shape_locked:
                        # Median window, NOT an EMA chase: the mask area flickers
                        # as fingers enter/leave the silhouette, and an EMA passes
                        # that straight into the drawn box size (visible shake).
                        r_hist.append(r_est)
                        r_fit = float(np.median(r_hist))
                        if a_est is not None and solid_ok:
                            a_samples.append(a_est); c_samples.append(c_est)
                        if a_samples:
                            a_cal = float(np.median(a_samples[-args.cal_max:]))
                            c_cal = float(np.percentile(c_samples[-args.cal_max:], 85))
                        lock_now = False
                        if len(a_samples) >= args.cal_min:
                            ra = np.asarray(a_samples[-args.cal_min:])
                            rc = np.asarray(c_samples[-args.cal_min:])
                            lock_now = (ra.std() / max(1e-6, ra.mean()) <= args.cal_cv
                                        and rc.std() / max(1e-6, rc.mean()) <= args.cal_cv)
                        if not lock_now and len(a_samples) >= args.cal_max:
                            lock_now = True
                        if lock_now:
                            shape_class = ("elongated"
                                           if (c_cal / max(1e-6, a_cal)) >= args.ca_threshold
                                           else "spheroid")
                            shape_locked = True

                    # Box + centre-fit geometry. Centre fit stays a STABLE sphere (r_fit)
                    # except once locked-elongated WITH a tag attitude (then the oriented
                    # ellipsoid fit is well-conditioned and gives the better centre).
                    if shape_locked and shape_class == "elongated":
                        if R_wo is not None:
                            box_half = np.array([a_cal, a_cal, c_cal]); box_R = R_wo
                            fit_a, fit_c, fit_R = a_cal, c_cal, R_wo
                        else:  # no orientation -> conservative box (doc 3.4), sphere centre
                            box_half = np.array([c_cal, c_cal, c_cal]); box_R = None
                            fit_a, fit_c, fit_R = r_fit, r_fit, None
                    else:
                        box_half = np.array([r_fit, r_fit, r_fit]); box_R = None
                        fit_a, fit_c, fit_R = r_fit, r_fit, None

                    band = (1.8 * c_cal if (shape_locked and shape_class == "elongated") else 0.07)
                    pts = deproject_zdepth(depth_m, valid, K)
                    if pts.size:
                        pts = pts[np.isfinite(pts).all(1) & (np.abs(pts[:, 2] - medz) < band)]
                    if pts.shape[0] >= 30:
                        fit = fit_ellipsoid_center(
                            pts, a=fit_a, c=fit_c,
                            shape_class=("elongated" if fit_R is not None else "spheroid"),
                            R_wo=fit_R, n_iterations=150, inlier_ratio_target=0.5,
                        )
                        if fit.ok and fit.center is not None and np.all(np.isfinite(fit.center)):
                            measured_C = np.asarray(fit.center, dtype=np.float64)
                            inl = (np.asarray(fit.inliers, dtype=bool)
                                   if fit.inliers is not None else np.ones(pts.shape[0], dtype=bool))
                            n_in = int(inl.sum())
                            used = pts[inl] if n_in >= 3 else pts
                            n_used = int(used.shape[0])
                            # surface-fit residual (object frame): | sqrt(f) - 1 | * a. GT-free.
                            pl = (used - measured_C) @ fit_R if fit_R is not None else (used - measured_C)
                            ff = (pl[:, 0] ** 2 + pl[:, 1] ** 2) / fit_a ** 2 + pl[:, 2] ** 2 / fit_c ** 2
                            d = np.abs(np.sqrt(ff) - 1.0) * fit_a
                            fit_resid_mm = float(np.sqrt(np.mean(d ** 2)) * 1000.0)
                            inlier_ratio = float(n_in) / float(pts.shape[0])

            # --- C4: position KF + measurement gating ---
            state = "lost"
            if measured_C is not None:
                if not kf.initialized:
                    kf.initialize(measured_C); state = "tracking"
                elif (n_in >= MIN_INLIERS
                      and np.linalg.norm(measured_C - kf.center())
                          < JUMP_BASE + JUMP_SPEED * max(dt, 1e-3)):
                    kf.predict(dt); kf.update(measured_C); state = "tracking"
                else:
                    kf.predict(dt); state = "occluded->predict"
            elif kf.initialized:
                kf.predict(dt); state = "occluded->predict"

            # Box-colour hysteresis: a SINGLE rejected/occluded frame must not flash
            # the box orange. Stay green until predict-only persists for >2 frames.
            predict_run = 0 if state == "tracking" else predict_run + 1
            box_tracking = (state == "tracking") or (predict_run <= 2)

            # Draw exactly ONE box: the silhouette-oriented rectangle (a single
            # clean rect that hugs the fruit). The full 3D box is PUBLISHED to
            # /inhand/bbox_corners (8 corners) but NOT drawn as a wireframe -- its
            # front+back faces projected as two rectangles and read as "two boxes".
            if mask_rect_px is not None:
                cv2.polylines(vis, [mask_rect_px.astype(np.intp)], True, (0, 255, 0), 2)
            if obb3d is not None and frame_i % 30 == 0:
                _cc = obb3d[1]
                print(f"[live] /inhand/bbox_corners published: 8x3, "
                      f"center=({_cc[0]*100:.1f},{_cc[1]*100:.1f},{_cc[2]*100:.1f})cm",
                      flush=True)

            jitter_mm = None
            if kf.initialized and state != "lost":
                C = kf.center()
                recent_centers.append(C.copy())
                if len(recent_centers) >= 5:
                    arr = np.asarray(recent_centers)
                    jitter_mm = float(np.linalg.norm(arr.std(axis=0)) * 1000.0)
                # Box from the shape model: ORIENTED ellipsoid (a,a,c) when a tag
                # attitude is available; else axis-aligned sphere / conservative cube.
                # Rendering uses the deadband-smoothed copies so the overlay does
                # not tremble on residual mm-level KF noise.
                Cd = smooth_draw("C", C)
                half_d = smooth_draw("half", box_half, dead=0.001)
                hx, hy, hz = float(half_d[0]), float(half_d[1]), float(half_d[2])
                col = (0, 255, 0) if box_tracking else (0, 200, 255)
                # EVERY projected point must be finite before int-casting: a point
                # behind the camera (z <= 0) projects to NaN, and int(NaN) crashes
                # cv2.line/drawMarker (observed: gizmo axis tip going behind the cam
                # took down the whole GUI). Guard the centre, box, circle, gizmo and
                # marker so a bad projection just skips that primitive for one frame.
                cuv = np.asarray(project_points(Cd[None], K)).reshape(-1, 2)[0]
                cu_ok = bool(np.all(np.isfinite(cuv))) and Cd[2] > 0
                cu_i = (int(cuv[0]), int(cuv[1])) if cu_ok else None
                # NOTE: the fruit bbox is drawn ONCE above as the silhouette rectangle
                # (mask_rect_px). The old box_R 3D wireframe here was a SECOND box on
                # screen (its front+back faces read as two boxes) -- removed. Only the
                # centre cross + orientation gizmo are drawn from here on.
                if cu_i is not None:
                    cv2.drawMarker(vis, cu_i, col, cv2.MARKER_CROSS, 16, 2)
                cv2.drawMarker(vis, px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
                # Orientation axes gizmo (x=red, y=green, z=blue): bright while the
                # tag is RECENTLY SEEN (fresh over the detect cadence), dimmed only
                # after the tag is genuinely lost for longer than one detect interval.
                # Keyed on tag_fresh (not tag_used) so it does NOT blink on the frames
                # skipped by --tag-every-n.
                if ori_init and R_wo is not None and cu_ok:
                    L = max(hx, hy, hz) * 1.5
                    ax = np.vstack([Cd, Cd + R_wo[:, 0] * L, Cd + R_wo[:, 1] * L, Cd + R_wo[:, 2] * L])
                    auf = np.asarray(project_points(ax, K)).reshape(-1, 2)
                    fin = np.isfinite(auf).all(axis=1)
                    g = 1.0 if tag_fresh else 0.45
                    axis_cols = [(0, 0, int(255 * g)), (0, int(255 * g), 0), (int(255 * g), 0, 0)]
                    if fin[0]:
                        o0 = tuple(auf[0].astype(int))
                        for j in range(3):
                            if fin[j + 1]:
                                cv2.line(vis, o0, tuple(auf[j + 1].astype(int)), axis_cols[j], 2)
                # AprilTag-seen marker (magenta square) drawn from the LAST known tag
                # centre while fresh -> stays put on skipped frames (no on/off blink).
                if tag_fresh and tag_last_center_px is not None:
                    tc = np.asarray(tag_last_center_px)
                    if np.all(np.isfinite(tc)):
                        cv2.drawMarker(vis, (int(tc[0]), int(tc[1])), (255, 0, 255),
                                       cv2.MARKER_SQUARE, 14, 2)

            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, dt))
            mcol = (0, 255, 0) if method.startswith("sam2") else (0, 0, 255)
            status = state if kf.initialized else "seed: click the fruit"
            # Keyed on freshness (not this-frame detection) so the label is steady
            # across --tag-every-n skipped frames instead of toggling every frame.
            ori_str = (f"tag{args.tag_id}" if tag_fresh
                       else ("ori-predict" if ori_init else "no-ori"))
            # Silhouette shape read straight off the mask OBB (no lock needed), so
            # the label matches the drawn orange box: elongated fruit -> "elong".
            sil_ratio = float(np.median(sil_aspect_hist)) if sil_aspect_hist else 1.0
            sil_shape = "elong" if sil_ratio >= args.ca_threshold else "round"
            seed_str = f"seed={seed_method}" if sam3 is not None else ""
            cv2.putText(vis, f"seg={method}  {status}  ori={ori_str}  "
                             f"bbox={sil_shape}({sil_ratio:.2f})  {seed_str}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mcol, 2)
            # 3D diagnostics (no GT live): centre XYZ, diameter, surface-fit RMS, inlier%, jitter
            if kf.initialized and state != "lost":
                C = kf.center()
                # Report the PUBLISHED box (silhouette-derived W x H x D) + shape,
                # consistent with the drawn box -- not the legacy shape-lock a/c.
                if obb3d is not None:
                    _bw, _bh, _bd = (2.0 * v for v in obb3d[2])
                    box_str = (f"{sil_shape} box={_bw*100:.1f}x{_bh*100:.1f}"
                               f"x{_bd*100:.1f}cm c/a={sil_ratio:.2f}")
                else:
                    box_str = f"{sil_shape}(no-depth) c/a={sil_ratio:.2f}"
                line2 = (f"C=({C[0]*100:+.1f},{C[1]*100:+.1f},{C[2]*100:+.1f})cm  {box_str}")
                rms = "fitRMS=n/a(predict)" if fit_resid_mm is None else f"fitRMS={fit_resid_mm:.1f}mm/{n_used}pt"
                inr = "" if inlier_ratio is None else f"  inlier={inlier_ratio*100:.0f}%"
                jit = "" if jitter_mm is None else f"  jitter={jitter_mm:.1f}mm"
                cv2.putText(vis, line2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(vis, rms + inr + jit, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # AprilTag diagnostics. Show the LAST detection's verdict + metrics and
            # HOLD it across the frames skipped by --tag-every-n, so the line does
            # not switch every frame between a readout and "predict". It only turns
            # to "not detected" after the tag is actually missed for longer than one
            # detect interval.
            tag_seen_fresh = (tag_last_obs is not None
                              and (frame_i - tag_seen_frame) <= max(1, args.tag_every_n))
            if tag_seen_fresh:
                o = tag_last_obs
                tg = ("acc" if o.accepted else
                      ("posonly" if o.position_only else f"REJ:{o.reject_reason}"))
                cv2.putText(
                    vis,
                    f"tag{args.tag_id} {tg} dm={o.decision_margin:.0f} "
                    f"rep={o.reproj_error_px:.1f}px sz={o.tag_size_px:.0f}px "
                    f"ang={o.view_angle_deg:.0f}deg",
                    (8, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            elif tag_det is not None:
                # steady across frames once the tag is genuinely gone (no toggle).
                cv2.putText(vis, f"tag{args.tag_id}: not detected", (8, H - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            s_hint = "STOP+save" if recording else "REC"
            cv2.putText(vis, f"{fps:.0f}fps  q=quit r=reset click=seed  s={s_hint}",
                        (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if recording:
                cv2.putText(vis, f"REC * {len(rec_buffer)}f", (W - 170, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif now - last_log_t < 1.5:                     # brief confirmation flash
                cv2.putText(vis, f"SAVED session {session_idx - 1}", (W - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, vis)
            if recording:
                rec_buffer.append(make_record())     # stack every frame of the session
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("r"):
                ui["reset"] = True
            elif k == ord("s"):
                if not recording:
                    recording = True
                    rec_buffer = []
                    last_log_t = now
                    print(f"[live] REC start (session {session_idx}) -- press 's' again to stop+save")
                else:
                    recording = False
                    base, ext = os.path.splitext(args.log)
                    ext = ext or ".jsonl"
                    out_path = f"{base}.session{session_idx}{ext}"
                    with open(out_path, "w") as _lf:
                        for r in rec_buffer:
                            _lf.write(json.dumps(r) + "\n")
                    # --- trend / noise summary (informs how much noise to inject in policy training) ---
                    coms = np.array([r["com_cam_m"] for r in rec_buffer if r.get("com_cam_m")])
                    summary = {"session": session_idx, "frames": len(rec_buffer), "file": out_path}
                    if len(coms) >= 2:
                        summary["dur_s"] = round(rec_buffer[-1]["t"] - rec_buffer[0]["t"], 2)
                        summary["com_std_mm"] = (coms.std(axis=0) * 1000).round(3).tolist()
                        summary["com_range_mm"] = ((coms.max(0) - coms.min(0)) * 1000).round(3).tolist()
                        summary["com_total_std_mm"] = round(float(np.linalg.norm(coms.std(0)) * 1000), 3)
                        diam = np.array([0.5 * (r["a_cm"] + r["c_cm"]) for r in rec_buffer])
                        summary["size_cm_mean"] = round(float(diam.mean()), 2)
                        summary["size_cm_std"] = round(float(diam.std()), 3)
                        quats = [r["mekf_quat_wxyz"] for r in rec_buffer if r.get("mekf_quat_wxyz")]
                        if len(quats) >= 2:
                            qa = np.array(quats)
                            steps = [np.degrees(2 * np.arccos(abs(float(np.clip(np.dot(qa[i], qa[i - 1]), -1, 1)))))
                                     for i in range(1, len(qa))]
                            summary["ori_step_deg_mean"] = round(float(np.mean(steps)), 2)
                            summary["n_tag_frames"] = int(sum(1 for r in rec_buffer if r.get("tag_used")))
                    with open(f"{base}.sessions_summary{ext}", "a") as _sf:
                        _sf.write(json.dumps(summary) + "\n")
                    print(f"[live] REC stop -> {out_path}  ({len(rec_buffer)} frames)")
                    print(f"[live] TREND/NOISE: {json.dumps(summary)}")
                    last_log_t = now
                    session_idx += 1
    finally:
        cv2.destroyAllWindows()
        cam.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
