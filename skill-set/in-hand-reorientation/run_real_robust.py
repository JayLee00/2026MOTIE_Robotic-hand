#!/usr/bin/env python
"""Robust ROS2 launcher for the in-hand fruit bbox demo (live camera topics).

This used to drive a RealSense directly via ``pyrealsense2``. The live rig now
publishes its RGB-D stream on ROS2 under the ``/front_cam/front`` namespace, so
this script instead SUBSCRIBES to those topics and draws a bounding box on the
camera's RGB frames:

    /front_cam/front/color/image_raw                    (Image, rgb8)   [required]
    /front_cam/front/color/camera_info                  (CameraInfo)    [for K]
    /front_cam/front/aligned_depth_to_color/image_raw   (Image, 16UC1)  [optional]

Pipeline per frame:

    color (rgb8) -> RealSenseSegmenter.segment(center prompt)  (SAM2 / HSV)
      -> 2D bbox = boundingRect(mask)          (ALWAYS drawn, RGB-only path)
      -> [optional, if depth present]
         deproject_zdepth(depth_m, mask, K) -> fit sphere -> aabb_from_sphere
         -> project 3D box corners with K -> draw the projected 2D rect + depth.

NOTE ON DEPTH: the 3D path expects depth ALIGNED to the color frame
(``aligned_depth_to_color``). The wrapper only publishes it when
``align_depth.enable`` is true -- enable it at runtime with

    ros2 param set /front_cam/front align_depth.enable true

(or persist it in the camera launch file). If the aligned topic is absent the
demo still runs: the 2D RGB bbox never depends on depth. Pass ``--no-depth``
to skip the 3D path entirely, or ``--depth-topic <ns>/depth/image_rect_raw``
to (approximately) use the unaligned depth.

Writes annotated PNGs + an MP4 + per-frame bbox JSON + a summary.json under
``--out``. No display needed -- unless you pass ``--gui``, which opens a live
cv2 window instead:

    python run_real_robust.py --gui                 # live view, no files
    python run_real_robust.py --gui --frames 0      # (0 = run until quit)

GUI keys:  q / ESC = quit    r = reseed the prompt (auto color-blob)
           LEFT-CLICK on the fruit = seed the SAM2 prompt there
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from in_hand_tracker.io.realsense_source import (  # noqa: E402
    deproject_zdepth,
    project_points,
)
from in_hand_tracker.perception.realsense_segmenter import (  # noqa: E402
    RealSenseSegmenter,
    hsv_center_blob,
)
from in_hand_tracker.perception.bbox import aabb_from_sphere  # noqa: E402
from in_hand_tracker.perception.ellipsoid_fit import fit_ellipsoid_center  # noqa: E402
from in_hand_tracker.calibration.per_object_init import calibrate_object  # noqa: E402

_DEFAULT_ESTIMATOR_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "in_hand_tracker",
    "config",
    "estimator.yaml",
)
_DEFAULT_NAMESPACE = "/front_cam/front"


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def draw_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    rect2d,
    box3d,
    K: Optional[np.ndarray],
    idx: int,
    center_depth: float,
    radius: float,
    method: str,
) -> np.ndarray:
    """Tint the mask, draw the 2D mask rect (green) + optional 3D-box rect (cyan)."""
    import cv2

    img = bgr.copy()
    if mask is not None and mask.any():
        overlay = img.copy()
        overlay[mask] = (0, 0, 255)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # RGB-only 2D bbox straight from the mask (the primary, always-on path).
    if rect2d is not None:
        x, y, w, h = rect2d
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Optional projected 3D box (cyan), when depth + K produced a fit.
    if box3d is not None and K is not None:
        pix = project_points(box3d.corners(), K)      # (8, 2), nan if behind cam
        good = np.isfinite(pix).all(axis=1)
        if good.sum() >= 2:
            pg = pix[good]
            u0, v0 = pg[:, 0].min(), pg[:, 1].min()
            u1, v1 = pg[:, 0].max(), pg[:, 1].max()
            cv2.rectangle(
                img,
                (int(round(u0)), int(round(v0))),
                (int(round(u1)), int(round(v1))),
                (255, 255, 0),
                2,
            )

    lines = [
        f"frame {idx}  [{method}]",
        (f"depth {center_depth:.3f} m" if np.isfinite(center_depth) else "depth n/a"),
        (f"radius {radius * 100:.1f} cm" if np.isfinite(radius) else "radius n/a"),
    ]
    y = 22
    for txt in lines:
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return img


def _rect_from_mask(mask: np.ndarray):
    """2D (x, y, w, h) bounding rect of the True region, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _inner_point(mask: np.ndarray):
    """The most interior (u, v) of a bool mask (distance-transform argmax).

    More robust than the centroid, which can fall OFF the object for concave
    masks (e.g. fruit + fingers merged into one blob).
    """
    import cv2

    if mask is None or not mask.any():
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (int(u), int(v))


def _seed_prompt(bgr: np.ndarray, segmenter: RealSenseSegmenter):
    """Initial SAM2 prompt point: interior of the dominant saturated color blob.

    The image center is a bad prompt when the fruit is off-center (it hits the
    gripper), so seed from the HSV blob nearest the center instead. Falls back
    to the image center when no blob is found.
    """
    h, w = bgr.shape[:2]
    blob = hsv_center_blob(
        bgr, sat_min=segmenter.config.sat_min, val_min=segmenter.config.val_min
    )
    return _inner_point(blob) or (w // 2, h // 2)


def _fit_center_camframe(points_cam: np.ndarray, radius: float):
    """Axis-fixed SPHERE fit in the camera frame (R_wo=None -> object==camera)."""
    return fit_ellipsoid_center(
        points_cam, a=radius, c=radius, shape_class="spheroid", R_wo=None,
    )


# --------------------------------------------------------------------------- #
# ROS2 subscriber node
# --------------------------------------------------------------------------- #
def _imgmsg_to_np(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image without cv_bridge (numpy-ABI safe).

    cv_bridge on ROS humble is compiled against numpy 1.x and hard-crashes
    under numpy 2.x (torch/conda envs), so we decode the few encodings this
    camera actually publishes by hand.
    """
    enc = msg.encoding.lower()
    dtype, channels = {
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "16uc1": (np.uint16, 1),
        "mono16": (np.uint16, 1),
        "32fc1": (np.float32, 1),
    }.get(enc, (None, None))
    if dtype is None:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    buf = np.frombuffer(msg.data, dtype=dt)
    itemsize = dt.itemsize
    row_elems = msg.step // itemsize
    img = buf.reshape(msg.height, row_elems)[:, : msg.width * channels]
    if channels > 1:
        img = img.reshape(msg.height, msg.width, channels)
    return img


def _build_node(color_topic, info_topic, depth_topic, use_depth):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo

    if not rclpy.ok():
        rclpy.init()

    class BboxCameraNode(Node):
        """Caches the latest color frame (+ optional depth) and camera K."""

        def __init__(self):
            super().__init__("in_hand_bbox_demo")
            self.K: Optional[np.ndarray] = None
            self._latest_bgr = None
            self._latest_depth_m = None
            self._latest_stamp = None
            self._color_frame_id = ""          # optical frame (for published Markers)
            self._n_color = 0

            self.create_subscription(
                CameraInfo, info_topic, self._on_info, qos_profile_sensor_data
            )
            self.create_subscription(
                Image, color_topic, self._on_color, qos_profile_sensor_data
            )
            if use_depth:
                self.create_subscription(
                    Image, depth_topic, self._on_depth, qos_profile_sensor_data
                )

        def _on_info(self, msg: CameraInfo):
            if self.K is None:
                self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        def _on_color(self, msg: Image):
            # Normalize the color encoding to BGR8 for cv2 / the segmenter.
            img = _imgmsg_to_np(msg)
            if msg.encoding.lower() == "rgb8":
                img = img[:, :, ::-1]
            self._latest_bgr = np.ascontiguousarray(img)
            self._latest_stamp = msg.header.stamp
            if msg.header.frame_id:
                self._color_frame_id = msg.header.frame_id
            self._n_color += 1

        def _on_depth(self, msg: Image):
            depth = _imgmsg_to_np(msg)
            if depth.dtype == np.uint16:            # 16UC1 -> meters (mm assumed)
                depth_m = depth.astype(np.float32) * np.float32(1e-3)
            else:                                   # 32FC1 already in meters
                depth_m = depth.astype(np.float32)
            self._latest_depth_m = depth_m

        # ----- accessors for the capture loop ----- #
        def has_color(self) -> bool:
            return self._latest_bgr is not None

        def grab(self):
            """Return (bgr, depth_m_or_None, K_or_None) snapshot of latest frame."""
            return (
                None if self._latest_bgr is None else self._latest_bgr.copy(),
                None if self._latest_depth_m is None else self._latest_depth_m.copy(),
                None if self.K is None else self.K.copy(),
            )

        def color_count(self) -> int:
            return self._n_color

    return rclpy, BboxCameraNode()


def _spin_until(rclpy, node, predicate, timeout_s: float) -> bool:
    """Spin the node until ``predicate()`` is true or timeout; returns success.

    Ctrl+C / SIGTERM makes rclpy's signal handler invalidate the context, after
    which ``spin_once`` raises -- treat that as "no more frames coming" instead
    of dumping a traceback, so callers wind down through their cleanup path.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not rclpy.ok():
            return bool(predicate())
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            if not rclpy.ok():          # shutdown raced the spin
                return bool(predicate())
            raise
        if predicate():
            return True
    return bool(predicate())


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def run_ros2_bbox_demo(
    frames: int,
    out: str,
    color_topic: str,
    info_topic: str,
    depth_topic: str,
    use_depth: bool,
    radius: Optional[float],
    estimator_yaml: str,
    fps: int,
    wait_s: float,
    prompt_uv: Optional[tuple] = None,
    gui: bool = False,
) -> dict:
    import cv2

    save = out is not None
    if save:
        os.makedirs(out, exist_ok=True)
    segmenter = RealSenseSegmenter.from_yaml(estimator_yaml)

    win = "in-hand fruit bbox (q/ESC quit, r reseed, click = prompt)"
    click: dict = {}
    if gui:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click["uv"] = (int(x), int(y))

        cv2.setMouseCallback(win, _on_mouse)

    rclpy, node = _build_node(color_topic, info_topic, depth_topic, use_depth)
    try:
        print(f"[robust] waiting for first color frame on {color_topic} ...")
        if not _spin_until(rclpy, node, node.has_color, wait_s):
            raise RuntimeError(
                f"no image received on {color_topic} within {wait_s:.0f}s "
                "(is the camera driver running? check `ros2 topic hz`)."
            )
        # Give camera_info a brief chance to arrive (falls back to 2D-only if not).
        _spin_until(rclpy, node, lambda: node.K is not None, 2.0)
        if node.K is None:
            print(f"[robust] WARN: no CameraInfo on {info_topic}; "
                  "3D depth path disabled (2D RGB bbox still drawn).")
        if use_depth:
            _spin_until(rclpy, node, lambda: node._latest_depth_m is not None, 2.0)
            if node._latest_depth_m is None:
                print(f"[robust] WARN: no depth on {depth_topic}; 3D path will be "
                      "skipped. If this is the aligned topic, enable it with:\n"
                      "         ros2 param set /front_cam/front align_depth.enable true")

        fitted_radius = float(radius) if radius is not None else None
        seg_methods, coverage_flags, radii, fps_times = [], [], [], []
        writer, frame_size = None, None
        prompt_pt = tuple(prompt_uv) if prompt_uv is not None else None

        frame_iter = range(int(frames)) if int(frames) > 0 else itertools.count()
        for i in frame_iter:
            # Block for a genuinely NEW color frame (not a re-processed cache).
            target = node.color_count() + 1
            if not _spin_until(rclpy, node, lambda: node.color_count() >= target, 5.0):
                print("[robust] shutting down." if not rclpy.ok() else
                      f"[robust] frame {i}: timed out waiting for a new image; "
                      "stopping.")
                break

            t0 = time.time()
            bgr, depth_m, K = node.grab()
            if bgr is None:
                break
            if frame_size is None:
                frame_size = (bgr.shape[1], bgr.shape[0])

            # Prompt policy: click > explicit --prompt > tracked interior > HSV seed.
            if gui and "uv" in click:
                prompt_pt = click.pop("uv")
                print(f"[robust] frame {i}: prompt reseeded by click at {prompt_pt}")
            if prompt_pt is None:
                prompt_pt = _seed_prompt(bgr, segmenter)
                print(f"[robust] frame {i}: seeding prompt at {prompt_pt}")
            seg = segmenter.segment_at(bgr, [prompt_pt])
            seg_methods.append(seg.method)
            mask = seg.mask
            n_mask = int(mask.sum())
            rect2d = _rect_from_mask(mask) if n_mask > 0 else None
            # Track: next frame's prompt follows the object; reseed if lost.
            prompt_pt = _inner_point(mask) if n_mask > 0 else None

            box3d = None
            center_depth = float("nan")
            frame_radius = float("nan")
            fit_ok = False
            n_points = 0

            if use_depth and depth_m is not None and K is not None and n_mask > 0:
                if depth_m.shape == mask.shape:
                    pts_cam = deproject_zdepth(depth_m, mask, K)
                    n_points = int(pts_cam.shape[0])
                    if n_points >= 5:
                        if fitted_radius is None:
                            calib = calibrate_object(pts_cam)
                            if calib.ok and calib.a is not None:
                                fitted_radius = float(max(calib.a, calib.c))
                                print(f"[robust] free-fit radius from frame {i}: "
                                      f"{fitted_radius * 100:.1f} cm")
                        if fitted_radius is not None:
                            fit = _fit_center_camframe(pts_cam, fitted_radius)
                            if fit.ok and fit.center is not None:
                                fit_ok = True
                                frame_radius = fitted_radius
                                box3d = aabb_from_sphere(fit.center, fitted_radius)
                                center_depth = float(fit.center[2])

            coverage_flags.append(bool(rect2d is not None))
            if np.isfinite(frame_radius):
                radii.append(frame_radius)

            img = draw_overlay(
                bgr, mask, rect2d, box3d, K, i, center_depth, frame_radius, seg.method
            )

            if save:
                cv2.imwrite(os.path.join(out, f"frame_{i:04d}.png"), img)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        os.path.join(out, "real_demo.mp4"), fourcc, float(fps),
                        frame_size,
                    )
                writer.write(img)
                rec = {
                    "idx": i,
                    "method": seg.method,
                    "sam2_score": seg.score,
                    "prompt_uv": (list(seg.prompt_points[0])
                                  if seg.prompt_points else None),
                    "n_mask_px": n_mask,
                    "bbox_2d_xywh": list(rect2d) if rect2d is not None else None,
                    "n_points": n_points,
                    "fit_ok": fit_ok,
                    "center": None if box3d is None else box3d.center.tolist(),
                    "center_depth_m": (None if not np.isfinite(center_depth)
                                       else center_depth),
                    "radius_m": (None if not np.isfinite(frame_radius)
                                 else float(frame_radius)),
                    "bbox_3d": None if box3d is None else box3d.to_dict(),
                }
                with open(os.path.join(out, f"bbox_{i:04d}.json"), "w") as f:
                    json.dump(rec, f, indent=2)

            fps_times.append(time.time() - t0)
            if not gui or i % 30 == 0:
                print(f"[robust] frame {i}: method={seg.method} mask_px={n_mask} "
                      f"bbox2d={rect2d} fit3d={fit_ok}")

            if gui:
                cv2.imshow(win, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):                     # q / ESC
                    print(f"[robust] quit requested at frame {i}.")
                    break
                if key == ord("r"):                           # reseed
                    prompt_pt = None
                    print(f"[robust] frame {i}: prompt reset (auto reseed).")
    finally:
        if writer is not None:
            writer.release()
        if gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    n_bbox = int(sum(coverage_flags))
    n_done = len(coverage_flags)
    dt = np.asarray(fps_times, dtype=np.float64)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    summary = {
        "frames_requested": int(frames),
        "frames_processed": n_done,
        "n_bbox_2d": n_bbox,
        "bbox_2d_coverage_pct": 100.0 * n_bbox / max(1, n_done),
        "median_radius_m": (float(np.median(radii)) if radii else None),
        "median_fps": float(1.0 / np.median(dt)) if dt.size else None,
        "segmenter_methods": {m: seg_methods.count(m) for m in sorted(set(seg_methods))},
        "topics": {"color": color_topic, "camera_info": info_topic,
                   "depth": depth_topic if use_depth else None},
        "depth_used": bool(use_depth),
        "out_dir": os.path.abspath(out) if save else None,
        "video": (os.path.abspath(os.path.join(out, "real_demo.mp4"))
                  if save else None),
    }
    if save:
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=None,
                   help="frames to process; 0 = until quit "
                        "(default: 30 headless, 0 with --gui)")
    p.add_argument("--out", default=None,
                   help="output dir for PNG/MP4/JSON "
                        "(default: real_out headless, no files with --gui)")
    p.add_argument("--gui", action="store_true",
                   help="show a live cv2 window (q/ESC quit, r reseed, "
                        "left-click = SAM2 prompt)")
    p.add_argument("--namespace", default=_DEFAULT_NAMESPACE,
                   help="camera namespace (default: %(default)s)")
    p.add_argument("--color-topic", default=None,
                   help="override; default <ns>/color/image_raw")
    p.add_argument("--info-topic", default=None,
                   help="override; default <ns>/color/camera_info")
    p.add_argument("--depth-topic", default=None,
                   help="override; default <ns>/aligned_depth_to_color/image_raw")
    p.add_argument("--no-depth", action="store_true",
                   help="skip the (approximate) 3D depth path; RGB 2D bbox only")
    p.add_argument("--radius", type=float, default=None,
                   help="fixed sphere radius (m) for the 3D path; free-fit if omitted")
    p.add_argument("--prompt", default=None, metavar="U,V",
                   help="initial SAM2 prompt pixel on the fruit (e.g. 290,140); "
                        "auto-seeded from the dominant color blob if omitted")
    p.add_argument("--estimator-yaml", default=_DEFAULT_ESTIMATOR_YAML)
    p.add_argument("--fps", type=int, default=15, help="output MP4 fps")
    p.add_argument("--wait", type=float, default=15.0,
                   help="seconds to wait for the first frame")
    args = p.parse_args(argv)

    prompt_uv = None
    if args.prompt:
        u, v = (int(x) for x in args.prompt.split(","))
        prompt_uv = (u, v)

    frames = args.frames if args.frames is not None else (0 if args.gui else 30)
    out = args.out if args.out is not None else (None if args.gui else "real_out")
    if frames <= 0 and not args.gui:
        p.error("--frames 0 (run until quit) only makes sense with --gui")

    ns = args.namespace.rstrip("/")
    color_topic = args.color_topic or f"{ns}/color/image_raw"
    info_topic = args.info_topic or f"{ns}/color/camera_info"
    depth_topic = args.depth_topic or f"{ns}/aligned_depth_to_color/image_raw"

    summary = run_ros2_bbox_demo(
        frames=frames,
        out=out,
        color_topic=color_topic,
        info_topic=info_topic,
        depth_topic=depth_topic,
        use_depth=not args.no_depth,
        radius=args.radius,
        estimator_yaml=args.estimator_yaml,
        fps=args.fps,
        wait_s=args.wait,
        prompt_uv=prompt_uv,
        gui=args.gui,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["frames_processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
