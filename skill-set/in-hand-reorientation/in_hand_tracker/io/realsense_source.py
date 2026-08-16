"""Live RealSense (D4xx) RGB-D source for the REAL bbox-tracking path.

PORTED (reimplemented), NOT imported, from the sim-stack RealSense collector's
pyrealsense2 capture/align/intrinsics pattern. NO sim-side (recorder / Isaac /
Omniverse) imports -- the AC-2 clean boundary holds here too.

CRITICAL geometry note (read before touching :func:`deproject_zdepth`):
    A RealSense depth value is the PLANAR Z-DEPTH (distance along the optical
    axis to the surface), NOT the euclidean ray length the Replicator sim
    records. So you MUST NOT reuse ``io.deproject.deproject`` (which scales a
    *unit* ray direction by the value, i.e. treats it as euclidean). For
    RealSense the back-projection is

        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        Z = depth_value                         (the planar depth, in meters)

    i.e. ``point_cam = Z * Kinv @ [u, v, 1]`` (NO normalization). Equivalently,
    ``rs.rs2_deproject_pixel_to_point(intrin, [u, v], Z)`` does the same and
    additionally undoes lens distortion. The D435 *color* stream is typically a
    near-pinhole (Brown-Conrady) model with ~zero coefficients, so the
    vectorized ``Z * Kinv`` path is numerically equivalent in practice; we use
    the per-pixel ``rs2_deproject_pixel_to_point`` when an rs intrinsics handle
    is available (correct + distortion) and fall back to the vectorized path
    otherwise. The path actually taken is recorded in the returned points'
    provenance via the ``prefer_rs`` argument / availability of ``rs``.

The camera frame here is OpenCV convention (+X right, +Y down, +Z forward),
which matches the projection used by :func:`project_points` and the demo
overlay. No GL->CV flip is applied (unlike the sim's USD/OpenGL frame).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "RealSenseSource",
    "deproject_zdepth",
    "project_points",
]

# Default live profile (matches the sim-stack RealSense collector default).
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FPS = 30


def _intrin_to_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Assemble a 3x3 pinhole intrinsics matrix K (float64)."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def deproject_zdepth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K_or_intrin: Union[np.ndarray, dict, object],
    *,
    prefer_rs: bool = True,
) -> np.ndarray:
    """Deproject masked pixels to the CAMERA frame using PLANAR Z-DEPTH.

    This is the RealSense counterpart of :func:`io.deproject.deproject`. The
    sim function assumes euclidean ray length; this one assumes planar depth
    (``point = Z * Kinv @ [u, v, 1]``) and is the ONLY correct choice for a
    RealSense depth map.

    Args:
        depth_m: (H, W) planar Z-depth per pixel, in METERS. Zero / non-finite
            pixels are treated as "no data" and dropped (RealSense uses 0 for
            invalid / clipped).
        mask: (H, W) bool; only True pixels with valid depth are deprojected.
        K_or_intrin: either a 3x3 intrinsics matrix, a dict with ``fx, fy, cx,
            cy``, or a ``pyrealsense2`` video-stream ``rs.intrinsics`` handle.
            When an ``rs.intrinsics`` handle is given and ``prefer_rs`` is set,
            ``rs.rs2_deproject_pixel_to_point`` is used per pixel (handles lens
            distortion); otherwise the vectorized ``Z * Kinv`` path is used.
        prefer_rs: prefer the pyrealsense2 per-pixel deprojection when an
            ``rs.intrinsics`` handle is available (default True).

    Returns:
        (N, 3) camera-frame points (OpenCV convention, +Z forward), where N is
        the number of valid True mask pixels.
    """
    depth_m = np.asarray(depth_m)
    mask = np.asarray(mask, dtype=bool)
    if depth_m.shape != mask.shape:
        raise ValueError(
            f"depth_m {depth_m.shape} and mask {mask.shape} must match"
        )

    z_all = depth_m.astype(np.float64)
    valid = mask & np.isfinite(z_all) & (z_all > 0.0)
    vs, us = np.nonzero(valid)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    zs = z_all[vs, us]

    # Detect a pyrealsense2 intrinsics handle (duck-typed: has fx/fy/ppx/ppy).
    is_rs_intrin = (
        hasattr(K_or_intrin, "fx")
        and hasattr(K_or_intrin, "ppx")
        and hasattr(K_or_intrin, "ppy")
    )

    if is_rs_intrin and prefer_rs:
        try:
            import pyrealsense2 as rs

            pts = np.empty((vs.size, 3), dtype=np.float64)
            for i in range(vs.size):
                # rs expects pixel as [u (col), v (row)] and Z in meters.
                p = rs.rs2_deproject_pixel_to_point(
                    K_or_intrin, [float(us[i]), float(vs[i])], float(zs[i])
                )
                pts[i, 0] = p[0]
                pts[i, 1] = p[1]
                pts[i, 2] = p[2]
            return pts
        except Exception:
            # Fall through to the vectorized pinhole path below.
            pass

    # Vectorized Z * Kinv path (near-pinhole; distortion assumed negligible).
    if is_rs_intrin:
        fx, fy = float(K_or_intrin.fx), float(K_or_intrin.fy)
        cx, cy = float(K_or_intrin.ppx), float(K_or_intrin.ppy)
    elif isinstance(K_or_intrin, dict):
        fx, fy = float(K_or_intrin["fx"]), float(K_or_intrin["fy"])
        cx, cy = float(K_or_intrin["cx"]), float(K_or_intrin["cy"])
    else:
        K = np.asarray(K_or_intrin, dtype=np.float64).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

    x = (us.astype(np.float64) - cx) / fx * zs
    y = (vs.astype(np.float64) - cy) / fy * zs
    return np.stack([x, y, zs], axis=1)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project CAMERA-frame points (N, 3) to pixels (N, 2) with K (pinhole).

    ``u = fx * X / Z + cx``, ``v = fy * Y / Z + cy``. Points with non-positive
    Z (behind the camera) map to ``nan`` so callers can drop them. This is the
    inverse of :func:`deproject_zdepth` for the no-distortion pinhole model and
    round-trips it exactly.
    """
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pts[:, 0] / z + cx
        v = fy * pts[:, 1] / z + cy
    bad = ~(z > 1e-9)
    u = np.where(bad, np.nan, u)
    v = np.where(bad, np.nan, v)
    return np.stack([u, v], axis=1)


class RealSenseSource:
    """Live aligned RGB-D source backed by a pyrealsense2 pipeline.

    Starts a depth+color pipeline, aligns depth to the COLOR stream, reads the
    COLOR intrinsics, and yields aligned (BGR, depth-in-meters) frames. If the
    requested ``width x height @ fps`` color/depth profile is not supported, it
    falls back to the device's first compatible color+depth profile pair and
    records the substitution in :attr:`fallback_info`.

    Usage:
        src = RealSenseSource()
        src.start()
        intr = src.intrinsics()                 # dict fx,fy,cx,cy,width,height
        bgr, depth_m, rs_intrin = src.capture()
        src.stop()
    """

    def __init__(
        self,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = _DEFAULT_FPS,
        serial: Optional[str] = None,
        warmup_frames: int = 15,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.warmup_frames = int(warmup_frames)

        self._pipeline = None
        self._align = None
        self._depth_scale: Optional[float] = None
        self._rs_intrinsics = None          # rs.intrinsics handle (color stream)
        self._intrinsics: Optional[dict] = None
        self.fallback_info: Optional[dict] = None
        self._started = False

    # ------------------------------------------------------------------ #
    def start(self) -> dict:
        """Start the pipeline (with fallback) and return the intrinsics dict."""
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial is not None:
            config.enable_device(str(self.serial))

        def _enable_requested(cfg):
            cfg.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            cfg.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

        profile = None
        try:
            _enable_requested(config)
            profile = pipeline.start(config)
        except Exception as exc:  # requested profile unsupported -> fall back.
            requested = (self.width, self.height, self.fps)
            config = rs.config()
            if self.serial is not None:
                config.enable_device(str(self.serial))
            # Let librealsense pick supported color + depth defaults.
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)
            profile = pipeline.start(config)
            color_vsp = (
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            )
            self.width = int(color_vsp.width())
            self.height = int(color_vsp.height())
            self.fps = int(color_vsp.fps())
            self.fallback_info = {
                "reason": str(exc),
                "requested": {
                    "width": requested[0],
                    "height": requested[1],
                    "fps": requested[2],
                },
                "actual": {
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                },
            }

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

        color_stream = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
        )
        self._rs_intrinsics = color_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        intr = self._rs_intrinsics
        self._intrinsics = {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
            "depth_scale": self._depth_scale,
            "model": str(intr.model),
            "coeffs": [float(x) for x in intr.coeffs],
        }

        # Consume warmup frames so auto-exposure / depth settle.
        for _ in range(max(0, self.warmup_frames)):
            self._pipeline.wait_for_frames(timeout_ms=15000)

        self._started = True
        return self._intrinsics

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Stop the pipeline (idempotent)."""
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            finally:
                self._started = False

    def __enter__(self) -> "RealSenseSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    def intrinsics(self) -> dict:
        """Return the COLOR intrinsics dict (fx, fy, cx, cy, width, height, ...)."""
        if self._intrinsics is None:
            raise RuntimeError("RealSenseSource.start() must be called first")
        return dict(self._intrinsics)

    @property
    def K(self) -> np.ndarray:
        """The 3x3 pinhole intrinsics matrix from the COLOR stream."""
        i = self.intrinsics()
        return _intrin_to_K(i["fx"], i["fy"], i["cx"], i["cy"])

    @property
    def rs_intrinsics(self):
        """The raw pyrealsense2 ``rs.intrinsics`` handle (or None before start)."""
        return self._rs_intrinsics

    @property
    def depth_scale(self) -> float:
        if self._depth_scale is None:
            raise RuntimeError("RealSenseSource.start() must be called first")
        return self._depth_scale

    # ------------------------------------------------------------------ #
    def capture(
        self, timeout_ms: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        """Capture one aligned (color, depth) frame.

        Returns:
            bgr: (H, W, 3) uint8 BGR color image.
            depth_m: (H, W) float32 PLANAR Z-depth in METERS (depth_scale
                already applied). Invalid pixels are 0.0.
            rs_intrinsics: the COLOR ``rs.intrinsics`` handle (for
                :func:`deproject_zdepth`).
        """
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseSource.start() must be called first")

        frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete aligned frame")

        bgr = np.asanyarray(color_frame.get_data())            # (H, W, 3) uint8
        depth_raw = np.asanyarray(depth_frame.get_data())      # (H, W) uint16
        depth_m = depth_raw.astype(np.float32) * np.float32(self._depth_scale)
        return bgr, depth_m, self._rs_intrinsics
