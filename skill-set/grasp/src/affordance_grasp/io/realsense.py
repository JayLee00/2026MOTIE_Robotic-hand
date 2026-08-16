from pathlib import Path
import os
import sys

import cv2
import numpy as np

from affordance_grasp.io.dataset_io import ensure_dir, save_json, save_rgbd_bundle


def _candidate_pyrealsense2_paths():
    candidates = []

    env_root = os.environ.get("AFF_GRASP_LIBREALSENSE_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend(
            [
                root / "build" / "Release",
                root / "build",
                root / "wrappers" / "python",
            ]
        )

    default_root = Path("/home/kist/librealsense")
    candidates.extend(
        [
            default_root / "build" / "Release",
            default_root / "build",
            default_root / "wrappers" / "python",
        ]
    )

    ordered = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _import_pyrealsense2():
    try:
        import pyrealsense2 as rs

        return rs
    except ImportError as first_exc:
        last_exc = first_exc
        for candidate in _candidate_pyrealsense2_paths():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                import pyrealsense2 as rs

                return rs
            except ImportError as exc:
                last_exc = exc

        searched = [str(path) for path in _candidate_pyrealsense2_paths()]
        raise ImportError(
            "Could not import pyrealsense2. Install it in the active environment, or set "
            "AFF_GRASP_LIBREALSENSE_ROOT to your local librealsense build root. "
            f"Searched paths: {searched}. Current Python: {sys.version.split()[0]}. "
            "If you built pyrealsense2 for a different Python version, rebuild it for this interpreter."
        ) from last_exc


rs = _import_pyrealsense2()


def make_depth_vis(depth_raw):
    valid = depth_raw[depth_raw > 0]
    if valid.size == 0:
        return cv2.applyColorMap(np.zeros_like(depth_raw, dtype=np.uint8),
                                 cv2.COLORMAP_TURBO)
    lo, hi = int(valid.min()), int(valid.max())
    if hi == lo:
        hi = lo + 1
    norm = np.clip(depth_raw.astype(np.float32), lo, hi)
    norm = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
    norm[depth_raw == 0] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def make_capture_path(output_dir, stem, capture_idx, suffix=".npz"):
    output_dir = ensure_dir(output_dir)
    name = f"{stem}_{capture_idx:03d}{suffix}"
    return output_dir / name


def save_capture_bundle(
    output_path,
    rgb_bgr,
    depth_raw,
    depth_m,
    K,
    save_rgb_png=True,
    save_depth_png=True,
):
    output_path = Path(output_path)
    save_rgbd_bundle(output_path, depth=depth_m, K=K, rgb=rgb_bgr)

    metadata = {
        "rgb_shape": list(rgb_bgr.shape),
        "depth_shape": list(depth_m.shape),
        "depth_dtype": str(depth_m.dtype),
        "rgb_dtype": str(rgb_bgr.dtype),
        "K": K.tolist(),
    }
    save_json(output_path.with_suffix(".json"), metadata)

    if save_rgb_png:
        cv2.imwrite(str(output_path.with_name(output_path.stem + "_rgb.png")), rgb_bgr)
    if save_depth_png:
        cv2.imwrite(
            str(output_path.with_name(output_path.stem + "_depth_vis.png")),
            make_depth_vis(depth_raw),
        )


def capture_realsense_interactive(
    output_dir,
    stem="realsense_frame",
    suffix=".npz",
    width=640,
    height=480,
    fps=30,
    warmup_frames=30,
    save_rgb_png=True,
    save_depth_png=True,
    window_name="RealSense RGBD Capture",
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[INFO] RealSense started")
        print("[INFO] Press 's' to save the current frame, 'q' to quit")

        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            align.process(frames)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        capture_idx = 0

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array(
                [
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            preview = np.hstack((rgb_bgr, make_depth_vis(depth_raw)))
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
                save_capture_bundle(
                    output_path,
                    rgb_bgr,
                    depth_raw,
                    depth_m,
                    K,
                    save_rgb_png=save_rgb_png,
                    save_depth_png=save_depth_png,
                )
                print(f"[INFO] Saved capture: {output_path}")
                capture_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense stopped")


def capture_realsense_once(
    output_dir,
    stem="realsense_frame",
    suffix=".npz",
    width=640,
    height=480,
    fps=30,
    warmup_frames=30,
    save_rgb_png=True,
    save_depth_png=True,
    capture_idx=0,
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[INFO] RealSense started")
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            align.process(frames)

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned color/depth frame from RealSense")

        rgb_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_m = depth_raw.astype(np.float32) * depth_scale
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
        save_capture_bundle(
            output_path,
            rgb_bgr,
            depth_raw,
            depth_m,
            K,
            save_rgb_png=save_rgb_png,
            save_depth_png=save_depth_png,
        )
        print(f"[INFO] Saved capture: {output_path}")
        return output_path
    finally:
        pipeline.stop()
        print("[INFO] RealSense stopped")


class RealSenseSession:
    """파이프라인을 세션 내내 열어두고 grab만 반복하는 컨텍스트 매니저.

    AE/AWB 수렴을 최초 1회만 기다리므로 반복 캡처 시 이미지가 일관됨.

    Usage:
        with RealSenseSession(warmup_frames=60) as cam:
            path = cam.capture(output_dir, stem, capture_idx)
    """

    def __init__(self, width=640, height=480, fps=30, warmup_frames=60,
                 video_path=None):
        self.width         = width
        self.height        = height
        self.fps           = fps
        self.warmup_frames = warmup_frames
        self._video_path   = Path(video_path) if video_path else None
        self._pipeline     = None
        self._align        = None
        self._depth_scale  = None
        self._K            = None
        self._writer       = None

    def __enter__(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)

        profile         = self._pipeline.start(config)
        self._align     = rs.align(rs.stream.color)
        depth_sensor    = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        print(f"[RealSense] 파이프라인 시작 — warmup {self.warmup_frames}프레임 대기 중...")
        for _ in range(self.warmup_frames):
            self._pipeline.wait_for_frames()
        print("[RealSense] warmup 완료.")

        # K는 첫 프레임에서 읽음
        frames      = self._pipeline.wait_for_frames()
        frames      = self._align.process(frames)
        color_frame = frames.get_color_frame()
        intr        = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self._K = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        if self._video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._video_path), fourcc, float(self.fps),
                (self.width, self.height))
            if self._writer.isOpened():
                print(f"[RealSense] 비디오 녹화 시작: {self._video_path}")
            else:
                print(f"[RealSense] WARNING: VideoWriter 열기 실패 → 녹화 건너뜀")
                self._writer = None

        return self

    def capture(self, output_dir, stem, capture_idx=0,
                suffix=".npz", save_rgb_png=True, save_depth_png=True):
        """현재 프레임을 저장하고 output_path를 반환."""
        frames      = self._pipeline.wait_for_frames()
        frames      = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense: 프레임 수신 실패")

        rgb_bgr   = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m   = depth_raw.astype(np.float32) * self._depth_scale

        if self._writer:
            self._writer.write(rgb_bgr)

        output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
        save_capture_bundle(output_path, rgb_bgr, depth_raw, depth_m, self._K,
                            save_rgb_png=save_rgb_png,
                            save_depth_png=save_depth_png)
        print(f"[RealSense] 캡처 저장: {output_path}")
        return output_path

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
            print(f"[RealSense] 비디오 저장 완료: {self._video_path}")
        if self._pipeline:
            self._pipeline.stop()
            print("[RealSense] 파이프라인 종료.")
