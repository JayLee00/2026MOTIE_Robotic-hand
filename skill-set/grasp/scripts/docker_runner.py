#!/usr/bin/env python3
"""Docker container utilities shared by send_to_robot*.py scripts.

Provides:
  - to_container_path      : host path → container path
  - ensure_running         : start container if stopped
  - ros_exec_cmd           : build the ROS2 bash command string
  - run_in_container       : docker exec -i
  - start_recording        : RealSense RGB → MP4 (background thread)
  - stop_recording         : stop recording thread
  - ask_and_record         : ask user, start if 'y'
"""

import subprocess
import sys
import threading
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from utils.paths import DOCKER_CONTAINER, MOUNT_MAP as _MOUNT_MAP, ROS_DOMAIN_ID

_ROS_ENV_PREFIX = (
    "unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER && "
    "export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/humble/bin && "
    f"export ROS_DOMAIN_ID={ROS_DOMAIN_ID} && "
    "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
    "export ROS_LOCALHOST_ONLY=0"
)


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------

def to_container_path(host_path: str) -> str:
    """호스트 절대경로 → 컨테이너 내 경로 (마운트 테이블 기반)."""
    p = str(Path(host_path).resolve())
    for host_pfx, ctr_pfx in _MOUNT_MAP:
        if p.startswith(host_pfx):
            return ctr_pfx + p[len(host_pfx):]
    return p


def ensure_running(container: str) -> None:
    """컨테이너가 정지 상태이면 자동 시작. 없으면 sys.exit."""
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[ERROR] 컨테이너 '{container}' 를 찾을 수 없습니다.")
        sys.exit(1)
    if r.stdout.strip() == "true":
        print(f"[INFO]  컨테이너 '{container}' 실행 중.")
        return
    print(f"[INFO]  컨테이너 '{container}' 시작 중...")
    r2 = subprocess.run(["docker", "start", container])
    if r2.returncode != 0:
        print("[ERROR] 컨테이너 시작 실패.")
        sys.exit(1)
    print(f"[INFO]  컨테이너 '{container}' 시작 완료.")


def ros_exec_cmd(executor_ctr: str, summary_ctr: str,
                 kistar_ws_ctr: str, extra_args: str = '') -> str:
    """컨테이너 내에서 실행할 bash 명령문 반환.

    Args:
        executor_ctr  : 컨테이너 내 executor Python 스크립트 경로
        summary_ctr   : 컨테이너 내 summary JSON 경로
        kistar_ws_ctr : 컨테이너 내 kistar_ws 경로
        extra_args    : executor 에 넘길 추가 인자 문자열
    """
    ws_setup = f"{kistar_ws_ctr}/install/setup.bash"
    return (
        f"{_ROS_ENV_PREFIX} && "
        f"source /opt/ros/humble/setup.bash && "
        f"source {ws_setup} && "
        f"python3 {executor_ctr} "
        f"--summary_json {summary_ctr} "
        f"{extra_args}"
    ).strip()


def run_in_container(container: str, bash_cmd: str) -> int:
    """docker exec -i <container> bash -c <bash_cmd>. 종료 코드 반환."""
    return subprocess.run(
        ["docker", "exec", "-i", container, "bash", "-c", bash_cmd]
    ).returncode


# ---------------------------------------------------------------------------
# Recording helpers (host-side RealSense → MP4)
# ---------------------------------------------------------------------------

def _next_recording_path(rec_dir: Path) -> Path:
    idx = 1
    while True:
        p = rec_dir / f"recording_{idx:03d}.mp4"
        if not p.exists():
            return p
        idx += 1


def start_recording(rec_dir: Path) -> tuple:
    """RealSense RGB 녹화 시작 (백그라운드 스레드).

    Returns: (stop_event, thread, rec_path) — 실패 시 (None, None, None).
    """
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as e:
        print(f"[REC] 라이브러리 없음 — 녹화 불가: {e}")
        return None, None, None

    rec_path   = _next_recording_path(rec_dir)
    stop_event = threading.Event()

    def _record():
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(cfg)
        except Exception as e:
            print(f"[REC] RealSense 시작 실패: {e}")
            return
        writer = None
        try:
            while not stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=500)
                except RuntimeError:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                if writer is None:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(rec_path), fourcc, 30.0, (w, h))
                writer.write(img)
        finally:
            if writer:
                writer.release()
            pipeline.stop()
        print(f"[REC] 저장 완료: {rec_path}")

    t = threading.Thread(target=_record, daemon=True)
    t.start()
    print(f"[REC] 녹화 시작: {rec_path}")
    return stop_event, t, rec_path


def stop_recording(stop_event, thread) -> None:
    if stop_event is None:
        return
    print("[REC] 녹화 종료 중...")
    stop_event.set()
    if thread and thread.is_alive():
        thread.join(timeout=5.0)


def ask_and_record() -> tuple:
    """녹화 여부를 묻고 'y'이면 시작. (stop_event, thread, path) 반환."""
    try:
        resp = input("[REC] 동영상을 녹화하시겠습니까? (y/n): ").strip().lower()
    except EOFError:
        resp = 'n'
    if resp == 'y':
        rec_dir = ROOT / "data" / "fruit_vid"
        rec_dir.mkdir(parents=True, exist_ok=True)
        return start_recording(rec_dir)
    return None, None, None
