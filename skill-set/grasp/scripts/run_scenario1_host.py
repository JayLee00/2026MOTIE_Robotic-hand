#!/usr/bin/env python3
"""
시나리오1 (Pick + 시퀀스 제어권 handoff) — **호스트 / py3.10 버전** (Docker 미사용).

원본 run_scenario1.py 는 grasp_fruit(py3.12) 한 프로세스에서 SAM3 + 카메라(rclpy) +
SequenceClient(rclpy) 를 모두 돌리는데, 이 머신은 ROS Humble rclpy 가 py3.10 이라
py3.12 에서 rclpy import 가 안 된다. 그래서 역할을 분리한다:

  [이 프로세스: /usr/bin/python3 = 3.10]  카메라 캡처 + SequenceClient + 로봇 실행 (전부 rclpy)
  [subprocess: grasp_fruit = 3.12]         SAM3 검출 + Top-down 파지계산

흐름 (원본 run_scenario1.py 와 동일):
  SequenceClient(SEQ_PICK) Start(제어권+하트비트)
    → 카메라 캡처(ROS) → NPZ
    → SAM3(subprocess) → mask
    → 파지계산(subprocess) → summary JSON
    → 로봇 pick (robot_executor_scenario1.py, 물체 쥔 채 유지)
    → 1회 성공 → with 정상 탈출 = End(DONE) → Inhand(2) 이어받음
  'exit'/'quit'/'q' = Pick 취소(abort → IDLE, DONE 아님)

전제 — 이 스크립트 실행 전, 아래가 켜져 있어야 함:
  1. move_group :  ros2 launch scripts/launch_moveit.py   (fr_ws + dex_soldering kistar_ws 소싱)
  2. relay      :  /usr/bin/python3 scripts/franka_joint_state_relay.py
  3. 제어 PC    :  shm + control_pc.launch (require_control:=true) + 카메라 + Franka unlock

실행 (소싱: /opt/ros/humble + Dual_Arm_Hand_Ctrl/ros2/install):
  /usr/bin/python3 scripts/run_scenario1_host.py \
      --calibration configs/calibration/extrinsic_20260612_170053.json --execute_robot
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

from sequence_client import SequenceClient, SequenceError
from dual_arm_msgs.msg import SequenceState

SCRIPTS = Path(__file__).resolve().parent
ROOT    = SCRIPTS.parent

# ── 환경 defaults (플랜 B 호스트 MoveIt) ─────────────────────────────────────────
GRASP_FRUIT_PY = "/home/user/miniconda3/envs/grasp_fruit/bin/python"   # SAM3/파지계산 (3.12)
HOST_PY        = "/usr/bin/python3"                                   # 로봇 실행 (3.10)
ROS_SETUP      = "/opt/ros/humble/setup.bash"
FRANKA_WS      = "/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash"
KISTAR_WS      = "/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash"
DOMAIN_ID      = 9

DEFAULT_COLOR = "/front_cam/front/color/image_raw"
DEFAULT_DEPTH = "/front_cam/front/aligned_depth_to_color/image_raw"
DEFAULT_INFO  = "/front_cam/front/color/camera_info"
_SPIN_DT = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 카메라 노드 (rclpy, py3.10) — ros_camera_grab.py 와 동일 디코딩
# ─────────────────────────────────────────────────────────────────────────────
class CamNode(Node):
    def __init__(self, color_topic, depth_topic, info_topic):
        super().__init__("scenario1_camera")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color = None; self.depth = None; self.K = None
        self.create_subscription(Image, color_topic, self._color_cb, qos)
        self.create_subscription(Image, depth_topic, self._depth_cb, qos)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, qos)

    def _color_cb(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        a = np.ascontiguousarray(a[:, : m.width * 3]).reshape(m.height, m.width, 3)
        if m.encoding == "rgb8":
            a = a[..., ::-1]
        self.color = np.ascontiguousarray(a)

    def _depth_cb(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        a = np.ascontiguousarray(a[:, : m.width * 2])
        self.depth = a.view(np.uint16).reshape(m.height, m.width)

    def _info_cb(self, m):
        self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)

    def ready(self):
        return self.color is not None and self.depth is not None and self.K is not None

    def capture(self, out_dir: Path, stem: str, idx: int, timeout=30.0,
                depth_scale=0.001) -> Path:
        """최신 프레임 1장 → NPZ (rgb BGR uint8, depth meters float32, K)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.color = None; self.depth = None
        t0 = time.time()
        while rclpy.ok() and not self.ready() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=_SPIN_DT)
        if not self.ready():
            raise RuntimeError(
                f"[Camera] {timeout:.0f}s 안에 프레임 수신 실패 "
                f"(color={self.color is not None}, depth={self.depth is not None}, K={self.K is not None})")
        out = out_dir / f"{stem}_{idx:03d}.npz"
        depth_m = self.depth.astype(np.float32) * depth_scale
        np.savez_compressed(out, rgb=self.color, depth=depth_m, K=self.K.astype(np.float32))
        print(f"[Camera] 캡처 저장: {out}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clean_env() -> dict:
    """grasp_fruit(3.12) 실행용 — ROS(3.10) PYTHONPATH/LD 오염 제거."""
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH",
              "AMENT_PREFIX_PATH", "AMENT_CURRENT_PREFIX", "CMAKE_PREFIX_PATH"):
        env.pop(k, None)
    return env


def stage_sam3(npz: Path, query: str, interim: Path, gui: bool = False,
               instruction: 'str | None' = None, show_result: bool = True,
               model="facebook/sam3", thr=0.5, mthr=0.5) -> 'Path | None':
    """[grasp_fruit] SAM3 텍스트 검출 → mask PNG.

    instruction : run_sam3_qwen_select.py --no_grasp
                → SAM3 후보 전부 검출 → Qwen2-VL 이 자연어 지시로 자동 선택
                  → 선택 결과 창(Enter 확인) → mask 저장. gui 보다 우선.
    gui=True  : run_sam3_gui_select.py --no_grasp
                → 검출된 모든 인스턴스를 OpenCV 창에 표시, 사용자가 마우스 클릭 선택.
                  (grasp/robot 은 이 스크립트가 이어서 담당하므로 --no_grasp 로 mask 까지만)
    gui=False : run_sam3_only_stage.py → 자동 score 1위 선택 (기존 headless).
    모든 경우 {stem}_mask.png 를 interim 에 저장 (동일 포맷).
    """
    interim.mkdir(parents=True, exist_ok=True)
    if instruction:
        cmd = [GRASP_FRUIT_PY, str(SCRIPTS / "run_sam3_qwen_select.py"),
               "--input", str(npz), "--query", query,
               "--instruction", instruction,
               "--sam3_model_id", model,
               "--sam3_threshold", str(thr), "--sam3_mask_threshold", str(mthr),
               "--output_dir", str(interim),
               "--no_grasp"]     # Qwen 선택 → mask 까지만 (grasp/robot 은 이 스크립트가 담당)
        if not show_result:
            cmd += ["--show_sec", "3"]   # 창은 띄우되 3초 후 자동 진행 — 무정지 실행
        print(f"\n[SAM3+Qwen] '{query}' 검출 → Qwen2-VL 자동 선택: {instruction!r}")
    elif gui:
        cmd = [GRASP_FRUIT_PY, str(SCRIPTS / "run_sam3_gui_select.py"),
               "--input", str(npz), "--query", query,
               "--sam3_model_id", model,
               "--sam3_threshold", str(thr), "--sam3_mask_threshold", str(mthr),
               "--output_dir", str(interim),
               "--no_grasp"]     # mask 선택까지만
        print(f"\n[SAM3-GUI] '{query}' 검출 → 창에서 마우스로 클릭 선택 (Enter 확정, q/ESC 취소)...")
    else:
        cmd = [GRASP_FRUIT_PY, str(SCRIPTS / "run_sam3_only_stage.py"),
               "--input", str(npz), "--query", query,
               "--sam3_model_id", model,
               "--sam3_threshold", str(thr), "--sam3_mask_threshold", str(mthr),
               "--output_dir", str(interim)]
        print(f"\n[SAM3] '{query}' 검출 중 (grasp_fruit, 자동 score 1위)...")
    rc = subprocess.run(cmd, env=_clean_env()).returncode
    mask = interim / f"{npz.stem}_mask.png"
    if rc != 0 or not mask.exists():
        print("  [WARN] SAM3 마스크 없음 (또는 GUI 선택 취소)")
        return None
    return mask


def stage_grasp(npz: Path, mask: Path, query: str, outputs: Path,
                calibration: 'str | None') -> 'Path | None':
    """[grasp_fruit] Top-down 파지계산 → summary JSON."""
    outputs.mkdir(parents=True, exist_ok=True)
    cmd = [GRASP_FRUIT_PY, str(SCRIPTS / "run_topdown_grasp.py"),
           "--input", str(npz), "--mask", str(mask),
           "--depth_scale", "1.0", "--output", str(outputs), "--query", query]
    if calibration:
        cmd += ["--calibration", calibration]
    print(f"\n[Grasp] 파지계산 중 (grasp_fruit)...")
    rc = subprocess.run(cmd, env=_clean_env()).returncode
    summary = outputs / f"{npz.stem}_topdown_summary.json"
    if rc != 0 or not summary.exists():
        print("  [WARN] 파지계산 실패")
        return None
    return summary


def stage_robot_pick(summary: Path, args) -> int:
    """[호스트 py3.10] robot_executor_scenario1.py 로 pick (물체 쥔 채 유지)."""
    executor = str(SCRIPTS / "robot_executor_scenario1.py")
    extra = (f"--mode grasp --execute_mode {args.execute_mode} "
             f"--speed_factor {args.speed_factor} --approach_offset {args.approach_offset}")
    if args.disable_collision:
        extra += " --disable_collision"
    auto_yes = "export GRASP_AUTO_YES=1 && " if args.yes else ""
    bash = (
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER && "
        f"source {ROS_SETUP} && source {FRANKA_WS} && source {KISTAR_WS} && "
        f"export ROS_DOMAIN_ID={DOMAIN_ID} && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
        "export ROS_LOCALHOST_ONLY=0 && "
        f"{auto_yes}"
        f"exec {HOST_PY} {executor} --summary_json {summary} {extra}"
    )
    print(f"\n[Robot] Scenario1 Pick (물체 쥔 채 유지) — 호스트 직접 실행")
    return subprocess.run(["bash", "-c", bash]).returncode


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", default=None,
                   help="파지계산용 캘리브 JSON (run_topdown_grasp 로 전달)")
    p.add_argument("--execute_robot", action="store_true",
                   help="로봇 pick 실행 (없으면 비전+파지계산까지만)")
    p.add_argument("--execute_mode", default="direct_franka_topic",
                   choices=["trajectory_forwarder", "direct_franka_topic"])
    p.add_argument("--speed_factor",    type=float, default=0.2)
    p.add_argument("--approach_offset", type=float, default=0.10)
    p.add_argument("--disable_collision", action="store_true")
    p.add_argument("--gui", action="store_true",
                   help="SAM3 검출 인스턴스를 GUI 창에 띄워 마우스 클릭 선택 (기본: 자동 score 1위)")
    p.add_argument("--instruction", default=None,
                   help="자연어 선택 지시 (예: 'pick the kiwi on the dark table'). "
                        "지정 시 Qwen2-VL 이 자동 선택 (run_sam3_qwen_select.py, --gui 보다 우선)")
    p.add_argument("--query", default=None,
                   help="SAM3 검출 물체명 (예: kiwi). 지정 시 Query> 프롬프트 없이 바로 1회 실행, "
                        "실패 시 재입력 루프 대신 즉시 종료(abort). --yes 와 함께 쓰면 무정지 실행")
    p.add_argument("--yes", action="store_true",
                   help="로봇 실행 중 모든 y/n 확인을 자동 승인 (멈춤 없이 끝까지). RViz 경로 확인 생략됨")
    p.add_argument("--raw_dir",     default=str(ROOT / "data" / "raw"))
    p.add_argument("--interim_dir", default=str(ROOT / "data" / "interim"))
    p.add_argument("--output_dir",  default=str(ROOT / "data" / "outputs"))
    p.add_argument("--color_topic", default=DEFAULT_COLOR)
    p.add_argument("--depth_topic", default=DEFAULT_DEPTH)
    p.add_argument("--info_topic",  default=DEFAULT_INFO)
    return p.parse_args()


def main():
    args    = parse_args()
    raw_dir = Path(args.raw_dir); interim = Path(args.interim_dir); outputs = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("  Scenario 1 — Pick (시퀀스 제어권, 호스트/py3.10)")
    print("  Pick 1회 성공 시 제어권 반납(DONE) → Inhand(2) 차례")
    print("  'exit'/'quit'/'q' = Pick 취소(abort)")
    print("=" * 60)

    rclpy.init()
    cam = CamNode(args.color_topic, args.depth_topic, args.info_topic)
    seq = SequenceClient(SequenceState.SEQ_PICK)   # Pick=1, 첫 시퀀스라 wait 없음
    capture_idx = 0

    try:
        print("\n[Scenario1] 제어권 획득 중 (Start)...")
        with seq:                                  # Start: request_control + 하트비트 자동
            while True:
                if args.query:                      # 비대화 모드: CLI 로 받은 물체명 1회 실행
                    query = args.query.strip()
                    print(f"\n  Query(CLI)> {query}")
                else:
                    try:
                        query = input("\n  Query> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        raise SequenceError("Pick 취소 (사용자 중단)")
                    if query.lower() in ("exit", "quit", "q"):
                        raise SequenceError("Pick 취소 (사용자 종료)")
                    if not query:
                        continue

                stem = f"scenario1_{capture_idx:03d}"
                capture_idx += 1

                # ── Stage 0: 카메라 캡처 ──
                try:
                    npz = cam.capture(raw_dir, stem, capture_idx - 1)
                except RuntimeError as e:
                    if args.query:
                        raise SequenceError(f"카메라 캡처 실패: {e}")
                    print(f"  [WARN] {e} — 다시 입력하세요."); continue

                # ── Stage 1: SAM3 (subprocess) ──
                mask = stage_sam3(npz, query, interim, gui=args.gui,
                                  instruction=args.instruction,
                                  show_result=not args.yes)
                if mask is None:
                    if args.query:
                        raise SequenceError("SAM3 마스크 없음 (검출 실패)")
                    print("  [WARN] 마스크 없음 — 다시 입력하세요."); continue

                # ── Stage 2: 파지계산 (subprocess) ──
                summary = stage_grasp(npz, mask, query, outputs, args.calibration)
                if summary is None:
                    if args.query:
                        raise SequenceError("파지계산 실패")
                    print("  [WARN] 파지계산 실패 — 다시 입력하세요."); continue

                # ── Stage 3: 로봇 pick (물체 쥔 채 유지) ──
                if args.execute_robot:
                    rc = stage_robot_pick(summary, args)
                    if rc != 0:
                        if args.query:
                            raise SequenceError(f"로봇 pick 실패 (rc={rc})")
                        print(f"  [WARN] 로봇 pick 실패 (rc={rc}) — 다시 입력하세요."); continue

                print(f"  ✓ Pick 성공: query={query!r}  →  {summary.name}")
                break   # 1회 성공 → with 정상 탈출 → End(DONE) → Inhand 이어받음

        print("\n[Scenario1] ✅ Pick DONE — 제어권 반납. Inhand(2) 차례.")

    except SequenceError as e:
        print(f"\n[Scenario1] ⚠️ Pick 중단: {e}  (abort → IDLE 회수, Inhand 진행 안 함)")
    finally:
        try: seq.shutdown()
        except Exception: pass
        cam.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
