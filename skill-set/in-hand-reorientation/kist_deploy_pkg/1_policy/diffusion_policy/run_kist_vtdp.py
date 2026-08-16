#!/usr/bin/env python3
"""KIST VTDP 실기 배포 (ROS2) — 시각+촉각 diffusion policy, KISTAR 오른손 16관절 in-hand 회전.

이 파일은 **입력·출력 백엔드일 뿐**이다. 전처리·추론 계약(링버퍼 깊이·hp EMA·정규화·역정규화)은
전부 `kist-vtdp-wrapper/run.py` 의 `Deployer` 가 갖고 있고 여기서 그대로 import 한다.
계약을 두 벌로 만들면 조용히 어긋난다(이 프로젝트가 반복해서 밟은 실패형 — `docs/LEARNINGS.md`).
정본 문서: `kist-vtdp-wrapper/docs/DEPLOY.md`, 계약 출력: `python3 tools/best_model.py`.

옆 파일 `run.py`(46-D 순수 state 정책)와는 **다른 모델**이다. 저쪽은 obs 가 46차원 한 벡터이고
이쪽은 {state 16, tactile 12, rgb 3×224×224} 세 모달리티 × 각기 다른 시간창이다.

═══════════════════════════════════════════════════════════════════════════════
구독 — **config 의 obs_spec 에서 자동으로 정해진다** (아래 SOURCES 표로 매핑)
    /hand/right/joint_states                     JointState        → 03_hand_j_pos   (16)
    /paxini/right/ft                             Float32MultiArray → 10_hand_paxini_ft(12)
    /front_cam/front/color/image_raw/compressed  CompressedImage   → 40_image (3,224,224)
    /teleop/hand_engage/right                    Bool (latched)    → 인게이지 게이트
발행
    /hand/right/q_target   Float32MultiArray  (16, 절대 관절 타겟 [count])
    /hand/right/cmd_mode   Int32(1) · /hand/right/cmd_servo  Bool(True)   (--servo_on)
    /kist_vtdp/debug       Float64MultiArray  (진단)
═══════════════════════════════════════════════════════════════════════════════

시간축 (config 값. 100Hz 프레임 격자가 기준이다)
    100Hz 샘플러      토픽의 마지막 값을 10ms 마다 스냅샷 → 링버퍼 (= 레코더 `_tick` 과 동일)
                      ⚠️ 이 시계가 hp EMA 의 시계다. 끊기면 촉각 입력이 학습과 달라진다.
    링버퍼 깊이       tactile 21 프레임(200ms) · state/rgb 7 프레임(60ms) — 차기 전 추론 ❌
    20Hz 정책 틱      액션 1스텝 = 50ms. exec_horizon 만큼 소비하면 다시 추론
    100Hz 발행        정책 타겟 사이를 선형보간 → 계단 없이 매끄럽게

안전 가드
    인게이지      --enable_topic 이 True 여야 발행. 발판 오른쪽=ON / 왼쪽=OFF
    시작 램프     **실제로 타겟이 나가는 첫 틱**의 손 자세 → 정책 타겟으로 --ramp_sec 동안.
                  타겟이 안 나가는 틱(예열·starve·워치독·디스인게이지)마다 기준점을 현재
                  자세로 재앵커한다 — 속도 가드는 |타겟−직전타겟| 만 보므로 기준이 얼면
                  첫 발행이 실측에서 멀리 떨어진 채 출발한다.
    명령/측정 분리 앵커·램프 시작점은 측정 관절각을 **명령 한계 안으로 clip 한 값**이다.
                  JOINT_LIMITS 는 명령 봉투이고(학습 액션 위반 0/101,660), 측정값은 35.4% 가
                  그 밖이다(j5 최대 8018 count). 안 나누면 첫 틱이 속도 상한을 우회한다.
    속도 한계     --max_rate_cps. 기본 20000 count/s = 학습 데이터 |Δ|/50ms 의 p99.9(963 count)
    관절 한계     dp_config.JOINT_LIMITS (glove_teleop HAND_LIMITS 출처) 로 클램프
    워치독        손 상태 --stale_sec 이상 낡음 → 새 타겟 중단 + **남은 플랜 폐기**(홀드)
                  RGB --rgb_stale_sec 이상 안 갱신 → 새 타겟 중단 (얼어붙은 그림으로 운전 금지)
                  촉각은 낡아도 **멈추지 않는다** — hp 가 상수를 0 으로 수렴시켜 학습 분포에
                  가까워진다. ⚠️ 단 "정확히 0"은 **에피소드 시작부터 죽어 있던** 경우이고,
                  중간에 죽으면 τ=1 s 로 감쇠하는 과도구간이 남는다(+0.5 s 에 0.12 ≈ 0.2σ).
    NaN/차원      토픽에 NaN·Inf 가 오거나 차원이 기대와 다르면 그 메시지를 **버린다**.
                  hp EMA 는 한 번 NaN 이 되면 복구 경로가 없고, 관절·속도 클램프는
                  NaN 을 **둘 다 통과시킨다**(abs(nan)>x 도 nan<lo 도 False).
    카메라 신원   camera_info 의 intrinsics 를 **학습 h5 attrs 와 대조**한다. 토픽 이름은
                  증거가 아니다 — 2026-08-11 에 /camera/camera → /front_cam/front 리네임이 있었다.
    중복 발행     시작 시 q_target 의 다른 퍼블리셔 감지 → **경고만** 한다(참조 run.py 와 동일).
                  강제는 절차의 몫 — 실행 전 `bash run.sh status` 로 glove_teleop 이 0 인지 볼 것.
    --dry_run     상태기계는 **똑같이** 굴리고 마지막 발행 호출만 건너뛴다. 그래야 ②(dry_run)
                  리허설이 ③의 첫 틱을 실제로 검증한다(발행 여부로 분기하면 램프·앵커 경로가
                  dry_run 에서 한 번도 안 돌아 미검증인 채로 실기에 나간다).

⚠️ **정지(발판 왼쪽)는 홀드지 릴리즈가 아니다.** 이 스크립트에 서보를 내리는 경로는 없다 —
디스인게이지·halt·종료 모두 손이 마지막 타겟을 계속 잡고 있다. 손을 풀려면 **다른 터미널에서**:
    ros2 topic pub -1 /hand/right/cmd_servo std_msgs/Bool "{data: false}"
③ 시작 전에 이 명령을 두 번째 터미널에 **미리 쳐 놓고** 시작할 것.

실행:
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0

  python3 run_kist_vtdp.py --self_test                       # ① 로봇·ROS 없이 계약 검증
  python3 run_kist_vtdp.py --dry_run                         # ② 토픽 점검 + 주기 확인 (무발행)
  python3 run_kist_vtdp.py                                   # ③ 실제 발행 (인게이지 필요)
  ros2 topic pub -1 /teleop/hand_engage/right std_msgs/Bool "{data: true}" --qos-durability transient_local
                                                             #    발판 노드가 없을 때 수동 인게이지

⚠️ ③ 전에 글러브 텔레옵(glove_teleop.py)을 반드시 끌 것 — q_target 퍼블리셔가 둘이면 손이 튄다.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

DEFAULT_REPO = os.environ.get("KIST_VTDP_REPO", "/home/js/Desktop/kist-vtdp-wrapper")
DEFAULT_RUN = "runs/loop/r6_x_tacdrop_vt"          # tools/best_model.py 의 1위 (J_demo 405.5)


# ══════════════════════════════════════════════════════════════════════════════
# 배포 계약 코어 — wrapper repo 의 run.py 를 그대로 적재한다.
#
# 왜 import 냐: `Deployer` 가 링버퍼 깊이·hp EMA 연속성·ckpt 정규화·역정규화를 이미
# 계약대로 구현해 두었다. 여기 복사하면 두 번째 전처리 경로가 생겨 조용히 갈라진다.
# 이름을 'kist_vtdp_deploy' 로 주는 이유: 이 폴더에도 run.py 가 있어 `import run` 이 충돌한다.
# ※ 부작용: sys.path 맨 앞이 wrapper repo 가 되므로 이 프로세스에서 `import train` 은
#   wrapper 쪽 train.py 다(우리가 원하는 것). dp_config 등 이 폴더 모듈은 그대로 잡힌다.
# ══════════════════════════════════════════════════════════════════════════════
def load_deploy_core(repo: str):
    root = Path(repo).expanduser().resolve()
    rp = root / "run.py"
    if not rp.exists():
        sys.exit(f"❌ wrapper repo 를 못 찾았다: {rp}\n"
                 f"   --repo 로 경로를 주거나 KIST_VTDP_REPO 를 설정할 것.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))              # vtdp 패키지
    spec = importlib.util.spec_from_file_location("kist_vtdp_deploy", rp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# RGB — 학습 `VTWindowDataset._decode` 와 **같은 연산**이어야 한다.
#   PIL open → RGB → crop(l,u,r,lo) → resize(W,H) BILINEAR → (3,H,W) uint8 → /255
# cv2 로 바꾸면 안 된다: cv2.INTER_LINEAR 은 축소 시 안티에일리어싱을 안 해서 PIL 과
# 값이 다르고, imdecode 는 BGR 이다. (--self_test 가 학습 경로와 픽셀 단위로 대조한다.)
# ImageNet 정규화는 인코더 안에서 한다 — 여기서 하면 이중 적용(DEPLOY §2).
# ══════════════════════════════════════════════════════════════════════════════
class RgbDecoder:
    def __init__(self, crop, size_hw, expect_wh=(640, 480)):
        from PIL import Image
        self._Image = Image
        self.crop = None if crop is None else tuple(int(v) for v in crop)
        self.w, self.h = int(size_hw[1]), int(size_hw[0])
        self.expect_wh = expect_wh
        self.src_wh = None

    def __call__(self, jpeg: bytes) -> np.ndarray:
        im = self._Image.open(io.BytesIO(jpeg)).convert("RGB")
        self.src_wh = im.size
        if self.crop is not None:
            im = im.crop(self.crop)
        im = im.resize((self.w, self.h), self._Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        return arr.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# h5 키 → ROS 토픽. 레코더(`record/ros2_hdf5_recorder.py` FIELDS)와 **같아야** 한다.
# 학습이 본 신호와 다른 토픽을 물면 정책은 안 터지고 조용히 엉뚱한 값을 먹는다.
# 여기 없는 키를 config 가 요구하면 시작 자체를 거부한다(추측 금지).
# ══════════════════════════════════════════════════════════════════════════════
def build_sources(side: str, rgb_topic: str):
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float32MultiArray
    return {
        "03_hand_j_pos":     dict(topic=f"/hand/{side}/joint_states", typ=JointState,
                                  dim=16, get=lambda m: np.asarray(m.position, np.float32)),
        "10_hand_paxini_ft": dict(topic=f"/paxini/{side}/ft", typ=Float32MultiArray,
                                  dim=12, get=lambda m: np.asarray(m.data, np.float32)),
        "11_hand_paxini_raw": dict(topic=f"/paxini/{side}/raw", typ=Float32MultiArray,
                                   dim=1524, get=lambda m: np.asarray(m.data, np.float32)),
        "40_image":          dict(topic=rgb_topic, typ=CompressedImage, dim=None, get=None),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 신원 — 토픽 이름은 증거가 아니다.
#
# 2026-08-11 리그의 카메라 네임스페이스가 `/camera/camera` → `/front_cam/front` 로 **리네임**됐다
# (`record/ros2_hdf5_recorder.py` diff). 학습 h5 attrs 는 여전히 옛 이름을 적고 있어서, 토픽
# 이름만으로는 "같은 카메라를 리네임한 것"과 "다른 카메라"를 구분할 수 없다. 그런데 DEPLOY §2 는
# crop [71,0,551,480] 이 **이 카메라 위치 전용**이라고 못박는다 — 다른 카메라면 정책은 안 터지고
# 엉뚱한 화각으로 운전한다. 그래서 **intrinsics 로 대조**한다(해상도만으론 두 realsense 가 같다).
# ══════════════════════════════════════════════════════════════════════════════
def train_intrinsics(cfg) -> dict | None:
    """학습 데이터 h5 의 attrs 에서 카메라 intrinsics 를 읽는다. 못 읽으면 None."""
    try:
        import h5py
        root = Path(cfg["data"]["root"])
        names = list((cfg["data"].get("clamp_spec") or {}).keys())
        for n in names:
            p = root / n
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                a = f.attrs
                if "rgb_fx" not in a:
                    continue
                return {"fx": float(a["rgb_fx"]), "fy": float(a["rgb_fy"]),
                        "cx": float(a["rgb_cx"]), "cy": float(a["rgb_cy"]),
                        "width": int(a["rgb_width"]), "height": int(a["rgb_height"]),
                        "topic": str(a.get("rgb_topic", "?")), "src": n}
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
def make_node_class():
    """rclpy import 를 --self_test 경로에서 피하려고 클래스 정의를 늦춘다."""
    import rclpy                                                          # noqa: F401
    from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                       ReentrantCallbackGroup)
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, Int32

    SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
    # 인게이지는 latched — 발판이 먼저 떠 있었어도 마지막 상태를 받는다
    # (foot_pedal_glove.py:48-51 과 같은 QoS. 다르면 아예 수신이 안 된다.)
    ENGAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            history=HistoryPolicy.KEEP_LAST, depth=1)

    class VTDPRunner(Node):
        def __init__(self, dep, args, joint_limits):
            super().__init__("kist_vtdp_runner")
            self.dep, self.args = dep, args
            self.jl = np.asarray(joint_limits, np.float32)          # (16,2)

            self.lock = threading.Lock()
            self.sources = build_sources(args.side, args.rgb_topic)
            self.key_of = {}                     # 모달리티 이름 → h5 키
            for name, s in dep.ospec.items():
                ks = list(s["keys"])
                if len(ks) != 1 or ks[0] not in self.sources:
                    sys.exit(f"❌ 이 배포기가 못 대는 obs 키다: {name} → {ks}\n"
                             f"   지원: {sorted(self.sources)}  (레코더 FIELDS 와 같은 표)")
                self.key_of[name] = ks[0]
            self.kind_key = {s["kind"]: self.key_of[n] for n, s in dep.ospec.items()}
            # 손 관절각은 obs 에 없더라도 **항상** 구독한다 — 램프 시작점·워치독이 이 값이다.
            self.pos_key = "03_hand_j_pos"
            self.keys = sorted(set(self.key_of.values()) | {self.pos_key})

            # ── 최신 센서값 (레코더와 같은 hold 의미) ──
            self.latest = {k: np.zeros(self.sources[k]["dim"], np.float32)
                           for k in self.keys if self.sources[k]["dim"]}
            self.rgb = None
            self.t_recv = {k: 0.0 for k in self.keys}
            self.n_recv = {k: 0 for k in self.keys}
            self.rgb_decode_ms = deque(maxlen=60)
            self.tac_hist = deque(maxlen=300)             # 3s @100Hz — 부위 생사 판정용
            vis = next((s for s in dep.ospec.values() if s["kind"] == "vision"), None)
            self.dec = None if vis is None else RgbDecoder(
                dep.cfg["data"].get("rgb_crop"),
                (int(vis["shape"][1]), int(vis["shape"][2])))
            # 🔴 하드 스톱 래치 — 하나라도 서면 새 타겟을 만들지 않는다.
            # "감지했는데 계속 돈다"가 이 프로젝트가 반복해서 밟은 실패형이다.
            self.halt = None
            self.cam_ref = train_intrinsics(dep.cfg)   # 학습 h5 attrs (없으면 None)
            self.cam_checked = False

            self.enabled = not args.require_enable
            self.t_pub0 = None                   # 램프 기준 = **첫 발행 틱**
            self.n_pushed = self.n_lost = self.n_bad = 0
            self.sample_dt = deque(maxlen=500)
            self._t_last_sample = None
            self.warm_done = False
            self.warm_ms = 0.0
            self.gap_warn = float(args.gap_warn)
            self.jref_warned = False

            self.act_buf = deque()
            self.next_target = None
            self.have_new = False
            self.cur_target = None
            self.prev_target = None
            self.interp_from = None
            self.interp_i = 0
            self.n_interp = max(1, int(round(args.publish_hz / dep.hz)))
            self.max_delta_pub = args.max_rate_cps / args.publish_hz
            self._plock = threading.Lock()

            self.infer_ms = deque(maxlen=50)
            self.n_infer = self.n_clamp_j = self.n_clamp_v = 0
            self.n_stale_hand = self.n_stale_rgb = self.n_stale_tac = 0
            self.n_starve = self.n_overrun = self.pub_cnt = 0
            self._stop = False

            # ── 구독 ──
            cb = ReentrantCallbackGroup()
            for k in self.keys:
                src = self.sources[k]
                if k == "40_image":
                    self.create_subscription(src["typ"], src["topic"],
                                             self._cb_rgb, SENSOR_QOS, callback_group=cb)
                    self.create_subscription(CameraInfo, args.caminfo_topic,
                                             self._cb_caminfo, SENSOR_QOS, callback_group=cb)
                else:
                    self.create_subscription(
                        src["typ"], src["topic"],
                        lambda m, kk=k: self._cb_vec(kk, m), SENSOR_QOS, callback_group=cb)
            self.create_subscription(Bool, args.enable_topic, self._cb_enable,
                                     ENGAGE_QOS, callback_group=cb)

            # ── 발행 ──
            self.pub_target = self.create_publisher(Float32MultiArray,
                                                    args.target_topic, SENSOR_QOS)
            self.pub_debug = self.create_publisher(Float64MultiArray, "/kist_vtdp/debug", 10)
            self.pub_mode = self.create_publisher(Int32, f"/hand/{args.side}/cmd_mode", 1)
            self.pub_servo = self.create_publisher(Bool, f"/hand/{args.side}/cmd_servo", 1)
            self.servo_sent = False

            # ── 타이머 ──
            # 두 100Hz 타이머는 **서로 다른** MutuallyExclusive 그룹이다. 같은 그룹에 두면
            # 하나가 밀릴 때 다른 하나도 같이 밀리고, Reentrant 로 두면 콜백이 겹쳐 쌓인다.
            self.create_timer(0.01, self._sample,
                              callback_group=MutuallyExclusiveCallbackGroup())
            self.create_timer(1.0 / args.publish_hz, self._tick,
                              callback_group=MutuallyExclusiveCallbackGroup())
            self.create_timer(2.0, self._conflict_check, callback_group=cb)
            self._report_timer = self.create_timer(4.0, self._topic_report,
                                                   callback_group=cb)

            self._banner()
            self._thread = threading.Thread(target=self._policy_loop, daemon=True)
            self._thread.start()

        # ── 로그 ───────────────────────────────────────────────────────────
        def _banner(self):
            a, d, L = self.args, self.dep, self.get_logger().info
            L("=" * 78)
            L("  KIST VTDP Runner — 레몬 in-hand 회전 (시각+촉각)")
            L(f"  run           : {a.run}  ({d.prefer} 가중치)")
            L(f"  모달리티      : " + " · ".join(
                f"{n}[{self.key_of[n]}] T={s['horizon']}×s{s['stride']}"
                f"→{d.rings[n].depth}프레임" for n, s in d.ospec.items()))
            L(f"  hp EMA        : " + (", ".join(d.hp) if d.hp else "없음")
              + f"  (τ={d.cfg['obs_spec'].get('tactile', {}).get('transform_kwargs', {}).get('tau', '-')}"
                f" 프레임, 에피소드 동안 연속)")
            L(f"  제어/발행     : {d.hz:.0f} Hz / {a.publish_hz:.0f} Hz "
              f"(정책 타겟 사이 {self.n_interp}틱 선형보간)")
            L(f"  pred/exec     : {d.pred_h} / {d.exec_h} "
              f"→ 추론 {d.exec_h * d.tick_budget_ms:.0f} ms 마다")
            L(f"  DDIM steps    : {a.ddim_steps if a.ddim_steps else '체크포인트 값'}")
            L(f"  예열          : 링버퍼 {d.warmup_frames}프레임 + hp {a.hp_warmup_sec:.1f}s "
              f"→ {self.warm_need}프레임 ({self.warm_need*10} ms)")
            L(f"  속도 한계     : {a.max_rate_cps:.0f} count/s (발행 틱당 {self.max_delta_pub:.0f})")
            L(f"  램프          : {a.ramp_sec:.1f}s (**첫 발행 틱**부터, 현재 자세에서 출발)")
            L(f"  워치독        : 손 {a.stale_sec:.2f}s / RGB {a.rgb_stale_sec:.2f}s "
              f"(촉각은 홀드만) · |타겟−실측| 경고 {a.gap_warn:.0f} count")
            L(f"  카메라 대조   : " + ("학습 h5 attrs 없음 — 건너뜀" if self.cam_ref is None else
                                       f"{self.cam_ref['src']} fx={self.cam_ref['fx']:.2f} "
                                       f"{self.cam_ref['width']}x{self.cam_ref['height']} "
                                       f"(수집 당시 토픽 {self.cam_ref['topic']})"))
            L(f"  인게이지      : {a.enable_topic} "
              f"{'(필요)' if a.require_enable else '(무시 — 즉시 동작)'}")
            L(f"  발행          : " + ("🟡 DRY-RUN (발행 안 함)" if a.dry_run
                                       else f"🔴 {a.target_topic}"))
            if a.dry_run:
                L("  ⚠ DRY-RUN 은 상태기계를 실행과 **똑같이** 굴린다(발행 호출만 건너뜀).")
                L("    손이 따라오지 않으므로 |Δ|max 와 clamp 가 계속 커지는 것이 **정상**이다.")
                L("    ②에서 볼 것: 토픽 점검표 · 샘플러 p90 10.x ms · infer < 50 ms · 카메라 대조 ✅")
            L("=" * 78)

        @property
        def warm_need(self) -> int:
            return max(self.dep.warmup_frames, int(self.args.hp_warmup_sec * 100))

        def _conflict_check(self):
            n = self.count_publishers(self.args.target_topic)
            if n > 1:            # 내 퍼블리셔는 dry_run 에서도 만들어지므로 항상 1
                self.get_logger().error(
                    f"⚠ {self.args.target_topic} 퍼블리셔 {n}개 — 글러브 텔레옵이 아직 떠 "
                    "있으면 먼저 끄세요. 동시 스트리밍은 손이 두 타겟 사이에서 튑니다.")

        def _tactile_report(self):
            """부위별 생사 — DEPLOY §7 "배포 직전 센서 생사 확인"의 자동화.

            수신 건수만 세면 **얼어붙은 부위가 초록불로 보인다**. 이 프로젝트는 part1/part3 가
            세션 도중 죽은 데이터로 학습했고(진행 기록), 죽은 부위는 hp 가 정확히 0 으로 만들어
            정책이 그 손가락을 못 본다. 그 사실을 실행 전에 사람이 알아야 한다.
            """
            key = self.kind_key.get("tactile")
            if key is None or len(self.tac_hist) < 100:
                return
            w = np.stack(list(self.tac_hist))             # (T, D) — **원시** ft (hp 전)
            name = next(n for n, k in self.key_of.items() if k == key)
            n_part = int((self.dep.ospec[name].get("encoder_kwargs") or {}).get("n_part", 4))
            if w.shape[1] % n_part:
                return
            ch = w.shape[1] // n_part
            std = [float(w[:, p * ch:(p + 1) * ch].std()) for p in range(n_part)]
            dead = [p for p, s in enumerate(std) if s == 0.0]
            marks = " ".join(f"part{p}={s:.4f}{'💀' if s == 0.0 else ''}"
                             for p, s in enumerate(std))
            print(f"  {'✓' if not dead else '⚠ 정지 부위':12s} "
                  f"{'촉각 부위별 std':20s}    {marks}   ({len(w)/100:.1f}s 창, 원시 ft)")
            if dead:
                print(f"      ↳ part{dead} 가 {len(w)/100:.1f}초 내내 **정확히 상수**다 = 그 손가락은"
                      f" 정책에 안 보인다(hp 가 0 으로 만든다). 배선·전원 확인.")
                print(f"        참고: 이 프로젝트는 part1·part3 가 죽은 데이터로 학습했다 —"
                      f" 죽은 것 자체는 학습 분포와 어긋나지 않는다.")

        def _topic_report(self):
            self._report_timer.cancel()
            print("\n===== 토픽 점검 (4초) =====")
            for k in self.keys:
                src = self.sources[k]
                ok = "✓" if self.n_recv[k] else "✗ 수신 없음"
                extra = ""
                if k == self.pos_key and k not in self.key_of.values():
                    extra += "  (obs 아님 — 램프·워치독용)"
                if k == "40_image" and self.n_recv[k]:
                    ms = np.mean(self.rgb_decode_ms) if self.rgb_decode_ms else 0.0
                    extra += f"  (원본 {self.dec.src_wh}, decode {ms:.1f} ms/장)"
                print(f"  {ok:12s} {k:20s} <- {src['topic']}"
                      f"   {self.n_recv[k]}건/4s{extra}")
            self._tactile_report()
            en = self.count_publishers(self.args.enable_topic)
            print(f"  {'✓' if en else '✗ 퍼블리셔 없음':12s} {'engage':20s} "
                  f"<- {self.args.enable_topic}   현재 {'ON' if self.enabled else 'OFF'}")
            if not en and self.args.require_enable:
                print("      ↳ 발판 노드(record/foot_pedal_glove.py)가 안 떠 있다. 수동 인게이지:")
                print(f"         ros2 topic pub -1 {self.args.enable_topic} std_msgs/Bool "
                      "\"{data: true}\" --qos-durability transient_local")
            print("===========================\n")

        # ── 콜백 ───────────────────────────────────────────────────────────
        def _halt(self, why: str):
            """되돌릴 수 없는 불일치 — 새 타겟 생성을 영구히 멈춘다(마지막 값 홀드)."""
            if self.halt is None:
                self.halt = why
                self.get_logger().error(f"🛑 정지: {why}  — 새 타겟을 만들지 않는다(홀드).")

        def _cb_vec(self, key, msg):
            v = self.sources[key]["get"](msg)
            # 차원 검증 — 레코더는 zeros 에 덮어써서 결측 채널이 0 이지만, 여기는 latest 를
            # 재사용하므로 짧게 오면 뒷부분이 **직전 값으로 홀드**되어 학습과 다른 신호가 된다.
            if len(v) != self.sources[key]["dim"]:
                self._halt(f"{key} 차원 {len(v)} != 기대 {self.sources[key]['dim']} "
                           f"({self.sources[key]['topic']})")
                return
            # NaN/Inf 는 여기서 버린다. 학습은 로드 단계에서 예외로 막지만(vtdp/data.py)
            # 배포엔 그 관문이 없고, hp EMA 는 한 번 NaN 이 되면 **복구 경로가 없다**
            # (m = m + a·(x−m) 이 영구 NaN). 그러면 16관절 전부 NaN 타겟이 나간다.
            if not np.isfinite(v).all():
                self.n_bad += 1
                self.get_logger().warn(f"{key} 에 NaN/Inf — 이 메시지 버림(직전 값 홀드)",
                                       throttle_duration_sec=2.0)
                return
            with self.lock:
                self.latest[key][:] = v
                self.t_recv[key] = time.perf_counter()
                self.n_recv[key] += 1
            if key == self.kind_key.get("tactile"):
                self.tac_hist.append(v.copy())   # 부위 생사 판정용 (3s 창, 원시값)

        def _cb_rgb(self, msg):
            t0 = time.perf_counter()
            try:
                arr = self.dec(bytes(msg.data))
            except Exception as e:                       # 깨진 프레임 1장에 죽지 않는다
                self.get_logger().warn(f"RGB 디코드 실패: {e}", throttle_duration_sec=2.0)
                return
            if self.dec.src_wh != self.dec.expect_wh:
                self._halt(f"카메라 해상도 {self.dec.src_wh} != 학습 {self.dec.expect_wh} — "
                           f"crop {self.dec.crop} 무효")
                return
            if not np.isfinite(arr).all():
                self.n_bad += 1
                return
            with self.lock:
                self.rgb = arr
                self.t_recv["40_image"] = time.perf_counter()
                self.n_recv["40_image"] += 1
            self.rgb_decode_ms.append((time.perf_counter() - t0) * 1000)

        def _cb_caminfo(self, msg):
            """카메라 **신원** 확인 — 토픽 이름이 아니라 intrinsics 로."""
            if self.cam_checked:
                return
            self.cam_checked = True
            r = self.cam_ref
            if r is None:
                self.get_logger().warn(
                    f"학습 h5 attrs 를 못 읽어 카메라 대조를 건너뛴다 — "
                    f"live {msg.width}x{msg.height} fx={msg.k[0]:.2f}. "
                    f"crop_reference.png 로 화각을 눈으로 확인할 것.")
                return
            live = (msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            want = (r["width"], r["height"], r["fx"], r["fy"], r["cx"], r["cy"])
            same = (live[0], live[1]) == (want[0], want[1]) and all(
                abs(a - b) < 0.5 for a, b in zip(live[2:], want[2:]))
            self.get_logger().info(
                f"카메라 대조 — live {live[0]}x{live[1]} fx {live[2]:.2f} cx {live[4]:.2f} vs "
                f"학습({r['src']}, 토픽 {r['topic']}) {want[0]}x{want[1]} fx {want[2]:.2f} "
                f"cx {want[4]:.2f}  {'✅ 같은 카메라' if same else '❌ 다르다'}")
            if not same:
                self._halt("camera_info 가 학습 때와 다르다 — crop 이 무효다(DEPLOY §2: "
                           "카메라를 옮기면 재수집·재학습)")

        def _cb_enable(self, msg):
            if bool(msg.data) != self.enabled:
                self.get_logger().info(f"인게이지 {'ON' if msg.data else 'OFF'}")
            self.enabled = bool(msg.data)

        # ── 100 Hz 샘플러: 링버퍼 + hp EMA 의 시계 ──────────────────────────
        # 레코더 `_tick` 과 같은 의미다 — 토픽의 **마지막 값을 홀드**해 10ms 격자에 찍는다.
        # 학습 h5 가 정확히 이렇게 만들어졌다(30Hz 카메라·8~19Hz 인식도 홀드로 들어갔다).
        def _sample(self):
            now = time.perf_counter()
            miss = 0
            if self._t_last_sample is not None:
                dt = (now - self._t_last_sample) * 1000
                self.sample_dt.append(dt)
                # rcl 타이머는 주기를 놓치면 **밀린 만큼 건너뛴다**(따라잡기 호출 없음).
                # 그냥 두면 hp EMA 가 덜 굴러 τ 가 늘어나고 링버퍼 lookback 이 200→210ms 가
                # 된다 — 조용히 학습과 다른 창이 된다. 놓친 만큼 **홀드 값을 더 밀어** 격자를
                # 지킨다(레코더가 100Hz 로 홀드해 기록한 것과 같은 의미).
                miss = min(int(round(dt / 10.0)) - 1, 20)
                if miss > 0:
                    self.n_lost += miss
            self._t_last_sample = now

            if any(self.n_recv[k] == 0 for k in self.keys):
                self.get_logger().warn(
                    "센서 대기: " + ", ".join(self.sources[k]["topic"]
                                              for k in self.keys if self.n_recv[k] == 0),
                    throttle_duration_sec=3.0)
                return
            with self.lock:
                frame = {}
                for name, key in self.key_of.items():
                    frame[name] = self.rgb if key == "40_image" else self.latest[key].copy()
            for _ in range(max(0, miss) + 1):
                self.dep.push(frame)      # hp EMA 는 여기서 매 프레임 굴러간다
                self.n_pushed += 1

        # ── 20 Hz 정책 스레드 ──────────────────────────────────────────────
        # 발행 경로와 분리한다: 추론(≈14ms, 콜드스타트 265ms)을 발행 타이머 안에서 하면
        # 콜백이 밀려 쌓이고 지연이 눈덩이처럼 커진다(참조 run.py 주석: 실기에서 11→272ms).
        def _policy_loop(self):
            period = 1.0 / self.dep.hz
            nxt = time.perf_counter()
            while not self._stop:
                nxt += period
                try:
                    self._policy_step()
                except Exception as e:
                    self.get_logger().error(f"정책 루프 오류: {e}")
                slack = nxt - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    self.n_overrun += 1
                    nxt = time.perf_counter()

        def _policy_step(self):
            if self.halt is not None:
                self.act_buf.clear()
                return
            if self.n_pushed < self.warm_need or not self.dep.ready:
                return
            if not self.warm_done:
                # 🔴 첫 추론은 CUDA 콜드스타트로 ~265ms = 20Hz 예산의 5배(DEPLOY §6).
                # 버퍼가 찬 직후·**잡기 전에** 굴려 둔다.
                t0 = time.perf_counter()
                for _ in range(3):
                    self.dep.infer()
                self.warm_ms = (time.perf_counter() - t0) * 1000
                self.warm_done = True
                self.get_logger().info(
                    f"예열 완료 — 추론 3회 {self.warm_ms:.0f} ms "
                    f"(첫 회에 CUDA 콜드스타트 포함). 인게이지하면 즉시 동작합니다.")
                return

            now = time.perf_counter()
            with self.lock:
                t_hand = self.t_recv[self.pos_key]
                t_rgb = self.t_recv.get("40_image", now)
            # 워치독에 걸리면 **남은 플랜을 버린다.** 안 버리면 센서 복구 후 최대 15스텝
            # (=750ms)짜리 낡은 궤적을 눈 감고 먼저 실행한다 — 그 사이 레몬 자세는 달라져 있다.
            if now - t_hand > self.args.stale_sec:
                self.n_stale_hand += 1
                self.act_buf.clear()
                self.get_logger().warn("손 joint_states 가 낡았다 — 새 타겟 중단(홀드)",
                                       throttle_duration_sec=2.0)
                return
            if self.dec is not None and now - t_rgb > self.args.rgb_stale_sec:
                self.n_stale_rgb += 1
                self.act_buf.clear()
                self.get_logger().warn("RGB 가 낡았다 — 새 타겟 중단(얼어붙은 그림으로 운전 ❌)",
                                       throttle_duration_sec=2.0)
                return
            tac_key = self.kind_key.get("tactile")
            if tac_key and now - self.t_recv[tac_key] > self.args.stale_sec:
                self.n_stale_tac += 1      # 멈추지 않는다 — hp 가 상수를 0 으로 만든다

            if not self.act_buf:
                t0 = time.perf_counter()
                seq = self.dep.infer()                     # (pred_horizon, 16) [count]
                self.infer_ms.append((time.perf_counter() - t0) * 1000)
                self.n_infer += 1
                for a in seq[:self.dep.exec_h]:
                    self.act_buf.append(np.asarray(a, np.float32))
            nxt = self.act_buf.popleft()
            with self._plock:
                self.next_target = nxt
                self.have_new = True

        # ── 안전 가드 ──────────────────────────────────────────────────────
        def _guard(self, target, ref):
            t = np.asarray(target, np.float32).copy()
            # 🔴 NaN 은 아래 두 클램프를 **둘 다 통과한다** —
            # `abs(nan) > x` 도 False, `(nan < lo)|(nan > hi)` 도 False, `clip(nan)` = nan.
            # 카운터에도 안 잡혀 로그에 흔적이 없다. 그래서 여기서 먼저 막는다.
            if not np.isfinite(t).all():
                self.n_bad += 1
                self._halt("정책 출력에 NaN/Inf — 관절·속도 클램프가 NaN 을 못 잡는다")
                return np.asarray(ref, np.float32).copy()
            d = t - ref
            big = np.abs(d) > self.max_delta_pub
            if big.any():
                self.n_clamp_v += int(big.sum())
                t = ref + np.clip(d, -self.max_delta_pub, self.max_delta_pub)
            out = (t < self.jl[:, 0]) | (t > self.jl[:, 1])
            if out.any():
                self.n_clamp_j += int(out.sum())
            return np.clip(t, self.jl[:, 0], self.jl[:, 1]).astype(np.float32)

        # ── 100 Hz 발행: 보간 + 가드 + 발행만. 추론은 절대 하지 않는다 ──────
        def _tick(self):
            now = time.perf_counter()
            with self.lock:
                j_pos = self.latest[self.pos_key].copy()
                t_hand = self.t_recv[self.pos_key]
            if t_hand == 0.0:
                return

            # 🔴 명령 공간과 측정 공간을 섞지 않는다.
            # `JOINT_LIMITS` 는 **명령**(`04_hand_j_tar`)의 봉투다 — 학습 57데모 101,660 프레임에서
            # 액션 위반 **0건**. 반면 **측정**(`03_hand_j_pos`)은 35.4% 가 그 봉투 밖이고 j5 는 최대
            # 8018 count 벗어난다(물리적으로 불가능한 176° = 글리치이거나 다른 규약). 그래서 앵커·
            # 램프 시작점은 **측정값을 봉투 안으로 clip 한 값**이어야 한다. 안 그러면 첫 발행 틱에서
            # `_guard` 의 속도 클램프가 d=0 이라 놀고 마지막 `np.clip` 만 값을 옮겨 **속도 상한을
            # 통째로 우회하는 계단**이 생긴다(정지 자세에선 ≤105 count 라 작지만, 작업 중 재인게이지에선
            # 수천 count).
            j_ref = np.clip(j_pos, self.jl[:, 0], self.jl[:, 1])
            if not self.jref_warned and float(np.abs(j_pos - j_ref).max()) > 200.0:
                self.jref_warned = True
                self.get_logger().warn(
                    f"측정 관절각이 명령 한계 밖이다 (최대 {np.abs(j_pos-j_ref).max():.0f} count, "
                    f"관절 {int(np.argmax(np.abs(j_pos-j_ref)))}) — 명령은 한계 안에서 출발한다.")

            with self._plock:
                if self.have_new:
                    self.cur_target = self.next_target.copy()
                    self.have_new = False
                    got_new = True
                else:
                    got_new = False

            # 🔴 "발행 자격"이 아니라 **실제로 타겟이 나가는 틱**을 기준으로 앵커·램프를 잡는다.
            # 1라운드 수정은 `publishing`(자격)만 봤는데, 자격이 생긴 뒤에도 정책 첫 타겟까지는
            # 최소 1.0 s(hp 예열 100프레임) + 콜드스타트가 걸린다. 그 구간은 `cur_target is None`
            # 으로 빠져나가므로 **앵커는 얼고 램프 시계만 흘렀다** — 발판을 미리 밟아 둔(latched)
            # 흔한 경우에 첫 발행이 실측에서 멀리 떨어진 채, 램프도 이미 50~100% 소진된 상태로 나갔다.
            # 또 `--dry_run` 은 이 상태기계를 **똑같이** 굴린다(마지막 publish 호출만 건너뛴다) —
            # 그래야 ②(--dry_run) 리허설이 ③의 첫 틱을 실제로 검증한다.
            engaged = self.enabled and self.halt is None
            armed = engaged and self.cur_target is not None
            if not armed:
                self.prev_target = j_ref.copy()
                self.interp_from = j_ref.copy()
                self.interp_i = 0
                self.t_pub0 = None
                self.servo_sent = False          # 재인게이지 때 서보 다시 무장
                if engaged:
                    self.n_starve += 1
                self._beat(now, j_pos, None)     # 대기 중에도 심장은 뛴다 (아래 주석)
                return
            if self.t_pub0 is None:              # ← 실제 첫 발행 틱 = 램프 기준시각
                self.t_pub0 = now
                self.prev_target = j_ref.copy()
                self.interp_from = j_ref.copy()
                self.interp_i = 0
            elif got_new:
                self.interp_from = self.prev_target.copy()
                self.interp_i = 0

            self.interp_i += 1
            frac = min(1.0, self.interp_i / self.n_interp)
            target = self.interp_from + (self.cur_target - self.interp_from) * frac

            el = now - self.t_pub0                       # 시작 램프 (현재 자세 → 정책 타겟)
            if el < self.args.ramp_sec:
                target = j_ref + (target - j_ref) * (el / self.args.ramp_sec)

            target = self._guard(target, self.prev_target)
            gap = float(np.abs(target - j_pos).max())
            if gap > self.gap_warn:
                # 정지가 아니라 경고다: 학습 데이터에도 |q_target − q_pos| 가 p99 1242 ·
                # max 10103 count 로 크게 벌어지는 구간이 있다(사람이 실제로 그렇게 몰았다).
                # 여기서 멈추면 학습 분포를 스스로 부정하게 된다. 손가락이 걸린 것과
                # 정상 구동을 이 값만으로는 못 가른다 → 사람이 보고 판단하라는 신호.
                self.get_logger().warn(f"|타겟−실측| {gap:.0f} count > {self.gap_warn:.0f} — "
                                       f"손가락이 걸렸는지 확인", throttle_duration_sec=2.0)

            if not self.args.dry_run:            # dry_run 은 여기'만' 건너뛴다
                if not self.servo_sent and self.args.servo_on:
                    self.pub_mode.publish(Int32(data=1))
                    self.pub_servo.publish(Bool(data=True))
                    self.servo_sent = True
                    self.get_logger().info("핸드 servo ON (mode=position)")
                m = Float32MultiArray()
                m.data = [float(v) for v in target]
                self.pub_target.publish(m)
                self.pub_cnt += 1
            self.prev_target = target.copy()      # 속도 가드는 '직전 타겟' 기준

            self._beat(now, j_pos, target)

        # ── 1초 심장박동 ──────────────────────────────────────────────────
        # 🔴 **타겟이 안 나가는 동안에도 뛴다.** 이걸 발행 경로 뒤에 두면 인게이지 전에는
        # 노드가 완전히 침묵해서 "정상 대기"와 "죽었다"를 사람이 구분할 수 없다
        # (실기 첫 시도에서 실제로 헷갈렸다 — 예열 완료 뒤 아무 출력이 없었다).
        # `/kist_vtdp/debug` 도 같은 이유로 여기서 발행한다.
        def _beat(self, now, j_pos, target):
            infer = float(np.mean(self.infer_ms)) if self.infer_ms else 0.0
            gap = 0.0 if target is None else float(np.abs(target - j_pos).max())
            dbg = Float64MultiArray()
            dbg.data = [float(self.enabled), float(self.warm_done), infer,
                        float(len(self.act_buf)), gap,
                        float(self.n_clamp_j), float(self.n_clamp_v),
                        float(self.n_stale_hand), float(self.n_stale_rgb),
                        float(self.n_stale_tac), float(self.n_overrun),
                        float(self.n_starve), float(self.pub_cnt),
                        float(self.n_lost), float(self.n_bad),
                        float(self.halt is not None)]
            self.pub_debug.publish(dbg)

            self.tick_cnt = getattr(self, "tick_cnt", 0) + 1
            if self.tick_cnt % int(self.args.publish_hz):
                return
            dt = np.array(self.sample_dt) if self.sample_dt else np.array([10.0])
            rgb_age = (now - self.t_recv.get("40_image", now)) * 1000
            if self.halt is not None:
                state = f"🛑 정지({self.halt[:30]})"
            elif target is not None:
                state = f"발행중 tgt[0:4]={np.round(target[:4], 0)}"
            elif not self.enabled:
                state = f"⏸ 대기: 인게이지 OFF ({self.args.enable_topic} 를 true 로)"
            elif not self.warm_done:
                state = "⏸ 대기: 예열 중"
            else:
                state = "⏸ 대기: 정책 타겟 없음 (워치독? stale 카운터 확인)"
            self.get_logger().info(
                f"en={int(self.enabled)} warm={int(self.warm_done)} "
                f"push={self.n_pushed} 유실={self.n_lost} "
                f"샘플러 p90={np.percentile(dt,90):.1f}ms "
                f"infer={infer:5.1f}ms×{self.n_infer} buf={len(self.act_buf)} "
                f"rgb_age={rgb_age:4.0f}ms |Δ|max={gap:6.1f} "
                f"clamp(j/v)={self.n_clamp_j}/{self.n_clamp_v} "
                f"stale(h/r/t)={self.n_stale_hand}/{self.n_stale_rgb}/{self.n_stale_tac} "
                f"over={self.n_overrun} starve={self.n_starve} bad={self.n_bad} "
                f"pub={self.pub_cnt} | {state}")

        def shutdown(self):
            self._stop = True
            self._thread.join(timeout=1.0)
            dt = np.array(self.sample_dt) if self.sample_dt else np.array([np.nan])
            inf = np.array(self.infer_ms) if self.infer_ms else np.array([np.nan])
            print(f"\n  100Hz 샘플러 주기  p50 {np.nanpercentile(dt,50):.2f} "
                  f"p90 {np.nanpercentile(dt,90):.2f} max {np.nanmax(dt):.2f} ms (목표 10.0)")
            print(f"  추론 {self.n_infer}회  p50 {np.nanpercentile(inf,50):.2f} "
                  f"p90 {np.nanpercentile(inf,90):.2f} ms (예산 {self.dep.tick_budget_ms:.0f} ms)")
            print(f"  100Hz 프레임 {self.n_pushed} (타이머 유실 보정 {self.n_lost}) · "
                  f"버린 메시지(NaN/차원) {self.n_bad}"
                  + (f" · 🛑 {self.halt}" if self.halt else ""))
            print(f"  발행 {self.pub_cnt} · 클램프 j/v {self.n_clamp_j}/{self.n_clamp_v} · "
                  f"stale h/r/t {self.n_stale_hand}/{self.n_stale_rgb}/{self.n_stale_tac} · "
                  f"overrun {self.n_overrun} · starve {self.n_starve}")
            if not self.args.dry_run:
                print("  종료 — 마지막 타겟 유지 (서보는 끄지 않는다)")

    return VTDPRunner


# ══════════════════════════════════════════════════════════════════════════════
def _gpu_apps() -> list[str]:
    """이 GPU 를 같이 쓰는 프로세스 — 추론이 느릴 때 첫 용의자다(학습이 같이 돌면 3배)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
        return [l.strip() for l in out.splitlines() if l.strip()] or ["(없음)"]
    except Exception as e:
        return [f"(nvidia-smi 실패: {e})"]


def apply_ddim(dep, args) -> str | None:
    """--ddim_steps override. self_test 와 실행이 **같은 값**을 봐야 한다(지연 측정의 의미)."""
    if not args.ddim_steps:
        return None
    if not hasattr(dep.policy.head, "infer_steps"):
        return f"이 head 는 infer_steps 가 없다: {type(dep.policy.head).__name__}"
    dep.policy.head.infer_steps = int(args.ddim_steps)
    return None


def self_test(core, args) -> bool:
    """ROS·로봇 없이 계약을 검증한다. ① hp 스트리밍 ② RGB 디코드 ③ 추론 파이프라인."""
    ok = True
    print("\n[1] hp 고역통과 — 스트리밍 vs 배치 (DEPLOY §3-1)")
    ok &= core.self_test()

    print("\n[2] 정책 조립 + 링버퍼 + 추론 (합성 프레임)")
    dep = core.Deployer(Path(args.repo) / args.run, args.ckpt, args.device, args.exec_horizon)
    err = apply_ddim(dep, args)
    if err:
        print(f"  ❌ {err}")
        return False
    print(f"  DDIM steps {dep.policy.head.infer_steps} · exec_horizon {dep.exec_h} "
          f"· 제어 {dep.hz:.0f} Hz")
    rng = np.random.default_rng(0)
    shapes = {n: (tuple(s["shape"]) if s["kind"] == "vision" else (int(s["shape"]),))
              for n, s in dep.ospec.items()}
    for i in range(dep.warmup_frames):
        dep.push({n: rng.random(sh, dtype=np.float32) for n, sh in shapes.items()})
    print(f"  링버퍼 준비 {dep.warmup_frames}프레임 후 ready={dep.ready}   "
          f"{'✅' if dep.ready else '❌'}")
    ok &= bool(dep.ready)
    ms = []
    for _ in range(3 + 20):                        # 예열 3회는 배포 계약과 같은 횟수
        t0 = time.perf_counter()
        a = dep.infer()
        ms.append((time.perf_counter() - t0) * 1000)
    ms = np.array(ms)
    p50, p90 = np.percentile(ms[3:], 50), np.percentile(ms[3:], 90)
    shape_ok = a.shape == (dep.pred_h, dep.cfg["action"]["dim"])
    speed_ok = p90 < dep.tick_budget_ms
    print(f"  액션 shape {a.shape} == ({dep.pred_h}, {dep.cfg['action']['dim']})  "
          f"{'✅' if shape_ok else '❌'}")
    print(f"  추론 예열 3회 {np.round(ms[:3], 0)} ms (첫 회 = CUDA 콜드스타트)")
    print(f"  이후 20회  p50 {p50:.1f}  p90 {p90:.1f}  max {ms[3:].max():.1f} ms "
          f"(20Hz 틱 예산 {dep.tick_budget_ms:.0f} ms)  {'✅' if speed_ok else '❌'}")
    if not speed_ok:
        print(f"     ↳ GPU 를 누가 같이 쓰고 있나 확인할 것 (v5 실측은 유휴 GPU 에서 p90 14.4 ms):")
        for ln in _gpu_apps():
            print(f"       {ln}")
    ok &= shape_ok and speed_ok
    # 합성 입력이라 값 자체는 의미 없다 — 관절 한계 안인지만(NaN·발산 감지)
    print(f"  액션 범위 [{a.min():.0f}, {a.max():.0f}] count  "
          f"{'✅ 유한' if np.isfinite(a).all() else '❌ NaN/inf'}")
    ok &= bool(np.isfinite(a).all())

    print("\n[3] RGB 디코드 — 배포(JPEG 바이트) vs 학습(`_decode` 파일 경로)")
    crop = dep.cfg["data"].get("rgb_crop")
    vis = next((s for s in dep.ospec.values() if s["kind"] == "vision"), None)
    if vis is None:
        print("  이 config 는 vision 이 없다 — 건너뜀")
    else:
        dec = RgbDecoder(crop, (int(vis["shape"][1]), int(vis["shape"][2])))
        try:
            from vtdp.data import build_datasets
            _, val_ds, _, _, _ = build_datasets(dep.cfg, verbose=False)
            di = 0
            d = val_ds.demos[di]
            fi = int(d.rgb_index[d.rgb_index >= 0][0])
            path = os.path.join(d.rgb_dir, f"{fi:06d}.jpg")
            mine = dec(Path(path).read_bytes())
            train = val_ds._decode(di, fi)
            err = float(np.abs(mine - train).max())
            print(f"  {Path(path).name} ({d.name})  max|Δ| = {err:.3e}   "
                  f"{'✅ 동일' if err == 0.0 else '❌ 경로가 갈렸다'}")
            print(f"  범위 [{mine.min():.3f}, {mine.max():.3f}] (ImageNet 정규화 ❌ — 인코더 몫)")
            ok &= err == 0.0
        except Exception as e:
            print(f"  ⚠️ 학습 데이터를 못 읽어 대조 생략: {e}")
    print("\n" + ("  ✅ 전부 통과" if ok else "  ❌ 실패 항목 있음") + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description="KIST VTDP 실기 배포 (ROS2)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="kist-vtdp-wrapper 경로")
    ap.add_argument("--run", default=DEFAULT_RUN, help="run 디렉터리 (config.yaml + best.pt)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--side", default="right")
    # 시간축 / 추론
    ap.add_argument("--exec_horizon", type=int, default=None, help="기본: config 값(16)")
    ap.add_argument("--ddim_steps", type=int, default=None, help="기본: config 값(10)")
    ap.add_argument("--publish_hz", type=float, default=100.0)
    ap.add_argument("--hp_warmup_sec", type=float, default=1.0,
                    help="hp EMA(τ=1s) 예열. 이만큼 센서를 읽고 나서 추론을 시작한다")
    # 안전
    ap.add_argument("--max_rate_cps", type=float, default=20000.0,
                    help="관절 속도 상한 [count/s]. 기본 = 학습 |Δ|/50ms p99.9(963 count)")
    ap.add_argument("--ramp_sec", type=float, default=2.0)
    ap.add_argument("--stale_sec", type=float, default=0.3)
    ap.add_argument("--rgb_stale_sec", type=float, default=1.0)
    ap.add_argument("--gap_warn", type=float, default=1250.0,
                    help="|타겟−실측| 이 이보다 크면 경고(정지 아님). 기본 = 학습 p99 1242 count")
    ap.add_argument("--enable_topic", default="/teleop/hand_engage/right")
    ap.add_argument("--require_enable", type=int, default=1)
    ap.add_argument("--servo_on", type=int, default=1)
    ap.add_argument("--target_topic", default=None, help="기본 /hand/<side>/q_target")
    ap.add_argument("--rgb_topic", default="/front_cam/front/color/image_raw/compressed")
    ap.add_argument("--caminfo_topic", default="/front_cam/front/color/camera_info")
    ap.add_argument("--dry_run", action="store_true", help="계산만 하고 발행하지 않음")
    ap.add_argument("--self_test", action="store_true", help="ROS 없이 계약 검증 후 종료")
    ap.add_argument("--model_type", default=None,
                    help="--run 대신 짧은 id 로 모델을 고른다 (예: 000). 표는 --list_models. "
                         "정본 표는 <repo>/run.py 의 MODEL_TYPES 하나뿐이다")
    ap.add_argument("--list_models", action="store_true", help="--model_type 표를 찍고 종료")
    args, ros_args = ap.parse_known_args()

    if args.cpu or not torch.cuda.is_available():
        args.device = "cpu"
    args.require_enable = bool(args.require_enable)
    args.servo_on = bool(args.servo_on)
    if args.target_topic is None:
        args.target_topic = f"/hand/{args.side}/q_target"

    torch.set_num_threads(1)          # ROS 콜백과의 GIL·CPU 경쟁 줄이기
    torch.set_grad_enabled(False)
    core = load_deploy_core(args.repo)

    # `--model_type` → `--run`. 표는 저장소 쪽 `run.py MODEL_TYPES` 가 정본이다 (두 벌 금지).
    if args.list_models:
        print(core.format_model_types(Path(args.repo)))
        return 0
    if args.model_type is not None:
        args.run = core.resolve_model_type(args.model_type, Path(args.repo))
        print(f"[model_type {args.model_type}] {core.MODEL_TYPES[args.model_type][1]}")
        print(f"                → {args.run}")

    if args.self_test:
        return 0 if self_test(core, args) else 1

    run_dir = Path(args.repo) / args.run
    if not (run_dir / args.ckpt).exists():
        print(f"[ERROR] 체크포인트 없음: {run_dir / args.ckpt}")
        return 1
    dep = core.Deployer(run_dir, args.ckpt, args.device, args.exec_horizon)
    err = apply_ddim(dep, args)
    if err:
        print(f"[ERROR] {err}")
        return 1
    if dep.exec_h > dep.pred_h:
        print(f"[ERROR] exec_horizon({dep.exec_h}) > pred_horizon({dep.pred_h})")
        return 1

    # 🔴 촉각 transform 순서 가드.
    # `Deployer` 는 hp 를 **원시 스트림에** 굴리고(상태 유지) 나머지 transform 은 창에 나중에 건다.
    # 그 분해는 hp 가 **맨 앞**일 때만 학습과 같다. config 가 `[rawstat, hp]`(= rawstat 먼저)면
    # 배포는 hp→rawstat 이 되어 값이 달라진다 — 실측: 접촉 탁셀 수 채널 std 0.099 → 7.32(74배),
    # max|Δ| 128. 선형인 벡터합 3칸만 우연히 같다. 안 터지고 조용히 틀리는 계열이라 여기서 막는다.
    # (배포 대상 r6 은 `transform: hp` 하나뿐이라 해당 없음 — self_test 가 max|Δ|=0 으로 확인한다.)
    for name, s in dep.ospec.items():
        t = s.get("transform")
        t = [] if t is None else ([t] if isinstance(t, str) else list(t))
        if "hp" in t and t[0] != "hp":
            print(f"[ERROR] {name}: transform {t} — 스트리밍 hp 분해가 학습과 어긋난다.\n"
                  f"        hp 가 맨 앞이 아니면 배포 경로는 hp→{t[0]} 순서가 된다.\n"
                  f"        이 arm 을 실기에 쓰려면 wrapper `run.py` 의 Deployer 를 먼저 고칠 것.")
            return 1

    # 관절 한계: 이 데이터의 action 을 실제로 만든 표(tools/glove_teleop.py HAND_LIMITS).
    # 학습 57데모 실측 범위가 이 표 안에 정확히 들어간다(min −4096/−2048/−1000, max 4096).
    import dp_config as C
    jl = np.asarray(C.JOINT_LIMITS, np.float32)
    if jl.shape != (dep.cfg["action"]["dim"], 2):
        print(f"[ERROR] JOINT_LIMITS shape {jl.shape}")
        return 1

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    Runner = make_node_class()
    rclpy.init(args=ros_args or None)
    node = Runner(dep, args, jl)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    except Exception:
        # SIGTERM 으로 죽이면 rclpy 가 컨텍스트를 먼저 닫아 executor 가 RCLError 를 올린다
        # (정상 종료인데 traceback 이 뜬다). 컨텍스트가 살아 있으면 그건 진짜 오류다.
        if rclpy.ok():
            raise
    finally:
        node.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
