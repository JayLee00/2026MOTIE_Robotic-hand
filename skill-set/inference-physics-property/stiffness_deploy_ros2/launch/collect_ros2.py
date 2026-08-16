#!/usr/bin/env python3
"""collect_ros2.py — 배포와 '동일 경로'로 학습 데이터를 수집하는 하이브리드 레코더.

시퀀스 (한 run) — 8단계, phase 코드도 이 순서대로 1~8:
  1 [palm-up 이동]       팔 이동(MoveIt) — ★ **safe 경유** (직접 이동은 수직으로 빠지지
                         않아 물체와 충돌)
  2 [손 펴기]            손만 safe(initial_pose) — 물체를 palm-up 손바닥 위에 내려놓는다
  3 [물체 파지]          손만 — **파지 임계힘 도달 판정/재시도**
                         (임계 도달 손가락 < GRIP_MIN_FINGERS → 손 펴고 재시도,
                          재시도 소진 시 스퀴즈로 가지 않고 run 중단 = grip_fail)
                         여기서 나온 위치가 스퀴즈 A·B 의 복귀 기준이 된다
  4 [스퀴즈A ★]          손 스퀴즈 — 기록 + flag. 임계 = 파지임계 + delta(3~5N)
  5 [엄지 복귀 A]        ★ 스퀴즈로 움직인 **엄지만** 스퀴즈 직전 위치로 되돌린다(힘 기준 아님)
                         힘이 사실상 0 이면(놓침) 스퀴즈 B 생략하고 run 폐기(grip_lost)
  6 [palm-down 이동 ★]   팔 이동(MoveIt) — flag (bag)
  7 [스퀴즈B ★]          손 스퀴즈 — 기록 + flag
  8 [엄지 복귀 B]        ★ 같은 복귀 + 놓침 확인. 단 여기서는 **자동 폐기하지 않는다** —
                         스퀴즈 A·B 가 이미 기록됐으므로 grip_lost 를 판정 프롬프트의
                         기본값으로 '제안' 만 하고 확정은 사람이 한다

  ※ 이 시퀀스에는 [safe 시작]·[grip 자세 이동]·[물체 내려놓기]·[safe 끝] 이 **없다**
    (2026-07-29 사용자 지정). 그 결과:
      · run 은 palm-up 에서 시작해 palm-down 에서 **물체를 든 채** 끝난다.
      · 다음 run 이 그 상태를 이어받아 palm-up 이동부터 시작한다.
      · **첫 run 전에 물체를 손안(또는 palm-up 손바닥 위)에 놓아 두어야 한다.**
      · 세션 종료 시(전 조합 완료·Ctrl-C·'q' 모두) 팔 palm-up(기본, safe 경유)
        + 손 safe 로 복귀 후 종료한다. 복귀 중 Ctrl-C 를 또 누르면 그 자리에서 멈춘다.
    단 중단(grip_fail/grip_lost)은 예외 경로라 손을 펴고 팔을 safe 로 뺀다.

자세 스케줄 (POSE_SCHEDULE=True, 기본):
  개체 하나당 palm-up 3자세 × palm-down 3자세 × 파지 5자세 = **45 조합을 빠짐없이 1회씩**.
    · palm-up/down 3자세 = 기본 + {tilt_left|right} 택1 + {tilt_fwd|back} 택1 (개체 세션당 고정)
    · 파지 자세 = GRIP_POSE_CANDIDATES 5개 전부
    · 파지 실패(grip_fail)·discard 판정이면 그 조합을 **소진하지 않고 재시도**
  → 완전 랜덤과 달리 조합 커버리지가 보장된다(어떤 조합이 0번인 일이 없다).

원칙:
  · 모든 로봇 '팔' 이동은 moveit_arm_mover(MoveIt + 충돌회피, Option B: plan-only → q_target 재생).
  · 로봇 '손' 파지/스퀴즈는 배포와 '동일'(deploy(D) 프리미티브 + Ros2 브리지). 변경 없음.
  · flag: 이제 스퀴즈 하나가 아니라 ★ 구간 '각각' 라벨. /collect/segment(String) 토픽으로
    구간명을 발행 → rosbag 에 담겨 '전처리 시 구간별 포함여부'를 결정. 스퀴즈 ★ 는 추가로
    HDF5 그룹의 'segment' attr 로도 태깅.

데이터:
  · Option 1 (HDF5): 스퀴즈 ★(A,B) 를 각각 데모 그룹으로 기록(모델 입력 프레임). palm-down ★ 는
    손 add_sample 이 없어 HDF5 엔 없음 → rosbag(+segment 라벨)에서 슬라이스.
  · Option 2 (rosbag): 계약 토픽 + /collect/segment·demo_marker raw 기록.

실행:
  source env.sh
  python3 stiffness_deploy_ros2/launch/collect_ros2.py --fruit tomato --num-demos 20
  
  python3 collect_ros2.py --fruit tomato --grip-pose tomato

명령어 옵션 (argparse):
  --fruit NAME        물체 종류(tomato/kiwi/plum/lemon/ecoflex…). 생략 시 대화형 선택.
                      미등록 이름도 '<이름>.txt' 자세 파일이 있으면 수집 가능.
  --specimen S        개체 이름/번호. '1' → '<물체>_1', 'ecoflex_1' 은 그대로.
                      폴더·파일명이 <개체>_<파지자세>_<ts>. 생략 시 시작할 때 묻는다.
  --grip-pose POSE    파지 손 자세를 이 pose txt 하나로 고정(랜덤화 끔). 확장자 생략 가능.
                      미지정 시 GRIP_POSE_CANDIDATES 중 run 마다 랜덤.
  --num-demos N       run 수 상한(테스트용). 스케줄 모드 기본은 '조합 45개 전부'.
  --out-dir DIR       세션 저장 루트 (기본 <repo>/collect_logs).
  --paxini {ft,raw}   촉각 소스. ft=/paxini/right/ft(4×3, 기본),
                      raw=/paxini/right/raw(4×127×3 진짜 127점, 제어 PC 발행 전제).
  --no-bag            rosbag 동시 기록 끄기 (HDF5 만 저장).
  --bag-storage S     rosbag storage (기본 mcap; mcap 플러그인 없으면 sqlite3).
  --bag-topics ...    기록 토픽 재정의 (기본: 계약 토픽 + /collect/* 마커).
  --no-judge          run 종료 후 성공 판정 프롬프트 끄기 (전부 'unjudged').

수집 도중 그만두려면 (데이터는 어느 쪽이든 보존된다):
  · 권장 — run 끝 판정 프롬프트에서 숫자 뒤에 `q`: `1q`(성공으로 기록하고 종료) / `q`(기본값).
    남은 run 을 건너뛰고 정상 경로로 마감한다.
  · 비상 — Ctrl-C: 그 run 은 outcome='interrupted' 로 기록되고 h5/bag/outcomes.json 이
    중단 시점까지 정상 저장된다. 단 팔·손은 그 자세에 그대로 멈춘다(자동 복귀 안 함).

상단 편집 상수 (이 파일 위쪽에서 직접 수정):
  ARM_POSES               팔 자세 4개(safe/grip/palm_up/palm_down). capture_pose.py 로 채움.
                          미설정(None)이면 그 팔 이동은 '생략'(손 시퀀스만 먼저 검증 가능).
  GRIP_FORCE_RANGE        파지 임계 랜덤 범위 [N] (기본 6.0~8.0).
  SQUEEZE_DELTA_RANGE     스퀴즈 추가 압축량 delta [N] (기본 3.0~5.0).
                          스퀴즈 임계 = 파지 임계 + delta → 9~13N (절대값 독립 추출 아님).
  THUMB_RETURN_AFTER_SQUEEZE / THUMB_RETURN_DURATION / RELEASE_SETTLE_SEC / GRIP_LOST_FORCE_N
                          스퀴즈 후 엄지만 파지 위치로 복귀 + 놓침 확인.
                          (힘 기준 재조임은 실측에서 물체를 놓쳐 폐기했다 — 코드 주석 참고)
  OUTCOMES                데모 판정 라벨 (1=success/2=grip_fail/3=not_judged/4=discard).

전제: 제어 PC 스택(shm_state_publisher + receiver + paxini writer) + (팔 이동 시) 플래닝 PC
  move_group(joint_state_mode:=direct)이 같은 ROS_DOMAIN_ID(9)로 실행 중.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import threading
import unicodedata
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# deploy_ros2 재사용을 위한 sys.path 규약 (deploy_ros2.py / deploy_task3_ros2.py 와 동일).
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_LAUNCH_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))  # package root (core.* 해석)
sys.path.insert(0, _LAUNCH_DIR)                       # launch/ (deploy* 모듈)

import rclpy                                                           # noqa: E402
from rclpy.node import Node                                            # noqa: E402
from rclpy.executors import SingleThreadedExecutor                    # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy    # noqa: E402
import numpy as np                                                   # noqa: E402
from std_msgs.msg import String, Int8                                 # noqa: E402

# deploy_ros2 를 import 하면 그 모듈이 sys.path/shim 설정 + `import deploy as D` 를
# 이미 수행한다. 그 결과(D, 브리지, _grip)를 그대로 가져다 쓴다 (코드 중복 0).
import deploy_ros2 as DR                                              # noqa: E402
import real_deploy_inference_final as RE                              # noqa: E402  (상수/provenance)
from recording_engine import RecordingEngine, HDF5DemoWriter          # noqa: E402
from moveit_arm_mover import MoveItArmMover                           # noqa: E402  (팔 이동)

D = DR.D
Ros2ShmBridge = DR.Ros2ShmBridge
Ros2PaxiniBridge = DR.Ros2PaxiniBridge
_grip = DR._grip

# ── 팔 목표 자세: MoveItArmMover 로 이동(충돌회피). ★ capture_pose.py 로 값 채우기. ──
#   형식 두 가지 (둘 중 하나로 채움):
#     {"joints": [j0..j6]}                                 ← 방법 B(권장): 관절각 (capture_pose.py 출력)
#     {"position": (x,y,z), "orientation": (qx,qy,qz,qw)}  ← Cartesian(플랜지 right_fr3_link8, world)
#   둘 다 None 이면 그 팔 이동은 '생략'(안전 — 손 시퀀스만 먼저 검증 가능).
#   ※ 아래 값은 2026-07-27 capture_pose.py 실측(quick_start.txt 하단 기록).
#     캡처 당시 이름 → 이 키 매핑: 'safe'→safe, 'grasp'→grip, 'palm-up'→palm_up.
#   ★★ 키는 **끝에만 추가(append-only)**. 재정렬·중간삽입·삭제 금지 —
#     bag_to_session.py 가 이 선언 순서로 자세 숫자코드를 매긴다(위 GRIP_POSE_CANDIDATES 와 동일 이유).
ARM_POSES = {
    "safe":      {"joints": [-0.1026, 0.9079, 0.6511, -1.7808, -1.2049, 1.8748, 0.4749]},
    "grip":      {"joints": [-0.0605, 1.1827, 0.4211, -1.5709, -1.3295, 1.8854, 0.4948]},
    "palm_up":   {"joints": [-0.0002, 1.0334, 0.2085, -2.1937, 1.6606, 1.1778, 0.7250]},
    # ── palm_up 변형(스퀴즈 A 손목자세 랜덤화 후보). ★ capture_pose.py 로 채우기. ──
    #   비워두면(joints=None) 랜덤 풀에서 자동 제외되어 이동도 생략(안전). 값을 채우면 그때부터
    #   palm_up 과 함께 랜덤 선택 대상이 된다. 물체는 이미 손안이라 손목만 돌려도 안 놓친다.
    "palm_up_tilt_left":  {"joints": [0.0025, 1.0254, 0.2071, -2.2368, 1.6147, 0.9051, 0.7510]},   # 손목 좌로 기울임
    "palm_up_tilt_right": {"joints": [0.0065, 1.0284, 0.2175, -2.2064, 1.6573, 1.5282, 0.7033]},   # 손목 우로 기울임
    "palm_up_tilt_fwd":   {"joints": [0.0019, 1.0779, 0.2461, -2.1959, 1.8657, 1.0963, 0.8237]},   # 손목 앞으로 기울임
    "palm_up_tilt_back":  {"joints": [0.0035, 1.0643, 0.2583, -2.1409, 1.4372, 1.1590, 0.6369]},   # 손목축(마지막 관절) 롤 회전
    # ⚠ palm_down 은 전용 캡처가 없어 **임시로 safe 와 동일 값**을 넣었다(placeholder).
    #   손목축(joint5)이 palm_up(+1.070) ↔ 여기(-1.223) 로 약 2.29 rad 뒤집히므로 '팔이 안 움직이는'
    #   문제는 없고 ★'move_palm_down' 구간도 실제 모션으로 기록된다. 단 자세 자체는 safe 와 같으므로
    #   분석 시 "palm_down ≡ safe" 임을 감안할 것. 전용 자세를 잡으면 capture_pose.py 로 교체.
    "palm_down": {"joints": [-0.1026, 0.9079, 0.6511, -1.7808, -1.2049, 1.8748, 0.4749]},
    # ── palm_down 변형(스퀴즈 B 손목자세 랜덤화 후보). ★ capture_pose.py 로 채우기. ──
    "palm_down_tilt_left":  {"joints": [-0.0981, 0.8318, 0.7396, -1.8140, -1.1456, 1.5133, 0.3104]},   # 손목 좌로 기울임
    "palm_down_tilt_right": {"joints": [-0.1015, 0.8955, 0.6780, -1.8149, -0.9831, 2.1995, 0.4166]},   # 손목 우로 기울임
    "palm_down_tilt_fwd":   {"joints": [-0.1011, 0.8233, 0.7338, -1.8052, -1.2654, 1.7374, 0.3075]},   # 손목 앞으로 기울임
    "palm_down_tilt_back":  {"joints": [-0.1046, 0.8236, 0.7249, -1.7881, -0.7491, 1.6074, 0.2892]},   # 손목축(마지막 관절) 롤 회전
}
# MoveIt 파라미터 (dex_ros 조사 결과: 오른팔 그룹/플랜지/월드 프레임)
ARM_GROUP, ARM_EE_LINK, ARM_FRAME = "right_arm", "right_fr3_link8", "world"

# ★ 표시가 붙는(전처리 포함여부 대상) 구간. rosbag /collect/segment 라벨로 마킹.
STAR_SEGMENTS = ("squeeze_A", "move_palm_down", "squeeze_B")

# ── 파지/스퀴즈 힘 임계값 랜덤화 범위 [N]. ★ 여기 범위를 직접 입력해 고정. ──
#   (min, max) 튜플이면 그 범위에서 매번 uniform 랜덤. None 이면 랜덤화 끄고 과일별 고정값 사용.
#   파지 임계 = run 당 1회, 스퀴즈 임계 = 스퀴즈(A,B) 마다 각각 뽑음.
#   실제 사용한 값은 각 스퀴즈 HDF5 그룹 attr(grip/squeeze_force_threshold_n)에 기록(재현성).
GRIP_FORCE_RANGE = (6.0, 8.0)       # 파지 접촉력 임계 랜덤 범위 [N] (None=끔)
# ★ 스퀴즈 임계는 **파지 임계에 delta 를 더해** 만든다 (2026-07-28 사용자 규칙).
#   왜: 스퀴즈는 '파지 상태에서 얼마나 더 눌러 강성을 재는가' 이므로, 절대값을 독립으로
#   뽑으면 파지가 세게 잡힌 run 에서는 추가 압축이 거의 없고(squeeze ≈ grasp) 약하게 잡힌
#   run 에서는 과하게 눌리는 등 **'추가 압축량' 이 제어되지 않는다.**
#   결합하면 delta = 실제 추가 압축량이 되어 강성 신호의 크기를 직접 통제할 수 있다.
#     grasp   = uniform(6.0, 8.0)
#     delta   = uniform(3.0, 5.0)
#     squeeze = grasp + delta            → 9 ~ 13N
#   delta 는 스퀴즈(A,B) 마다 각각 뽑는다 — 한 run 안에서도 두 압축량을 다르게 하면
#   같은 파지 상태에 대한 서로 다른 압축 응답을 얻는다(A/B 다양성 유지).
SQUEEZE_DELTA_RANGE = (3.0, 5.0)     # 스퀴즈 추가 압축량 delta 랜덤 범위 [N] (None=끔)
SQUEEZE_FORCE_RANGE = None           # (구) 스퀴즈 절대 임계 범위. None = delta 방식 사용.
#   값을 넣으면 예전처럼 절대값으로 뽑는다(delta 무시) — 되돌릴 때만 쓸 것.

# ── palm 제시 자세(스퀴즈 측정 중 손목자세) 랜덤화 ─────────────────────────────
#   왜: 스퀴즈 시점엔 물체가 이미 손안이라 손목 방향을 바꿔도 물체를 놓치지 않는다(안전).
#     방향이 달라지면 in-hand 물체에 걸리는 중력방향이 바뀌어 stiffness/squeeze 일반화에
#     직접 기여한다. grip(파지) 자세는 물체가 고정 위치라 건드리지 않는다.
#   동작: run 당 palm_up(스퀴즈 A)·palm_down(스퀴즈 B) 를 각각 후보 중 1개 uniform 랜덤 선택.
#     후보 = 아래 목록 중 ARM_POSES 값이 '채워진' 키만(빈 변형은 자동 제외). 채워진 게
#     base(palm_up/palm_down) 하나뿐이면 사실상 고정 = 기존 동작과 동일 → 앵커 채우기 전까지 안전.
#   기록: 실제 뽑힌 키를 스퀴즈 그룹 attr(present_pose_up/present_pose_down)로 남긴다(재현성).
PRESENT_POSE_RANDOMIZE = True        # False = 끔(항상 base palm_up/palm_down 사용)
PALM_UP_CANDIDATES = ("palm_up", "palm_up_tilt_left", "palm_up_tilt_right",
                      "palm_up_tilt_fwd", "palm_up_tilt_back")
PALM_DOWN_CANDIDATES = ("palm_down", "palm_down_tilt_left", "palm_down_tilt_right",
                        "palm_down_tilt_fwd", "palm_down_tilt_back")

# ── 파지(손) 자세 랜덤화: run 당 pose txt 파일 1개를 후보 중 uniform 랜덤 선택. ─────
#   왜: 같은 물체라도 손가락 파지 형상을 바꿔 학습 일반화. set_pose_for_fruit 이 그 파일로
#     HAND_GRIP_POINT(파지) + HAND_SAVE_POINT(스퀴즈=thumb_3 curl)까지 함께 재계산한다.
#   후보 = launch 디렉터리의 pose txt 파일명. 존재하지 않는 파일은 자동 제외되고, 후보가
#     비거나 랜덤 off 면 그 과일의 기본 pose 를 그대로 쓴다(=기존 동작).
#   기록: 실제 뽑힌 파일명을 스퀴즈 그룹 attr(hand_pose_file)로 남긴다(재현성).
#   ★ 예전에는 '다른 과일 pose 재사용'으로 형상을 다양화했는데(tomato/plum/kiwi/lemon/
#     ecoflex), 2026-07-29 부터 **전용 파지 변형 txt(pose1~5)** 로 교체했다.
#     (initial_pose.txt = 손 펴기 자세이므로 제외 — 넣으면 '벌린 손으로 파지' 가 되어
#      실제 파지가 안 됨.)
#   ★ 파일명에 '_' 를 쓰지 않는다(pose1, e_pose1 ✗). 폴더명 '<개체>_<파지자세>_<ts>' 를
#     '_' 로 토큰 분해해 되읽기 때문에, 자세 이름에 '_' 가 있으면 _pose_tag 가 그걸 '-' 로
#     바꿔야 하고 그러면 '여러 자세를 이은 구분자(-)' 와 뒤섞인다.
GRIP_POSE_RANDOMIZE = True           # False = 끔(과일 기본 pose 고정)
#   ★★ 이 목록은 **끝에만 추가(append-only)** 가 원칙이다. bag_to_session.py 가 이
#     '선언 순서' 로 자세 숫자코드를 매기므로, 순서를 바꾸면 그 이후 세션의 숫자 뜻이
#     이전 세션과 달라진다(실제로 sorted 였을 때 ecoflex.txt 를 추가하자 tomato.txt 가
#     4→5 로 밀렸다).
#   ⚠ 2026-07-29 그 원칙을 **의도적으로 깨고 목록을 통째로 교체했다**(사용자 지시).
#     그래서 hand_pose 코드 1~5 의 뜻이 바뀌었다:
#       구(舊) 1~5 = tomato / plum / kiwi / lemon / ecoflex
#       신(新) 1~5 = pose1 / pose2 / pose3 / pose4 / pose5
#     → **세션을 합칠 때는 숫자가 아니라 session.h5 /runs_names(문자열)로 join 할 것.**
#       (/codes 표도 더는 저장하지 않으므로 숫자만으로는 구·신 구분이 불가능하다)
GRIP_POSE_CANDIDATES = ("pose1.txt", "pose2.txt", "pose3.txt",
                        "pose4.txt", "pose5.txt")

# ── 자세 스케줄: '완전 랜덤' 대신 **조합을 빠짐없이 한 번씩** 돈다 (2026-07-28 사용자 규칙) ──
#   왜 바꿨나: run 마다 uniform 랜덤이면 개체 하나를 다 수집해도 어떤 조합은 여러 번,
#   어떤 조합은 0번이 된다(커버리지 보장 없음). 학습 일반화를 노리는 축(손목 방향 × 파지 형상)
#   이라면 **균등 커버리지**가 낫다.
#
#   규칙:
#     · palm_up  = 5개 중 3개만 쓴다 — 기본(palm_up) **항상** + {tilt_left|tilt_right} 택1
#                  + {tilt_fwd|tilt_back} 택1. (택1 은 개체 세션 시작 시 1회 뽑아 고정)
#     · palm_down = 같은 규칙으로 3개.
#     · 파지 자세 = GRIP_POSE_CANDIDATES **5개 모두** 사용.
#     · 조합 수 = 3 × 3 × 5 = 45. 개체 하나당 이 45개를 전부 완료할 때까지 반복.
#     · 파지 실패(grip_fail)·데모 실패 판정이면 그 조합을 **소진하지 않고 재시도**한다.
#
#   ※ 'tilt 택1 을 세션마다 고정' 인 이유: 매 run 다시 뽑으면 조합 수가 3×3 이 아니라
#     5×5 로 늘어나 45 라는 계획이 성립하지 않는다. 개체별로 좌/우·앞/뒤 중 어느 쪽을
#     쓸지는 랜덤이므로 개체를 여러 개 모으면 5방향이 모두 데이터에 들어온다.
POSE_SCHEDULE = True                 # False = 예전 동작(run 마다 uniform 랜덤)
PALM_UP_BASE, PALM_DOWN_BASE = "palm_up", "palm_down"
#   택1 짝 (좌/우, 앞/뒤). ARM_POSES 에 값이 없는 키는 자동 제외된다.
PALM_UP_PAIRS = (("palm_up_tilt_left", "palm_up_tilt_right"),
                 ("palm_up_tilt_fwd", "palm_up_tilt_back"))
PALM_DOWN_PAIRS = (("palm_down_tilt_left", "palm_down_tilt_right"),
                   ("palm_down_tilt_fwd", "palm_down_tilt_back"))
#   이 판정이 나오면 조합을 소진하지 않고 같은 자세로 다시 시퀀스를 돈다.
#   grip_lost = 스퀴즈 후 파지력 복원 실패(물체가 흘러내렸다) — 그 조합은 데이터를
#   못 얻었으므로 grip_fail 과 같이 재수집 대상이다.
RETRY_OUTCOMES = ("grip_fail", "grip_lost", "discard")

# ── 파지 성공 판정 & 재시도 ──────────────────────────────────────────────
#   왜 필요한가: deploy.move_hand_to_target_until_force 는 내부 settled[] 로 도달을
#   추적하지만 **반환값은 16관절 position 뿐**이고 미도달은 print 경고만 한다. 즉 호출자가
#   파지 성공을 알 방법이 없어, 예전에는 파지가 안 된 상태로 palm-up 으로 넘어갔다.
#   → 여기서 paxini 를 직접 읽어 finger별 normal force(Σ127 Fz)로 독립 판정한다.
#     (deploy 내부 normal_forces() 와 동일 계산 — deploy.py 는 수정하지 않는다)
GRIP_MIN_FINGERS = 2        # 판정 힘 도달 손가락이 이 개수 이상이면 파지 성공
# ★ 판정 기준을 '파지 목표 힘' 과 **분리**한다 (2026-07-28).
#   왜: GRIP_FORCE_RANGE 의 랜덤값은 원래 '얼마나 세게 쥘지'(데이터 다양성) 용인데,
#   그 값을 성공 판정에도 그대로 쓰면 **난수가 성공/실패를 결정**해 버린다. 실측 로그:
#     임계 7.92N 으로 뽑힌 run → 3회 시도 모두 2번째 손가락이 6.4~7.7N 이라 1개만 도달 → 중단.
#     같은 파지인데 임계가 6.0 으로 뽑혔으면 2~3개 도달로 성공이었다.
#   목표 힘은 계속 랜덤화하고(다양성 유지), 판정은 아래 고정값으로 한다.
#   None = 예전 동작(목표 힘으로 판정). 실측 근거: 2번째 손가락 최소 6.4N, 3번째 최소 5.8N.
GRIP_JUDGE_FORCE_N = None   # None = **뽑힌 파지 임계 그대로 판정**(범위 기반, 기본)
#   변경 이력: None → 5.0 고정(07-28) → **None 으로 되돌림**(07-28, 사용자 지시).
#   되돌린 이유: 새 설계가 grasp_threshold 를 기준으로 스퀴즈 임계(=grasp+delta)와
#   해제 검증(=grasp×0.8)까지 만들므로, 판정만 딴 고정값을 쓰면 기준이 두 개가 되어 어긋난다.
#   범위를 6.0~8.0 으로 넓힌 것도 '실측상 충분' 이라는 판단이었다.
#   ※ 다만 알고 있어야 할 것: 임계가 범위 위쪽(8.0)으로 뽑히면 실패할 수 있다 —
#     실측 2번째 손가락 힘이 6.4~7.7N 이었다. 그때는 파지 재시도(GRIP_MAX_RETRY)로 다시
#     시도하고, 그래도 안 되면 run 이 grip_fail 이 되며 **자세 조합은 소진되지 않고 재수집**
#     되므로 데이터에 구멍은 남지 않는다(재시도 시간만 든다).
#     실패가 잦으면 GRIP_FORCE_RANGE 상한을 낮추는 것이 정공법이다(판정만 낮추면 '임계에
#     도달했다' 는 기록과 실제가 어긋난다).
#   변경 이력: 2 → 3 (07-28) → **2 로 되돌림**(07-28, 사용자 요청 — 파지 실패가 잦았다).
#   실측 근거: 접촉 없는 손가락이 늘 하나 이상 있다 — 예 [11.40, 0.00, 16.50, 2.20]N,
#   [9.3, 12.6, 8.0, 0.00]N. 그래서 4 는 비현실적이고, 3 은 물체·자세에 따라 걸린다.
#   2 는 '물체를 두 손가락 이상으로 확실히 눌렀다' 는 최소 조건.
# ── 스퀴즈 후 엄지 위치 복귀 ────────────────────────────────────────────────
#   변경 이력(2026-07-28): '힘 기준 재조임' 으로 만들었다가 **실측에서 물체를 놓쳐** 폐기했다.
#     문제 로그: 재조임 1회차 [5.0 8.1 5.1]N → 2회차 **[0.0 0.0 0.0]N**(=놓침) → 3회차 [5.1 5.3 0]N.
#     원인: move_hand_to_target_until_force(HAND_GRIP_POINT, grip_curl=True) 는 **전 손가락을
#     다시 닫는다.** 이미 물체를 물고 있는 상태에서 그러면 파지 형상이 흐트러져 물체가 빠진다.
#     교훈: 물체를 든 상태에서는 파지 형상을 다시 만들려 하면 안 된다.
#   현재 동작(사용자 지시): **엄지(SQUEEZE_FORCE_FINGERS)만** 스퀴즈 직전 위치로 살짝 되돌린다.
#     · 힘 기준 없음(그래서 물체를 흐트러뜨리지 않는다), 다른 손가락은 손대지 않는다.
#     · 되돌린 뒤 정착 대기 후 힘을 **관측·기록만** 한다(판정에 쓰지 않는다).
#     · 단, 힘이 사실상 0 이면(= 정말 놓친 것) run 을 grip_lost 로 폐기해 쓰레기 데이터를 막는다.
#   ※ deploy.move_hand_to_squeeze 도 내부에서 엄지를 return_position 으로 되돌리지만,
#     변형되는 물체는 그 복귀가 덜 먹을 수 있어 여기서 한 번 더 '살짝' 눌러 자리를 잡아준다.
THUMB_RETURN_AFTER_SQUEEZE = True    # False = 끔(deploy 내부 복귀에만 맡긴다)
THUMB_RETURN_DURATION = 1.0          # 엄지 복귀 이동 시간 [s] (짧으면 급하게 튕긴다)
RELEASE_SETTLE_SEC = (0.3, 0.5)      # 복귀 후 정착 대기 [s]. (min,max) 면 랜덤
GRIP_LOST_FORCE_N = 1.0              # 이 힘조차 못 넘는 손가락이 전부면 '놓쳤다' 로 본다
#   (grasp_threshold 로 판정하지 않는 이유: 스퀴즈 후 힘이 낮아지는 것은 정상이다.
#    여기서 잡아야 하는 것은 '약해짐' 이 아니라 '빠짐' 이므로 아주 낮은 절대값을 쓴다.)

GRIP_SETTLE_SEC = 0.8       # 판정 전 안정화/피크 관측 시간 [s] (단일 노이즈 프레임 방지)
#   0.3 → 0.8 (2026-07-28): 힘이 다 오르기 전에 판정해 3회 다 실패하는 일이 있었다.
#   조건(min_fingers 도달)을 만족하면 즉시 빠지므로 **성공한 파지는 느려지지 않는다.**
GRIP_CLOSE_DURATION = 2.5   # 파지 close 시간 [s]. None = deploy 기본(HAND_MOVE_DURATION=1.5)
#   왜: move_hand_to_target_until_force 는 steps=duration×100tick 으로 닫고,
#     curl(grip_curl=True) 은 steps×HAND_PRESS_TIMEOUT_FACTOR(=2.0) 까지만 더 시도한다.
#     즉 1.5s → 램프 1.5s + curl 최대 3.0s. 2.5s 로 올리면 램프 2.5s + curl 최대 5.0s 가 되어
#     접촉이 늦게 잡히는 물체도 임계까지 갈 시간이 생긴다. (deploy.py 는 수정하지 않고
#     D.HAND_MOVE_DURATION 을 파지 구간에만 임시로 바꿔 끼운다 — 손 펴기도 같이 느려진다.)
GRIP_MAX_RETRY = 6          # 실패 시 재시도 횟수 (매 시도 = 손 펴고 다시 파지, 총 7회 시도)
#   ※ 향후: 시도마다 '다른 파지 자세'로 바꿔 재시도할 예정. _grip_with_retry 의 attempt
#     인덱스에서 자세를 바꿔 끼우면 된다(아래 TODO 참고).

# ── 데모(run) 성공 판정: 매 run 종료 후 터미널에서 사용자에게 물어봄. ──
#   숫자→라벨. 판정은 HDF5 그룹 attr(outcome) + /collect/demo_outcome 토픽(bag) +
#   세션 폴더의 outcomes.json(sidecar)에 저장 → 학습 시 outcome 으로 필터(제외/분리).
#   ★ 키는 **끝에만 추가**(append-only) — 기존 세션의 판정 숫자 뜻이 밀리지 않게.
OUTCOMES = {"1": "success", "2": "grip_fail", "3": "not_judged", "4": "discard",
            "5": "grip_lost"}
OUTCOME_DEFAULT = "success"          # Enter 만 치면 이 값(단, 아래 '제안' 이 있으면 그것)

DEFAULT_BAG_TOPICS = [
    "/hand/right/joint_states", "/hand/right/kin",
    "/paxini/right/ft", "/paxini/right/raw",
    "/franka/right/joint_states", "/franka/right/q_target",
    "/hand/right/q_target", "/hand/right/cmd_servo", "/hand/right/cmd_mode",
    "/collect/demo_marker", "/collect/squeeze_on", "/collect/segment",
    "/collect/demo_outcome",
]
PAUSE_BETWEEN_RUNS_SEC = 0.5


class MarkerPublisher(Node):
    """구간 라벨 / 데모 경계 / 스퀴즈 flag 를 토픽으로 발행 → rosbag 에 담겨 분할·전처리에 쓰인다.
    (기존 /tmp squeeze flag 대체.) 이제 ★ 구간 '각각' /collect/segment 라벨로 구분."""

    def __init__(self):
        super().__init__("collect_marker_pub")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self._demo = self.create_publisher(String, "/collect/demo_marker", qos)
        self._sq = self.create_publisher(Int8, "/collect/squeeze_on", qos)
        self._seg = self.create_publisher(String, "/collect/segment", qos)
        self._out = self.create_publisher(String, "/collect/demo_outcome", qos)
        self._last_sq = None
        self._last_seg = None

    def wait_for_recorder(self, topics, timeout: float = 15.0) -> bool:
        """기록 대상 /collect/* 토픽에 구독자(=ros2 bag record)가 붙을 때까지 대기.

        ★ 왜 필요한가: 마커 퍼블리셔는 durability=VOLATILE 이라, record 가 구독을 붙이기
          전에 발행한 메시지는 bag 에 남지 않는다. 예전에는 고정 `time.sleep(1.0)` 이었는데
          실측(collect_tomato_20260728_160837)에서 부족했다 —— run 시작의 'S' 마커와 첫
          segment 라벨('safe')이 통째로 유실됐고, 그러면 bag_to_hdf5 의 run_of() 가 -1 을
          돌려 **outcome 태깅과 --skip-outcomes 가 조용히 무력화**된다(실패 run 이 학습
          데이터에 그대로 남는다). 그래서 시간이 아니라 구독자 수를 직접 확인한다.
        """
        want = set(topics)
        pubs = [(t, p) for t, p in (("/collect/demo_marker", self._demo),
                                    ("/collect/squeeze_on", self._sq),
                                    ("/collect/segment", self._seg),
                                    ("/collect/demo_outcome", self._out))
                if t in want]
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            missing = [t for t, p in pubs if p.get_subscription_count() == 0]
            if not missing:
                print(f"  [bag] record 구독 확인 ({len(pubs)}개 마커 토픽, "
                      f"{time.monotonic() - t0:.1f}s)")
                return True
            time.sleep(0.1)
        print(f"  ⚠ [bag] record 구독 미확인({timeout}s) — 미구독: {missing}. "
              "run 시작 마커가 bag 에 안 남아 outcome 태깅이 안 될 수 있음.")
        return False

    def pub_demo(self, event: str, run_id: int, t_ns: int):
        self._demo.publish(String(data=f"{event},{run_id},{t_ns}"))

    def pub_outcome(self, run_id: int, outcome: str, t_ns: int):
        # 형식: "run_id,outcome,t_mono_ns". bag_to_hdf5 가 읽어 run 별 outcome attr 로 태깅.
        self._out.publish(String(data=f"{run_id},{outcome},{t_ns}"))

    def pub_squeeze(self, on: int):
        v = int(on)
        if v != self._last_sq:
            self._sq.publish(Int8(data=v)); self._last_sq = v

    def pub_segment(self, label: str):
        # 구간 진입 시 라벨, 이탈 시 "" (idle). 변할 때만 발행.
        if label != self._last_seg:
            self._seg.publish(String(data=label)); self._last_seg = label


@contextmanager
def _segment(marker: MarkerPublisher, label: str):
    """구간 라벨을 진입/이탈에 발행. rosbag 에 그 구간이 라벨로 남는다."""
    marker.pub_segment(label)
    try:
        yield
    finally:
        marker.pub_segment("")


# ── 콘솔 출력 ────────────────────────────────────────────────────────────────
#   터미널이 rosbag2/MoveIt INFO 와 deploy.py 배너로 뒤덮여 진행 상황이 안 보였다.
#   기본은 '9단계 진행 표시'만 깔끔하게 띄우고, 원본 로그는 --verbose 로 되살린다.
#   (억제되는 것들: rosbag2 record 출력 → 세션 폴더 bag_record.log,
#    MoveIt/브리지 노드 INFO → 로거 레벨 WARN, deploy.py 배너 → 아래 _NOISE 필터)
VERBOSE = False

#   deploy.py 는 수정 금지 대상이라 그 stdout 을 줄 단위로 걸러낸다. 여기 없는 문장은
#   그대로 보이므로 예상 못한 메시지가 묻히지 않는다.
_NOISE = ("=================== ", "[move_hand_to_target_until_force]",
          "[pose] 과일 포즈 적용", "[squeeze] ")

STEPS_TOTAL = 8          # 진행 표시용 단계 수 (= phase 코드 1~8 과 같은 순서)
#   palm_up→손펴기→파지→스퀴즈A→엄지복귀A→palm_down→스퀴즈B→엄지복귀B


def _pad(text: str, width: int) -> str:
    """터미널 표시폭 기준 왼쪽 정렬. 한글·★ 은 2칸을 차지하므로 f"{s:<18}" 로는 열이 어긋난다."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


class _Console:
    """sys.stdout 래퍼 — 상용구 필터 + 한 줄 진행 표시.

    진행 줄은 '   3/9  파지            ' 까지 먼저 쓰고(개행 없이) 완료 시 결과를 이어 쓴다.
    그 사이에 다른 출력(경고 등)이 끼면 줄이 깨지므로, 열린 진행 줄이 있으면 개행을 먼저 넣는다.
    """

    def __init__(self, real):
        self._real, self._buf, self._open = real, "", False
        # ★ 락 필수: ROS 로거는 **executor spin 스레드**에서 stdout 에 쓰고, 진행 줄은 메인
        #   스레드가 쓴다. 락 없이 self._buf 를 동시에 고치면 출력이 깨지고 최악에는
        #   콜백 안에서 예외가 나 spin 스레드가 죽는다('Exception in thread Thread-1 (spin)').
        self._lock = threading.RLock()

    def write(self, s):
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit(line + "\n")
        return len(s)

    def _emit(self, line):                  # 호출자가 락을 쥔 상태로만 호출된다
        if not VERBOSE and any(k in line for k in _NOISE):
            return
        if self._open:                      # 진행 줄이 열려 있으면 줄바꿈 후 출력
            self._real.write("\n")
            self._open = False
        self._real.write(line)

    def flush(self):
        with self._lock:
            self._real.flush()

    def step_start(self, n, title):
        with self._lock:
            self._real.write(f"   {n}/{STEPS_TOTAL}  {_pad(title, 20)}")
            self._real.flush()
            self._open = True

    def step_end(self, note):
        with self._lock:
            if not self._open:              # 중간 출력으로 줄이 끊겼으면 들여써서 이어준다
                self._real.write(" " * 8)
            self._real.write(note + "\n")
            self._real.flush()
            self._open = False

    def __getattr__(self, k):               # isatty/encoding 등은 원본에 위임
        return getattr(self._real, k)


def _con() -> _Console:
    """설치된 _Console (미설치면 표준 stdout 을 감싼 임시 객체)."""
    return sys.stdout if isinstance(sys.stdout, _Console) else _Console(sys.stdout)


@contextmanager
def _step(marker, label, n, title):
    """진행 표시 + 구간 라벨 발행을 한 번에. `with _step(...) as st:` 후 st["note"] 로 결과 지정."""
    con = _con()
    st = {"note": ""}
    con.step_start(n, title)
    marker.pub_segment(label)
    t0 = time.monotonic()
    try:
        yield st
    except BaseException as e:              # 예외 종류·메시지를 그 자리에 보여준다(진단용)
        con.step_end(f"✘ {type(e).__name__}: {e}"[:140])
        raise
    else:
        el = time.monotonic() - t0
        con.step_end(f"{st.get('mark', '✔')} {st['note'] or f'{el:.1f}s'}")
    finally:
        marker.pub_segment("")


class _NullWriter:
    """라이브 HDF5 를 만들지 않을 때 쓰는 대체 writer (파일을 안 만든다).

    ★ 왜 라이브 h5 를 폐지했나: bag 기반 `session.h5`(연속 타임라인)가 표준이 된 뒤로
      라이브 h5 는 스퀴즈 A/B 만 담는 **중복**이었다. 둘을 다 둔 이유는 서로 검증(parity)
      이었고 그건 끝났다(todolist 1·9·11번).
    ★ 그런데 RecordingEngine 은 계속 필요하다 — 스퀴즈 첫 유효 프레임에
      `/collect/squeeze_on=1` 을 발행하는 콜백이 거기 있고, 그 마커가 bag→session.h5 의
      squeeze_on 열이 된다. 그래서 engine 은 살리고 writer 만 껍데기로 바꾼다.
      진행 표시에 쓰는 프레임 수는 여기서 세므로 화면 출력도 그대로 유지된다.
    """

    def __init__(self):
        self.n_demos, self._n = 0, 0

    def start_demo(self, demo_id, attrs=None, name=None):
        self._n = 0

    def append(self, **_kw):
        self._n += 1

    def end_demo(self, attrs=None) -> int:
        self.n_demos += 1
        return self._n

    def set_group_attr(self, name, key, value):
        pass

    def close(self):
        pass


def _resolve_pose_file(fruit: str) -> str:
    """물체 이름 → 파지 자세 txt 파일명.

    ★ FRUIT_CONFIG(추론 엔진) 에 등록된 물체는 그 `pose` 를 쓰고, **미등록 물체(학습 전
      새 물체: ecoflex 등)는 '<이름>.txt' 가 자세 디렉터리에 있으면 그대로 수집한다.**
      수집에 필요한 건 파지 자세뿐이고 모델은 배포에서만 필요하다. 임계값도
      D._threshold_for 가 미등록 이름에 기본값(파지 7.0N / 스퀴즈 10.0N)을 주므로 안전하다.
      → 물체를 새로 추가할 때 추론 엔진을 건드리지 않아도 된다.
    """
    cfg = RE.FRUIT_CONFIG.get(fruit)
    if cfg is not None:
        pose = cfg.get("pose")
        if pose is None:
            raise SystemExit(f"'{fruit}' 파지 포즈(pose) 미설정(FRUIT_CONFIG).")
        return pose
    pose = f"{fruit}.txt"
    if not (D._POSE_DIR / pose).exists():
        raise SystemExit(
            f"'{fruit}' 은 FRUIT_CONFIG 미등록이고 자세 파일도 없다: {D._POSE_DIR / pose}\n"
            f"  → 그 이름의 pose txt 를 만들거나 등록된 이름을 쓰세요"
            f" (등록: {list(RE.FRUIT_CONFIG)}).")
    print(f"[collect] ℹ '{fruit}' 은 FRUIT_CONFIG 미등록(=미학습) 물체 — 자세 {pose} 로 "
          "수집만 진행(배포/추론은 불가). 임계값은 기본값 사용.")
    return pose


def _draw(rng, fallback) -> float:
    """(min,max) 범위면 그 안에서 uniform 랜덤, None 이면 fallback(과일 고정값)."""
    return round(random.uniform(rng[0], rng[1]), 3) if rng else float(fallback)


_TTY = None                  # (읽기, 쓰기) 제어 터미널 핸들. 첫 사용 시 1회 open.


def _open_tty():
    """제어 터미널(/dev/tty) 핸들 (read, write). 없으면 None.

    ★ 읽기·쓰기를 **따로** 연다: `open("/dev/tty", "r+")` 는 BufferedRandom 을 만들어
      seek 가능성을 요구하므로 터미널에서 OSError("File or stream is not seekable") 로 실패한다
      (실측). "r" / "w" 단방향은 각각 BufferedReader/Writer 라 정상이다.
    """
    global _TTY
    if _TTY is None:
        try:
            _TTY = (open("/dev/tty", "r"), open("/dev/tty", "w"))
        except OSError:
            _TTY = False                        # 재시도 안 하도록 표시
    return _TTY or None


def _prompt(text: str):
    """프롬프트를 띄우고 한 줄 입력받아 반환. 입력 수단이 아예 없으면 None.

    ★ stdin 이 대화형이 아니면(파이프·리다이렉트·백그라운드·IDE 태스크) input() 은 **기다리지
      않고 즉시 EOFError** 를 낸다. 예전에는 그 예외가 main() 을 빠져나가 판정·outcomes.json·
      group attr·bag 의 demo_outcome 이 통째로 유실됐다(세션 195101: E 마커 후 0.21초에 종료,
      사용자가 누른 Enter 는 셸로 갔다).
      그래서 stdin 이 막혀 있어도 **제어 터미널에서 직접 읽어 계속 기다린다** — 파이프로
      실행해도 판정을 놓치지 않는다. (EOF 를 반복 호출로 버티려 하면 무한 루프가 된다.)
      프롬프트도 같은 터미널에 쓰므로 stdout 이 파이프여도 화면에 보인다.
    """
    asked = False
    if sys.stdin.isatty():
        try:
            return input(text)
        except EOFError:
            # stdin 이 tty 인데도 EOF — 다른 프로세스가 터미널 입력을 가로챈 경우다
            # (실측: `ros2 bag record` 가 SPACE 단축키용으로 키보드를 읽어갔다. 그건
            #  start_bag 의 stdin=DEVNULL 로 막았지만, 다른 원인일 수도 있으니 포기하지
            #  않고 아래 /dev/tty 로 한 번 더 시도한다.)
            asked = True
    tty = _open_tty()
    if tty is None:
        return None
    rd, wr = tty
    try:
        if asked:                               # input() 이 이미 찍은 프롬프트를 다시 안내
            wr.write("\n  (터미널에서 직접 입력) ")
        wr.write(text)
        wr.flush()
        line = rd.readline()
    except OSError:
        return None
    return None if line == "" else line         # tty 마저 EOF 면 포기


def _ask_outcome(run_id: int, *, remaining: int = 0, suggest: str | None = None,
                 why: str = ""):
    """run 종료 후 터미널에서 성공 여부 + 계속할지를 물어봄.

    반환: (outcome, stop) — stop=True 면 이 run 을 기록한 뒤 수집을 끝낸다.
      숫자 뒤에 `q` 를 붙이면 종료: `1q`(성공으로 기록하고 종료) / `q`(기본값으로 기록하고 종료).
      → 남은 run 을 포기할 때 Ctrl-C 를 쓰지 않아도 되므로 **정상 경로로 깔끔히 마감**된다
        (Ctrl-C 도 데이터는 보존되지만 그건 비상용이다).

    suggest: 코드가 의심하는 판정(예: 엄지 복귀 B 에서 힘≈0 → 'grip_lost').
      **자동 확정하지 않고 Enter 기본값으로만** 쓴다 — 시퀀스가 끝까지 돌아 데이터가 다
      남은 run 을 사람 확인 없이 폐기하면 안 된다(그러면 판정 없이 다음 run 이 시작된다).
      why 는 그 제안의 근거를 한 줄로 보여주기 위한 것.

    입력이 올 때까지 **기다린다**(stdin 이 끊겨 있어도 /dev/tty 로). 터미널이 아예 없으면
    그때만 'not_judged' 로 기록하고 계속한다 — 판정을 못 받아도 기록은 유실되지 않게.
    """
    opts = "  ".join(f"[{k}]{v}" for k, v in OUTCOMES.items())
    tail = f"  (뒤에 q=기록 후 종료, 남은 run {remaining}개)" if remaining > 0 else "  (q=종료)"
    default = suggest or OUTCOME_DEFAULT
    if suggest:
        print(f"\n  ⚠ 코드 의심: {suggest}"
              + (f" — {why}" if why else "")
              + f"\n     스퀴즈 A·B 는 이미 기록돼 있습니다. 폐기할지 살릴지 확인해 주세요"
                f" (Enter = {suggest}).")
    while True:
        try:
            line = _prompt(f"\n[판정] run {run_id} 결과?  {opts}  "
                           f"[Enter={default}]{tail} : ")
        except KeyboardInterrupt:
            # 프롬프트에서 Ctrl-C — 예외를 올려보내면 이 run 의 판정·outcomes.json 이 빠진다.
            print("\n  ⚠ Ctrl-C — outcome='interrupted' 로 기록하고 종료합니다.")
            return "interrupted", True
        if line is None:
            # '터미널 없음' 과 '터미널은 있는데 입력이 EOF(다른 프로세스가 가로챔)' 를 구분해
            # 알린다 — 예전 메시지는 후자에도 '터미널이 없다' 고 해서 오진을 유도했다.
            why_tty = ("제어 터미널(/dev/tty)이 없음" if _open_tty() is None else
                       "터미널은 있으나 입력이 EOF — 다른 프로세스가 키보드를 가로챘을 수 있음")
            # 터미널이 없으면 사람 판정은 불가 → 코드 제안이 있으면 그걸 쓴다(없으면 not_judged).
            #   제안(grip_lost)이 not_judged 보다 정보량이 많고, 재수집 대상이라 구멍도 안 남는다.
            fallback = suggest or "not_judged"
            print(f"\n  ⚠ 판정 입력을 받을 수 없음({why_tty}) → outcome={fallback!r} 로 기록하고 "
                  "계속. (--no-judge 로 명시하거나 대화형 터미널에서 실행하세요)")
            return fallback, False
        c = line.strip().lower()
        stop = c.endswith("q")
        if stop:
            c = c[:-1].strip()
        if c == "":
            return default, stop
        if c in OUTCOMES:
            return OUTCOMES[c], stop
        print(f"  {'/'.join(OUTCOMES)} 중 하나 또는 Enter"
              " (종료하려면 뒤에 q — 예: 1q, 2q, q).")


def _write_outcomes(session_dir, fruit, outcomes: dict, session_attrs: dict | None = None,
                    specimen: str | None = None):
    """세션 폴더에 outcomes.json 갱신(증분) — bag 옆에 사람이 읽는 판정 기록.

    ★ session 블록(수집 시점 provenance) 도 함께 남긴다. 예전에는 이 정보가 **라이브 h5 의
      root attr 에만** 있었는데, 라이브 h5 를 폐지하면 git_sha·FACTOR·USE_JKIN·
      t_offset_ns·kin 소스 같은 재현 정보가 통째로 사라진다. bag_to_session 이 이 블록을
      읽어 session.h5 root attr 로 옮긴다(provenance 체인 유지).
    """
    # fruit = 물체 종류(자세·임계 조회 키) / specimen = 개체 이름(폴더·라벨). 둘 다 남긴다 —
    #   같은 재료의 개체가 여러 개면 일반화 평가를 개체 단위로 나눠야 한다.
    payload = {"fruit": fruit, "specimen": specimen or fruit}
    if session_attrs:
        payload["session"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                              for k, v in session_attrs.items()}
    payload["runs"] = outcomes
    (session_dir / "outcomes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _finger_normal_forces(paxini):
    """finger별 접촉 normal force(127점 Fz 합) [N]. 무효 프레임이면 0.
    deploy.move_hand_to_target_until_force 내부 normal_forces() 와 동일 계산."""
    tactile, _t, valid, _seq = paxini.read()
    if not int(valid):
        return np.zeros(D.Finger_Num, dtype=np.float32)
    return np.nan_to_num(tactile, nan=0.0)[:, :, 2].sum(axis=1)


def _peak_forces(paxini, settle_sec: float = GRIP_SETTLE_SEC, *,
                 threshold: float | None = None, min_fingers: int | None = None):
    """settle_sec 동안 관측한 finger별 **최대** normal force. 단일 노이즈 프레임 방지.

    threshold·min_fingers 를 주면 조건을 만족한 순간 **즉시 반환** → 성공한 파지는
    창을 다 기다리지 않는다(관측창을 늘려도 정상 파지는 느려지지 않는다).

    반환: (peak, still_rising).
      still_rising=True = 창이 끝날 때까지 힘이 계속 오르고 있었다는 뜻 →
      GRIP_SETTLE_SEC / GRIP_CLOSE_DURATION 을 더 늘릴 여지가 있다(진단 힌트).
    """
    peak = np.zeros(D.Finger_Num, dtype=np.float32)
    t0, dt = time.monotonic(), 1.0 / D.CONTROL_RATE_HZ
    half, peak_half = settle_sec * 0.5, None
    while True:
        el = time.monotonic() - t0
        if el >= settle_sec:
            break
        peak = np.maximum(peak, _finger_normal_forces(paxini))
        if peak_half is None and el >= half:
            peak_half = peak.copy()
        if (threshold is not None and min_fingers is not None
                and int((peak >= threshold).sum()) >= min_fingers):
            return peak, False                      # 조건 충족 → 더 기다릴 이유 없음
        time.sleep(dt)
    rising = bool(peak_half is not None and (peak.max() - peak_half.max()) > 0.1)
    return peak, rising


def _thumb_return(bridge, paxini, grip_position):
    """스퀴즈 후 **엄지만** 스퀴즈 직전(파지) 위치로 되돌린다. 위 THUMB_RETURN_AFTER_SQUEEZE 참고.

    힘 기준으로 조이지 않는다 — 물체를 든 상태에서 파지 형상을 다시 만들면 놓친다(실측).
    다른 손가락은 **직전 명령값(grip_position)으로 고정**해 건드리지 않는다.
    ⚠ 예전처럼 '읽은 실측값'을 나머지 손가락에 재명령하면 안 된다 — 쥔 손가락은
      실측<명령이라 그 재명령은 서보 목표를 후퇴시켜 손가락을 푼다('그대로 두기'는
      명령값 유지이지 실측값 명령이 아니다). 이게 스퀴즈 B 때 다른 손가락이 다시
      확 조여지던 원인(스퀴즈 pre-wait 가 grip_position 을 재명령하므로).

    반환: (held, peak, n_hold)
      held   = 물체를 아직 들고 있나(힘이 GRIP_LOST_FORCE_N 조차 안 되면 False = 놓침)
      n_hold = GRIP_LOST_FORCE_N 을 넘는 손가락 수(참고용 기록)
    """
    # 엄지 관절만 실측 위치→파지 위치로 보간(변형 물체에 '살짝' 재밀착), 나머지는 고정.
    D.move_fingers_to(bridge, grip_position, D.SQUEEZE_FORCE_FINGERS,
                      THUMB_RETURN_DURATION, hold_position=grip_position)

    settle = (random.uniform(*RELEASE_SETTLE_SEC)
              if isinstance(RELEASE_SETTLE_SEC, (tuple, list)) else float(RELEASE_SETTLE_SEC))
    peak, _rising = _peak_forces(paxini, settle)             # 관측만(조건 없음)
    n_hold = int((peak >= GRIP_LOST_FORCE_N).sum())
    return n_hold > 0, peak, n_hold


def _grip_close(bridge, paxini):
    """손을 **현재 위치에서** 파지 자세까지 닫는다(앞의 '손 펴기' 없음).

    deploy_ros2._grip 은 '손 펴기(HAND_SAFE_POSITION) + 파지' 를 한 함수로 묶어 놨다.
    palm-up 재파지는 펴기를 **별도 시퀀스 단계**로 분리해야 해서(구간 라벨을 따로 남기려면
    한 단계 안에 두 동작이 들어가면 안 된다) 그 후반부만 떼어 쓴다.
    임계는 _grip 과 똑같이 전역 D.GRIP_FORCE_THRESHOLD(=run 당 뽑힌 grip_thr)를 쓴다.
    """
    return D.move_hand_to_target_until_force(
        bridge, paxini, D.HAND_GRIP_POINT, D.HAND_MOVE_DURATION,
        D.GRIP_FORCE_THRESHOLD, grip_curl=True)


def _grip_with_retry(bridge, paxini, *, threshold: float,
                     judge_force: float | None = GRIP_JUDGE_FORCE_N,
                     min_fingers: int = GRIP_MIN_FINGERS,
                     max_retry: int = GRIP_MAX_RETRY,
                     open_first: bool = True):
    """파지 → 판정 힘 도달 손가락 수 판정 → 실패면 손 펴고 재시도.

    open_first=False = '손은 이미 펴져 있다' — 닫기만 한다(palm-up 재파지 단계용).
      재시도 경로는 실패할 때마다 아래에서 손을 펴므로, 그 다음 시도도 닫기만 하면 된다
      (True 로 두면 매 시도 앞에 펴기가 한 번 더 들어가 1.5s 씩 헛돈다).

    threshold  = **파지 목표 힘**(랜덤화 대상). 각 손가락은 이 힘에 도달하면 그 자리에 멈춘다.
    judge_force = **성공 판정 힘**(고정). None 이면 threshold 로 판정(예전 동작).
      두 값을 나눈 이유: 목표 힘을 랜덤화하면 그 난수가 성공/실패까지 결정해 버린다.
      실측 — 목표 7.92N 로 뽑힌 run 은 3회 모두 2번째 손가락이 6.4~7.7N 이라 1개만 도달해
      중단됐는데, 같은 파지가 목표 6.0N 이었으면 성공이었다. 판정은 하드웨어가 실제로
      낼 수 있는 힘 기준으로 고정해야 한다.

    성공 조건: normal force >= judge_force 인 손가락이 min_fingers 개 이상.
      (4개 전부를 요구하면 비현실적 — 실측에서 접촉 없는 손가락이 늘 존재한다:
       예 [11.40, 0.00, 16.50, 2.20]N. 그래서 '개수 기준'을 쓴다.)

    반환: (grip_position, ok, peak_forces, n_reached)
    """
    judge = float(threshold if judge_force is None else judge_force)
    grip_position, peak, n_ok = None, np.zeros(D.Finger_Num, np.float32), 0
    for attempt in range(max_retry + 1):
        # 파지 자세는 이미 run 단위로 랜덤화됨(GRIP_POSE_CANDIDATES → _run_sequence 상단에서
        #   set_pose_for_fruit 적용). 여기 재시도는 '같은 run 자세'를 유지한다(판정 일관성).
        #   TODO(선택): attempt 별로도 자세를 바꾸려면 여기서 set_pose_for_fruit 을 다시 호출.
        # close 시간을 파지 구간에만 늘려 끼운다(deploy.py 는 건드리지 않음). finally 로 복원.
        _saved_dur = D.HAND_MOVE_DURATION
        if GRIP_CLOSE_DURATION:
            D.HAND_MOVE_DURATION = float(GRIP_CLOSE_DURATION)
        try:
            # open_first=True  : 손 펴기 → 파지력까지 close (기존 3단계 파지)
            # open_first=False : 이미 펴져 있으므로 close 만 (palm-up 재파지 단계)
            grip_position = (_grip(bridge, paxini) if open_first
                             else _grip_close(bridge, paxini))
        finally:
            D.HAND_MOVE_DURATION = _saved_dur
        peak, rising = _peak_forces(paxini, threshold=judge, min_fingers=min_fingers)
        n_ok = int((peak >= judge).sum())
        tag = (f"{n_ok}/{D.Finger_Num}개 도달  [{' '.join(f'{v:.1f}' for v in peak)}] N  "
               f"(판정 {judge:.2f}N / 목표 {threshold:.2f}N)")
        if n_ok >= min_fingers:
            if VERBOSE:                     # 기본 출력에서는 진행 표시의 3/9 줄이 같은 내용을 보여준다
                print(f"[collect]   파지 성공: {tag}")
            return grip_position, True, peak, n_ok
        left = max_retry - attempt
        print(f"   ⚠ 파지 시도 {attempt + 1}/{max_retry + 1}: {tag} — {min_fingers}개 미만"
              + (" → 손 펴고 재시도" if left > 0 else " → 재시도 소진")
              # 창이 끝날 때까지 힘이 오르고 있었다면 '시간 부족' 이지 '임계 과다' 가 아니다.
              + ("  [힘이 계속 상승 중 — GRIP_SETTLE_SEC/GRIP_CLOSE_DURATION 을 더 늘려볼 것]"
                 if rising else ""))
        if left > 0:
            D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)   # 손 펴기
    return grip_position, False, peak, n_ok


def _pose_is_set(pose_key: str) -> bool:
    """ARM_POSES[pose_key] 에 실제 좌표(joints 또는 position)가 채워져 있으면 True."""
    p = ARM_POSES.get(pose_key) or {}
    return p.get("joints") is not None or p.get("position") is not None


def _draw_present(candidates, base_key: str) -> str:
    """스퀴즈 제시 자세를 후보 중 '값이 채워진' 것만 대상으로 uniform 랜덤 선택.
    랜덤 off 이거나 채워진 변형이 없으면 base_key(=기존 동작)로 폴백 → 앵커 미입력 시 안전."""
    if not PRESENT_POSE_RANDOMIZE:
        return base_key
    pool = [k for k in candidates if _pose_is_set(k)]
    return random.choice(pool) if pool else base_key


def _draw_grip_pose(candidates, fallback: str) -> str:
    """파지 손 자세 txt 를 '실제 존재하는' 후보 중 uniform 랜덤 선택.
    랜덤 off 이거나 존재하는 후보가 없으면 fallback(과일 기본 pose)로 폴백 → 안전."""
    if not GRIP_POSE_RANDOMIZE:
        return fallback
    pool = [n for n in candidates if (D._POSE_DIR / n).exists()]
    return random.choice(pool) if pool else fallback


def _ask_specimen(fruit: str, given: str | None) -> str:
    """개체 이름을 정한다. 폴더·파일명이 `collect_<specimen>_<ts>` 가 된다.

    입력 형태 셋 다 받는다 (예: fruit='ecoflex'):
      · `ecoflex_1` → 그대로 사용            · `1` / `3` → `ecoflex_1` 로 조합
      · 빈 줄       → fruit 그대로(`ecoflex`)
    --specimen 으로 주면 묻지 않는다(스크립트·반복 실행용).
    ※ fruit 는 자세·임계 조회 키라 바뀌지 않는다 — specimen 은 **이름·라벨 전용**이다.
    """
    raw = (given or "").strip()
    if not raw:
        line = _prompt(f"\n[개체] 번호 또는 이름?  [Enter={fruit}] "
                       f"(예: 1 → {fruit}_1) : ")
        raw = (line or "").strip()
    if not raw:
        return fruit
    if raw.isdigit():
        return f"{fruit}_{raw}"
    # 파일·폴더명에 안전한 문자만 남긴다(공백·경로 구분자 사고 방지).
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in raw)
    if safe != raw:
        print(f"   ℹ 개체 이름을 '{safe}' 로 정규화(파일명 안전 문자만).")
    return safe


def _pick_present_set(base: str, pairs, label: str) -> list:
    """제시 자세 3개 세트: 기본 + 각 짝에서 랜덤 택1. (짝의 두 자세가 다 비어 있으면 그 짝은 생략)

    ARM_POSES 에 값이 안 채워진 변형은 후보에서 빠지므로, 앵커를 아직 안 잡았어도 안전하게
    (기본 1개만으로) 동작한다 — 그러면 조합 수가 그만큼 줄어든다.
    """
    out = [base] if _pose_is_set(base) else []
    for pair in pairs:
        pool = [k for k in pair if _pose_is_set(k)]
        if pool:
            out.append(random.choice(pool))
    if not out:                              # 전부 미설정 → base 로 폴백(이동은 생략됨)
        out = [base]
    print(f"   {label} 자세 {len(out)}개: {', '.join(out)}")
    return out


def _pose_tag(grips) -> str:
    """이 세션이 쓴 파지 자세를 폴더·파일명에 넣을 짧은 토큰으로.

    폴더명이 '<개체>_<파지자세>_<타임스탬프>' 가 되도록 하려는 것(사용자 요청 2026-07-29):
      ecoflex_1 + ecoflex.txt  →  ecoflex_1_ecoflex_20260729_000753

    규칙:
      · '.txt' 를 떼고 stem 만 쓴다. 2~3개면 '-' 로 잇는다(tomato-plum).
      · 4개 이상이면 이름이 길어지기만 하니 개수로 줄인다('5pose'). 실제 자세는
        session.h5 /runs_names 와 outcomes.json 에 run 단위로 남으므로
        폴더명은 '한눈 라벨' 이면 된다 — 폴더명이 유일한 출처가 아니다.
      · stem 안의 '_' 는 '-' 로 바꾼다. 폴더명을 '_' 로 토큰 분해해서 되읽는
        곳(bag_to_session._guess_names)이 있어서, 태그는 **항상 1토큰**이어야 한다.
    """
    stems = []
    for n in grips:
        s = Path(n).stem.replace("_", "-")
        if s and s not in stems:
            stems.append(s)
    if not stems:
        return "nopose"
    return "-".join(stems) if len(stems) <= 3 else f"{len(stems)}pose"


def _build_schedule(grip_poses, *, limit: int | None = None):
    """이 개체(specimen)에서 돌 조합 목록. 반환: (up_set, down_set, grips, combos).

    combos = [(present_up, present_down, hand_pose), ...] 를 셔플한 것.
      · 셔플 이유: 순서대로 돌면 앞쪽 조합만 모인 채 중단될 때 특정 자세에 편향된다.
        셔플해도 **전부 한 번씩** 도는 것은 그대로 보장된다.
      · limit: --num-demos 로 조합 수를 줄일 때(테스트용). None = 전부.
    """
    up = _pick_present_set(PALM_UP_BASE, PALM_UP_PAIRS, "palm-up")
    down = _pick_present_set(PALM_DOWN_BASE, PALM_DOWN_PAIRS, "palm-down")
    grips = [n for n in grip_poses if (D._POSE_DIR / n).exists()]
    if not grips:
        grips = [Path(D.HAND_GRIP_POINT_FILE).name]
    combos = [(u, d, g) for u in up for d in down for g in grips]
    random.shuffle(combos)
    if limit is not None and limit < len(combos):
        combos = combos[:limit]
    print(f"   파지 자세 {len(grips)}개: {', '.join(grips)}")
    print(f"   → 조합 {len(up)}×{len(down)}×{len(grips)} = {len(up) * len(down) * len(grips)}개"
          + (f" 중 {len(combos)}개만 수집(--num-demos)" if len(combos) != len(up) * len(down) * len(grips)
             else " 전부 수집"))
    return up, down, grips, combos


def _arm_move(mover: MoveItArmMover, bridge, pose_key: str) -> bool:
    """팔을 ARM_POSES[pose_key] 로 이동(MoveIt 충돌회피). joints 우선, 없으면 position, 둘 다 없으면 생략."""
    p = ARM_POSES.get(pose_key) or {}
    if p.get("joints") is not None:
        ok = mover.move_to_joints(bridge, p["joints"])
    elif p.get("position") is not None:
        ok = mover.move_to_pose(bridge, p["position"], p.get("orientation"))
    else:
        print(f"[collect]   ⚠ 팔 포즈 '{pose_key}' 미설정 — 이동 생략(placeholder). "
              "capture_pose.py 로 ARM_POSES 채우기.")
        return False
    if not ok:
        print(f"[collect]   ⚠ '{pose_key}' 이동 실패(플랜/재생) — 시퀀스 계속.")
    return ok


def _record_squeeze(marker, writer, rec, bridge, paxini, grip_position, *,
                    label: str, seg_id: int, squeeze_threshold: float, attrs: dict) -> str:
    """스퀴즈 ★: 구간 라벨 + HDF5 그룹 열고, 배포와 동일한 손 스퀴즈(engine=rec 로 프레임 기록).
       squeeze_threshold = 이번 스퀴즈에 쓸 힘 임계(랜덤값). 그룹 attr 로 기록."""
    marker.pub_segment(label)
    # 그룹 이름 = bag_to_hdf5 와 동일 규칙 '{segment}__run{NNN}' (run+구간이 이름에 드러남).
    grp_name = f"{label}__run{int(attrs.get('run', 0)):03d}"
    writer.start_demo(seg_id, name=grp_name,
                      attrs={**attrs, "segment": label,
                             "squeeze_force_threshold_n": float(squeeze_threshold)})
    try:
        D.move_hand_to_squeeze(
            bridge, paxini, D.HAND_SAVE_POINT, D.HAND_SQUEEZE_DURATION,
            squeeze_threshold, grip_position,
            return_duration=D.HAND_SQUEEZE_RETURN_DURATION, engine=rec)
    finally:
        n = writer.end_demo()
        marker.pub_squeeze(0)
        marker.pub_segment("")
    return grp_name, n


def _squeeze_threshold(grasp_thr: float):
    """스퀴즈 임계 = 파지 임계 + delta. 반환 (threshold, delta).

    SQUEEZE_FORCE_RANGE 에 값이 있으면 예전처럼 절대값으로 뽑는다(그때 delta 는 차이값).
    """
    if SQUEEZE_FORCE_RANGE:                        # 구 동작(되돌림용)
        thr = _draw(SQUEEZE_FORCE_RANGE, D.SQUEEZE_FORCE_THRESHOLD)
        return thr, thr - grasp_thr
    delta = (round(random.uniform(*SQUEEZE_DELTA_RANGE), 3) if SQUEEZE_DELTA_RANGE
             else max(0.0, D.SQUEEZE_FORCE_THRESHOLD - grasp_thr))
    return round(grasp_thr + delta, 3), delta


def _squeeze_step(n, title, marker, writer, rec, bridge, paxini, grip_position,
                  label, seg_id, base, grasp_thr: float):
    """스퀴즈 ★ 한 단계 = 진행 표시 + 기록. 구간 라벨은 _record_squeeze 가 직접 발행하므로
       여기서는 _step(=라벨 발행 포함) 을 쓰지 않고 표시만 한다(이중 발행 방지).

    반환: (그룹명, 임계, delta) — 임계·delta 를 run meta 에 남겨야 '추가 압축량' 을 추적할 수 있다.
    """
    con = _con()
    con.step_start(n, title)
    thr, delta = _squeeze_threshold(grasp_thr)
    try:
        name, frames = _record_squeeze(marker, writer, rec, bridge, paxini, grip_position,
                                       label=label, seg_id=seg_id, attrs=base,
                                       squeeze_threshold=thr)
    except BaseException as e:                  # _step 과 같은 형식으로 원인을 보여준다
        con.step_end(f"✘ {type(e).__name__}: {e}"[:140])
        raise
    con.step_end(f"✔ {frames} 프레임 · 임계 {thr:.2f}N "
                 f"(= 파지 {grasp_thr:.2f} + Δ{delta:.2f})")
    return name, thr, delta


def _thumb_return_step(n, title, marker, bridge, paxini, grip_position, base, tag):
    """엄지 위치 복귀 한 단계. 결과를 run meta(base)에 남기고 (위치, held) 반환.

    스퀴즈는 엄지만 움직이므로 되돌릴 것도 엄지뿐이다 → grip_position(=스퀴즈 직전 위치)의
    엄지 관절만 다시 명령한다. 파지 위치는 바뀌지 않으므로 grip_position 을 그대로 돌려준다.
    THUMB_RETURN_AFTER_SQUEEZE=False 면 아무것도 하지 않고 통과.
    구간 라벨 'thumb_return_A'/'thumb_return_B' 로 bag 에 남아 session.h5 에서 이 구간을
    잘라 볼 수 있다. ★ A·B 를 한 라벨로 합치지 않는 이유: 합치면 한 run 안에 같은 phase 가
    2구간 나와 '스퀴즈 A 뒤' 와 'B 뒤' 를 코드만으로 못 가른다(safe_start/safe_end 를
    분리한 것과 같은 이유 — bag_to_session.PHASE 주석 참고).
    """
    if not THUMB_RETURN_AFTER_SQUEEZE:
        return grip_position, True
    with _step(marker, f"thumb_return_{tag}", n, title) as st:
        held, peak, n_hold = _thumb_return(bridge, paxini, grip_position)
        base[f"thumb_return_{tag}_held"] = int(held)
        base[f"thumb_return_{tag}_peak_force_n"] = float(peak.max())
        base[f"thumb_return_{tag}_fingers"] = int(n_hold)
        st["mark"] = "✔" if held else "✘"
        st["note"] = (f"엄지 복귀 · 접촉 {n_hold}/{D.Finger_Num}개 "
                      f"[{' '.join(f'{v:.1f}' for v in peak)}] N"
                      + ("" if held else f"  ← 전 손가락 < {GRIP_LOST_FORCE_N:g}N = 놓침"))
    return grip_position, held


def _run_sequence(run_id, *, bridge, paxini, mover, marker, writer, rec, fruit, seg_ids,
                  combo: tuple | None = None, out: dict | None = None):
    """한 run 의 시퀀스. 팔=MoveIt, 손=배포 프리미티브, ★=구간 라벨.

    반환: (스퀴즈 그룹 이름들, auto_outcome, meta)
      auto_outcome=None → 사용자 판정 / "grip_fail" → 파지 실패로 run 중단됨(자동 태깅).

    out: 호출자가 넘기는 dict. 진행하면서 out["names"]/out["meta"] 에 **같은 객체**를 담아둔다.
      → Ctrl-C 로 중간에 예외가 나가도 호출자가 '여기까지 만든 그룹·자세·임계값' 을 회수해
        outcomes.json 에 기록할 수 있다(중단된 run 이 무기록으로 사라지지 않게).
    """
    # 힘 임계 랜덤화: 파지 = run 당 1회(전역값 → _grip 이 사용), 스퀴즈 = 스퀴즈마다 각각.
    grip_thr = _draw(GRIP_FORCE_RANGE, D.GRIP_FORCE_THRESHOLD)
    D.GRIP_FORCE_THRESHOLD = grip_thr
    names: list[str] = []
    if out is not None:
        out["names"] = names
    base = {"run": int(run_id), "fruit": fruit,
            "grip_force_threshold_n": float(grip_thr),          # 목표 힘(랜덤)
            # 판정 힘은 목표와 별개로 고정 — 둘 다 남겨야 나중에 '왜 성공/실패였나' 를 안다.
            "grip_judge_force_n": float(grip_thr if GRIP_JUDGE_FORCE_N is None
                                        else GRIP_JUDGE_FORCE_N)}
    if out is not None:
        out["meta"] = base
    # 자세는 **호출자(스케줄)가 정해서 넘긴다.** combo=None 이면 예전 동작(run 마다 랜덤).
    #   스케줄 모드에서 여기서 다시 뽑으면 '조합을 한 번씩' 이라는 계획이 깨진다.
    if combo is not None:
        present_up, present_down, hand_pose = combo
    else:
        present_up = _draw_present(PALM_UP_CANDIDATES, PALM_UP_BASE)
        present_down = _draw_present(PALM_DOWN_CANDIDATES, PALM_DOWN_BASE)
        hand_pose = _draw_grip_pose(GRIP_POSE_CANDIDATES, Path(D.HAND_GRIP_POINT_FILE).name)
    base["present_pose_up"] = present_up
    base["present_pose_down"] = present_down
    # 파지 손 자세: set_pose_for_fruit 이 그 txt 로 GRIP/SAVE 포인트를 재계산한다.
    #   (파지·재시도·스퀴즈 전에 적용해야 그 자세가 실제로 쓰인다.)
    D.set_pose_for_fruit(hand_pose)
    base["hand_pose_file"] = hand_pose
    print(f"\n── run {run_id} " + "─" * 34 + f"  파지 임계 {grip_thr:.2f}N")
    print(f"   자세   palm-up={present_up} · palm-down={present_down} · 파지={hand_pose}")

    # ── 8단계 시퀀스 (2026-07-29 사용자 지정) ────────────────────────────────────
    #   palm-up → 손 펴기 → 파지 → 스퀴즈A → 엄지복귀A → palm-down → 스퀴즈B → 엄지복귀B
    #   예전의 [safe 시작(팔+손)]·[grip 이동]·[grip 자세 파지]·[물체 내려놓기]·[safe 끝] 은
    #   **빠졌다.** 즉 한 run 은 palm-up 에서 시작해 palm-down 에서 끝나고, 물체를 내려놓지
    #   않으므로 **다음 run 은 물체를 든 채로 palm-up 이동부터 시작**한다.
    #   → 첫 run 전에는 물체가 손안(또는 palm-up 손바닥 위)에 있어야 한다. 팔도 safe 로
    #     자동 복귀하지 않으니, 세션이 끝나면 팔은 palm-down 자세에 그대로 선다.

    # 1) palm-up 이동. ★ safe 경유 유지 — 직접 이동은 수직으로 빠지지 않아 물체와 부딪힌다.
    with _step(marker, "move_palm_up", 1, "palm-up 이동") as st:
        t0 = time.monotonic()
        _arm_move(mover, bridge, "safe")
        _arm_move(mover, bridge, present_up)
        st["note"] = f"{time.monotonic() - t0:.1f}s (safe 경유)"

    # 2) 손만 펴기. '펴기' 와 '파지' 를 한 단계에 묶으면 구간 라벨이 하나뿐이라 전처리에서
    #    둘을 못 가른다 → 별도 단계로 나눈다(그래서 아래 파지는 _grip_close 로 '닫기만').
    #   ⚠ 손을 펴는 순간 물체는 palm-up 손바닥 위에 놓인다 — 손바닥이 위를 보고 있어야
    #     굴러떨어지지 않는다. present_up 후보(palm_up_tilt_*)를 너무 기울이면 위험하다.
    with _step(marker, "hand_safe", 2, "safe 이동 (손만)") as st:
        D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
        st["note"] = "손 펴기 (물체를 palm-up 손바닥에 내려놓음)"

    # 3) 파지 — 파지 임계힘(grip_thr) 도달 판정 + 실패 시 손 펴고 재시도.
    #    손은 이미 펴져 있으므로 open_first=False(닫기만). 재시도 경로가 다시 펴 준다.
    #    ★ 여기서 나온 grip_position 이 스퀴즈 A·B 의 복귀 기준이 된다.
    with _step(marker, "grip", 3, "물체 파지 (palm-up)") as st:
        grip_position, grip_ok, grip_peak, grip_n = _grip_with_retry(
            bridge, paxini, threshold=grip_thr, open_first=False)
        st["mark"] = "✔" if grip_ok else "✘"
        st["note"] = (f"{grip_n}/{D.Finger_Num}개 도달  "
                      f"[{' '.join(f'{v:.1f}' for v in grip_peak)}] N")
    base["grip_reached_fingers"] = int(grip_n)
    base["grip_peak_force_n"] = float(grip_peak.max())

    if not grip_ok:
        # 임계 미달 → 스퀴즈로 가지 않는다(잡지도 못한 상태의 스퀴즈는 쓰레기 데이터).
        #   grip_fail 은 RETRY_OUTCOMES 라 그 자세 조합은 소진되지 않고 재수집된다.
        #   ※ 이 시퀀스에는 safe 단계가 없지만, 중단은 예외 경로이므로 팔을 safe 로 빼
        #     정의된 자세에서 멈추게 한다(놓친 물체를 사람이 회수하기도 쉽다).
        with _step(marker, "grip_fail_abort", 4, "파지 실패 → 중단") as st:
            D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)   # 손 열기
            _arm_move(mover, bridge, "safe")
            st["note"] = "손 펴고 safe 복귀 (스퀴즈 생략, grip_fail 로 기록)"
        return [], "grip_fail", base

    # 4) 스퀴즈 A ★
    _nm, base["squeeze_A_threshold_n"], base["squeeze_A_delta_n"] = _squeeze_step(
        4, "스퀴즈 A ★", marker, writer, rec, bridge, paxini,
        grip_position, "squeeze_A", next(seg_ids), base, grip_thr)
    names.append(_nm)          # ★ 재대입 금지 — out["names"] 와 같은 객체를 유지해야 한다

    # 5) ★ 스퀴즈 후 엄지만 원위치로. 물체를 놓친 경우(힘≈0)에만 폐기한다.
    grip_position, ok_a = _thumb_return_step(5, "엄지 복귀 A", marker, bridge, paxini,
                                             grip_position, base, "A")
    if not ok_a:
        with _step(marker, "grip_fail_abort", 6, "파지 상실 → 중단") as st:
            D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
            _arm_move(mover, bridge, "safe")
            st["note"] = "손 펴고 safe 복귀 (물체 놓침 — 스퀴즈 B 생략, grip_lost 로 기록)"
        return names, "grip_lost", base

    # 6) palm-down 이동 ★ (팔 이동도 구간 라벨)
    with _step(marker, "move_palm_down", 6, "palm-down 이동 ★"):
        _arm_move(mover, bridge, present_down)

    # 7) 스퀴즈 B ★. 앞에 '펴기+파지' 를 다시 넣지 않는다(사용자 지정 순서) — 즉 A 는
    #    '펴고 다시 잡은 직후' 의 스퀴즈, B 는 '그 파지를 유지한 채' 의 스퀴즈다.
    _nm, base["squeeze_B_threshold_n"], base["squeeze_B_delta_n"] = _squeeze_step(
        7, "스퀴즈 B ★", marker, writer, rec, bridge, paxini,
        grip_position, "squeeze_B", next(seg_ids), base, grip_thr)
    names.append(_nm)

    # 8) 스퀴즈 B 뒤에도 엄지를 되돌리고 확인한다 — 여기서 놓쳤다면 **B 구간 데이터가 이미
    #    의심스럽다**. run 이 곧 끝나 복귀 자체의 필요는 작지만 판정 기록을 남긴다.
    grip_position, ok_b = _thumb_return_step(8, "엄지 복귀 B", marker, bridge, paxini,
                                             grip_position, base, "B")
    if not ok_b:
        # ★ 여기서는 **자동 폐기하지 않는다**(2026-07-29 사용자 지적: "판정 안했는데 왜 시작해?").
        #   A 와 달리 스퀴즈 A·B 가 **둘 다 이미 기록됐다** — 놓친 시점이 스퀴즈 B 중인지
        #   B 가 끝난 뒤 복귀 중인지 힘값만으로는 구분이 안 되고, 후자면 B 데이터는 멀쩡하다.
        #   그 판단은 물체를 직접 본 사람이 해야 한다 → 판정을 '제안' 으로만 띄운다.
        base["suggest_outcome"] = "grip_lost"

    # ※ [물체 내려놓기]·[safe 끝] 은 이 시퀀스에 없다 — run 은 palm-down 에서 물체를 든 채
    #   끝나고, 다음 run 이 그 상태에서 palm-up 이동으로 이어받는다.

    # (names, auto_outcome, meta). auto_outcome=None 이면 사용자 판정(또는 --no-judge).
    #   meta(=base) 는 outcomes.json 에 그대로 실린다 — 자세·임계값이 **라이브 h5 group attr
    #   에만** 있으면 bag 기반 재구성(bag_to_session.py)에서 유실되기 때문.
    return names, None, base


def _git_sha(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def start_bag(bag_dir: Path, topics, storage: str):
    """rosbag record 기동. 기본은 그 출력을 세션 폴더의 bag_record.log 로 보낸다 —
       'Subscribed to topic ...' 12줄 등이 진행 표시를 덮어버리기 때문(--verbose 면 터미널).

    ★ stdin=DEVNULL 이 필수다: `ros2 bag record` 는 일시정지 단축키를 위해 **터미널 키보드를
      읽는다**('Press SPACE for pausing/resuming' 을 로그에 찍는다). stdin 을 물려주면
      레코더가 터미널 입력을 가로채, run 끝 판정 프롬프트에서 사용자가 누른 Enter 가
      레코더로 가고 우리 input() 은 EOF 를 받는다(=판정 유실. 실측으로 확인).
    """
    cmd = ["ros2", "bag", "record", "-o", str(bag_dir)]
    if storage:
        cmd += ["-s", storage]
    cmd += list(topics)
    if VERBOSE:
        print(f"[collect] rosbag record 시작: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    log = bag_dir.parent / "bag_record.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w", encoding="utf-8")
    f.write(f"$ {' '.join(cmd)}\n\n")
    f.flush()
    print(f"  bag 기록 {len(topics)}개 토픽  (rosbag2 로그 → {log.name})")
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=f, stderr=subprocess.STDOUT)


def stop_bag(proc):
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGINT)   # SIGINT 이라야 bag metadata 가 정상 마감됨
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    print("[collect] rosbag record 종료")


def parse_args():
    p = argparse.ArgumentParser(description="배포 동일 경로 하이브리드 데이터 수집기 (8단계 시퀀스)")
    p.add_argument("--fruit", default=None, help="과일명(생략 시 대화형 선택)")
    p.add_argument("--grip-pose", default=None, metavar="POSE",
                   help="파지 손 자세를 이 pose txt 하나로 고정(랜덤화 끔). "
                        "예: --grip-pose kiwi.txt (확장자 생략 가능). "
                        "미지정 시 GRIP_POSE_CANDIDATES 중 run 마다 랜덤.")
    p.add_argument("--specimen", default=None,
                   help="개체 이름/번호 (예: 1 → '<물체>_1', 또는 'ecoflex_1' 그대로). "
                        "폴더·파일명이 <개체>_<파지자세>_<ts> 가 된다. 생략 시 시작할 때 묻는다")
    p.add_argument("--num-demos", type=int, default=None,
                   help="run 수 상한. 스케줄 모드에서는 조합 45개 중 이만큼만 수집(테스트용). "
                        "생략 시 조합 전부")
    p.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "collect_logs"),
                   help="세션 저장 루트")
    # ★ 기본값 ft → raw (2026-07-28). §F7 에서 raw(Σ127)가 표준으로 확정됐는데 기본값이
    #   ft 로 남아 있어, --paxini raw 를 빼먹으면 파지 게이트가 5~15배 작은 값을 읽고
    #   임계에 도달하지 못해 **매번 grip_fail** 이 됐다(실측: 같은 파지에서
    #   ft [1.8,1.7,1.4]N vs Σ127 [9.3,12.6,8.0]N).
    p.add_argument("--paxini", choices=["ft", "raw"], default="raw",
                   help="촉각 소스: raw=/paxini/right/raw(4×127×3, 표준) | "
                        "ft=/paxini/right/ft(4×3, 값이 5~15배 작다 — 임계 재조정 필요)")
    p.add_argument("--no-bag", action="store_true", help="rosbag 동시 기록 끄기")
    p.add_argument("--bag-storage", default="mcap",
                   help="rosbag storage (mcap 플러그인 없으면 sqlite3)")
    p.add_argument("--bag-topics", nargs="*", default=None, help="기록 토픽 재정의")
    p.add_argument("--live-h5", action="store_true",
                   help="라이브 HDF5(collect_*.h5)도 같이 쓴다. 기본은 안 쓴다 — "
                        "session.h5(bag 변환)와 중복이고 parity 검증이 끝났다")
    p.add_argument("--verbose", action="store_true",
                   help="rosbag2/MoveIt INFO·deploy 배너까지 원본 그대로 출력(기본=진행 표시만)")
    p.add_argument("--no-judge", action="store_true",
                   help="run 종료 후 성공 판정 프롬프트 끄기(모두 unjudged)")
    a = p.parse_args()
    # --num-demos 를 '명시했는지' 를 구분해야 한다: 스케줄 모드의 기본은 '조합 전부' 이고
    # 레거시 모드에서는 횟수가 필요하다(기본 10).
    a.num_demos_given = a.num_demos is not None
    if a.num_demos is None:
        a.num_demos = 10
    return a


def main():
    global VERBOSE, GRIP_POSE_CANDIDATES
    args = parse_args()
    VERBOSE = args.verbose
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 진행 표시(줄 관리)는 두 모드 다 필요하므로 항상 설치하고, --verbose 면 필터만 끈다.
    sys.stdout = _Console(sys.stdout)

    # 판정 프롬프트 입력 경로를 **로봇을 33초 움직이기 전에** 확인해 둔다
    # (위 _prompt 주석의 판정 유실 사고 참고).
    if not args.no_judge and not sys.stdin.isatty():
        if _open_tty() is not None:
            print("[collect] ℹ stdin 이 대화형이 아니지만(파이프·리다이렉트 등) 판정은 제어 "
                  "터미널(/dev/tty)에서 받는다 — 입력할 때까지 기다린다.")
        else:
            print("[collect] ⚠ 제어 터미널이 없어 run 끝 판정을 받을 수 없다 — 모든 run 이\n"
                  "          outcome='not_judged' 로 기록된다(데이터 자체는 정상 저장).\n"
                  "          판정을 남기려면 터미널에서 실행할 것. 로그가 필요하면\n"
                  "          `script -qc \"python3 ... collect_ros2.py ...\" collect.log` (TTY 유지).")

    rclpy.init()
    bridge = Ros2ShmBridge()
    if args.paxini == "raw":
        from deploy_ros2_exp_rawft import Ros2RawPaxiniBridge   # /raw(127점) 구독
        paxini = Ros2RawPaxiniBridge(bridge)
    else:
        paxini = Ros2PaxiniBridge(bridge, "/paxini/right/ft")
        # ★ ft 는 값이 Σ127 의 1/5~1/15 다(실측: 같은 파지에서 ft [1.8,1.7,1.4]N vs
        #   Σ127 [9.3,12.6,8.0]N). 파지/스퀴즈 게이트가 이 값을 그대로 쓰므로 raw 기준으로
        #   맞춘 임계는 **절대 도달하지 못하고 매 run grip_fail** 이 된다. 조용히 실패하지
        #   않도록 여기서 알린다(예전에 실제로 당했다).
        print(f"[collect] ⚠ --paxini ft 선택 — 촉각 힘이 Σ127 의 1/5~1/15 로 작다.\n"
              f"          현재 임계 파지 {GRIP_FORCE_RANGE or D.GRIP_FORCE_THRESHOLD}N / "
              f"스퀴즈 = 파지+Δ{SQUEEZE_DELTA_RANGE or SQUEEZE_FORCE_RANGE} 이 ft 스케일에\n"
              f"          맞는지 확인할 것. 표준은 --paxini raw 다(§F7).")
    marker = MarkerPublisher()
    mover = MoveItArmMover(group=ARM_GROUP, ee_link=ARM_EE_LINK, frame=ARM_FRAME)
    if not VERBOSE:
        # 팔 이동마다 '플랜 성공 N waypoints' / '도달 완료 …' 2줄이 진행 표시를 덮는다.
        # WARN 으로 낮춰 문제(플랜 실패·미도달·stall)만 보이게 한다.
        for _n in (mover, bridge, marker):
            _n.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)

    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(marker)
    executor.add_node(mover)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    D._set_squeeze_flag(False)
    writer = None
    bag_proc = None
    try:
        if not bridge.attach():
            raise SystemExit(
                "상태 토픽 미수신 — Dual_Arm_Hand_Ctrl 스택/ROS_DOMAIN_ID 확인.")
        if not paxini.attach():
            print(f"[collect] 경고: paxini({args.paxini}) 미수신 — 힘=0/무효 프레임으로 진행"
                  " (수집 데이터 대부분 스킵될 수 있음).")
        # 팔 이동 준비 확인 (move_group). 없어도 진행하되 팔 이동은 실패/생략됨.
        if not mover.wait_ready(timeout=3.0):
            print("[collect] 경고: /move_action(move_group) 미수신 — 팔 이동은 전부 생략/실패. "
                  "플래닝 PC 에서 move_group 기동 필요(joint_state_mode:=direct).")

        # 안전 서보-온 (현재 손 자세를 q_target 으로 먼저 발행 → servo on).
        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # 과일 → 파지 포즈/임계값. 수집엔 '파지 포즈'만 필요 → FRUIT_CONFIG 에서 pose 만 직접 읽음
        # (D.resolve_fruit_config 는 model=None 과일에서 SystemExit — 미학습 과일도 수집 위해 우회).
        fruit = args.fruit or D.ask_fruit()
        pose_file = _resolve_pose_file(fruit)
        # --grip-pose: 파지 손 자세를 지정 pose 하나로 고정(랜덤 후보를 1개로 대체).
        #   _draw_grip_pose 가 1개 풀에서 매 run 그 파일을 뽑으므로 사실상 고정 = 랜덤화 끔.
        if args.grip_pose:
            name = args.grip_pose if args.grip_pose.endswith(".txt") else f"{args.grip_pose}.txt"
            if not (D._POSE_DIR / name).exists():
                raise SystemExit(f"--grip-pose '{name}' 파일 없음: {D._POSE_DIR}")
            GRIP_POSE_CANDIDATES = (name,)
            print(f"[collect] 파지 자세 고정: {name} (--grip-pose, 랜덤화 대체)")
        D.set_pose_for_fruit(pose_file)
        D.set_thresholds_for_fruit(fruit)
        # ★ specimen = '개체' 이름. fruit(물체 종류)는 자세·임계 조회에 쓰고, 폴더·파일·라벨은
        #   specimen 을 쓴다. 같은 재료라도 개체가 다르면 일반화 평가 때 개체 단위로 나눠야 한다.
        specimen = _ask_specimen(fruit, args.specimen)
        # ★ session_dir 은 **자세 스케줄을 확정한 뒤** 만든다(아래) — 폴더명에 파지 자세를
        #   넣으려면 어떤 pose txt 를 쓸지 먼저 알아야 한다.

        _fmt = (lambda r, fb, unit="N":
                f"{r[0]:g}~{r[1]}{unit} 랜덤" if r else f"{fb:g}{unit} 고정")
        print("\n" + "═" * 70)
        print(f"  데이터 수집   개체 {specimen}  (물체 {fruit})  ·  촉각 "
              f"{'raw(4×127×3)' if args.paxini == 'raw' else 'ft(4×3)'}")
        # 스퀴즈 임계는 '파지 + delta' 이므로 그 계산식과 실제 범위를 같이 보여준다.
        #   (예전 문구는 SQUEEZE_FORCE_RANGE=None 일 때 과일 기본값을 '고정' 으로 찍어
        #    실제(9~13N)와 다른 값을 표시했다)
        if SQUEEZE_FORCE_RANGE:
            _sq = f"{_fmt(SQUEEZE_FORCE_RANGE, D.SQUEEZE_FORCE_THRESHOLD)} (절대값)"
        elif SQUEEZE_DELTA_RANGE and GRIP_FORCE_RANGE:
            _lo = GRIP_FORCE_RANGE[0] + SQUEEZE_DELTA_RANGE[0]
            _hi = GRIP_FORCE_RANGE[1] + SQUEEZE_DELTA_RANGE[1]
            _sq = (f"파지+Δ({SQUEEZE_DELTA_RANGE[0]:g}~{SQUEEZE_DELTA_RANGE[1]:g}N) "
                   f"= {_lo:g}~{_hi:g}N")
        else:
            _sq = f"파지+Δ({SQUEEZE_DELTA_RANGE or '고정'})"
        print(f"  힘 임계       파지 {_fmt(GRIP_FORCE_RANGE, D.GRIP_FORCE_THRESHOLD)}"
              f"  ·  스퀴즈 {_sq}")
        print(f"  파지 성공     판정 "
              f"{'목표힘 그대로' if GRIP_JUDGE_FORCE_N is None else f'{GRIP_JUDGE_FORCE_N:g}N 고정'}"
              f" 도달 손가락 {GRIP_MIN_FINGERS}개 이상 "
              f"(실패 시 손 펴고 최대 {GRIP_MAX_RETRY}회 재시도)")
        # ── 자세 스케줄 확정 (개체 세션당 1회) ──
        if POSE_SCHEDULE:
            print("  자세 스케줄   조합을 빠짐없이 1회씩 (실패 판정 시 같은 조합 재시도)")
            up_set, down_set, grip_set, combos = _build_schedule(
                GRIP_POSE_CANDIDATES,
                limit=args.num_demos if args.num_demos_given else None)
        else:
            # 레거시 랜덤 모드에도 폴더명용 자세 목록은 필요하다(존재하는 후보만).
            up_set, down_set, combos = [], [], []
            grip_set = [n for n in GRIP_POSE_CANDIDATES if (D._POSE_DIR / n).exists()] \
                or [Path(D.HAND_GRIP_POINT_FILE).name]
            print(f"  자세          레거시 랜덤 모드 (POSE_SCHEDULE=False) · run {args.num_demos}개")

        #   폴더·파일명 = '<개체>_<파지자세>_<타임스탬프>'
        #     예: ecoflex_1_ecoflex_20260729_000753  (개체 ecoflex_1, 파지 자세 ecoflex.txt)
        #   파지 자세를 넣는 이유(사용자 요청): 어떤 pose txt 로 잡은 데이터인지 파일 열지 않고
        #   구분해야 한다. 자세가 여러 개인 세션은 '-' 로 잇거나 개수로 줄인다(_pose_tag).
        #   'collect_' 접두사는 붙이지 않는다 — 상위 폴더가 이미 collect_logs 다.
        pose_tag = _pose_tag(grip_set)
        session_dir = Path(args.out_dir) / f"{specimen}_{pose_tag}_{session}"
        session_dir.mkdir(parents=True, exist_ok=True)
        h5_path = session_dir / f"{specimen}_{pose_tag}_{session}.h5"
        print(f"  저장          {session_dir}")
        print("═" * 70)

        # provenance (재현성: 데이터 옆에 코드/설정 이력을 freeze).
        session_attrs = {
            "fruit": fruit,
            "specimen": specimen,   # 개체 이름(폴더·라벨). fruit=물체 종류와 구분
            "paxini_source": args.paxini,
            "grip_force_range_n": str(GRIP_FORCE_RANGE),          # 랜덤 범위(=고정 입력값)
            "squeeze_force_range_n": str(SQUEEZE_FORCE_RANGE),
            "squeeze_delta_range_n": str(SQUEEZE_DELTA_RANGE),   # 스퀴즈 = 파지 + delta
            "release_settle_sec": str(RELEASE_SETTLE_SEC),
            "thumb_return_after_squeeze": int(bool(THUMB_RETURN_AFTER_SQUEEZE)),
            "thumb_return_duration_sec": float(THUMB_RETURN_DURATION),
            "grip_lost_force_n": float(GRIP_LOST_FORCE_N),
            "force_randomized": int(bool(GRIP_FORCE_RANGE) or bool(SQUEEZE_FORCE_RANGE)),
            "present_pose_randomized": int(bool(PRESENT_POSE_RANDOMIZE)),   # palm 손목자세 랜덤 on/off
            "palm_up_candidates": ",".join(PALM_UP_CANDIDATES),            # 후보(빈 앵커 포함, 실제 뽑힘은 그룹 attr)
            "palm_down_candidates": ",".join(PALM_DOWN_CANDIDATES),
            "grip_pose_randomized": int(bool(GRIP_POSE_RANDOMIZE)),         # 파지 손 자세 랜덤 on/off
            "grip_pose_candidates": ",".join(GRIP_POSE_CANDIDATES),        # pose txt 후보(실제 뽑힘은 그룹 attr hand_pose_file)
            "grip_pose_used": ",".join(grip_set),   # ★ 이 세션이 실제로 쓴 pose txt(존재하는 것만)
            "grip_pose_tag": pose_tag,             # ★ 폴더·파일명에 들어간 자세 토큰(폴더명 되읽기용)
            "fruit_grip_force_n": float(D.GRIP_FORCE_THRESHOLD),   # 과일 baseline(랜덤 off 시)
            "fruit_squeeze_force_n": float(D.SQUEEZE_FORCE_THRESHOLD),
            "control_rate_hz": float(getattr(D, "CONTROL_RATE_HZ", 0.0)),
            "USE_TACTILE": int(RE.USE_TACTILE),
            "USE_JKIN": int(RE.USE_JKIN),
            "FACTOR": int(RE.FACTOR),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", ""),
            "git_sha": _git_sha(_REPO_ROOT),
            "created_wall": datetime.now().isoformat(),
            # 라이브(A)는 t_mono_ns=time.monotonic_ns, bag(B)은 t_ns=epoch 를 쓴다.
            # 세션 시작 시 1회 기록해 두면 parity 가 A 를 epoch 로 환산: epoch = t_mono_ns + t_offset_ns.
            "t_offset_ns": int(time.time_ns() - time.monotonic_ns()),
            "raw_hand_j_kin_mN_present": int(RE.RAW_HAND_J_KIN_FILE.exists()),
            # phase 1~8 과 같은 순서. release/safe 단계 없음 → run 이 물체를 든 채 끝난다.
            "sequence": ("palm_up→hand_safe→grip→squeezeA★→thumb_return_A"
                         "→palm_down★→squeezeB★→thumb_return_B"),
            "arm_moves_via": f"moveit_arm_mover(/move_action plan-only→q_target) group={ARM_GROUP} ee={ARM_EE_LINK} frame={ARM_FRAME}",
            "star_segments": ",".join(STAR_SEGMENTS),
            "flags": "per-segment label on /collect/segment (bag) + HDF5 group 'segment' attr",
            "demo_outcome": "per-run judgment → group attr 'outcome' + /collect/demo_outcome (bag) + outcomes.json",
            # OUTCOMES(사용자 선택) + 자동 태깅 라벨(unjudged / interrupted=Ctrl-C 중단)
            "outcome_labels": ",".join(OUTCOMES.values()) + ",unjudged,interrupted,grip_lost",
            "note": "HDF5 groups = squeeze ★ (A,B) only; palm_down ★ arm-move is in bag(segment label).",
        }
        # 기본 산출물 = session.h5(bag 변환) + bag/ + outcomes.json.
        #   라이브 h5 는 중복이라 만들지 않는다(--live-h5 로 되살릴 수 있음. _NullWriter 주석 참고).
        writer = HDF5DemoWriter(h5_path, session_attrs) if args.live_h5 else _NullWriter()
        rec = RecordingEngine(writer=writer,
                              on_squeeze_start=lambda: marker.pub_squeeze(1))
        # provenance 는 outcomes.json 의 session 블록으로 남긴다 → bag_to_session 이 옮긴다.
        _write_outcomes(session_dir, fruit, {}, session_attrs, specimen)

        if not args.no_bag:
            topics = args.bag_topics if args.bag_topics else DEFAULT_BAG_TOPICS
            bag_proc = start_bag(session_dir / "bag", topics, args.bag_storage)
            # 고정 sleep 이 아니라 구독자 수로 확인한다(위 wait_for_recorder 주석 참고).
            marker.wait_for_recorder(topics)

        # ── run 루프 ────────────────────────────────────────────────────────────
        #   스케줄 모드: combos(=자세 조합)를 하나씩 소진한다. 실패 판정이면 **같은 조합을
        #   다시 큐에 넣어** 성공할 때까지 반복 → 개체당 모든 조합이 최소 1회 성공 기록을 갖는다.
        #   레거시 모드(POSE_SCHEDULE=False): 예전처럼 --num-demos 회 반복(자세는 run 마다 랜덤).
        seg_ids = itertools.count()          # HDF5 그룹 id 전역 카운터
        outcomes = {}                        # run_id → {outcome, wall, groups} (sidecar)
        interrupted = False
        queue = list(combos) if POSE_SCHEDULE else [None] * args.num_demos
        total_planned, done_combos, run_id = len(queue), 0, -1
        while queue:
            combo = queue.pop(0)
            run_id += 1
            if POSE_SCHEDULE:
                print(f"\n[진행] 조합 {done_combos + 1}/{total_planned} "
                      f"(남은 {len(queue)}개, 재시도 포함 run #{run_id})")
            marker.pub_demo("S", run_id, time.monotonic_ns())
            # ★ Ctrl-C 로 중간에 멈춰도 '여기까지 모은 데이터' 를 정상 마감한다.
            #   out 으로 부분 결과(그룹 이름·자세·임계값)를 회수해 아래 기록 경로를 그대로 탄다
            #   → E 마커·demo_outcome 토픽·HDF5 group attr·outcomes.json 이 모두 남는다.
            #   (h5/bag 자체는 예전에도 main 의 finally 에서 닫혔지만, 중단된 run 은
            #    판정·자세 기록이 통째로 빠져 학습에서 걸러낼 수도 없었다.)
            out: dict = {}
            try:
                names, auto_outcome, run_meta = _run_sequence(
                    run_id, bridge=bridge, paxini=paxini, mover=mover, marker=marker,
                    writer=writer, rec=rec, fruit=fruit, seg_ids=seg_ids,
                    combo=combo, out=out)
            except KeyboardInterrupt:
                interrupted = True
                names = out.get("names", [])
                run_meta = out.get("meta", {"run": int(run_id), "fruit": fruit})
                auto_outcome = "interrupted"
                print("\n  ⚠ Ctrl-C — 여기까지 기록한 데이터를 저장하고 종료합니다"
                      f"{f' (스퀴즈 {len(names)}개 기록됨)' if names else ' (스퀴즈 기록 없음)'}."
                      "\n     ※ 팔 palm-up(기본) + 손 safe 로 복귀 후 종료합니다.")
            marker.pub_demo("E", run_id, time.monotonic_ns())
            # ── 데모 판정. 파지 실패/중단된 run 은 프롬프트 없이 자동 태깅. ──
            #   stop=True → 이 run 을 기록한 뒤 수집 종료(판정 프롬프트에서 'q' 로 선택).
            stop = False
            #   ★ suggest = '코드가 의심하지만 확정하지 않는' 판정. 시퀀스가 끝까지 돌아
            #     스퀴즈 A·B 가 다 기록된 run 은 사람 확인 없이 폐기하지 않는다
            #     (2026-07-29: 엄지 복귀 B 힘≈0 을 자동 grip_lost 로 확정해 판정 프롬프트
            #      없이 다음 run 이 시작되는 문제가 있었다).
            suggest = run_meta.pop("suggest_outcome", None)
            if auto_outcome is not None:
                outcome = auto_outcome          # 파지 실패·중단 = 스퀴즈 기록 자체가 없음
            elif args.no_judge:
                outcome = suggest or "unjudged"
            else:
                outcome, stop = _ask_outcome(
                    run_id, remaining=len(queue), suggest=suggest,
                    why=("엄지 복귀 B 에서 전 손가락 접촉력 < "
                         f"{GRIP_LOST_FORCE_N:g}N" if suggest == "grip_lost" else ""))
            if outcome == "interrupted":        # 판정 프롬프트에서 Ctrl-C 한 경우도 포함
                interrupted = True
            marker.pub_outcome(run_id, outcome, time.monotonic_ns())     # bag 토픽
            for nm in names:
                writer.set_group_attr(nm, "outcome", outcome)           # HDF5 그룹 attr
            # ★ 자세·임계값·파지결과를 여기 함께 남긴다 → bag_to_session.py 가 /runs 를 채운다.
            outcomes[str(run_id)] = {"outcome": outcome,
                                     "wall": datetime.now().isoformat(), "groups": names,
                                     **{k: v for k, v in run_meta.items() if k != "run"}}
            _write_outcomes(session_dir, fruit, outcomes, session_attrs, specimen)  # sidecar
            print(f"   판정 = {outcome}"
                  + (f"  ·  기록 {', '.join(names)}" if names else "  ·  스퀴즈 기록 없음"))
            # ★ 실패 판정이면 그 조합을 소진하지 않고 **다시 큐에 넣어** 같은 자세로 재실행한다.
            #   (파지 실패·discard 는 '그 자세 조합의 데이터를 못 얻은 것' 이므로 소진 처리하면
            #    개체당 조합 커버리지에 구멍이 남는다.)
            if POSE_SCHEDULE and combo is not None and outcome in RETRY_OUTCOMES \
                    and not interrupted and not stop:
                queue.append(combo)
                print(f"   ↻ '{outcome}' — 같은 자세 조합을 큐 뒤로 재투입 "
                      f"(남은 {len(queue)}개)")
            elif POSE_SCHEDULE:
                done_combos += 1

            if interrupted or stop:             # 중단·사용자 종료면 다음 run 을 시작하지 않는다
                if stop and not interrupted:
                    print(f"   ▸ 사용자 종료 선택 — 남은 조합 {len(queue)}개를 건너뜁니다.")
                break
            if run_id < args.num_demos - 1:
                time.sleep(PAUSE_BETWEEN_RUNS_SEC)

        # ── 마무리 자세: 팔 palm-up(기본) + 손 safe ──────────────────────────────
        #   전 조합 완료·Ctrl-C·'q' 조기 종료 모두 공통으로 복귀한다(2026-07-30 사용자 지정).
        #   run 은 palm-down 에서 물체를 든 채 끝나므로, 팔을 먼저 palm-up 으로 되돌린 뒤
        #   손을 펴야 물체가 손바닥 위에 놓인다(순서 반대면 palm-down 에서 물체 낙하).
        #   palm-down→palm-up 직접 이동은 물체와 부딪히므로 safe 경유(_run_sequence 1단계와
        #   동일 경로). 복귀 중 Ctrl-C 를 또 누르면 그 자리에서 멈춘다(finally 로 정상 마감).
        print("\n[collect] "
              + ("전 조합 완료" if not (interrupted or queue) else "조기 종료")
              + " — 팔 palm-up(기본) + 손 safe 로 복귀합니다.")
        _arm_move(mover, bridge, "safe")
        _arm_move(mover, bridge, PALM_UP_BASE)
        D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)

        ok = sum(1 for v in outcomes.values() if v["outcome"] == "success")
        print("\n" + "═" * 70)
        print(f"  {'중단' if (interrupted or queue) else '완료'}   개체 {specimen}  ·  "
              f"run {len(outcomes)}회 (success {ok})  ·  스퀴즈 그룹 {writer.n_demos}개")
        if POSE_SCHEDULE:
            print(f"  조합          {done_combos}/{total_planned} 완료"
                  + (f"  ·  미완 {len(queue)}개 남음 — 같은 개체로 다시 실행하면 이어서"
                     " 수집할 수 있다(자세 조합은 매 실행 새로 뽑힘)" if queue else "  (전부 수집)"))
        print("  파일   bag/  +  outcomes.json"
              + (f"  +  {h5_path.name}(라이브)" if args.live_h5 else "")
              + ("   ← 중단 시점까지 정상 저장됨" if interrupted else ""))
        print("  ↳ 학습용 session.h5 는 아래 변환으로 만든다(bag 이 원본)")
        print(f"  다음   python3 stiffness_deploy_ros2/launch/bag_to_session.py {session_dir}")
        print("═" * 70)
    finally:
        D._set_squeeze_flag(False)
        if writer is not None:
            writer.close()
        stop_bag(bag_proc)
        bridge.detach()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    # deploy_ros2.py 와 동일: 라이브 DDS 에서 네이티브 teardown 대기가 길 수 있어 즉시 종료.
    _code = 0
    try:
        main()
    except SystemExit as e:
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except KeyboardInterrupt:
        _code = 130
    sys.stdout.flush(); sys.stderr.flush()   # P2#4: os._exit 전 flush (파이프/리다이렉트 시 마지막 출력 유실 방지)
    os._exit(_code)
