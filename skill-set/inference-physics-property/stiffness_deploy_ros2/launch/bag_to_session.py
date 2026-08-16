#!/usr/bin/env python3
"""bag_to_session.py — rosbag → **session.h5** (연속 타임라인 1개 파일).

왜 이 파일이 생겼나 (기존 2개 h5 의 문제):
  · `collect_<ts>.h5`(A, 라이브)  = 스퀴즈 A/B 만, 손 채널만, 시계 monotonic.
  · `from_bag.h5`(B, 구간별)      = 스퀴즈 A/B + palm_down, 시계 epoch. A 와 스퀴즈가 중복.
  → 둘 다 있었던 이유는 서로 검증(parity)용이었고 그건 끝났다. 그리고 **둘 중 어느 쪽도
    '스퀴즈 중 팔 관절/목표각'을 담지 않는다**(arm_* 은 move_palm_down 그룹에만 있었다).

session.h5 가 담는 것:
  /data   한 행 = 한 tick(고정 100Hz), **시퀀스 전체가 끊기지 않는 하나의 타임라인**
          run(데모 번호) + phase(시퀀스 상태 코드) 열로 나중에 자유롭게 자를 수 있다.
  /runs   한 행 = 한 run. 행 범위(row0,row1) + 자세·임계값·판정 → **데모 분할 기준표**.
  /runs_names  /runs 와 같은 행 순서의 **이름 문자열**(자세·판정). 세션을 합칠 때는
          숫자 코드가 아니라 이 이름으로 join 한다.
  ※ 예전에 있던 /codes(숫자↔이름 대응표)는 저장하지 않는다 — 2026-07-29 사용자 지시.

tick = 100Hz 고정 그리드인 이유 (실측 근거, 세션 195101):
  제어 루프(q_target) 실측 간격 10.15ms = 98.5Hz → 학습 스펙(CONTROL_RATE_HZ=100,
  FACTOR=10)과 이미 일치. 가장 느린 실센서 paxini 는 87.5Hz(11.12ms) 라 100Hz 그리드면
  모든 paxini 샘플이 1 tick 안에 들어온다. joint/kin 은 195.7Hz 로 절반이 버려지지만
  학습은 FACTOR=10 으로 10Hz 까지 내려 쓰므로 무의미하고, 원본은 bag 에 그대로 남는다.

샘플링 규칙: 모든 채널 zero-order-hold(그 시각 이전 마지막 메시지) — 라이브 리더와 동일.
  채널이 아직 한 번도 안 온 구간은 0 으로 채우고 `age_ms` 에 -1 을 넣는다.

실행:
  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
  python3 stiffness_deploy_ros2/launch/bag_to_session.py collect_logs/<개체>_<파지자세>_<ts>
  #   opt: --rate 100  --out other.h5  --no-raw(paxini_raw 제외로 용량 1/10)

  # ★ 여러 세션을 한 번에 + **h5/json 만 있는 가벼운 트리로 재구성**(--out-root)
  #   수집 폴더는 세션당 bag 이 140~440MB 라 그대로는 학습 PC 로 못 옮긴다. bag 은 원본
  #   자리에 두고, 학습에 필요한 session.h5 + outcomes.json 만 별도 트리에 쌓는다.
  python3 .../bag_to_session.py ~/Desktop/collect_logs_recovered/collect_logs \
          --out-root ~/Desktop/collect_h5
  #   → <out-root>/<세션폴더명>/session.h5 + outcomes.json   (bag 없음)
  #   이미 변환해 둔 session.h5 를 옮기기만 할 때는 --reuse-h5 (bag 을 다시 안 읽는다)
  #   중간에 끊겼다 이어서 돌릴 때는 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_LAUNCH_DIR, "..")))
sys.path.insert(0, _LAUNCH_DIR)

import rosbag2_py                                                     # noqa: E402
from rclpy.serialization import deserialize_message                    # noqa: E402
from rosidl_runtime_py.utilities import get_message                    # noqa: E402
from std_msgs.msg import String                                        # noqa: E402

import real_deploy_inference_final as RE                               # noqa: E402
import collect_ros2 as CO             # noqa: E402  ARM_POSES/후보목록 = 코드표의 단일 출처

ARM_DOF, HAND_DOF = 7, 16
KIN = RE.Kinesthetic_Sensor_Num * RE.Kinesthetic_Sensor_DOF            # 4×3 = 12
FINGERS, POINTS = 4, 127
RAW_N = FINGERS * POINTS * 3                                           # 1524

T = {
    "arm_joint":     "/franka/right/joint_states",
    "arm_q_target":  "/franka/right/q_target",
    "hand_joint":    "/hand/right/joint_states",
    "hand_q_target": "/hand/right/q_target",
    "kin":           "/hand/right/kin",
    "paxini_raw":    "/paxini/right/raw",
    "segment":       "/collect/segment",
    "demo":          "/collect/demo_marker",
    "squeeze_on":    "/collect/squeeze_on",
    "outcome":       "/collect/demo_outcome",
}

# ── phase 코드: 8단계 시퀀스를 **실행 순서대로** 1~8. 0 = run 밖/유휴. ──
#   ★★ 2026-07-29 재번호 매김(사용자 지시). 시퀀스가 8단계로 줄면서 예전 코드표
#     (grip=1, move_palm_up=2, squeeze_A=3 …)를 버리고 순서대로 다시 매겼다.
#     ⚠ append-only 규칙을 **의도적으로 깬 변경**이다 — 그 전에 만든 session.h5 의 phase
#       숫자와 뜻이 다르다. 옛 세션을 이 스크립트로 **다시 변환하면** 사라진 구간
#       (safe_start/safe_end/move_grip/release)은 표에 없으므로 0(idle)이 된다.
#       그래서 세션마다 /codes 표를 함께 저장한다 — 숫자 해석은 항상 그 파일의 /codes 로.
#   ★ thumb_return 을 A/B 로 나눈 이유: 한 라벨이면 한 run 에 같은 phase 가 2구간 나와
#     '스퀴즈 A 뒤' 와 'B 뒤' 를 코드만으로 못 가른다(예전 safe_start/safe_end 와 같은 문제).
PHASE = {
    "":                 0,   # 구간 사이(idle)
    "move_palm_up":     1,   # palm-up 으로 이동(safe 경유 포함)
    "hand_safe":        2,   # 손만 safe(펴기) — 물체를 palm-up 손바닥에 내려놓음
    "grip":             3,   # palm-up 에서 파지 임계힘으로 파지
    "squeeze_A":        4,   # 스퀴즈 A
    "thumb_return_A":   5,   # 스퀴즈 A 후 엄지만 파지 위치로 복귀(+놓침 확인)
    "move_palm_down":   6,   # palm-down 으로 이동
    "squeeze_B":        7,   # 스퀴즈 B
    "thumb_return_B":   8,   # 스퀴즈 B 후 엄지 복귀(+놓침 확인)
    # 9 = 정상 시퀀스가 아니라 **중단 경로**. 1~8 과 섞이지 않게 뒤에 둔다.
    "grip_fail_abort":  9,   # 파지 실패/파지 상실로 run 중단(손 펴고 safe 복귀)
}
#   interrupted = Ctrl-C 로 중간에 멈춘 run (중단 시점까지의 데이터는 정상 저장돼 있다).
OUTCOME_CODE = {"success": 1, "grip_fail": 2, "not_judged": 3, "discard": 4,
                "interrupted": 5, "grip_lost": 6, "unjudged": 0}

#   스퀴즈 구간 촉각 stale 경고 기준 [ms]. paxini 정상 간격이 ~11ms 이므로 20ms = 약 2 샘플.
STALE_WARN_MS = 20

# 채널 단위 — 지금 counts/raw/N/rad 이 섞여 있어 전처리 사고가 나기 쉽다. 파일에 박아둔다.
UNITS = {
    "arm_joint": "rad", "arm_q_target": "rad",
    "hand_joint": "encoder_counts", "hand_q_target": "encoder_counts",
    "kin": "shm_raw_int16",                      # = 예전 'ft' 데이터셋. mN 아님(소스는 root attr)
    "paxini_resultant": "N", "paxini_raw": "N",
}


def _pose_codes():
    """자세 이름 → 숫자. ARM_POSES(팔)·GRIP_POSE_CANDIDATES(손)의 **선언 순서**를 그대로 쓴다.

    ★ 정렬하지 않는 이유(실제로 당한 일): 손 자세를 `sorted()` 로 번호 매기다가
      GRIP_POSE_CANDIDATES 에 'ecoflex.txt' 를 추가하자 알파벳 앞이라 끼어들어
      tomato.txt 코드가 4→5 로 밀렸다. 물체는 계속 추가되므로 매번 밀린다.
      선언 순서 + **목록 끝에만 추가(append-only)** 규칙이면 기존 번호가 보존된다.
      ※ 목록을 재정렬·중간삽입·삭제하면 그 시점 이후 세션의 숫자 뜻이 달라진다 —
        그래서 /codes(파일별 표)와 /runs_names(이름)를 항상 함께 저장한다. 세션을 합칠
        때는 이름으로 join 하는 것이 원칙이고, 숫자는 단일 세션 내 편의용이다.
    """
    arm, hand, seen = {}, {}, set()
    for i, name in enumerate(CO.ARM_POSES):                            # 0 = 미기록
        arm[name] = i + 1
    for name in CO.GRIP_POSE_CANDIDATES:                               # 중복은 첫 등장 기준
        if name not in seen:
            seen.add(name)
            hand[name] = len(hand) + 1
    return arm, hand


def _splits_into_stems(tok: str, stems: set) -> bool:
    """tok 이 '자세 stem 들을 '-' 로 이은 것' 인가.

    ★ 단순히 tok.split("-") 로 못 가르는 이유: collect_ros2._pose_tag 는 stem 안의 '_' 를
      '-' 로 바꿔 태그를 **항상 1토큰**으로 만든다(폴더명을 '_' 로 분해해 되읽기 때문).
      그래서 stem 자체에 '-' 가 들어간다 — 'e_pose1' → 'e-pose1'. 여기서 '-' 로 쪼개면
      ['e','pose1'] 이 되어 어느 쪽도 stem 이 아니고, 자세 태그를 못 알아본다
      (→ 개체 이름이 'ecoflex_1_e-pose1' 로 오염됐다. 2026-07-29 e_pose1~5 도입 시 발견).
      '-' 는 stem 안에도 stem 사이에도 나오므로, 쪼개는 대신 **앞에서부터 stem 을 맞춰
      나간다**(긴 stem 우선 — 짧은 stem 이 접두사로 먼저 걸려 오탐하는 것을 막는다).
    """
    order = sorted(stems, key=len, reverse=True)
    i, n = 0, len(tok)
    while i < n:
        for s in order:
            # stem 이 딱 끝나거나 다음이 '-'(다음 stem 과의 경계)여야 한다.
            if s and tok.startswith(s, i) and (i + len(s) == n or tok[i + len(s)] == "-"):
                i += len(s) + 1
                break
        else:
            return False
    return True


def _guess_names(folder: str):
    """세션 폴더명 → (물체 종류, 개체 이름). outcomes.json 이 없을 때의 폴백.

    ★ 물체/개체 이름은 학습 라벨의 근간이라 폴더명에만 두면 안 된다(폴더를 옮기면 사라진다).
      그래서 여기서 추정한 값도 session.h5 root attr 로 박아둔다.
    폴더명 형식 세 가지를 모두 받는다:
      현: '<개체>_<파지자세>_<YYYYmmdd>_<HHMMSS>' 예: ecoflex_1_ecoflex_20260729_000753
      구: '<개체>_<YYYYmmdd>_<HHMMSS>'            예: ecoflex_1_20260728_231155
      구: 'collect_<물체>_<YYYYmmdd>_<HHMMSS>'    예: collect_ecoflex_20260728_224840
    개체 이름 자체에 '_' 가 들어가므로(ecoflex_1) **뒤에서 날짜·시각 2토큰을 떼는** 방식이다
    (앞에서 자르면 'ecoflex_1' 의 '_1' 을 잃는다).

    ★ 파지자세 토큰을 개체 이름으로 오인하면 안 된다. 그래서 남은 마지막 토큰이
      '자세 태그처럼 보일 때만' 뗀다 — 판단 기준은 **실제 pose txt 의 stem**
      (여러 개면 '-' 로 이어진 형태) 또는 '5pose' 같은 개수 표기.
      후보 목록(GRIP_POSE_CANDIDATES)만 보면 --grip-pose 로 지정한 목록 밖의 txt 를
      놓치므로, pose 디렉터리에 실제로 있는 txt 까지 합쳐서 본다.
      (collect_ros2._pose_tag 가 stem 안의 '_' 를 '-' 로 바꿔 **항상 1토큰**으로 만든다)
      전부 떼면 이름이 사라지는 경우(개체명 == 자세명, 예: 'ecoflex_20260728_224840')는
      떼지 않는다 — 자세를 살리려고 개체 라벨을 잃는 건 손해다.

    ※ 한계: 개체 이름의 마지막 토큰이 자세 이름과 같으면(예: 개체 'ecoflex_tomato' 를
      자세 토큰 없는 옛 폴더명으로 저장) 그 토큰을 자세로 오인해 뗀다. 폴더명만으로는
      구분할 방법이 없다. 실제 세션은 outcomes.json 의 specimen 이 우선이라 이 함수가
      쓰이지 않고, 그래서 이 추정값은 outcomes.json 이 사라진 폴더에만 적용된다.
    """
    parts = folder.split("_")
    if len(parts) >= 3 and parts[0] == "collect":
        parts = parts[1:]
    if (len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8
            and parts[-1].isdigit() and parts[-2].isdigit()):
        parts = parts[:-2]                          # 날짜·시각 제거
        names = set(CO.GRIP_POSE_CANDIDATES)
        try:                                        # --grip-pose 로 쓴 목록 밖 txt 까지 포함
            names |= {p.name for p in CO.D._POSE_DIR.glob("*.txt")}
        except Exception:
            pass
        stems = {Path(n).stem.replace("_", "-") for n in names}
        tok = parts[-1]
        looks_like_pose = bool(tok) and (re.fullmatch(r"\d+pose", tok) is not None
                                         or _splits_into_stems(tok, stems))
        if len(parts) >= 2 and looks_like_pose:
            parts = parts[:-1]                      # 자세 태그 제거 → 개체 이름만 남는다
    spec = "_".join(parts)
    return (spec.split("_")[0] if spec else ""), spec


def _detect_storage(bag_dir: Path) -> str:
    try:
        import yaml
        d = yaml.safe_load((bag_dir / "metadata.yaml").read_text())
        return d["rosbag2_bagfile_information"]["storage_identifier"]
    except Exception:
        return "sqlite3"


def _read_bag(bag_dir: Path):
    """bag → {채널: (시각[ns], 값 ndarray)} + 마커/구간/판정 목록."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=_detect_storage(bag_dir)),
                rosbag2_py.ConverterOptions("", ""))
    tmap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = {}

    def _m(topic, data):
        if topic not in cls:
            cls[topic] = get_message(tmap[topic])
        return deserialize_message(data, cls[topic])

    ch = {k: ([], []) for k in ("arm_joint", "arm_q_target", "hand_joint",
                                "hand_q_target", "kin", "paxini_raw")}
    seg, demo, sq, outc = [], [], [], {}
    t_min = t_max = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        t_min = t if t_min is None else min(t_min, t)
        t_max = t if t_max is None else max(t_max, t)
        if topic == T["arm_joint"]:
            m = _m(topic, data)
            if len(m.position) >= ARM_DOF:
                ch["arm_joint"][0].append(t)
                ch["arm_joint"][1].append(np.asarray(m.position[:ARM_DOF], np.float32))
        elif topic == T["hand_joint"]:
            m = _m(topic, data)
            if len(m.position) >= HAND_DOF:
                ch["hand_joint"][0].append(t)
                # 라이브 기록과 동일하게 정수 반올림(엔코더 counts)
                ch["hand_joint"][1].append(
                    np.asarray([round(m.position[j]) for j in range(HAND_DOF)], np.float32))
        elif topic in (T["arm_q_target"], T["hand_q_target"], T["kin"], T["paxini_raw"]):
            key = {T["arm_q_target"]: ("arm_q_target", ARM_DOF),
                   T["hand_q_target"]: ("hand_q_target", HAND_DOF),
                   T["kin"]: ("kin", KIN),
                   T["paxini_raw"]: ("paxini_raw", RAW_N)}[topic]
            name, n = key
            m = _m(topic, data)
            if len(m.data) >= n:
                ch[name][0].append(t)
                ch[name][1].append(np.asarray(m.data[:n], np.float32))
        elif topic == T["segment"]:
            seg.append((t, str(_m(topic, data).data)))
        elif topic == T["demo"]:
            p = str(_m(topic, data).data).split(",")
            if len(p) >= 2:
                demo.append((t, p[0].strip(), int(p[1])))
        elif topic == T["squeeze_on"]:
            sq.append((t, int(_m(topic, data).data)))
        elif topic == T["outcome"]:
            p = str(_m(topic, data).data).split(",")
            if len(p) >= 2:
                outc[int(p[0])] = p[1]
    out = {k: (np.asarray(v[0], np.int64), np.asarray(v[1], np.float32))
           for k, v in ch.items() if v[0]}
    return out, sorted(seg), demo, sorted(sq), outc, (t_min or 0), (t_max or 0)


def _zoh(ts, vals, grid):
    """zero-order-hold 샘플링. 반환 (값[len(grid), ...], age_ms[len(grid)]).
       그리드 시각 이전 메시지가 없으면 값=0, age=-1."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    safe = np.clip(idx, 0, len(ts) - 1)
    v = vals[safe]
    v[~ok] = 0.0
    age = ((grid - ts[safe]) / 1e6).astype(np.int32)
    age[~ok] = -1
    return v, age


def _step_series(events, grid, default=0, code=None):
    """(시각, 값) 이벤트를 그리드에 계단 함수로 펼친다. code 가 있으면 값→코드 변환."""
    out = np.full(len(grid), default, np.int16)
    if not events:
        return out
    ts = np.asarray([e[0] for e in events], np.int64)
    vs = [e[1] for e in events]
    idx = np.searchsorted(ts, grid, side="right") - 1
    for i, k in enumerate(idx):
        if k >= 0:
            v = vs[k]
            out[i] = code.get(v, default) if code else v
    return out


def _run_series(demo, grid):
    """demo_marker S/E → 그리드별 run 번호(밖은 -1)."""
    run = np.full(len(grid), -1, np.int16)
    opens = {}
    for t, ev, rid in demo:
        if ev == "S":
            opens[rid] = t
        elif ev == "E" and rid in opens:
            run[(grid >= opens.pop(rid)) & (grid <= t)] = rid
    for rid, t0 in opens.items():                    # 닫히지 않은 run (중단)
        run[grid >= t0] = rid
    return run


def _git_sha(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class ConvertError(RuntimeError):
    """이 세션 하나만 변환 불가(bag 없음/토픽 없음 등). 배치에서는 건너뛰고 계속 간다."""


def _find_sessions(paths):
    """인자 → 세션 폴더 목록(중복 제거). 세션 폴더(bag/ 보유) 자체도, 그것들을 담은 상위
    폴더(collect_logs)도 받는다.

    ★ 상위 폴더를 통째로 받는 이유: 세션이 수십 개라 셸 for 루프로 돌리면 **한 세션이
      죽는 순간 나머지가 안 돈다**(실제로 metadata.yaml 유실 bag 하나 때문에 겪었다).
      여기서 모아 돌면 실패한 세션만 건너뛰고 끝에 목록으로 보고할 수 있다.
    """
    found, seen = [], set()
    for p in paths:
        d = Path(str(p).rstrip("/")).expanduser()
        if not d.is_dir():
            raise SystemExit(f"폴더 없음: {d}")
        cands = [d] if (d / "bag").is_dir() else [
            c for c in sorted(d.iterdir()) if c.is_dir() and (c / "bag").is_dir()]
        if not cands:
            raise SystemExit(f"bag 폴더가 없음: {d}\n"
                             "  세션 폴더(bag/ 를 가진 폴더)나 그 세션들을 담은 상위 폴더"
                             "(collect_logs)를 지정하세요.")
        for c in cands:
            r = c.resolve()
            if r not in seen:
                seen.add(r)
                found.append(c)
    return found


def _export_json(sess: Path, out_dir: Path) -> int:
    """세션 폴더의 *.json(outcomes.json)을 출력 폴더로 복사 → 복사한 개수.

    ★ h5 만 옮기면 안 되는 이유: outcomes.json 의 run 항목에는 /runs 에 **없는 값**이 있다
      (thumb_return_A/B_held·_peak_force_n·_fingers, squeeze_*_delta_n, 판정 시각 wall,
      groups …). /runs 는 학습에 바로 쓰는 열만 추린 것이라 원문은 json 뿐이다.
      bag 없는 트리로 재구성할 때 json 이 빠지면 그 정보가 bag 폴더에 남아 같이 못 간다.
    """
    n = 0
    for src in sorted(sess.glob("*.json")):
        dst = out_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)
        n += 1
    return n


def convert_one(sess: Path, out_path: Path, rate: float, no_raw: bool) -> Path:
    """세션 폴더 하나 → out_path 에 session.h5 작성. 이 세션만의 문제는 ConvertError."""
    bag_dir = sess / "bag"
    if not bag_dir.is_dir():
        raise ConvertError(f"bag 폴더 없음: {bag_dir}")

    ch, seg, demo, sq, outc, t_min, t_max = _read_bag(bag_dir)
    if not ch:
        raise ConvertError("bag 에 센서 토픽이 없음 — 수집 설정 확인.")
    if not seg:
        raise ConvertError("bag 에 /collect/segment 라벨 없음 — 새 collect_ros2 로 수집한 bag 필요.")

    step = int(1e9 / rate)
    grid = np.arange(int(t_min), int(t_max) + 1, step, dtype=np.int64)
    print(f"[session] bag={bag_dir}  {(t_max - t_min) / 1e9:.1f}s  → "
          f"{len(grid)} tick @ {rate:g}Hz")

    # ── 채널 샘플링 ──
    want = [k for k in ("arm_joint", "arm_q_target", "hand_joint", "hand_q_target",
                        "kin", "paxini_raw") if k in ch]
    data, ages = {}, {}
    for k in want:
        ts, vals = ch[k]
        v, age = _zoh(ts, vals, grid)
        data[k], ages[k] = v, age
        print(f"[session]   {k:14s} 원본 {len(ts):5d}개 "
              f"({len(ts) / ((t_max - t_min) / 1e9):5.1f}Hz) → hold, "
              f"미수신 {int((age < 0).sum())} tick")

    # paxini: raw(4,127,3) → resultant(4,3) 는 배포와 같은 함수로 계산(Σ127 = §F7 표준)
    if "paxini_raw" in data:
        tac = data["paxini_raw"].reshape(-1, FINGERS, POINTS, 3)
        data["paxini_resultant"] = np.stack(
            [RE.resultant_from_tactile(f).astype(np.float32) for f in tac])
        ages["paxini_resultant"] = ages["paxini_raw"]
        valid = (ages["paxini_raw"] >= 0).astype(np.int8)
    else:
        valid = np.zeros(len(grid), np.int8)

    phase = _step_series(seg, grid, default=0, code=PHASE).astype(np.int8)
    run = _run_series(demo, grid)
    squeeze_on = _step_series(sq, grid, default=0).astype(np.int8)
    phase[run < 0] = 0                          # run 밖은 무조건 idle

    # ── /runs (데모 분할 기준표) ──
    meta_path = sess / "outcomes.json"
    side, fruit, specimen, collect_attrs = {}, "", "", {}
    if meta_path.exists():
        try:
            _j = json.loads(meta_path.read_text(encoding="utf-8"))
            side = _j.get("runs", {})
            fruit = str(_j.get("fruit", ""))
            specimen = str(_j.get("specimen", ""))   # 개체 이름(fruit 와 구분)
            # 수집 시점 provenance(git_sha·FACTOR·USE_JKIN·t_offset_ns·힘범위 …).
            #   라이브 h5 폐지 후 이 경로가 유일한 전달 통로다 → collect_* 접두사로 root attr 에.
            collect_attrs = {f"collect_{k}": v for k, v in (_j.get("session") or {}).items()}
        except Exception as e:
            print(f"[session]   ⚠ outcomes.json 읽기 실패({e}) — /runs 는 bag 정보만 채운다")
    if not fruit or not specimen:
        g_fruit, g_spec = _guess_names(sess.name)
        specimen = specimen or g_spec
        fruit = fruit or g_fruit
        if g_spec:
            print(f"[session]   ℹ 폴더명에서 추정: 물체={fruit!r} 개체={specimen!r} "
                  "(outcomes.json 이 있으면 그 값을 쓴다)")
    arm_code, hand_code = _pose_codes()
    runs = []
    for rid in sorted({int(r) for r in run if r >= 0}):
        rows = np.nonzero(run == rid)[0]
        s = side.get(str(rid), {})
        oc = s.get("outcome") or outc.get(rid, "unjudged")
        runs.append((
            rid, int(rows[0]), int(rows[-1]) + 1, int(grid[rows[0]]), int(grid[rows[-1]]),
            OUTCOME_CODE.get(oc, 0),
            arm_code.get(s.get("present_pose_up", ""), 0),
            arm_code.get(s.get("present_pose_down", ""), 0),
            hand_code.get(s.get("hand_pose_file", ""), 0),
            float(s.get("grip_force_threshold_n", np.nan)),
            float(s.get("squeeze_A_threshold_n", np.nan)),
            float(s.get("squeeze_B_threshold_n", np.nan)),
            int(s.get("grip_reached_fingers", -1)),
            float(s.get("grip_peak_force_n", np.nan)),
        ))
    RUN_COLS = ("run", "row0", "row1", "t0_ns", "t1_ns", "outcome_code",
                "pose_palm_up", "pose_palm_down", "pose_hand",
                "grip_thr_n", "squeezeA_thr_n", "squeezeB_thr_n",
                "grip_reached_fingers", "grip_peak_force_n")
    # ★ 자세·판정을 **이름으로도** 남긴다. 숫자 코드는 후보 목록이 바뀌면 밀린다 —
    #   실제로 GRIP_POSE_CANDIDATES 에 ecoflex.txt 를 추가하자 tomato.txt 코드가 4→5 로
    #   바뀌었다. /codes 는 파일마다 저장되니 단일 세션은 안전하지만, **여러 세션을 합쳐
    #   학습할 때는 숫자가 서로 다른 뜻**이 된다 → 병합은 이 이름 열로 join 할 것.
    run_names = [[str(side.get(str(int(r[0])), {}).get(k, ""))
                  for k in ("present_pose_up", "present_pose_down", "hand_pose_file")]
                 + [str(side.get(str(int(r[0])), {}).get("outcome")
                        or outc.get(int(r[0]), "unjudged"))]
                 for r in runs]
    NAME_COLS = ("pose_palm_up", "pose_palm_down", "pose_hand", "outcome")

    # ── 저장 ──
    with h5py.File(out_path, "w") as f:
        f.attrs.update(
            schema="session_v1 (연속 타임라인)",
            fruit=fruit,                 # ★ 물체 종류 = 학습 라벨의 근간. 폴더명에만 두면
            specimen=(specimen or fruit),  # ★ 개체 이름(같은 재료의 개체 구분 — 일반화 평가용)
            session=sess.name,           #   폴더를 옮기는 순간 사라진다 → 파일 안에 박는다.
            source_bag=str(bag_dir),
            tick_hz=float(rate),
            tick_rule="zero-order-hold (그 시각 이전 마지막 메시지)",
            clock="epoch_ns (bag 수신시각)",
            n_ticks=len(grid), n_runs=len(runs),
            created_wall=datetime.now().isoformat(),
            git_sha=_git_sha(str(Path(_LAUNCH_DIR).parents[1])),
            kin_source=("mN_side_channel" if RE.USE_MN_SIDE_CHANNEL else "SHM_raw"),
            note="자세/판정 이름은 /runs_names 참조. 데모 분할은 /runs. phase 숫자는 "
                 "bag_to_session.PHASE 표 참조(파일에는 코드표를 넣지 않는다).",
        )
        f.attrs.update(collect_attrs)        # 수집 시점 provenance (collect_* 접두사)
        g = f.create_group("data")
        g.create_dataset("t_ns", data=grid, compression="gzip")
        g.create_dataset("run", data=run, compression="gzip")
        g.create_dataset("phase", data=phase, compression="gzip")
        g.create_dataset("squeeze_on", data=squeeze_on, compression="gzip")
        g.create_dataset("valid", data=valid, compression="gzip")
        shapes = {"paxini_resultant": (FINGERS, 3), "paxini_raw": (FINGERS, POINTS, 3)}
        for k in list(data):
            if k == "paxini_raw" and no_raw:
                continue
            arr = data[k].reshape((len(grid),) + shapes.get(k, (-1,)))
            d = g.create_dataset(k, data=arr, compression="gzip")
            d.attrs["units"] = UNITS.get(k, "?")
            d.attrs["source_topic"] = T.get(k, "computed(resultant_from_tactile)")
        # 채널별 '마지막 갱신 후 경과' — 센서 정지(값 0 인데 valid) 를 사후에 걸러내려면 필요.
        ak = [k for k in want if k in ages]
        ag = g.create_dataset("age_ms", data=np.stack([ages[k] for k in ak], axis=1),
                              compression="gzip")
        ag.attrs["channels"] = ",".join(ak)
        ag.attrs["note"] = "-1 = 그 tick 까지 해당 채널 메시지 없음"

        r = f.create_dataset("runs", data=np.asarray(runs, np.float64), compression="gzip") \
            if runs else f.create_dataset("runs", shape=(0, len(RUN_COLS)), dtype="f8")
        r.attrs["columns"] = ",".join(RUN_COLS)
        r.attrs["note"] = "row0:row1 로 /data 를 잘라 데모별 분리. NaN = 정보 없음(구 세션)"
        rn = f.create_dataset("runs_names",
                              data=np.asarray(run_names or [[]], dtype=h5py.string_dtype()),
                              compression="gzip") if run_names else \
            f.create_dataset("runs_names", shape=(0, len(NAME_COLS)),
                             dtype=h5py.string_dtype())
        rn.attrs["columns"] = ",".join(NAME_COLS)
        rn.attrs["note"] = ("/runs 와 같은 행 순서. **세션을 여러 개 합칠 때는 숫자 코드가 아니라 "
                            "이 이름으로 join 할 것** — 후보 목록이 바뀌면 코드가 밀린다.")

        # ※ /codes 그룹(숫자↔이름 대응표 + arm_pose_joints)은 **저장하지 않는다**
        #   (2026-07-29 사용자 지시 — 데이터만 남기고 코드표는 불필요).
        #   자세·판정 숫자의 뜻은 /runs_names(문자열)로 그대로 확인할 수 있고, phase 숫자의
        #   뜻은 이 스크립트의 PHASE 표가 유일한 출처가 된다(위 PHASE 주석 참고).

    mb = out_path.stat().st_size / 1048576
    print(f"[session] 완료: {out_path}  ({mb:.1f} MiB)")
    print(f"[session]   /data {len(grid)} tick · 채널 {sorted(data)}")
    print(f"[session]   /runs {len(runs)} run · phase 분포 "
          + ", ".join(f"{k}={int((phase == v).sum())}"
                      for k, v in sorted(set(PHASE.items()), key=lambda x: x[1])
                      if (phase == v).any()))

    # ── 품질 점검: ★스퀴즈 구간에서 촉각이 stale 하지 않은가 ──────────────────
    #   왜 이 구간만 보나: paxini 발행 공백(todolist 12번)은 실측상 **팔 이동 구간에만** 몰린다
    #   (제어 PC 가 궤적 스트리밍으로 바쁠 때 tactile 타이머가 밀림). 힘 신호가 학습에 쓰이는
    #   곳은 스퀴즈 A/B 뿐이라, 거기가 깨끗하면 데이터는 쓸 수 있다. 실측 2세션 모두 0.0% 였다.
    #   → 이후 세션에서 그 전제가 깨지면 **여기서 자동으로 걸리게** 한다(12번 조사 대신 감시).
    if "paxini_raw" in ages:
        stale_ms, worst = STALE_WARN_MS, []
        for code, name in ((PHASE["squeeze_A"], "squeeze_A"), (PHASE["squeeze_B"], "squeeze_B")):
            m = phase == code
            if not m.any():
                continue
            a = ages["paxini_raw"][m]
            frac = float((a > stale_ms).mean())
            worst.append((name, int(m.sum()), frac, int(a.max())))
        for name, n, frac, mx in worst:
            tag = "✔" if frac == 0 else ("⚠" if frac < 0.05 else "✘")
            print(f"[session]   {tag} {name}: 촉각 stale(>{stale_ms}ms) "
                  f"{frac * 100:.1f}% ({n} tick 중) · 최대 {mx}ms")
        if any(f >= 0.05 for _n, _c, f, _m in worst):
            print(f"[session]   ✘ 스퀴즈 구간 촉각이 {stale_ms}ms 넘게 정체한 tick 이 5% 이상 —\n"
                  "            힘 신호가 오래된 값으로 채워졌다. age_ms 로 걸러내거나 재수집 검토.\n"
                  "            (원인 추적: docs/todolist.md 12번 — 제어 PC 의 tactile 발행 경로)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="rosbag → session.h5 (연속 타임라인)")
    ap.add_argument("session", nargs="+",
                    help="collect 세션 폴더(bag/ 포함) 또는 세션들을 담은 상위 폴더(collect_logs)")
    ap.add_argument("--rate", type=float, default=100.0, help="tick 그리드 [Hz] (기본 100)")
    ap.add_argument("--out", default=None,
                    help="출력 h5 경로 (기본 <세션>/session.h5). 세션 1개일 때만 쓸 수 있다")
    ap.add_argument("--out-root", default=None, help=(
        "h5/json 만 모을 별도 트리의 뿌리. <out-root>/<세션폴더명>/session.h5 + outcomes.json "
        "으로 내보낸다(bag 은 원본 폴더에 그대로 둔다 — 학습 PC 로 옮길 가벼운 사본용)"))
    ap.add_argument("--no-raw", action="store_true",
                    help="paxini_raw(용량 90%%) 제외 — resultant 만으로 충분할 때")
    ap.add_argument("--reuse-h5", action="store_true", help=(
        "세션 폴더에 이미 있는 session.h5 를 그대로 복사한다(bag 재변환 없음). "
        "--out-root 로 '옮기기만' 할 때 쓴다 — 없는 세션은 실패로 보고"))
    ap.add_argument("--skip-existing", action="store_true",
                    help="출력 h5 가 이미 있으면 건너뛴다(중단된 배치 이어서 돌리기)")
    args = ap.parse_args()

    sessions = _find_sessions(args.session)
    out_root = Path(args.out_root).expanduser() if args.out_root else None
    if args.out and (out_root or len(sessions) > 1):
        raise SystemExit("--out 은 세션 1개 전용입니다 — 여러 세션은 --out-root 를 쓰세요.")
    if out_root:
        print(f"[session] 세션 {len(sessions)}개 → {out_root} (h5 + json 만, bag 은 원본 유지)")

    ok, skipped, failed = [], [], []
    for i, sess in enumerate(sessions, 1):
        out_dir = (out_root / sess.name) if out_root else sess
        out_path = Path(args.out).expanduser() if args.out else out_dir / "session.h5"
        if len(sessions) > 1:
            print(f"\n[{i}/{len(sessions)}] {sess.name}")
        try:
            if args.skip_existing and out_path.exists():
                print(f"[session] 이미 있음 — 건너뜀: {out_path}")
                skipped.append(sess.name)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.reuse_h5:
                src = sess / "session.h5"
                if not src.exists():
                    raise ConvertError(f"session.h5 없음(--reuse-h5): {src}")
                if src.resolve() != out_path.resolve():
                    shutil.copy2(src, out_path)
                print(f"[session] 기존 h5 복사: {src} → {out_path} "
                      f"({out_path.stat().st_size / 1048576:.1f} MiB)")
            else:
                convert_one(sess, out_path, args.rate, args.no_raw)
            if out_root:
                n = _export_json(sess, out_dir)
                print(f"[session]   json {n}개 복사 → {out_dir}")
            ok.append(sess.name)
        except (Exception, SystemExit) as e:
            # 세션이 하나뿐이면 원래대로 그대로 터뜨린다(traceback 이 있어야 원인을 본다).
            # 배치일 때만 삼키고 계속 — 마지막에 실패 목록을 다시 찍는다.
            if len(sessions) == 1:
                raise
            print(f"[session]   ✘ 실패 — 건너뜀: {e}")
            failed.append((sess.name, str(e)))

    if len(sessions) > 1 or out_root:
        print("\n" + "─" * 70)
        print(f"[session] 성공 {len(ok)} · 건너뜀 {len(skipped)} · 실패 {len(failed)} "
              f"/ 전체 {len(sessions)}")
        for name, err in failed:
            print(f"          ✘ {name}: {err}")
        if out_root:
            print(f"[session] 재구성한 트리: {out_root}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
