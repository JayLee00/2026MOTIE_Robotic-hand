#!/usr/bin/env python3
"""check_parity_timing.py — parity 불일치가 '시점 차'인지 **bit-exact 로** 증명한다.

verify_parity 의 PASS*/FAIL 은 허용오차 판정이라 경계에서 애매해진다. 이 도구는 bag 의
**원본 메시지 스트림**과 직접 대조하므로 판정이 아니라 사실 확인이다:

  A 값과 B 값 **둘 다** bag 원본 메시지에 존재하면
     → 같은 스트림에서 '다른 시점'을 뽑은 것(시점 차). 데이터 불일치가 아니다.
  A 는 있고 B 가 없으면 → **변환기(bag_to_hdf5) 쪽 문제**(리샘플·스케일·손상).
  A 가 없으면          → **라이브 기록 쪽 문제**.
  (A 만 검사하면 B 가 오염된 경우를 '시점 차'로 오판한다 — 양쪽을 본다.)

왜 A 가 더 오래된 값을 드나: 라이브는 tick 순간 executor 가 아직 처리하지 못한 메시지를
보지 못하고, bag 은 도착시각 기준이라 그 메시지를 포함한다 → 최대 1 메시지(≈5ms) staleness.

실행:
  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
  python3 tools/check_parity_timing.py collect_logs/<개체>_<파지자세>_<ts>
  #   종료코드 0 = 모든 불일치가 원본 스트림에 존재(시점 차) / 1 = 설명 안 되는 불일치 있음
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np

_LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "stiffness_deploy_ros2", "launch")
sys.path.insert(0, os.path.abspath(os.path.join(_LAUNCH, "..")))
sys.path.insert(0, os.path.abspath(_LAUNCH))

import rosbag2_py                                                    # noqa: E402
from rclpy.serialization import deserialize_message                   # noqa: E402
from rosidl_runtime_py.utilities import get_message                   # noqa: E402
import bag_to_hdf5 as B2                                              # noqa: E402
import real_deploy_inference_final as RE     # noqa: E402  resultant_from_tactile (동일 계산)

def _tactile(m):
    """/paxini/right/raw → (4,127,3). HDF5 'tactile' 과 같은 형상."""
    if len(m.data) < B2.RAW_N:
        return None
    return np.asarray(m.data[:B2.RAW_N], np.float64).reshape(B2.FINGERS, B2.POINTS, 3)


def _resultant(m):
    """raw → resultant(4,3). 라이브·변환기와 **같은 함수**로 계산해야 비교가 성립한다."""
    tac = _tactile(m)
    return None if tac is None else np.asarray(
        RE.resultant_from_tactile(tac.astype(np.float32)), np.float64)


# HDF5 채널 → (bag 토픽 키, 메시지에서 값 뽑는 함수)
#   ★ paxini 계열(tactile/resultant)까지 포함해야 한다 — 예전엔 joint·ft 만 봐서
#     "불일치 전부 시점 차" 라고 보고하면서 실제로는 그 두 채널을 검사하지 않았다.
SRC = {
    "joint": ("hand_joint",
              lambda m: (np.array([round(m.position[j]) for j in range(B2.HAND_DOF)],
                                  np.float64)
                         if len(m.position) >= B2.HAND_DOF else None)),
    "ft":    ("kin",
              lambda m: (np.asarray(m.data[:B2.KIN], np.float64)
                         if len(m.data) >= B2.KIN else None)),
    "tactile":   ("raw", _tactile),
    "resultant": ("raw", _resultant),
}


def read_raw(bag_dir, storage):
    """bag 원본 메시지 스트림 → {채널: (시각[ns], 값)}"""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
           rosbag2_py.ConverterOptions("", ""))
    tmap = {t.name: t.type for t in r.get_all_topics_and_types()}
    # ★ 한 토픽에서 여러 채널이 나온다(raw → tactile·resultant) → 토픽당 '리스트' 로 모은다.
    #   dict 를 토픽 키로 쓰면 뒤 채널이 앞 채널을 덮어써서 조용히 하나만 검사하게 된다.
    want: dict[str, list] = {}
    for ch, (k, fn) in SRC.items():
        if B2.T[k] in tmap:
            want.setdefault(B2.T[k], []).append((ch, fn))
    acc = {ch: ([], []) for lst in want.values() for ch, _ in lst}
    while r.has_next():
        topic, data, t = r.read_next()
        if topic not in want:
            continue
        msg = deserialize_message(data, get_message(tmap[topic]))
        for ch, fn in want[topic]:
            v = fn(msg)
            if v is not None:
                acc[ch][0].append(t)
                acc[ch][1].append(v)
    return {ch: (np.asarray(ts, np.int64), np.asarray(vs, np.float64))
            for ch, (ts, vs) in acc.items() if ts}


def main():
    ap = argparse.ArgumentParser(description="parity 불일치의 시점차 여부를 bag 원본으로 증명")
    ap.add_argument("session", help="수집 세션 폴더 (bag/ + 라이브 h5 + from_bag.h5)")
    ap.add_argument("--tol", type=float, default=1e-3, help="불일치로 볼 최소 오차")
    args = ap.parse_args()

    sess = args.session.rstrip("/")
    # 라이브 h5 이름: 현 '<개체>_<파지자세>_<ts>.h5' / 구 '<개체>_<ts>.h5' / 구 'collect_<물체>_<ts>.h5'.
    #   session.h5·from_bag.h5 는 제외해야 A 로 오인하지 않는다.
    a_paths = sorted(p for p in glob.glob(f"{sess}/*.h5")
                     if Path(p).name not in ("session.h5", "from_bag.h5"))
    if not a_paths:
        raise SystemExit(f"라이브 HDF5 없음: {sess} (--live-h5 로 수집한 세션만 대조 가능)")
    b_path = f"{sess}/from_bag.h5"
    if not os.path.exists(b_path):
        raise SystemExit(f"{b_path} 없음 — bag_to_hdf5.py 를 먼저 실행하세요.")
    bag_dir = f"{sess}/bag"

    print(f"세션: {sess}")
    raw = read_raw(bag_dir, B2.detect_storage(Path(bag_dir)))
    for ch, (ts, _v) in raw.items():
        print(f"  원본 {ch:5s}: {len(ts)}개, 간격 중앙 {np.median(np.diff(ts))/1e6:.2f}ms")

    A, B = h5py.File(a_paths[0]), h5py.File(b_path)
    off = int(A.attrs.get("t_offset_ns", 0))
    if not off:
        print("  ⚠ A 에 t_offset_ns 없음 — 구파일은 시각 대조 불가")

    total_bad = explained = 0
    for seg in sorted(set(A) & set(B)):
        ga, gb = A[seg], B[seg]
        if "t_mono_ns" not in ga or "t_ns" not in gb:
            continue
        ta = ga["t_mono_ns"][:].astype(np.int64) + off
        tb = gb["t_ns"][:].astype(np.int64)
        idx = np.abs(tb[None, :] - ta[:, None]).argmin(axis=1)
        print(f"\n=== {seg} ===")
        for ch in SRC:
            if ch not in ga or ch not in gb or ch not in raw:
                continue
            x = ga[ch][:].astype(np.float64).reshape(len(ga[ch]), -1)
            y = gb[ch][:].astype(np.float64).reshape(len(gb[ch]), -1)
            bad = np.where(np.abs(x - y[idx]).max(axis=1) > args.tol)[0]
            rts, rvs = raw[ch]
            rvs2 = rvs.reshape(len(rvs), -1)
            if not len(bad):
                print(f"  {ch:5s} 불일치 0 — bit-exact 일치")
                continue
            def in_raw(vec):
                """이 값이 원본 메시지에 정확히 존재하나 → 그 인덱스(없으면 None)."""
                eq = np.where(np.abs(rvs2 - vec).max(axis=1) < 1e-9)[0]
                return eq if len(eq) else None

            lags, miss_a, miss_b = [], [], []
            for j in bad:
                ea, eb = in_raw(x[j]), in_raw(y[idx[j]])
                if ea is None:
                    miss_a.append(int(j))
                if eb is None:
                    miss_b.append(int(j))
                if ea is not None and eb is not None:
                    k = ea[np.abs(rts[ea] - ta[j]).argmin()]
                    lags.append((rts[k] - ta[j]) / 1e6)
            total_bad += len(bad)
            explained += len(lags)
            dt = np.median(np.diff(rts)) / 1e6
            if not miss_a and not miss_b:
                tag = "✔ A·B 둘 다 원본에 존재(시점 차)"
            elif miss_b and not miss_a:
                tag = f"✘ B 값이 원본에 없음 → 변환기 문제: {miss_b[:8]}"
            elif miss_a and not miss_b:
                tag = f"✘ A 값이 원본에 없음 → 라이브 기록 문제: {miss_a[:8]}"
            else:
                tag = f"✘ A·B 둘 다 원본에 없음: A{miss_a[:5]} B{miss_b[:5]}"
            extra = (f" Δt {min(lags):+.2f}~{max(lags):+.2f}ms "
                     f"(≈{abs(np.mean(lags))/dt:.1f} 메시지)" if lags else "")
            print(f"  {ch:5s} 불일치 {len(bad):3d}/{len(x)}  {tag}{extra}")

    print("\n" + "=" * 72)
    if total_bad == 0:
        print("불일치 프레임 없음 — A 와 B 가 bit-exact 로 같다.")
    elif explained == total_bad:
        print(f"불일치 {total_bad}프레임 전부 bag 원본 스트림에 존재 → **시점 차**. "
              "데이터 불일치 아님(수집=bag 재현 성립).")
    else:
        print(f"불일치 {total_bad}프레임 중 {total_bad - explained}개가 원본에 없음 "
              "→ **진짜 불일치**. 출처/스케일/필터를 추적하세요.")
    A.close(); B.close()
    sys.exit(0 if explained == total_bad else 1)


if __name__ == "__main__":
    main()
