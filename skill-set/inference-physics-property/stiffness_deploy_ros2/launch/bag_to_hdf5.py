#!/usr/bin/env python3
"""bag_to_hdf5.py — rosbag(collect_ros2 기록) → '구간(segment)별' HDF5 변환기 (Option 2a).

collect_ros2 의 8단계 시퀀스는 run 당 여러 ★ 구간(squeeze_A / move_palm_down / squeeze_B)을
갖고, 각 구간을 /collect/segment(String) 라벨로 표시한다. 이 변환기는 bag 의 raw 토픽을
배포 read_live_sample 과 같은 causal-ZOH 로 재생하되, '구간별 레시피'로 추출한다:

  · 스퀴즈(squeeze_A/B): 손 tick(/hand/right/q_target), 손관절+kin+resultant+tactile.
      → 라이브 Option 1 HDF5(스퀴즈 그룹)와 동일 프레임(= 모델 입력). squeeze_on 로 창을 좁힘.
  · palm-down(move_palm_down): 팔 tick(/franka/right/q_target), 팔관절(+명령)+손관절+resultant+
      tactile. → 라이브 HDF5 엔 없는(손 add_sample 없음) '팔/촉각' 구간을 여기서 뽑는다.

출력: 구간마다 HDF5 그룹 하나(`{segment}__run{NNN}`) + attrs(run, segment, tick, n_samples ...).

전제: bag 에 /collect/segment 가 있어야 함(= 새 collect_ros2 로 수집). rosbag2_py 필요.
실행:
  source env.sh
  python3 stiffness_deploy_ros2/launch/bag_to_hdf5.py collect_logs/collect_tomato_<ts>/bag
  #  --segments squeeze_A,squeeze_B,move_palm_down (기본 전부)  | --paxini auto|ft|raw

  # ★ 여러 세션 한 번에 + **h5/json 만 있는 가벼운 트리로 재구성**(--out-root)
  #   bag 은 세션당 수백 MB 라 그대로는 학습 PC 로 못 옮긴다. bag 은 원본 자리에 두고
  #   결과 h5 와 outcomes.json 만 별도 트리에 쌓는다(bag_to_session.py 와 같은 규칙).
  python3 .../bag_to_hdf5.py ~/Desktop/collect_logs_recovered/collect_logs \
          --out-root ~/Desktop/collect_h5
  #   → <out-root>/<세션폴더명>/from_bag.h5 + outcomes.json   (bag 없음)
  #   인자는 bag 폴더 / 세션 폴더 / 세션들의 상위 폴더 아무거나 받는다. --skip-existing 로 이어서.
"""
from __future__ import annotations

import argparse
import bisect
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_LAUNCH_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))
sys.path.insert(0, _LAUNCH_DIR)

import rosbag2_py                                             # noqa: E402
from rclpy.serialization import deserialize_message          # noqa: E402
from rosidl_runtime_py.utilities import get_message          # noqa: E402

import real_deploy_inference_final as RE                      # noqa: E402  (resultant_from_tactile)

HAND_DOF, ARM_DOF, KIN = 16, 7, 12
FINGERS, POINTS = 4, 127
RAW_N = FINGERS * POINTS * 3          # 1524

T = {
    "hand_joint": "/hand/right/joint_states",
    "arm_joint":  "/franka/right/joint_states",
    "kin":        "/hand/right/kin",
    "ft":         "/paxini/right/ft",
    "raw":        "/paxini/right/raw",
    "hand_qtar":  "/hand/right/q_target",
    "arm_qtar":   "/franka/right/q_target",
    "segment":    "/collect/segment",
    "demo":       "/collect/demo_marker",
    "squeeze_on": "/collect/squeeze_on",
    "outcome":    "/collect/demo_outcome",
}

# 구간별 추출 레시피. tick=어느 q_target 발행시각을 샘플 시각으로 쓸지. refine=창 좁힘(squeeze_on).
#   channels=(HDF5 dataset 이름, 소스 kind). require_valid=paxini 무효 프레임 스킵 여부.
SEGMENT_SPECS = {
    "squeeze_A": dict(tick="hand_qtar", refine="squeeze_on", require_valid=True, channels=[
        ("joint", "hand_joint"), ("ft", "kin"), ("resultant", "resultant"), ("tactile", "tactile")]),
    "squeeze_B": dict(tick="hand_qtar", refine="squeeze_on", require_valid=True, channels=[
        ("joint", "hand_joint"), ("ft", "kin"), ("resultant", "resultant"), ("tactile", "tactile")]),
    "move_palm_down": dict(tick="arm_qtar", refine=None, require_valid=False, channels=[
        ("arm_joint", "arm_joint"), ("arm_q_target", "arm_qtar_v"), ("hand_joint", "hand_joint"),
        ("resultant", "resultant"), ("tactile", "tactile")]),
}


def detect_storage(bag_dir: Path) -> str:
    try:
        import yaml
        d = yaml.safe_load((bag_dir / "metadata.yaml").read_text())
        return d["rosbag2_bagfile_information"]["storage_identifier"]
    except Exception:
        return "sqlite3"


def read_streams(bag_dir: Path, storage: str, paxini_pref: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
                rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = set(type_map)
    pax_src = ("raw" if T["raw"] in present else "ft") if paxini_pref == "auto" else paxini_pref
    pax_topic = T[pax_src]
    # ★ P3#10: 요청한 paxini 소스가 bag 에 없으면 즉시 거부한다.
    #   예전에는 pax 스트림이 빈 채로 진행돼 **에러 없이** 스퀴즈 구간이 0프레임이 되고
    #   (require_valid=True 라 전 프레임 스킵) move_palm_down 1그룹만 나왔다 → 잘못된
    #   플래그가 조용히 절반짜리 데이터셋을 만든다. `--paxini raw` 가 표준(§F7)이 된 뒤로는
    #   오타/습관으로 `ft` 를 주기 쉬워 위험이 커졌다.
    if pax_topic not in present:
        have = sorted(t for t in present if t.startswith("/paxini/"))
        raise SystemExit(
            f"[bag2h5] paxini 소스 '{pax_src}' 토픽이 bag 에 없음: {pax_topic}\n"
            f"  bag 의 paxini 토픽: {have or '없음'}\n"
            + ("  → --paxini 를 bag 에 있는 소스로 지정하세요(표준은 raw)."
               if paxini_pref != "auto" else
               "  → paxini 토픽이 전혀 기록되지 않은 bag 입니다(수집 설정 확인)."))

    S = {k: ([], []) for k in ("hand_joint", "arm_joint", "kin", "pax", "arm_qtar_v")}
    hand_qtar_ts, arm_qtar_ts, seg, demos, sq, outcomes = [], [], [], [], [], {}
    t_min, t_max = None, None
    _cls = {}

    def cls(topic):
        if topic not in _cls:
            _cls[topic] = get_message(type_map[topic])
        return _cls[topic]

    while reader.has_next():
        topic, data, t = reader.read_next()
        t_min = t if t_min is None else min(t_min, t)
        t_max = t if t_max is None else max(t_max, t)
        if topic == T["hand_joint"]:
            m = deserialize_message(data, cls(topic))
            if len(m.position) >= HAND_DOF:
                S["hand_joint"][0].append(t)
                S["hand_joint"][1].append(np.array([int(round(m.position[j])) for j in range(HAND_DOF)], np.float32))
        elif topic == T["arm_joint"]:
            m = deserialize_message(data, cls(topic))
            if len(m.position) >= ARM_DOF:
                S["arm_joint"][0].append(t)
                S["arm_joint"][1].append(np.array([float(m.position[j]) for j in range(ARM_DOF)], np.float32))
        elif topic == T["kin"]:
            m = deserialize_message(data, cls(topic))
            if len(m.data) >= KIN:
                S["kin"][0].append(t); S["kin"][1].append(np.array(m.data[:KIN], np.float32))
        elif topic == pax_topic:
            m = deserialize_message(data, cls(topic))
            tac = np.zeros((FINGERS, POINTS, 3), np.float32)
            if pax_src == "raw" and len(m.data) >= RAW_N:
                tac = np.asarray(m.data[:RAW_N], np.float32).reshape(FINGERS, POINTS, 3)
            elif pax_src == "ft" and len(m.data) >= KIN:
                tac[:, 0, :] = np.asarray(m.data[:KIN], np.float32).reshape(FINGERS, 3)
            else:
                continue
            S["pax"][0].append(t); S["pax"][1].append(tac)
        elif topic == T["hand_qtar"]:
            hand_qtar_ts.append(t)
        elif topic == T["arm_qtar"]:
            m = deserialize_message(data, cls(topic))
            arm_qtar_ts.append(t)
            v = np.array(m.data[:ARM_DOF], np.float32) if len(m.data) >= ARM_DOF else np.zeros(ARM_DOF, np.float32)
            S["arm_qtar_v"][0].append(t); S["arm_qtar_v"][1].append(v)
        elif topic == T["segment"]:
            m = deserialize_message(data, cls(topic))
            seg.append((t, str(m.data)))
        elif topic == T["demo"]:
            m = deserialize_message(data, cls(topic))
            p = str(m.data).split(",")
            if len(p) >= 2:
                demos.append((t, p[0].strip(), int(p[1])))
        elif topic == T["squeeze_on"]:
            m = deserialize_message(data, cls(topic))
            sq.append((t, int(m.data)))
        elif topic == T["outcome"]:
            m = deserialize_message(data, cls(topic))
            p = str(m.data).split(",")
            if len(p) >= 2:
                outcomes[int(p[0])] = p[1]        # run_id → outcome

    return dict(S=S, hand_qtar_ts=sorted(hand_qtar_ts), arm_qtar_ts=sorted(arm_qtar_ts),
                seg=sorted(seg), demos=demos, sq=sorted(sq), outcomes=outcomes,
                pax_src=pax_src, t_min=t_min or 0, t_max=t_max or 0)


def _latest(pair, t):
    ts, vs = pair
    i = bisect.bisect_right(ts, t) - 1
    return vs[i] if i >= 0 else None


def _seen(ts, t):
    return bisect.bisect_right(ts, t) - 1 >= 0


def segment_windows(seg, t_max):
    """/collect/segment 전이 → [(label, t0, t1)] (빈 라벨 제외)."""
    out = []
    for i, (t, lab) in enumerate(seg):
        if not lab:
            continue
        t1 = seg[i + 1][0] if i + 1 < len(seg) else t_max
        out.append((lab, t, t1))
    return out


def run_of(demos, t0):
    """demo_marker S/E 로 t0 가 속한 run id (없으면 -1)."""
    opens = {}
    for t, ev, rid in demos:
        if ev == "S":
            opens[rid] = t
        elif ev == "E" and rid in opens:
            if opens[rid] <= t0 <= t:
                return rid
            del opens[rid]
    for rid, ts in opens.items():          # 아직 안 닫힌 run
        if ts <= t0:
            return rid
    return -1


def refine_squeeze(sq, t0, t1):
    """[t0,t1] 안에서 squeeze_on==1 [rise,fall] (add_sample 창). 없으면 원 창."""
    on = [t for t, v in sq if v == 1 and t0 <= t <= t1]
    if not on:
        return t0, t1
    rise = min(on)
    off = [t for t, v in sq if v == 0 and rise < t <= t1]
    return rise, (min(off) if off else t1)


def extract_segment(D, label, t0, t1, spec, rate_hz):
    """한 구간 창을 레시피대로 추출 → dict(dataset이름→ndarray)."""
    S = D["S"]
    lo, hi = (refine_squeeze(D["sq"], t0, t1) if spec["refine"] == "squeeze_on" else (t0, t1))
    tick_ts = [t for t in D[spec["tick"] + ("_ts" if spec["tick"].endswith("qtar") else "")]
               if lo <= t <= hi] if spec["tick"] in ("hand_qtar", "arm_qtar") else []
    if not tick_ts:   # tick 없으면 고정 rate 폴백
        step = int(1e9 / rate_hz)
        tick_ts = list(range(int(lo), int(hi) + 1, max(1, step)))

    def value(kind, t):
        if kind == "tactile":
            v = _latest(S["pax"], t)
            return v if v is not None else np.zeros((FINGERS, POINTS, 3), np.float32)
        if kind == "resultant":
            v = _latest(S["pax"], t)
            tac = v if v is not None else np.zeros((FINGERS, POINTS, 3), np.float32)
            return RE.resultant_from_tactile(tac).astype(np.float32)          # (4,3)
        if kind == "kin":
            return _latest(S["kin"], t)
        if kind in ("hand_joint", "arm_joint", "arm_qtar_v"):
            return _latest(S[kind], t)
        return None

    cols = {ds: [] for ds, _ in spec["channels"]}
    valids, tstamps = [], []
    for t in tick_ts:
        valid = 1 if _seen(S["pax"][0], t) else 0
        if spec["require_valid"] and valid != 1:
            continue
        row = {ds: value(src, t) for ds, src in spec["channels"]}
        if any(v is None for v in row.values()):   # 상태 아직 없음
            continue
        for ds, v in row.items():
            cols[ds].append(np.asarray(v, np.float32))
        valids.append(np.int8(valid)); tstamps.append(np.int64(t))

    if not tstamps:
        return None
    out = {ds: np.stack(vs) for ds, vs in cols.items()}
    out["valid"] = np.asarray(valids, np.int8)
    # bag 메시지 시각은 epoch(ns) → 라이브의 monotonic 't_mono_ns' 와 구분해 't_ns' 로 저장.
    # (parity 시각정렬은 collect 의 root attr t_offset_ns 로 A(monotonic)↔B(epoch) 환산)
    out["t_ns"] = np.asarray(tstamps, np.int64)
    return out


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _find_bags(paths):
    """인자 → [(세션폴더, bag폴더)]. bag 폴더 자체 / 세션 폴더(bag/ 보유) / 세션들을 담은
    상위 폴더(collect_logs) 를 모두 받는다(중복 제거).

    ★ 세션 폴더까지 같이 받는 이유: 출력 h5 는 bag 옆(= 세션 폴더)에 놓이고 outcomes.json 도
      거기 있다. 인자를 bag 폴더로만 받으면 '어느 세션이냐' 를 매번 .parent 로 되짚어야 해서
      --out-root 로 트리를 재구성할 때 폴더 이름이 'bag' 이 되는 사고가 난다.
    """
    found, seen = [], set()
    for p in paths:
        d = Path(str(p).rstrip("/")).expanduser()
        if not d.is_dir():
            raise SystemExit(f"폴더 없음: {d}")
        if (d / "metadata.yaml").exists():
            cands = [(d.parent, d)]                       # bag 폴더를 직접 준 경우
        elif (d / "bag" / "metadata.yaml").exists():
            cands = [(d, d / "bag")]                      # 세션 폴더
        else:                                             # 세션들을 담은 상위 폴더
            cands = [(c, c / "bag") for c in sorted(d.iterdir())
                     if c.is_dir() and (c / "bag" / "metadata.yaml").exists()]
        if not cands:
            raise SystemExit(f"bag 아님(metadata.yaml 없음): {d}\n"
                             "  bag 폴더 / 세션 폴더 / 세션들을 담은 상위 폴더 중 하나를 주세요.")
        for sess, bag in cands:
            r = bag.resolve()
            if r not in seen:
                seen.add(r)
                found.append((sess, bag))
    return found


def _export_json(sess: Path, out_dir: Path) -> int:
    """세션 폴더의 *.json(outcomes.json)을 출력 폴더로 복사 → 복사한 개수.
       (bag_to_session.py 의 같은 이름 함수와 같은 규칙 — 저쪽은 collect_ros2 를 import 해
        무거워서 여기서 부르지 않고 같은 몇 줄을 둔다.)
       ★ h5 에 없는 run 상세(thumb_return_*_held, squeeze_*_delta_n, 판정 시각 wall …)가
         json 에만 있으므로 bag 없는 트리에도 반드시 함께 가야 한다."""
    n = 0
    for src in sorted(sess.glob("*.json")):
        dst = out_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)
        n += 1
    return n


def convert_one(bag_dir: Path, out_path: Path, args):
    """bag 하나 → 구간별 h5 작성. 반환 (그룹 수, 총 프레임 수)."""
    storage = args.storage or detect_storage(bag_dir)
    want = [s.strip() for s in args.segments.split(",") if s.strip()]
    skip = {s.strip() for s in args.skip_outcomes.split(",") if s.strip()}   # 제외할 outcome

    print(f"[bag2h5] bag={bag_dir} storage={storage} segments={want}")
    D = read_streams(bag_dir, storage, args.paxini)
    print(f"[bag2h5] paxini={D['pax_src']} hand_joint={len(D['S']['hand_joint'][0])} "
          f"arm_joint={len(D['S']['arm_joint'][0])} pax={len(D['S']['pax'][0])} "
          f"hand_qtar={len(D['hand_qtar_ts'])} arm_qtar={len(D['arm_qtar_ts'])} "
          f"seg={len(D['seg'])} demo={len(D['demos'])}")

    windows = segment_windows(D["seg"], D["t_max"])
    if not windows:
        raise SystemExit("bag 에 /collect/segment 라벨 없음 — 새 collect_ros2 로 수집한 bag 필요.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(out_path, "w")
    f.attrs["schema_source"] = "bag_to_hdf5 segment-based (Option 2a)"
    f.attrs["source_bag"] = str(bag_dir)
    f.attrs["paxini_source"] = D["pax_src"]
    f.attrs["segments_extracted"] = ",".join(want)
    f.attrs["git_sha"] = _git_sha()
    f.attrs["created_wall"] = datetime.now().isoformat()
    n_groups, total = 0, 0
    try:
        for label, t0, t1 in windows:
            if label not in want:
                continue
            spec = SEGMENT_SPECS.get(label)
            if spec is None:
                print(f"[bag2h5]   (레시피 없음: {label} — 건너뜀)")
                continue
            rid = run_of(D["demos"], t0)
            outcome = D["outcomes"].get(rid, "unjudged")
            if outcome in skip:                         # 실패/제외 outcome 인 run 은 건너뜀
                print(f"[bag2h5]   {label} run{rid}: outcome={outcome} → 제외(--skip-outcomes)")
                continue
            data = extract_segment(D, label, t0, t1, spec, args.rate)
            if data is None:
                print(f"[bag2h5]   {label} run{rid}: 프레임 0 — 건너뜀")
                continue
            g = f.create_group(f"{label}__run{max(rid,0):03d}")
            for ds, arr in data.items():
                g.create_dataset(ds, data=arr, compression="gzip")
            n = len(data["t_ns"])
            g.attrs.update(run=int(rid), segment=label, tick=spec["tick"], outcome=outcome,
                           n_samples=int(n), t0_ns=int(t0), t1_ns=int(t1),
                           paxini_source=D["pax_src"])
            n_groups += 1; total += n
            print(f"[bag2h5]   {g.name}: {n} frames  ({list(data)})")
        f.attrs["n_groups"] = int(n_groups)
        print(f"[bag2h5] 완료: {n_groups} 그룹, {total} frames → {out_path}")
    finally:
        f.close()
    return n_groups, total


def main():
    ap = argparse.ArgumentParser(description="rosbag → 구간별 HDF5 (Option 2a, palm-down 포함)")
    ap.add_argument("bag", nargs="+",
                    help="bag 폴더 / 세션 폴더(bag/ 포함) / 세션들을 담은 상위 폴더(collect_logs)")
    ap.add_argument("-o", "--output", default=None,
                    help="출력 HDF5 (기본 <세션>/from_bag.h5). bag 1개일 때만 쓸 수 있다")
    ap.add_argument("--out-root", default=None, help=(
        "h5/json 만 모을 별도 트리의 뿌리. <out-root>/<세션폴더명>/from_bag.h5 + outcomes.json "
        "으로 내보낸다(bag 은 원본 폴더에 그대로 둔다 — 학습 PC 로 옮길 가벼운 사본용)"))
    ap.add_argument("--storage", default=None, help="mcap/sqlite3 (기본 metadata 자동)")
    ap.add_argument("--paxini", choices=["auto", "ft", "raw"], default="auto")
    ap.add_argument("--segments", default=",".join(SEGMENT_SPECS),
                    help="추출할 구간 라벨 CSV (기본 전부)")
    ap.add_argument("--rate", type=float, default=100.0, help="tick 없을 때 폴백 rate(Hz)")
    ap.add_argument("--skip-outcomes", default="",
                    help="이 outcome(CSV)인 run 은 추출 제외 (예: grip_fail,discard)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="출력 h5 가 이미 있으면 건너뛴다(중단된 배치 이어서 돌리기)")
    args = ap.parse_args()

    bags = _find_bags(args.bag)
    out_root = Path(args.out_root).expanduser() if args.out_root else None
    if args.output and (out_root or len(bags) > 1):
        raise SystemExit("-o/--output 은 bag 1개 전용입니다 — 여러 개는 --out-root 를 쓰세요.")
    if out_root:
        print(f"[bag2h5] 세션 {len(bags)}개 → {out_root} (h5 + json 만, bag 은 원본 유지)")

    ok, skipped, failed = [], [], []
    for i, (sess, bag_dir) in enumerate(bags, 1):
        out_dir = (out_root / sess.name) if out_root else sess
        out_path = Path(args.output).expanduser() if args.output else out_dir / "from_bag.h5"
        if len(bags) > 1:
            print(f"\n[{i}/{len(bags)}] {sess.name}")
        try:
            if args.skip_existing and out_path.exists():
                print(f"[bag2h5] 이미 있음 — 건너뜀: {out_path}")
                skipped.append(sess.name)
                continue
            convert_one(bag_dir, out_path, args)
            if out_root:
                n = _export_json(sess, out_dir)
                print(f"[bag2h5]   json {n}개 복사 → {out_dir}")
            ok.append(sess.name)
        except (Exception, SystemExit) as e:
            # bag 이 하나면 원래대로 터뜨린다(traceback 이 있어야 원인을 본다).
            # 배치일 때만 삼키고 계속 — 마지막에 실패 목록을 다시 찍는다.
            if len(bags) == 1:
                raise
            print(f"[bag2h5]   ✘ 실패 — 건너뜀: {e}")
            failed.append((sess.name, str(e)))

    if len(bags) > 1 or out_root:
        print("\n" + "─" * 70)
        print(f"[bag2h5] 성공 {len(ok)} · 건너뜀 {len(skipped)} · 실패 {len(failed)} "
              f"/ 전체 {len(bags)}")
        for name, err in failed:
            print(f"         ✘ {name}: {err}")
        if out_root:
            print(f"[bag2h5] 재구성한 트리: {out_root}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
