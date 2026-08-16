#!/usr/bin/env python3
"""check_setup.py — 이식한 PC 에서 "실행 준비가 됐는지" 를 로봇/ROS토픽 없이 확인.

  source env.sh && python3 check_setup.py

확인 항목
  1) python = 시스템 python3(conda 아님) / 버전
  2) 필수 패키지 import (numpy · yaml · torch · rclpy · tkinter)
  3) 번들 파일 존재 (모델 ckpt · sensors.json · 라벨 3종 · 포즈 txt)
  4) 추론엔진 실물 로드 (ckpt → 모델 생성 → 채널수/정규화 범위 출력)
  5) 라벨 ↔ ckpt 정규화 범위 정합(check_label_norm)
  6) (B) 전용 의존 dual_arm_msgs / sequence_client 유무 (없으면 (A) 만 실행 가능)

종료코드 0 = (A) 실행 가능. 1 = 필수 항목 실패.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAUNCH = ROOT / "stiffness_deploy_ros2" / "launch"

ok = True
warn = 0


def _p(good: bool, msg: str, fatal: bool = True) -> None:
    global ok, warn
    if good:
        print(f"  [ OK ] {msg}")
    elif fatal:
        ok = False
        print(f"  [FAIL] {msg}")
    else:
        warn += 1
        print(f"  [WARN] {msg}")


print("\n=== 1. python ===")
print(f"  실행 python : {sys.executable}")
print(f"  버전        : {sys.version.split()[0]}")
_p("conda" not in sys.executable and "miniconda" not in sys.executable,
   "conda python 아님 (env.sh 를 source 했는지 확인)")

print("\n=== 2. 패키지 ===")
for mod, fatal in (("numpy", True), ("yaml", True), ("torch", True),
                   ("rclpy", True), ("tkinter", False)):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "")
        _p(True, f"{mod} {ver}")
    except Exception as e:  # noqa: BLE001
        _p(False, f"{mod} import 실패 ({e})" +
           ("  → GUI 만 못 뜸(추론은 가능)" if not fatal else ""), fatal=fatal)

try:
    import torch  # noqa: E402
    print(f"  CUDA 사용가능 : {torch.cuda.is_available()}"
          f"{'  (CPU 로 동작 — 느려도 정상)' if not torch.cuda.is_available() else ''}")
except Exception:
    pass

print("\n=== 3. 번들 파일 ===")
need = [
    "stiffness_deploy_ros2/models/ecoflex2fruit/Champ_repair_s42.pth",
    "stiffness_deploy_ros2/models/ecoflex2fruit/gru_anchor_s42.pth",
    "stiffness_deploy_ros2/models/ecoflex2fruit/Anchor_s42.pth",
    "stiffness_deploy_ros2/models/ecoflex2fruit/RC_v2_5_s42.pth",
    "stiffness_deploy_ros2/models/ecoflex2fruit/sensors.json",
    "stiffness_deploy_ros2/labels/object_labels_oldstif/mass.yaml",
    "stiffness_deploy_ros2/labels/object_labels_oldstif/size.yaml",
    "stiffness_deploy_ros2/labels/object_labels_oldstif/stif.yaml",
    "stiffness_deploy_ros2/launch/initial_pose.txt",
    "stiffness_deploy_ros2/launch/pose1.txt",
    "stiffness_deploy_ros2/launch/deploy_ros2_demo.py",
    "stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py",
    "stiffness_deploy_ros2/gui/property_gui.py",
]
for rel in need:
    _p((ROOT / rel).exists(), rel)

if not ok:
    print("\n필수 파일/패키지 누락 — 위 [FAIL] 항목부터 해결하세요.")
    raise SystemExit(1)

print("\n=== 4. 추론엔진 로드 ===")
sys.path.insert(0, str(ROOT / "stiffness_deploy_ros2"))
sys.path.insert(0, str(LAUNCH))
try:
    from ecoflex_engine import (EcoflexPropertyEngine, load_labels,  # noqa: E402
                                check_label_norm)
    variant = os.environ.get("ECO_MODEL_CHECK", "gru")   # 두 데모의 기본값과 동일
    engine = EcoflexPropertyEngine(variant=variant)
    _p(True, f"엔진 로드 성공 (variant={variant}, device={engine.device})")
except Exception as e:  # noqa: BLE001
    _p(False, f"엔진 로드 실패 ({type(e).__name__}: {e})")
    raise SystemExit(1)

print("\n=== 5. 라벨 정합 ===")
try:
    labels, norm = load_labels(str(ROOT / "stiffness_deploy_ros2" / "labels"
                                   / "object_labels_oldstif"))
    _p(bool(labels), f"라벨 {len(labels)} 개체 로드")
    check_label_norm(engine, norm)   # 불일치면 자체 경고 출력
except Exception as e:  # noqa: BLE001
    _p(False, f"라벨 로드 실패 ({e}) — 추론은 되지만 실제값 대조가 빠짐", fatal=False)

print("\n=== 6. (B) 시퀀스 체인 의존 ===")
for mod in ("dual_arm_msgs.msg", "sequence_client"):
    try:
        __import__(mod)
        _p(True, f"{mod} 사용 가능")
    except Exception:  # noqa: BLE001
        _p(False, f"{mod} 없음 — (B) deploy_task3_ros2_demo 는 실행 불가 "
                  f"(Dual_Arm_Hand_Ctrl 워크스페이스 source 필요). (A) 는 영향 없음",
           fatal=False)

print("\n" + "=" * 60)
print(f"결과: (A) deploy_ros2_demo 실행 준비 완료. 경고 {warn} 건.")
print("=" * 60)
raise SystemExit(0)
