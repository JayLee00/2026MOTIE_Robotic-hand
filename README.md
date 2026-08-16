# RobotAgentSystem

과일 조작 데모 — **물체 파지 → 손 안 조작 → 물성 추론 → 물체 내려놓기** 를 단일 명령으로 실행한다.

```bash
./run_fruit_demo.sh --fruit orange
```

원래 3대(Control / Current / 산업부)로 나뉘어 있던 시스템을 **2대**로 통합했다.
이 PC(`prime-ws`, 192.168.0.101)가 구 Current PC + 구 산업부 PC 역할을 모두 맡고,
Control PC(192.168.0.100)는 그대로 유지된다.

| | 담당 |
|---|---|
| **Control PC** (192.168.0.100) | Franka 듀얼 암 + KISTAR 핸드 실기 제어, `sequence_arbiter`, PaXini 촉각, **front RealSense 카메라 발행** |
| **이 PC** (192.168.0.101) | 4개 skill 전부, MoveIt 디지털 트윈 + RViz, place 모델 서비스 5종, 파이프라인 러너 |

두 PC 는 `ROS_DOMAIN_ID=9` 한 버스에서 만난다.

---

## 디렉토리

```
RobotAgentSystem/
├── run_fruit_demo.sh          ← 단일 명령 진입점
├── pipeline/                  ← 순서 제어 (구 orchestrator 대체)
│   ├── run_pipeline.py            시퀀스 관측 + 단계별 spawn + 실패 처리
│   └── config.yaml                단계 명령 · 타임아웃 · 과일 매핑
├── skill-set/                 ← 실제로 로봇을 움직이는 4개 모듈
│   ├── grasp/                     seq 1  물체 파지 (SAM3 + top-down grasp)
│   ├── in-hand-reorientation/     seq 2  손 안 조작 (HDF5 손 관절 궤적)
│   ├── inference-physics-property/ seq 3 물성(강성) 추론 (LSTM/Transformer + PaXini)
│   └── place/                     seq 4  물체 내려놓기 (Molmo + SAM3 + AnyPlace + IGR)
├── tools/
│   ├── env/                       공통 환경 스크립트 (setup_env.sh, paths.sh, conda/)
│   ├── ros2/                      colcon 워크스페이스 2개 (fr_ws, dex_ros/…/kistar_ws) + build.sh
│   ├── camera/  moveit/  rviz/  urdf/   각 서브시스템 진입점 + 위치 안내
│   └── diagnostics/               preflight.sh, place_logger.py
├── docs/                      RUNBOOK · ARCHITECTURE · INSTALL · MIGRATION · TROUBLESHOOTING
└── logs/                      실행 로그 (run_MMDD_HHMMSS/)
```

## 문서

| 문서 | 내용 |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | **매일 쓰는 실행 절차** (Control PC 준비 → 단일 명령) |
| [docs/INSTALL.md](docs/INSTALL.md) | 최초 1회 설치 (apt · colcon · conda · 가중치) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시퀀스 규약, 데이터 흐름, 왜 이렇게 나뉘는가 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 3PC→2PC 통합 내역, 무엇을 어디로 옮겼는가, 남은 리스크 |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 증상별 원인/조치 |
| [docs/DEV_REPORT.md](docs/DEV_REPORT.md) | **개발 보고서** — 기술 간 연결(arm/hand 정렬), 2026-08-16 업데이트 skill 2종(VTDP 정책·ecoflex 숙도) 통합 내역, 정책 종료판정 검토 |

## 안전

- `run_fruit_demo.sh` 는 **실제 로봇을 움직인다.** E-stop 옆 인원 상주 필수.
- `/safety/estop` 토픽은 라이브 시스템에 아직 없다. 정지 권한은 **하드웨어 E-stop** 에 있다.
- 처음 브링업할 때는 `--dry-run` → 단계별 `--skip` 순으로 검증하고 전체 체인을 돌릴 것.

## 다음 단계 (VLM high-level planner)

`pipeline/run_pipeline.py` 는 자연어를 다루지 않는다 — 어떤 물체를 집을지 인자로만 받는다.
추후 VLM planner 는 이 스크립트를 그대로 호출하면 된다:

```bash
./run_fruit_demo.sh --fruit <planner 가 고른 물체명> --stiffness-fruit <매핑된 과일>
```

`--dry-run` 으로 사전 점검만, `--skip` 으로 단계 일부만 돌릴 수 있어 planner 통합 시
단계별 검증이 가능하다. 종료 코드는 성공 0 / 실패 1 / 사용자 중단 130 이다.
