# 개발 보고서 — 2026 산업부 촉각지능 워크샵 데모 통합

작성일: 2026-08-16 · 작성 범위: 전임자 통합 작업 인수 현황 + 업데이트 skill 2종(inhand VTDP 정책 · 숙도판별 ecoflex) 통합
관련 문서: [RUNBOOK](RUNBOOK.md) · [ARCHITECTURE](ARCHITECTURE.md) · [MIGRATION](MIGRATION.md) · [TROUBLESHOOTING](TROUBLESHOOTING.md)

---

## 1. 시스템 개요 (인수 시점 현황)

데모는 4개 기술을 한 체인으로 잇는다: **① 과일 인식 후 Pick → ② In-hand Manipulation(오리엔테이션 변경) → ③ 숙도(stiffness) 판별 → ④ 난좌 인식 후 Place**.

전임자가 3대(Control/Current/산업부)로 흩어져 있던 시스템을 2대로 통합해 두었다:

| PC | 역할 |
|---|---|
| **Control PC** (192.168.0.100) | Franka 듀얼 암 + KISTAR 핸드 실기 제어, `sequence_arbiter`(순서 강제), PaXini 촉각 writer, front RealSense 발행. **실기를 움직이는 유일한 주체** |
| **이 PC** (prime-ws, 192.168.0.101) | 4개 skill 전부, MoveIt 디지털 트윈(계획·시각화 전용), place 모델 서비스 5종, 파이프라인 러너 |

- 단일 명령 진입점: `./run_fruit_demo.sh --fruit <name>` (러너 `pipeline/run_pipeline.py` + `pipeline/config.yaml`)
- 인수 시점에 apt/colcon/conda/가중치 설치와 dry-run 검증까지 완료돼 있었고, **실기 전체 체인은 미검증** 상태였다.

## 2. 순서는 누가 강제하는가 — 시퀀스 규약

**Control PC 의 `sequence_arbiter`가 순서를 강제한다. 러너는 감독자다.**
고정 배정: **1=Pick, 2=Inhand, 3=Stiffness, 4=Place** (client_id 동일).

각 skill 은 `SequenceClient(n)` 으로:
1. `wait_for_previous_done(n-1)` — 직전 번호의 DONE 을 스스로 기다린다
2. `with client:` 진입 = 제어권 요청 + 1Hz+ 하트비트 (arbiter 가 `[n, RUNNING]` 전이)
3. 정상 탈출 = 제어권 반납 → `[n, DONE]` — **DONE 만 성공을 뜻한다**
4. 프로세스가 죽으면 하트비트 끊김 → 3초 내 arbiter 가 IDLE 회수 → **뒷단계 시작 안 함**

러너(`run_pipeline.py`)는 (a) 상시 서버 확인/기동, (b) 단계 프로그램 spawn, (c) `/sequence_state` 관측으로 성공/실패 판정, (d) 실패 시 프로세스 그룹 종료만 한다.
함정: 직전 체인의 DONE 이 latched 로 남으면 skill 쪽 대기가 그냥 통과한다 → **체인 재실행 전 arbiter 재시작**이 정석 (러너가 preflight 에서 경고).

## 3. 기술 간 연결 — 단계 전환 시 arm/hand 상태 정렬 (핵심)

"앞 기술이 끝났을 때의 로봇 상태"와 "다음 기술이 기대하는 시작 상태"가 어떻게 이어지는지가 통합의 핵심이다.

### 3-1. 전환 표

| 전환 | Arm | Hand | 정렬 메커니즘 |
|---|---|---|---|
| **① Pick 종료 → ② Inhand 시작** | pick 후 들어올린 자세에서 정지. ② 진입 시 `pose_commander` 가 우측 팔을 **고정 목표 pose 로 이동** (`right_fr3_link0` 기준 `right_fr3_link8`: xyz `0.2333 0.1590 -0.0668`, quat `-0.3373 0.2612 0.3084 0.8502` — `inhand_policy_sequence_2.py` 상수) | 파지 유지 (① 이 쥔 채 종료) | ② 는 팔 위치를 **자기가 직접 재정렬**하므로 ① 의 종료 팔 위치에 의존하지 않는다. 손은 encoder count 목표가 receiver 에 래치되어 유지 |
| **② Inhand 종료 → ③ Stiffness 시작** | **이동 없음** — ③ 은 팔을 움직이지 않는다(손 스퀴즈만) | ② 의 마지막 자세(goto+squeeze 재파지 후) 유지 | ③ 이 `safe_hand_servo_on`: **현재(파지 중) 측정 자세를 q_target 으로 먼저 발행 → servo ON** — ② 가 남긴 자세를 그대로 자기 기준점으로 재앵커 |
| **③ Stiffness 종료 → ④ Place 시작** | 이동 없음 (팔은 ②가 만든 pose 그대로 = place 의 `~child_pose` 근방) — ④ 는 **이동 없이 그 자리에서** child 캡처 | ③ 이 파지 유지 Position target 을 스트리밍하다가 제어권 이양과 함께 중단 | **그립 인수인계**: place 서버가 ③ 이 보내던 `/hand/right/q_target` 을 카메라 executor 스레드에서 캐시해 두었다가, 제어권 획득 직후 그 값을 **20Hz 로 재발행**해 release 직전까지 유지. 인수할 target 이 없거나 30s 이상 오래되면 **지어내지 않고** 래치에 맡긴다 (ARCHITECTURE §7) |
| **④ Place 내부** | 배치 후 `parent_pose` 복귀 | release 구간에서만 Voltage 모드 | §3-2 불변식 |

### 3-2. 손 모드 불변식 (시스템 전체 안전의 축)

`/hand/right/cmd_mode`: **1=Position(encoder counts)** / **0=Voltage(raw PWM duty)** — 같은 토픽·타입, 전혀 다른 단위.

> **불변식: Voltage 는 ④의 release 구간에서만 존재하고, 그 구간을 벗어나는 모든 경로에서 Position 으로 복귀해야 한다.** 깨지면 Position counts(수백~수천)가 duty 로 해석돼 핸드가 폭주한다.

3중 방어: ① place 정상경로 `try/finally` 의 `hand_safe_shutdown()`(duty 0 → servo OFF → mode=Position), ② `atexit`+SIGINT/SIGTERM 안전망(우리가 Voltage 로 바꾼 경우에만 복원), ③ 각 단계 진입 명령 앞의 `cmd_mode=1` 발행 (`pipeline/config.yaml`). 러너는 SIGTERM→5s→SIGKILL 순서로 ②가 동작할 시간을 준다.

### 3-3. 이번에 추가된 전환 배선 (VTDP 정책)

새 seq 2 래퍼(`inhand_policy_sequence_2.py`)의 정책 종료 시 손 상태는 **홀드(이완 아님)** 다 — `run_kist_vtdp.py` 는 disengage/Ctrl-C 시 q_target 발행만 멈추고 servo 는 켠 채 두므로(설계 의도: 힘 빠지면 과일 낙하) 수신기가 마지막 타겟을 유지한다. 래퍼는 이어서 `hand_goto_target`(3s 소프트 램프) → `hand_manual_squeeze`(오므리기 1회)로 **"살짝 폈다 다시 잡기"** 재파지를 수행한 뒤 DONE 을 낸다 → ③ 의 `safe_hand_servo_on` 재앵커와 그대로 정합.

## 4. 2026-08-16 통합 작업 내역

### 4-1. 숙도판별 업데이트 (`stiffness_predict_demo.zip`)

**정체**: 기존 "강성 등급(soft/mid/hard)" 추론을 **ecoflex2fruit 3속성(무게 g · 크기 mm · 강성) 모델**로 바꾼 데모 계열. 새 진입점 `deploy_task3_ros2_demo.py` 는 시퀀스 규약(SequenceClient(3))·stdin 과일번호·`--no-gui`·env.sh 계약이 구판과 **완전 동일**해 드롭인 교체가 성립한다. 팔은 여전히 움직이지 않고 손 스퀴즈만 한다.

**통합 방식 — add-only 병합** (통째 교체 금지): zip 에 기존 진입점 `deploy_task3_ros2.py`·`moveit_arm_mover.py`·수집 도구가 **미포함**이라 통째 교체 시 롤백 경로가 사라진다. 신규 20파일 + 갱신 2파일(`gui/property_gui.py`, `real_deploy_inference_final.py` — 촉각 포화 despike 순증)만 반영했다.

**바뀐 실행 전제**:
- 촉각 입력이 `/paxini/right/ft`(합력 12ch) → **`/paxini/right/raw`(4×127×3 분포)** 로 변경. **미발행이어도 힘=0 으로 조용히 DONE 까지 진행**하므로(결과만 무의미) 러너 preflight 에 발행자 점검을 추가했다.
- 힘 임계 고정(파지 6.0N / 스퀴즈 11.0N), `fruit_thresholds.yaml` 미사용. 과일번호는 포즈 파일 선택용으로만 유지.
- 실사용 모델: `models/ecoflex2fruit/gru_anchor_s42.pth` (**코드 기준 variant="gru"** — 문서 일부의 "Champ 고정" 표기는 구 설정. readme_demo §4.1 이 명시. 코드는 개발자 전달본 그대로 무수정)
- 배선: `pipeline/config.yaml` stiffness command 파일명 한 줄 교체 (구판 라인은 주석 롤백용 보존)
- 검증: `check_setup.py` 통과 (gru 엔진 109,352 파라미터 cuda 로드, 라벨 18개체 norm 정합)

### 4-2. In-hand Manipulation 정책 (`kist_deploy_pkg.zip`)

**정체**: 학습된 **visuo-tactile diffusion policy (VTDP)** 배포판. model 000(`r6_x_tacdrop_vt`, 레몬, 시각+촉각)만 가중치 포함 — 원본 PC(RTX 4080)에서 유일하게 실기 rollout 검증된 모델. 입력 = 손 관절(16) + PaXini 합력(12) + 카메라 compressed(640×480, crop→224²) / 출력 = `/hand/right/q_target` 16ch counts @100Hz. **FoundationPose 는 정책 입력이 아니며**(코드 전수 확인) 순수 데모 화면용이다.

**실행 계약의 요점** (래퍼 설계 근거):
- engage(`/teleop/hand_engage/right`, RELIABLE+TRANSIENT_LOCAL) 전엔 **아무것도 발행하지 않는다** → 미리 띄워 프리워밍해도 안전
- **정책은 스스로 종료하지 않는다** (성공판정·시간제한 없음 — 원래 사람이 발판으로 껐다) → 래퍼가 시간 기반 종료(기본 15s, `config.yaml inhand_policy.duration_s`)
- 정지 = **홀드** (이완 아님). 서보를 내리는 경로가 정책에 없다 → 인수인계와 정합
- `/kist_vtdp/debug`[15]=halt 래치 (NaN 타겟/카메라 이상) — 래퍼가 감시해 halt 시 체인 중단

**래퍼** `skill-set/in-hand-reorientation/scripts/inhand_policy_sequence_2.py` (원본 정책 코드 무수정, subprocess 구동):
프리워밍 spawn(체인 시작 시 — 러너 `spawn: early` 신설로 모델 로드가 Pick 과 겹침) → Pick DONE 대기 → Start → `pose_commander` 팔 이동(기존 유지) → "예열 완료" 확인 → engage → 15s(halt 감시) → disengage → SIGINT → goto+squeeze 재파지 → DONE.
기존 HDF5 재생은 `--inhand-legacy` 플래그로 즉시 복귀 가능 (`command_legacy`).

**검증 결과** (RTX 5090 + torch 2.7.1+cu128 — 원저자도 미검증이던 조합):
- `--self_test`: hp 필터 max|Δ|=0, 추론 **p50 18.2ms / p90 18.7ms / max 19.2ms** (예산 50ms), 액션 유한 — 전부 통과
- `--dry_run`(라이브 토픽): 센서 3종 수신 확인, **카메라 fx=614.68 = 학습값 614.678 일치**, engage 대기 정상
- 학습 h5 미동봉으로 RGB 파이프라인 자동 대조는 생략됨 → 실전 전 `crop_reference.png` 화각 육안 대조 권장

### 4-3. 과일 6DoF 오버레이 (FoundationPose — seq 2 데모 화면)

realsense 화면 위에 과일 CAD 정합 기반 **3D bbox + 오리엔테이션**을 표시한다(`record/fruit_overlay.py`). 러너가 **체인 시작 시 자동 기동**(`services.fruit_viz`, 등록 시간 확보 목적, `--no-fruit-viz` 로 끔). 보조 화면이므로 실패해도 체인은 계속 간다.

이 PC(RTX 5090, sm_120)용 재빌드를 수행했다:
- 기존 docker 이미지(torch2.0/sm_86)는 폐기. `build_sm120.sh` 신설 — conda env `foundationpose`(py3.11) + torch cu128 + pytorch3d/nvdiffrast 소스빌드 + mycpp. **CUDA 12.8 이 g++≥14 를 거부해 gcc 13 고정이 필요했다** (스크립트에 반영)
- fp_server 스모크: 레몬 메시 로드 → listening **9초** ✓
- 텍스처 복구: 4종 CAD 가 복숭아 텍스처를 공유하던 함정(README 함정 3)을 `Obj/` 원본 스캔에서 과일별 디렉토리로 재생성해 해소. `fruits.yaml` 갱신, 카탈로그 4종 전부 ✓
- 경로 보정: `fp_ros_node.py` FM_ROOT 환경변수화(기본=동봉 sam2), 런처/viz 스크립트 env 블록을 이 PC 표준 `setup_env.sh` 로 교체. **패키지의 `config/fastdds_lan_only.xml`·`fix_ros_net.sh` 는 사용 금지** — 원본 PC(192.168.0.1) 전용이라 이 PC 에서 쓰면 DDS 전멸
- `run_foundation_pose.sh --check` 전 항목 통과 (카메라 3스트림 30Hz), `live_viz.py --selftest` 렌더 OK
- GPU 배치: fp_server=GPU0(`FP_GPU` env), VTDP 정책=cuda:1 — place 서비스와의 경합은 실측 후 조정

### 4-4. 환경/인프라

- 시스템 python 의존 복구: `typing_extensions`/`filelock`/`fsspec` 부재로 **`import torch` 자체가 깨져 있던 것**을 복구(+ `torchvision 0.22.1+cu128`, `h5py`, `open3d`, `trimesh`, SAM2). open3d 가 끌고 온 numpy 2.2.6 은 cv2/cv_bridge ABI 위험이라 **1.26.4 로 강제** (1.x ABI 유지, sklearn 요구 충족). scipy 1.8 은 경고만 — 핵심 서브모듈 동작 확인
- `resnet18-f37072fd.pth` → `~/.cache/torch/hub/checkpoints/` (오프라인 필수)
- `rs` 별칭(~/.bashrc): 새 터미널에서 `rs` 한 번으로 데모 ROS2 환경(DOMAIN 9) 접속
- preflight 확장: `/paxini/right/raw` + VTDP 입력 3종 점검 (skip/legacy 시 자동 생략)
- git 저장소: 이 트리를 `git init` 하고 인수 스냅샷부터 단계별 커밋. 중첩 skill 저장소 5개는 `.git.disabled` 로 보존(외부 저장소가 코드 추적). 대용량 가중치(>40MB)·로그·colcon 산출물은 gitignore — zip/로컬 관리

### 4-5. seq 4 수동 place 체인 신규 개발 — 티칭 포인트 기반 (2026-08-16 저녁)

**기존 seq 4(vision_pipeline place 서버)는 무수정 보존**, 별도 실행 파일 추가: `skill-set/place/scripts/seq4_manual_place.py`. 모든 좌표는 파일 **최상단 변수**(티칭 → 붙여넣기), stiffness(seq 3) 종료 자세에서 그대로 이어받아 시작(기본값 = 제시 자세 관절각).

slot 당 시퀀스 (체인 1회전마다 slot 1→5 자동 진행, 상태파일 `.seq4_manual_slot`, `--slot N` 강제 가능):

1. 진입 시 arm/hand state 저장 → hand **mode 1 유지**한 채 지정 11개 관절(2,3,5,6,7,9,10,11,13,14,15번)에 **+100 counts** 더 꽉 쥠 (`--grip-delta`)
2. 공통 경유 `FRANKA_PLACE_1→2→3` → `FRANKA_PLACE_SAFE` (공통 4좌표)
3. `FRANKA_PLACE_TOP[slot]` → `FRANKA_PLACE_DOWN[slot]` (5쌍)
4. 릴리즈: 모드 전환 프로토콜(servo-OFF 창 + 50ms 틱 + 시드 ×2) 그대로 **mode 2(current)** 전환 → 정착 2s → 벌리는 타겟 `[4096,-4096,0,0, 0,1000,1000,1000, 0,1000,1000,1000, 0,1000,1000,1000]` 로 **3s 선형 램프** (천천히 놓기)
5. top 복귀 → mode 1 재진입(벌린 자세 시드, 재파지 아님) → safe 복귀 → done

임피던스 안전: 모든 팔 이동 min-jerk 보간 `/franka/right/q_target` **100Hz 스트리밍**(속도 배율 0.1), 첫 이동 전 측정자세 재앵커(sync_target 패턴), q_target 은 BEST_EFFORT.

```bash
# 포인트 티칭 (출력 줄을 파일 상단에 붙여넣기)
/usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --capture wp1
# 설정/토픽 검증 (로봇 안 움직임)
/usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --check
# ★ 4번만 1회 테스트 — stiffness 종료 자세로 이동 후 slot 1 place
/usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --goto-start --slot 1
# 체인 모드 (arbiter 규약: seq3 DONE 대기 → 제어권 4 → 반납=DONE)
/usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --chain
```

검증: 문법 통과 + `--check` 라이브 버스에서 arm/hand 상태 실시간 수신 확인(토픽 배선 정상). 좌표 14개(공통 4 + top/down 5쌍)는 티칭으로 채워야 실행(빠진 항목은 스크립트가 짚어줌). 실기 구동 미실시.

⚠ `--chain` 은 기존 place 서버(`skill_server`)와 **동시 실행 금지**(둘 다 client_id 4). 테스트는 기본(standalone) 모드로. mode 2(current) 릴리즈는 첫 사용 경로 — 첫 실행은 E-stop 옆에서.

## 5. 정책 종료판정 검토 (요청 항목)

원래 의도: "화면에서 토마토 꼭지·레몬 라벨 등 특징이 보이고 중앙 정렬되면 중지, 살짝 폈다 다시 잡기로 마무리". 세 가지 방안을 검토했다.

| 방안 | 방법 | 판정 |
|---|---|---|
| **① FoundationPose 6DoF 기반 (권장)** | 오버레이용으로 **이미 돌고 있는** `/fruit/pose` 에서, 과일 CAD 에 1회 주석한 "특징 축"(꼭지/라벨 방향)이 카메라 광축과 이루는 각 < 임계 를 T초 유지하면 disengage | 추가 런타임 비용 0(이미 추적 중), 과일 외관 변경 없음. 단 (a) CAD 별 특징 축 주석 1회 필요, (b) 레몬처럼 장축 회전대칭 과일은 장축 둘레 회전이 관측 불안정 — 실측 검증 필요. **구현 난이도 낮음: 래퍼의 15s 타이머 자리에 조건 하나 추가** |
| ② AprilTag | 과일에 태그 부착, 태그 검출+이미지 중앙 판정 | 검출 자체는 가장 강건. 그러나 **정책이 태그 없는 과일로 학습됨** — 시각 입력 분포가 흔들려 정책 성능 저하 위험이 있고, FoundationPose 텍스처 정합도 교란한다. 태그를 붙일 거면 **학습 데이터도 태그 붙인 과일로** 재수집하는 편이 안전 |
| ③ 꼭지/라벨 직접 검출 (SAM 쿼리/검출기) | 매 프레임 특징 검출 | 소형 특징의 프레임별 검출은 취약하고 GPU 추가 부담. 비권장 |

**현행 구현은 15초 고정 + goto/squeeze 재파지**다(사용자 결정). ①을 다음 단계로 권고하며, 필요한 것은 (1) 레몬 CAD 의 라벨 축 정의, (2) `/fruit/pose` 구독 + 각도 조건을 래퍼 monitor 루프에 추가, (3) 실과일로 임계각/유지시간 튜닝이다.

## 6. 실행 방법 (업데이트 반영)

```bash
# 새 터미널: rs 로 환경 접속
rs
cd ~/prime/Jaesung_Lee/RobotAgentSystem
tools/diagnostics/preflight.sh              # 로봇 미동작 점검
./run_fruit_demo.sh --fruit lemon           # 정책 데모 과일 = lemon (model 000)
#   --inhand-legacy       : seq 2 를 기존 HDF5 재생으로
#   --no-fruit-viz        : 3D bbox 오버레이 끔
#   --dry-run             : preflight 만
# 정책 시간/GPU: pipeline/config.yaml → inhand_policy.duration_s / device
```

실전 전 준비(DEPLOY.md 안전수칙): 글러브 텔레옵 OFF 확인, 별도 터미널에 `ros2 topic pub -1 /hand/right/cmd_servo std_msgs/Bool "{data: false}"` 대기(비상 이완), E-stop 인원 상주.

## 7. 남은 리스크 / 다음 단계

| # | 항목 | 상태 |
|---|---|---|
| 1 | **실기 전체 체인** (새 정책 포함) | 미검증 — dry-run/self-test 까지만. 사용자 입회 하에 `--skip` 단계별 검증 권장 (RUNBOOK §4 순서) |
| 2 | 정책 실기 거동 (5090) | 오프라인 지표는 원본 PC 와 동등 이상. 실기 rollout 은 미실시 |
| 3 | FoundationPose 첫 등록 | 서버 기동 검증 완료. 실카메라 과일 등록(클릭)과 nvdiffrast 첫 JIT 렌더는 실과일로 확인 필요 |
| 4 | RGB crop 대조 | 학습 h5 미동봉 → 자동 intrinsics 대조 생략됨 (라이브 fx 는 학습값과 일치 확인). `crop_reference.png` 육안 대조 1회 권장 |
| 5 | gru vs champ 모델 | 코드=gru(전달본 그대로). 성능 이슈 시 `deploy_task3_ros2_demo.py:168` 의 `variant="champ"` 한 줄 |
| 6 | GPU 경합 | seq 1 중 Molmo(GPU0) prewarm 과 fp_server(GPU0) 동시 → 등록 지연 가능. 정책은 GPU1 단독. 문제 시 `--ddim_steps 4` (품질 무손실 가속) |
| 7 | 정책 종료판정 고도화 | §5 ① 안 — CAD 특징 축 주석 + 래퍼 조건 추가 |
| 8 | numpy 1.26.4 승격 | scipy 1.8 경고 있음(동작 확인됨). place 실행 경로 첫 실기동 시 재확인 |
| 9 | `/teleop/hand_engage/right` 공유 | 발판 노드가 켜져 있으면 사람 발판도 정책을 engage/disengage 할 수 있다(현재 버스에 발행자 없음 확인). 데모 중 발판 운용 여부를 정할 것 |
