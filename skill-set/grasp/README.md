# Grasp_fruit

KISTAR Franka FR3 + KISTAR Hand를 이용한 과일 파지/배치 시스템.  
RealSense 카메라로 RGB-D를 촬영하고, SAM3 텍스트 프롬프트 세그멘테이션으로 대상 물체를 검출한 뒤,  
Top-down Grasp로 파지 위치를 계산하여 로봇이 자동으로 집어 내려놓는다.

---

## 시스템 구성

| 구성 요소 | 내용 |
|---|---|
| 로봇 | KISTAR Franka FR3 + KISTAR Dexterous Hand |
| 카메라 | Intel RealSense D-series (RGB-D) |
| 소프트웨어 스택 | ROS2 Humble / MoveIt2 (Docker `ros2_humble`) |
| 파이프라인 환경 | Conda `pipeline_all` (Python 3.12, CUDA 12.8) |
| 비전 모델 | SAM3 (text-prompted instance segmentation) |
| Grasp 알고리즘 | Top-down grasp — PCA 기반 방향 추정 + 포인트클라우드 높이 |

---

## 설치

### 1. Conda 환경 구성

```bash
cd HARILAB/Grasp_fruit
bash setup_pipeline_all.sh        # 환경 생성 + PyTorch(CUDA 12.8) + 패키지 설치
conda activate pipeline_all
```

이미 환경이 있다면:

```bash
bash setup_pipeline_all.sh --skip-conda
```

### 2. Docker 컨테이너 확인

ROS2 Humble + MoveIt2가 포함된 `ros2_humble` 컨테이너가 실행 중이어야 한다.

```bash
docker ps | grep ros2_humble
```

---

## 설정 파일

### `configs/paths.yaml` — 머신별 경로 설정

다른 PC에 클론할 경우 **이 파일만** 수정하면 된다.

```yaml
__CONDA_BASE__: /home/kist/miniforge3
__CONDA_ENV__:  pipeline_all
__DOCKER_CONTAINER__: ros2_humble
__ROS_DOMAIN_ID__:    9
__KISTAR_WS__:        /home/kist/HARILAB/dex_ros/isaac-ros/kistar_ws
__MOUNT_MAP__:
  - ["/home/kist/HARILAB", "/root/HARILAB"]
  - ["/home/kist/ros2_ws",  "/root/ros2_ws"]
```

### `configs/arm.yaml` — 로봇 팔 파라미터

| 파라미터 | 설명 |
|---|---|
| `home.joint_values` | HOME 자세 관절값 (rad) |
| `approach_offset_m` | 파지 위치 위 approach 시작 높이 (m) |
| `place_z_descent_m` | Place 시 HOME EE에서 world Z 기준 하강 거리 (m) |
| `grasp_z_offset_m` | SAM3 검출 z_top 위 실제 파지 목표까지 오프셋 (m, world Z) |
| `ee_correction.yaw_deg` | 핸드 장착 회전 오프셋 (°, world Z 기준) |
| `ee_correction.x_offset_m` / `y_offset_m` | EE 위치 보정 (m, EE 좌표계 기준 — yaw 회전 후 적용) |
| `pointcloud.top_z_pct` | 중심 계산에 사용할 상위 Z% 포인트 비율 |
| `pointcloud.z_top_pct` | z_top 계산 percentile (노이즈 제거) |

> **EE 오프셋 적용 순서**: PCA로 방향각 `alpha`를 먼저 구하고, `ee_correction.yaw_deg`를 더한 최종 yaw에 맞춰 x/y 오프셋 벡터를 회전하여 적용한다.  
> 즉 yaw가 바뀌면 x/y 이동 방향도 함께 바뀐다.

### `configs/fruits.yaml` — 과일별 EE 보정 오프셋

`--query`로 입력한 이름(소문자)과 정확 매칭되면 `arm.yaml` 기본값을 덮어쓴다.  
항목은 모두 optional이며, 없으면 `arm.yaml` 기본값을 유지한다.

```yaml
pear:
  yaw_deg: -45.0
  x_offset_m:  0.01
  y_offset_m: -0.03
  grasp_z_offset_m: 0.13
```

지원 항목: `yaw_deg`, `x_offset_m`, `y_offset_m`, `grasp_z_offset_m`

### `configs/hand.yaml` — KISTAR 핸드 자세

파지(`hand_grasp`), 초기(`hand_init`), 릴리즈(`hand_release`) 자세를 degrees 단위로 정의한다.  
`run_topdown_grasp.py`와 `robot_executor.py` 모두 이 파일에서 읽는다.

---

## 캘리브레이션

캘리브레이션 파일: `configs/calibration/extrinsic_20260612_170053.json`

| 필드 | 설명 | 변경 여부 |
|---|---|---|
| `T_base_camera` | 핸드-아이 캘리브레이션 결과 (카메라 → 베이스 변환) | **변경 금지** |
| `T_world_base` | 로봇 베이스 마운트 위치/자세 (world → base) | 자유롭게 변경 가능 |

### T_world_base 업데이트

로봇 베이스 위치가 바뀌었을 때 `update_world_base.py`로 행렬을 자동 계산하여 JSON에 기록한다.

```bash
# 미리보기 (파일 수정 없음)
python scripts/update_world_base.py \
    --calib configs/calibration/extrinsic_20260612_170053.json \
    --x 0.066 --y -0.122 --z 0.099 \
    --roll_deg 45.0 --dry_run

# 실제 적용
python scripts/update_world_base.py \
    --calib configs/calibration/extrinsic_20260612_170053.json \
    --x 0.066 --y -0.122 --z 0.099 \
    --roll_deg 45.0
```

---

## 사용법

### 인터랙티브 파이프라인 (권장)

SAM3 모델을 한 번 로드하고, 텍스트 쿼리를 반복 입력받아 실행한다.  
RealSense 파이프라인도 세션 내내 유지되므로 AE/AWB 재수렴 없이 쾌적하게 촬영된다.

```bash
conda activate pipeline_all
cd HARILAB/Grasp_fruit

# 비전만 (로봇 미실행)
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json

# Grasp 실행
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot

# Pick + Place 실행
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place

# MoveIt collision 비활성화 (로봇 베이스 이동 후 임시)
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place --disable_collision
```

**시작 시 동작:**

1. SAM3 모델 로드
2. "비디오 녹화 하시겠습니까?" 질문 (yes → `data/outputs/interactive_session.mp4`에 저장)
3. RealSense 파이프라인 시작 + warmup (60프레임, AE/AWB 수렴 대기)
4. `Query>` 프롬프트 반복 — `exit` / `quit` / `q` 입력 시 종료

> 세션 녹화(yes)를 선택하면 로봇 실행 시 매번 나오는 Docker 녹화 질문은 자동 생략된다.

### 단발 파이프라인

```bash
# 카메라 캡처 + Grasp
python scripts/run_pipeline.py \
    --capture \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot

# 카메라 캡처 + Pick + Place
python scripts/run_pipeline.py \
    --capture \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place

# 저장된 NPZ로 오프라인 처리 (로봇 미실행)
python scripts/run_pipeline.py \
    --input data/raw/scene_000.npz \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json
```

### 주요 공통 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--calibration` | 캘리브레이션 JSON 경로 | — |
| `--query` | SAM3 텍스트 쿼리 (예: `apple`) | — |
| `--execute_robot` | 로봇 실행 활성화 | 비활성 |
| `--place` | Pick+Place 모드 (하강 거리는 `arm.yaml` 참조) | 비활성 |
| `--speed_factor` | 로봇 속도 배율 | `0.1` |
| `--approach_offset` | Approach 높이 오프셋 (m) | `0.10` |
| `--z_offset` | Grasp Z 오프셋 (m, 미설정 시 `arm.yaml` 값 사용) | `arm.yaml` |
| `--disable_collision` | MoveIt collision 검사 비활성화 (임시) | 비활성 |

---

## 파이프라인 단계

```
Stage 0: RealSense 캡처       →  data/raw/<stem>_000.npz  (+  _rgb.png, _depth_vis.png)
Stage 1: SAM3 추론            →  data/interim/<stem>_mask.png  (+  _overlay.png)
Stage 2: Top-down Grasp 계산  →  data/outputs/<stem>_topdown_summary.json  (+  _overlay.png)
Stage 3: 로봇 실행            →  Docker exec → robot_executor.py
```

### Stage 2 — Grasp 계산 상세

1. 마스크 영역 포인트클라우드 backproject → SOR 필터로 outlier 제거
2. XY 평면 PCA → 장축 방향 = Grasp 접근 방향
3. `arm.yaml` `ee_correction` + `fruits.yaml` override 적용 → EE 자세 결정
4. z_top (상위 percentile) + `grasp_z_offset_m` → EE 높이 결정
5. Summary JSON (`T_world_ee`, joint angles, hand encoder) 저장

### Stage 3 — 로봇 실행 흐름

**Grasp 모드:**

```
HOME → Approach (물체 위 approach_offset_m) → 하강 (Grasp 위치) → 핸드 파지
→ 상승 (Approach 역재생) → HOME
```

**Place 모드 (Grasp 이후 연속 실행):**

```
(Grasp 완료) → HOME → world Z 기준 수직 하강 (place_z_descent_m) → 핸드 열기
→ 수직 상승 → HOME
```

> Place 하강은 world Z 기준으로 계산된다. `T_world_base`에 roll이 있어도 수직으로 내려간다.

---

## 발표 자료 이미지 생성

`_topdown_summary.json`으로부터 PCA 화살표 이미지와 좌표 이미지를 각각 생성한다.

```bash
python scripts/make_presentation_figs.py \
    data/outputs/interactive_012_012_topdown_summary.json
```

출력:
- `*_fig_pca.png` — 마스크 오버레이 + PCA 장축 화살표
- `*_fig_position.png` — 마스크 오버레이 + SAM3 검출 bbox + 파지 위치 좌표 (좌상단)

---

## 디렉토리 구조

```
Grasp_fruit/
├── configs/
│   ├── arm.yaml                     # 로봇 팔 파라미터 (HOME 자세, 오프셋 등)
│   ├── hand.yaml                    # 핸드 자세 파라미터
│   ├── fruits.yaml                  # 과일별 EE 오프셋 override
│   ├── paths.yaml                   # 머신별 경로 설정
│   ├── calibration/                 # 캘리브레이션 JSON 파일들
│   └── camera/
│       └── realsense.yaml           # RealSense 해상도/FPS 설정
│
├── scripts/
│   ├── run_pipeline_interactive.py  # 인터랙티브 파이프라인 (권장)
│   ├── run_pipeline.py              # 단발 파이프라인
│   ├── pipeline_core.py             # 공통 stage 함수 / argparse 빌더
│   ├── robot_executor.py            # Docker 내부 로봇 실행 진입점
│   ├── send_to_robot.py             # 호스트 → Docker exec 브릿지
│   ├── run_topdown_grasp.py         # Top-down grasp 계산 (Stage 2)
│   ├── run_sam3_only_stage.py       # SAM3 추론 subprocess용 (Stage 1)
│   ├── capture_realsense_once.py    # RealSense 단회 촬영 (Stage 0)
│   ├── make_presentation_figs.py    # 발표용 이미지 생성
│   ├── update_world_base.py         # T_world_base 업데이트 유틸
│   ├── launch_moveit.py             # MoveIt 런치 헬퍼
│   ├── docker_runner.py             # Docker exec / 녹화 유틸리티
│   └── utils/
│       ├── arm.py                   # arm.yaml 파싱 및 상수 export
│       ├── hand.py                  # hand.yaml 파싱 및 상수 export
│       ├── paths.py                 # paths.yaml 파싱
│       ├── grasp.py                 # GraspExecutor (ROS2 Node)
│       ├── place.py                 # PlaceExecutor (GraspExecutor 상속)
│       └── step.py                  # 원자 step 함수 (move_home, approach, …)
│
├── src/
│   └── affordance_grasp/
│       ├── io/
│       │   ├── dataset_io.py        # NPZ 저장/로드, JSON 유틸
│       │   └── realsense.py         # RealSense 캡처 / RealSenseSession
│       └── geometry/
│           └── frame_transform.py   # 변환 행렬 유틸 (xyzrpy, invert 등)
│
├── data/
│   ├── raw/                         # RealSense 캡처 NPZ + PNG
│   ├── interim/                     # SAM3 마스크, 오버레이
│   └── outputs/                     # Grasp summary JSON, overlay, session.mp4
│
├── docker/
│   ├── Dockerfile.moveit
│   └── entrypoint_moveit.sh
│
├── environment_pipeline_all.yml     # Conda 환경 정의
└── setup_pipeline_all.sh            # 환경 구성 스크립트
```

---

## 트러블슈팅

### IK 실패 (code=-31)

MoveIt collision scene이 로봇 베이스 위치와 맞지 않으면 IK가 차단된다.

- 임시: `--disable_collision` 플래그 사용
- 근본: RViz Planning Scene에서 collision object 위치를 새 베이스 위치에 맞게 업데이트

### depth_vis.png 이미지가 온통 파랑

고정 `alpha` 스케일링 문제. 현재 코드는 프레임별 min-max 정규화 + TURBO 컬러맵을 사용하므로 발생하지 않아야 한다. 구버전 코드(`alpha=0.03`, COLORMAP_JET)를 사용 중인 경우 `src/affordance_grasp/io/realsense.py`의 `make_depth_vis` 함수를 확인한다.

### RealSense 이미지가 뿌옇다

카메라 파이프라인을 시작 직후 바로 캡처하면 AE/AWB가 수렴하기 전이라 뿌옇게 나온다.  
인터랙티브 파이프라인(`run_pipeline_interactive.py`)은 `RealSenseSession`으로 세션 내내 파이프라인을 유지하므로 최초 warmup(60프레임) 이후에는 문제없다.  
단발 파이프라인은 `--warmup_frames` 옵션으로 조절할 수 있다.

### `/move_action`에 action server가 두 개

```
[WARN] Ignoring unexpected result response. There may be more than one action server for the action '/move_action'
```

Docker 컨테이너 안에 MoveGroup 노드가 두 개 실행 중인 경우다. 이전 세션이 정상 종료되지 않았을 때 발생한다.

```bash
# 컨테이너 안에서
ros2 node list | grep move_group
kill $(ps aux | grep move_group | grep -v grep | awk '{print $2}')
```

### Place가 world Z 방향이 아닌 base Z로 내려간다

`step_place_from_home`이 `T_world_base`를 찾지 못하는 경우다. Summary JSON에 `T_world_base`가 있는지 확인하고, 캘리브레이션 JSON에 해당 필드가 있는지 점검한다.
