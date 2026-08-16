# Topdown_Grasp 셋업 진행상황 (2026-06-15)

이 PC(`kist@kist`, `/home/kist/GW/Topdown_Grasp`)에서 KISTAR top-down 파지 파이프라인을
처음부터 구동 가능 상태로 만든 작업 기록.

---

## 1. 개요

**Topdown_Grasp** = 카메라 영상 → SAM3로 물체 분할 → top-down 파지자세 계산 →
ROS2(MoveIt)로 로봇 명령 전송 파이프라인.

**2-PC 설계:**
- **이 레포 (비전/계획)**: 영상 캡처, SAM3 분할, 파지자세 계산, MoveIt 경로계획, 명령 전송
- **GWB 레포 (`/home/kist/GW/Franka_KISTAR_R_Exp_GWB`)**: 실제 FR3+KISTAR 핸드를
  EtherCAT/libfranka로 구동하는 C++ 실시간 브리지 (토픽으로 명령 받음)

```
[카메라] → NPZ → SAM3 분할 → 파지자세 계산 → ROS2 토픽 → [GWB 브리지] → [실제 로봇]
   D435i         (grasp_fruit env, GPU)        (MoveIt, Docker)        (별도 PC/HW)
```

---

## 2. 하드웨어 / 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16GB (Blackwell sm_120), 드라이버 580, CUDA 13 |
| 카메라 | Intel RealSense **D435i** (serial 846112071515), **USB 3.0** 연결 필수 |
| 호스트 OS | Ubuntu, 커널 6.17, ROS2 **Jazzy** (호스트) |
| conda | miniconda3 (`/home/kist/miniconda3`) — miniforge3 없음 |
| Docker | 29.1.3 |
| 비전 env | `grasp_fruit` (Python 3.12, torch 2.10+cu128, transformers/SAM3) |
| 캡처 venv | `/home/kist/rs_env` (pyrealsense2) — grasp_fruit에도 설치됨 |
| MoveIt | Docker 이미지 `grasp_fruit_moveit:latest` (Humble + moveit + pick_ik + rviz2) |

---

## 3. 작업 내역

### 3-1. kistar_ws (MoveIt 워크스페이스) 구축
- GitHub `KIST-PRIME-Lab/dex_ros` 클론 → `/home/kist/GW/dex_ros`
- `isaac-ros/kistar_ws` 빌드 (Docker Humble 컨테이너 안에서 `colcon build`)
- **franka_description 버전 문제**: kistar URDF가 `arm_id`/`accelerometer_config`를
  넘기는데, 공식 태그 중 **2.2.0**만 호환 (2.1.0도 가능, 2.3.0+·1.x는 인터페이스 다름).
  → `kistar_ws/src/franka_description`에 2.2.0 vendored.
  (package.xml에 의존성 선언이 없어 `colcon build --packages-select`로 명시 빌드)
- **깨진 config 복구**: `franka_kistar_moveit_config/config/{kinematics,ompl_planning}.yaml`이
  원작자 PC(`/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/...`)를 가리키는 깨진 심볼릭 링크였음
  → 그룹 `fr3_arm` 기준으로 실파일 재작성.

### 3-1b. IK 솔버 (trac_ik)
- **솔버**: `trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin` (pick_ik 아님)
- **설정** (`franka_kistar_moveit_config/config/kinematics.yaml`):
  ```yaml
  fr3_arm:
    kinematics_solver: trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin
    kinematics_solver_timeout: 0.15
    kinematics_solver_attempts: 3
    position_threshold: 0.001
    orientation_threshold: 0.01
    solve_type: Manipulation2   # manipulability + 관절한계 여유 동시 최적화
  ```
- **빌드 버전 선택 히스토리** (중요):
  - trac_ik 레포엔 배포판별 브랜치(`rolling`/`jazzy`/`kilted`/`master`)가 있고,
    처음엔 **`rolling`(최신)** 으로 빌드하려 함.
  - 그런데 rolling/최신 브랜치는 헤더가 **`.hpp`** (`urdf/model.hpp`,
    `moveit/kinematics_base/kinematics_base.hpp`) → 우리 컨테이너 **Humble** 에선
    `.h` 헤더만 있어 **빌드 실패**.
  - → **태그 `2.0.1`** 로 교체해서 빌드 성공 (2.0.1은 `.h` 헤더 사용, Humble 호환).
  - 즉 **rolling 브랜치 ❌ → 태그 2.0.1 ✅** (2.0.2+ 부터 `.hpp` 라 안 됨).
- **deps**: `libnlopt-dev libnlopt-cxx-dev ros-humble-generate-parameter-library`
- **역할**: 6D 파지 pose → 팔 7관절 값 (IK). 경로계획은 OMPL(별개).
- 링크: https://github.com/traclabs/trac_ik

### 3-2. 경로 설정 (이 레포)
- `configs/paths.yaml`: `__KISTAR_WS__`=`/home/kist/GW/dex_ros/...`,
  `__CONDA_BASE__`=`/home/kist/miniconda3`, `MOUNT_MAP`=identity(`/home/kist/GW`)
- `scripts/start_moveit.sh`, `docker/entrypoint_moveit.sh`: ws 경로를 새 위치로(identity mount)

### 3-3. MoveIt + RViz
- `docker/Dockerfile.moveit`로 `grasp_fruit_moveit:latest` 빌드
- `bash scripts/start_moveit.sh` 한 방으로 MoveIt 구동 (`You can start planning now!`)
- **RViz 추가**: `launch_moveit.py`에 rviz2 노드 + joint_state_publisher(`fake_joints`) 추가,
  `docker/moveit.rviz` 설정파일, start_moveit.sh에 X11 마운트.
  - 로봇 없이 RViz 보기: `FAKE_JOINTS=true bash scripts/start_moveit.sh`
  - `/display_planned_path` Trajectory 디스플레이로 계획 경로 미리보기

### 3-4. 카메라 → NPZ
- USB 케이블 이슈로 고생: 처음 480M(USB 2.0)로만 잡힘 → **케이블이 USB 2.0이라** 깊이 안 나옴
  → 정상 USB 3.0 케이블 + 파란 포트로 **5000M(USB 3.0)** 연결 성공
- pyrealsense2 설치 (rs_env, 이후 grasp_fruit에도)
- 캡처 스크립트로 NPZ 생성 (rgb=BGR uint8, depth=미터 float32, K=3×3)

### 3-5. grasp_fruit 환경 (SAM3 + GPU)
- **디스크 부족**(5GB)으로 막힘 → 안전 정리로 확보:
  conda 캐시(3.5G) + docker `dex_moveit:humble`(4G) + pip 캐시(0.8G) + 설치파일(0.5G)
  + HF openvla-7b 캐시(15G) 삭제 → **29GB 확보**
- `CONDA_BASE=/home/kist/miniconda3 bash setup_pipeline_all.sh`로 설치
  → torch 2.10+cu128 (CUDA OK), SAM3 import OK, 전부 통과

### 3-5b. 서드파티 — TorchSDF
- PyTorch SDF(signed distance function) 라이브러리. 설치:
  ```bash
  git clone https://github.com/wrc042/TorchSDF.git thirdparty/TorchSDF
  cd thirdparty/TorchSDF && git checkout 3f3f83d && bash install.sh
  ```
- 커밋 `3f3f83d` 고정 + `install.sh`로 빌드.
- 링크: https://github.com/wrc042/TorchSDF

### 3-6. SAM3 모델 접근 (gated)
- `facebook/sam3`는 Meta 승인제(gated) 모델
- HF 계정(seok2299)으로 토큰 로그인 + 모델 페이지에서 접근 신청 → **승인됨**

### 3-7. 비전 파이프라인 검증
- 카메라 앞 3개 물체(캔/야구공/오렌지) 캡처
- SAM3로 "orange" 분할 (score 0.961) → 파지자세 계산 → 오버레이 확인 ✅

### 3-8. v1 병합
- `origin/v1`(리팩토링 + 새 캘리브 06-14) 받기. main이 v1의 조상이라 깔끔.
- 충돌 2개만 해결: `paths.yaml`=우리 머신값 유지, `capture_realsense_once.py`=v1 공식판 채택
- 우리 RViz/X11 변경은 v1이 안 건드려 그대로 유지
- v1이 추가: `scripts/utils/grasp.py`(파지 로직), `utils/place.py`,
  `update_world_base.py`, `configs/calibration/extrinsic_20260612_170053.json`(06-14 캘리브)
- 복구용 브랜치: `backup-pre-v1-merge`

### 3-9. v1 재검증
- grasp_fruit에 pyrealsense2 설치
- v1 코드 + 새 캘리브로 `--capture → SAM3 → 파지` 전체 동작 확인 ✅

---

## 4. 현재 상태

```
✅ kistar_ws + MoveIt + RViz
✅ 카메라 (D435i, USB 3.0)
✅ 카메라 → NPZ 캡처
✅ grasp_fruit 환경 (SAM3 + GPU)
✅ SAM3 gated 접근 승인
✅ v1 병합 (우리 셋업 + v1 코드/캘리브)
✅ 비전 파이프라인 E2E (v1 코드, 새 캘리브)
⬜ 실제 로봇 파지 — GWB 브리지 + FR3(172.16.0.1) + KISTAR 핸드 연결만 남음
```

---

## 5. 실행 방법

### 비전 파이프라인 (로봇 없이)
```bash
# 카메라 캡처 + 분할 + 파지계산
/home/kist/miniconda3/envs/grasp_fruit/bin/python scripts/run_pipeline.py \
  --capture --query "orange" \
  --calibration configs/calibration/extrinsic_20260612_170053.json

# 파일 입력으로
/home/kist/miniconda3/envs/grasp_fruit/bin/python scripts/run_pipeline.py \
  --input data/raw/scene_000.npz --query "orange" \
  --calibration configs/calibration/extrinsic_20260612_170053.json
```

### MoveIt + RViz
```bash
FAKE_JOINTS=true bash scripts/start_moveit.sh   # 로봇 없이 RViz 확인용
bash scripts/start_moveit.sh                    # 실제 로봇 연결 시
```

### 실제 로봇 파지 (HW 연결 후)
```bash
# 1) GWB 브리지 실행 (로봇 연결된 PC)
# 2) MoveIt 띄우기 (위)
# 3) --execute_robot 추가
... run_pipeline.py --capture --query "orange" --calibration ... --execute_robot
#    (STEP마다 RViz 확인 + y/n, --speed_factor 0.1 권장)
```

---

## 6. 알려진 이슈 / 팁

- **카메라 `errno=5` (xioctl VIDIOC_S_FMT I/O error)**: 카메라 stuck 상태.
  하드웨어 리셋으로 해결:
  ```bash
  /home/kist/miniconda3/envs/grasp_fruit/bin/python -c \
  "import pyrealsense2 as rs,time; rs.context().query_devices()[0].hardware_reset(); time.sleep(6); print('reset done')"
  ```
  (또는 USB 뺐다 꽂기)
- **USB 2.0로 잡히면 깊이 안 나옴**: 반드시 USB 3.0 케이블 + 파란 포트. `lsusb -t`에서
  카메라가 `Bus 002 ... 5000M`로 떠야 정상 (`Bus 001 480M`이면 2.0).
- **파지 위치가 빗나가면**: `configs/arm.yaml`의 `ee_correction`(x/y offset, yaw)로 미세조정.
- **SAM3 다운로드 실패(gated)**: HF 로그인 + facebook/sam3 접근 승인 필요.

---

## 7. 주요 경로

| 항목 | 경로 |
|---|---|
| 이 레포 | `/home/kist/GW/Topdown_Grasp` |
| kistar_ws (MoveIt) | `/home/kist/GW/dex_ros/isaac-ros/kistar_ws` |
| GWB 로봇 브리지 | `/home/kist/GW/Franka_KISTAR_R_Exp_GWB` |
| grasp_fruit env | `/home/kist/miniconda3/envs/grasp_fruit` |
| 캡처용 venv | `/home/kist/rs_env` |
| 새 캘리브 | `configs/calibration/extrinsic_20260612_170053.json` (06-14) |
| MoveIt 이미지 | `grasp_fruit_moveit:latest` (Docker) |
