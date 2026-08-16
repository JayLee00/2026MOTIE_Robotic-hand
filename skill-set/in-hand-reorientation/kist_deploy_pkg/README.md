# KIST in-hand manipulation — 새 PC 이관 패키지

이 zip 하나로 **① 학습된 정책 배포**, **② 과일 실시간 6DoF 포즈**, **③ 통합 시각화** 를 새 PC에서 돌린다.

대상: Ubuntu + ROS 2 Humble + **RTX 5090 × 2 (sm_120 / Blackwell)**

---

## 0. 먼저 읽을 것 — 이 패키지의 3가지 함정

### 🔴 함정 1. RTX 5090 은 sm_120 이라 기존 바이너리가 **하나도 안 돈다**
- 기존 PC 의 `torch 2.6.0+cu124` 는 5090 에서 **동작하지 않는다**. → `torch >= 2.7` **cu128** 빌드 필요.
- FoundationPose 의 docker 이미지(`foundationpose:local`, 20.4 GB)는 `torch 2.0.0+cu118` + `TORCH_CUDA_ARCH_LIST=8.6` 으로 굳어 있다. **가져오지 말 것.** 새로 빌드해야 한다.
- 재컴파일 필수: `mycpp`, `kaolin`, `nvdiffrast`, `pytorch3d` → 전부 `TORCH_CUDA_ARCH_LIST=12.0` (또는 `12.0+PTX`), CUDA **12.8+**.

### 🔴 함정 2. DDS 가 IP 화이트리스트로 묶여 있다
`config/fastdds_lan_only.xml` 이 **`192.168.0.1`** 만 허용한다. 새 PC 의 LAN IP 로 **반드시 수정**해야 토픽이 보인다. 안 고치면 "카메라/핸드 토픽이 아예 안 옴" 으로만 보인다.

### 🔴 함정 3. 과일 텍스처 4종이 전부 **복숭아 껍질**로 덮여 있다
`assets/{lemon,plum,mandarin,peach}.obj` 가 **하나의 `material.mtl` + `material0.jpeg`** 를 공유한다.
`prepare_mesh.py` 가 내보낼 때마다 이전 텍스처를 덮어써서, 현재 남은 건 마지막에 내보낸 **복숭아** 텍스처다.
→ 지금은 **peach 만 외형이 맞다.** FoundationPose 는 텍스처로 회전을 잡으므로 이건 성능에 직결된다.
→ 복구용 원본 스캔을 `2_fruit_pose/Obj/` 에 넣어뒀다. **§4-D** 참고.

---

## 1. 패키지 구성

```
1_policy/            학습된 정책 배포 (VTDP)                     ~231 MB
2_fruit_pose/        FoundationPose 실시간 과일 6DoF             ~625 MB
3_visualization/     통합 시각화 (새로 만든 것)                  ~0.5 MB
config/              DDS 프로파일 등                             ~2 KB
```

---

## 2. 새 PC 사전 준비

```bash
# NVIDIA 드라이버 >= 570 (Blackwell), CUDA Toolkit 12.8+
nvidia-smi                                   # 5090 2장 인식 확인

# ROS 2 Humble + 부속
sudo apt install ros-humble-desktop ros-humble-image-transport \
                 ros-humble-cv-bridge python3-yaml

# Python (ROS 의 rclpy 를 같이 import 할 수 있는 인터프리터에)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install numpy pyyaml h5py pillow opencv-python trimesh open3d
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 네트워크 (제어 PC 와 같은 DDS 도메인)
```bash
# 1) config/fastdds_lan_only.xml 의 <address> 를 새 PC LAN IP 로 수정
# 2) 매 터미널마다
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=<설치경로>/config/fastdds_lan_only.xml
ros2 topic list | grep -E "hand/right|paxini|front_cam"   # 안 보이면 위 IP부터 의심
```

> **제어 PC 가 계속 데이터 소스다.** 카메라(RealSense), 손 관절, paxini 는 전부 제어 PC 가 발행한다.
> 제어 PC 에서 `shm` → `nd` → `rs` 가 떠 있어야 한다.

---

## 3. ① 정책 배포 (`1_policy/`)

### 설치
```bash
# 두 저장소를 나란히 둔다
~/Desktop/kist-vtdp-wrapper/          <- 1_policy/kist-vtdp-wrapper/
~/Desktop/vive_franka_teleop/diffusion_policy/   <- 1_policy/diffusion_policy/

# ImageNet 사전학습 가중치 (오프라인이면 필수)
mkdir -p ~/.cache/torch/hub/checkpoints
cp 1_policy/torch_cache/resnet18-f37072fd.pth ~/.cache/torch/hub/checkpoints/

export KIST_VTDP_REPO=~/Desktop/kist-vtdp-wrapper    # 경로 하드코딩 회피
```

### 실행 (순서 지킬 것)
```bash
cd ~/Desktop/vive_franka_teleop/diffusion_policy

python3 run_kist_vtdp.py --list_models                  # 모델 표
python3 run_kist_vtdp.py --model_type 000 --self_test   # ROS/로봇 없이 계약 검증. p90 < 50ms 여야 함
python3 run_kist_vtdp.py --model_type 000 --dry_run     # 전체 상태머신, 발행만 안 함
python3 run_kist_vtdp.py --model_type 000               # 실전
```

**모델 000** = `runs/loop/r6_x_tacdrop_vt` — 레몬 · 시각+촉각 · J 405.45 · **유일하게 실기 검증됨**.
포함된 가중치는 000 뿐이다. 다른 model_type 을 쓰려면 원본 PC 에서 해당 `runs/loop/<name>/{best.pt,config.yaml}` 을 추가로 복사한다 (각 159~223 MB).

### ROS2 계약
```
IN   /hand/right/joint_states                     JointState            200 Hz
IN   /paxini/right/ft                             Float32MultiArray[12]  90 Hz
IN   /front_cam/front/color/image_raw/compressed  CompressedImage        30 Hz (640x480 JPEG 고정)
IN   /front_cam/front/color/camera_info           CameraInfo
IN   /teleop/hand_engage/right                    Bool  (RELIABLE + TRANSIENT_LOCAL 래치)
OUT  /hand/right/q_target                         Float32MultiArray[16] 100 Hz  [count]
OUT  /hand/right/cmd_mode(=1), /hand/right/cmd_servo(=true)   engage 시 1회
OUT  /kist_vtdp/debug                             Float64MultiArray[16] 100 Hz
```

### ⚠️ 안전
- **engage 되기 전엔 아무것도 발행하지 않는다.** 발판(`record/foot_pedal_glove.py --side right`) 오른쪽, 또는
  `ros2 topic pub -1 /teleop/hand_engage/right std_msgs/Bool "{data: true}" --qos-durability transient_local`
- **정지 = 홀드이지 이완이 아니다.** Ctrl-C/disengage 해도 손은 마지막 타겟을 유지한다. 힘을 빼려면 **다른 터미널에 미리 준비**해 둘 것:
  ```bash
  ros2 topic pub -1 /hand/right/cmd_servo std_msgs/Bool "{data: false}"
  ```
- **`tools/glove_teleop.py` 는 반드시 꺼져 있어야 한다.** `q_target` 발행자가 둘이면 손이 튄다.
- 카메라 내부파라미터가 학습 때와 0.5 이상 다르면 노드가 정지한다 (`fx=614.678 fy=614.930 cx=318.892 cy=240.121`).
  `crop_reference.png` 로 화각도 눈으로 대조할 것.
- 속도가 안 나오면 `--ddim_steps 4` (100→44.6 ms, 품질 손실 없음), GPU 분리는 `--device cuda:1`.

---

## 4. ② 과일 실시간 6DoF (`2_fruit_pose/`)

### 구조
```
카메라(compressed) → republish → fp_ros_node.py (호스트, SAM2 첫프레임 마스크)
      ↕ TCP :5577 pickle
   fp_server.py (FoundationPose, GPU)      →  /fruit/pose + /fruit/size
```

### A. 배치
```
~/Desktop/vive_franka_teleop/foundation_pose/     <- 2_fruit_pose/foundation_pose/
~/Desktop/vive_franka_teleop/record/fruit_overlay.py
~/Desktop/vive_franka_teleop/fruit-manipulation/  <- 2_fruit_pose/sam2/  (경로 주의, 아래 참고)
```

### B. SAM2 (호스트측, 필수)
`fp_ros_node.py:48-50` 이 아래 경로를 **하드코딩**한다 — 새 PC 경로에 맞게 수정하거나 같은 경로로 배치:
```
FM_ROOT   = /home/js/Desktop/vive_franka_teleop/fruit-manipulation
SAM2_CKPT = $FM_ROOT/sam2.1_hiera_tiny.pt          (149 MB, 동봉)
SAM2_CFG  = configs/sam2.1/sam2.1_hiera_t.yaml
```
```bash
pip install -e <설치경로>/fruit-manipulation/third_party/sam2
```

### C. FoundationPose 빌드 (sm_120) — **가장 오래 걸리는 단계**
동봉한 `FoundationPose/` 에는 **네트워크 가중치가 이미 들어있다** (258 MB):
```
weights/2023-10-28-18-33-37/model_best.pth   68 MB  (refiner)
weights/2024-01-11-20-02-45/model_best.pth  190 MB  (scorer)
```
빌드는 상류 공식 경로를 권장한다 (기존 docker 이미지는 버릴 것):
```bash
cd foundation_pose/FoundationPose
rm -rf mycpp/build                      # 구 py3.8/sm_86 산출물 — 반드시 삭제 후 재빌드
conda env create -f environment.yml     # python 3.11
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install -r requirements.txt
bash build_all_conda.sh
```
> nvdiffrast 는 내부에서 `TORCH_CUDA_ARCH_LIST` 를 덮어쓴다. 설치 후
> `site-packages/nvdiffrast/torch/ops.py` 의 `os.environ['TORCH_CUDA_ARCH_LIST'] = ''` 를 `= '12.0'` 로 패치할 것.

CUDA 12.8 이 호스트에 있으면 **docker 를 아예 안 써도 된다.** 그 경우 TCP 분리(1~6 ms)도 없앨 수 있다.

### D. 과일 텍스처 복구 (권장)
현재 4종이 복숭아 텍스처를 공유한다(§0 함정 3). 원본 스캔이 `Obj/` 에 있다:
```bash
# 과일마다 별도 디렉터리로 내보내야 material0.jpeg 충돌이 안 난다
python3 prepare_mesh.py --in Obj/Lemon/model_New\ Qlone_*.obj  --out assets/lemon/lemon.obj
python3 prepare_mesh.py --in Obj/Plum/...                      --out assets/plum/plum.obj
python3 prepare_mesh.py --in Obj/mandarin/...                  --out assets/mandarin/mandarin.obj
python3 prepare_mesh.py --in Obj/Peach/...                     --out assets/peach/peach.obj
# 그 다음 fruits.yaml 의 mesh: 경로를 새 위치로 수정
python3 fruit_label_node.py --list      # 확인
```

### E. 실행
```bash
bash foundation_pose/run_foundation_pose.sh --check          # 사전점검만
bash foundation_pose/run_foundation_pose.sh --fruit lemon    # 과일 지정
```
> ⚠️ `--fruit` 를 **값 없이** 주면 조용히 `lemon` 으로 떨어진다 (`run_foundation_pose.sh:52,60`). 항상 이름을 붙일 것.

### 발행 토픽
```
/fruit/pose       PoseStamped              ~27 Hz  (카메라 광학 프레임)
/fruit/size       Float32MultiArray[a,b,c] ~27 Hz  [m]
/fruit/type       Int32        (latched)   /fruit/type_name String
```

---

## 5. ③ 통합 시각화 (`3_visualization/`) — 새로 만든 것

과일 포즈 + Paxini 촉각 + FT 를 **한 창**에 띄운다.

```
┌────────────────────────┬──────────────────┐
│ 카메라 + 과일 바운딩박스│ Paxini 촉각 3D   │
│  + XYZ축 + 수치        │  (4개 파트)      │
├──────┬──────┬──────┬───┴──────────────────┤
│Thumb │Index │Middle│ Ring                 │  ← 손가락 4칸
│Tx Ty Fz (각 칸에 3줄, 링버퍼 6초)         │
└──────┴──────┴──────┴──────────────────────┘
```

- **과일 포즈**: `record/fruit_overlay.py` 의 `Overlay` 클래스를 그대로 임포트 (기존 표시 방식 유지)
- **센서**: `kist-vtdp-wrapper/tools/viz_demo.py` 의 `Scene3D`(Open3D) 를 이식 — 지문 CAD 4개에 127 탁셀을 힘 크기로 색칠 + 상위 2개 힘 화살표. `fruit_overlay` 자체 FT 패널은 끄고(`show_ft=False`) 이걸로 대체했다.

**실시간성 (실측 근거)**: 원본은 HDF5→MP4 오프라인 렌더러였고 병목은 Open3D 가 아니라 **matplotlib 전체 재그리기(31~51 ms)** 였다. Open3D 렌더 자체는 **5.6 ms(179 fps)**. 그래서 합성을 cv2 로 바꿔 **8~12 ms** 로 만들었다 → 30 Hz 여유.

### ⭐ 한 줄 실행 — 과일 인식 → 시각화 순차 기동 (권장)
```bash
bash scripts/run_fruit_viz_all.sh                 # 기본 lemon
bash scripts/run_fruit_viz_all.sh --fruit peach   # 과일 지정
bash scripts/run_fruit_viz_all.sh --check         # 전제조건 점검만
bash scripts/run_fruit_viz_all.sh --keep-overlay  # 기존 fruit_overlay 창도 함께
```
1) `run_foundation_pose.sh --no-overlay` (SAM2+FoundationPose → `/fruit/pose`, `/fruit/size`)
2) `/fruit/pose` 수신 대기 (최대 180초)
3) `live_viz.py` 통합 시각화
Ctrl+C 하면 docker `fp_server` 포함 전부 정리된다.

### 시각화만 따로
```bash
bash scripts/run_live_viz.sh --selftest     # ROS 없이 렌더 검증 (PNG 저장)
bash scripts/run_live_viz.sh                # 실시간
bash scripts/run_live_viz.sh --no-tactile   # 과일만
bash scripts/run_live_viz.sh --tactile-topic /glove/paxini/right/raw   # 글러브 촉각으로
```
키: `q`/`ESC` 종료, `s` 스냅샷 (`Visualization/snapshots/`)

> 실행 권한이 없다는 오류(`Permission denied`)가 나면 `chmod +x scripts/*.sh Visualization/*.py`.
> **sudo 로 돌리지 말 것** — root 는 X 접근이 막혀 창이 안 뜬다.

필요 에셋(동봉, 432 KB): `assets/fingertip-PX6AX-GEN3-DP-M2826-Omega.stl`, `assets/taxel_m2826_127.csv`

**주의**
- 촉각 색상 정규화 상한은 `--vmax` (기본 1.3). 실측 분포 p99=0.49, p99.9=1.33, max=2.82.
- **Paxini 4블록이 어느 손가락인지는 미확인**이다 (UART 가 스트림 순서로만 준다). 그래서 촉각은 part0~3 으로만 표시한다. FT(`kin`) 쪽 손가락 대응은 확인됨.
- `06_hand_j_kin` 축 순서는 **(Fz, Tx, Ty)**. (구 teleop 저장소 주석의 `[Ty,Tx,Fz]` 는 잘못된 표기다.)

---

## 6. 새 PC 에서 고쳐야 할 하드코딩 목록

| 파일:줄 | 값 | 조치 |
|---|---|---|
| `config/fastdds_lan_only.xml:11` | `192.168.0.1` | **새 PC LAN IP** (최우선) |
| `run_kist_vtdp.py:90` | `/home/js/Desktop/kist-vtdp-wrapper` | `export KIST_VTDP_REPO=` 로 회피 가능 |
| `runs/loop/*/config.yaml` `data.root` | `/home/js/.../record/logs` | 자가검증용. 없으면 경고만 |
| `dp_config.py:90` | `DATA_ROOT` | 배포 경로에선 미사용 |
| `fp_ros_node.py:48-50` | `FM_ROOT`, SAM2 ckpt/cfg | SAM2 경로 |
| `run_foundation_pose.sh:43` | `/home/js/franka_ros2_ws` | 워크스페이스 경로 |
| `run_foundation_pose.sh:46` | `DISPLAY=:1` | 새 PC 는 보통 `:0` |
| `run_foundation_pose.sh:28-32` | `NS=/front_cam/front` | 카메라 네임스페이스 |
| `run_foundation_pose.sh:34-35` | docker 이미지, conda `my`, py3.8 | sm_120 재빌드 시 전면 수정 |
| `scripts/fix_ros_net.sh:2,17` | NIC `enp6s0`, `192.168.0.1/24` | NIC 이름·IP |
| `FoundationPose/mycpp/build/` | 구 sm_86 산출물 | **삭제 후 재빌드** |

---

## 7. 미검증 / 알려진 문제

- **정책 배포는 이 PC(4080 SUPER)에서만 검증됐다.** 5090 + torch cu128 조합은 미검증 — 반드시 `--self_test` → `--dry_run` 순으로 확인할 것.
- **통합 시각화는 셀프테스트(가짜 데이터)까지만 검증**했다. 실제 토픽 연결 검증은 새 PC에서 필요. 특히 `/paxini/right/raw`(1524) 가 로봇 핸드 것인지 확인할 것 — 글러브 쪽은 `/glove/paxini/right/raw` 로 별도다.
- `fruit_label_node.py:63` 은 `/fruit/reset` 을 하드코딩하는데 `fp_ros_node.py:152` 는 `<ns>/reset` 을 구독한다. `--compare`(ns=`/fruit_fp`) 로 띄우면 과일 CAD 교체가 조용히 안 먹는다.
- FoundationPose 는 upstream tarball 을 받은 것이라 **커밋 핀이 없다**(`.git` 없음).

---

## 8. 첫날 순서 (권장)

1. 드라이버/CUDA 12.8/ROS2 설치 → `nvidia-smi`, `ros2 topic list`
2. DDS IP 수정 → 제어 PC 토픽이 보이는지 확인 (여기서 막히면 다음 단계 무의미)
3. `1_policy` → `--self_test` 통과시키기 (ROS/로봇 불필요, 여기서 torch/GPU 검증)
4. `3_visualization` → `--selftest` 통과 → 실시간 연결
5. `2_fruit_pose` → FoundationPose 재빌드 (반나절 예상) → `--check` → `--fruit lemon`
6. 마지막에 `1_policy` 실전 (`--dry_run` 먼저, 서보 OFF 명령 대기 터미널 준비)
