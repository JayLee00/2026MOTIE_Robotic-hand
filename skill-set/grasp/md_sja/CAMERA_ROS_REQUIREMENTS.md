# 다른 PC 세팅 요구사양 — 카메라를 ROS 토픽으로 받기

> 현재는 RealSense를 USB로 직접 열지만(pyrealsense2), 새 PC에서는 **카메라를 다른 머신이 ROS로 발행 → 파이프라인 PC가 구독**하는 구조로 바꾼다. 이때 필요한 스펙·토픽·코드변경 총정리.

---

## 0. 아키텍처 변경 (핵심 개념)

**현재 (직접 연결):**
```
[비전 PC] ── USB ── RealSense
  capture_realsense_once.py → pyrealsense2 직접 open → NPZ(rgb,depth,K)
```

**변경 (ROS 연결):**
```
[카메라 호스트 PC]                         [비전/계획 PC = 새 PC]
 RealSense USB                              (카메라 없음)
 realsense2_camera 노드 ──ROS2 토픽──▶      구독:
   /…/color/image_raw                         color 이미지
   /…/aligned_depth_to_color/image_raw        depth (color 정렬)
   /…/color/camera_info                       K(내부파라미터)
                                            → NPZ(rgb,depth,K) 변환
                                            → SAM3 → PCA → IK → 로봇
```

→ **파이프라인은 그대로, "캡처 단계"만 pyrealsense2 → ROS 구독으로 교체.**

---

## 1. 하드웨어 스펙

### 1-A. 비전/계획 PC (새 PC) — 가장 중요
| 항목 | 최소 | 권장 | 이유 |
|---|---|---|---|
| **GPU** | NVIDIA VRAM 8GB | **12~16GB** (RTX 4070/4080/5060Ti+) | SAM3 추론. 현재 RTX 5060 Ti 16GB |
| **CUDA** | 12.x | 현재 **cu128** (Blackwell용) | torch가 GPU 세대 맞아야 |
| **CPU** | 6코어 | 8코어+ | ROS+파이썬+docker |
| **RAM** | 16GB | 32GB | 모델+ROS+컨테이너 |
| **디스크** | SSD 50GB 여유 | SSD 100GB+ | HF모델캐시(수GB)+docker이미지+conda |
| **네트워크** | 1Gbps 유선 | 1Gbps 유선 | 카메라 스트리밍 대역폭(아래) |

> ⚠️ GPU가 핵심. SAM3는 큰 모델이라 VRAM 8GB는 빠듯. 12GB+ 권장.
> ⚠️ 카메라 호스트와 **같은 GPU 세대일 필요는 없음** (카메라 호스트는 GPU 불필요).

### 1-B. 카메라 호스트 PC (RealSense 꽂힌 머신)
| 항목 | 요구 |
|---|---|
| GPU | **불필요** (그냥 카메라 발행만) |
| USB | **USB 3.0(파란) 포트** (RealSense 필수) |
| CPU/RAM | 낮아도 됨 (4코어/8GB) |
| 네트워크 | 1Gbps 유선 (비전 PC와 같은 대역) |

---

## 2. 소프트웨어 요구사항

### 2-A. 비전/계획 PC
```
OS         : Ubuntu 22.04(Humble) 또는 24.04(Jazzy)  ← 현재 호스트 Jazzy
ROS2       : Jazzy(호스트) — 토픽 모니터/구독용
conda      : grasp_fruit (py3.12, torch cu128, transformers/SAM3)
SAM3       : facebook/sam3 (HF gated — huggingface-cli login + 모델페이지 접근승인)
Docker     : ros2_humble 컨테이너 (MoveIt + trac_ik, kistar_ws 빌드)
파이썬 패키지: rclpy, cv_bridge, sensor_msgs, numpy, opencv  ← ROS 이미지 변환용
```

### 2-B. 카메라 호스트 PC
```
OS      : Ubuntu (ROS2 되는 버전)
ROS2    : 비전 PC와 통신되는 배포판 (DDS라 버전 달라도 되지만 맞추는게 편함)
패키지  : ros-<distro>-realsense2-camera + librealsense2
          (sudo apt install ros-humble-realsense2-camera 등)
```

---

## 3. 필요한 ROS 토픽 (카메라)

realsense2_camera 노드가 **반드시 발행**해야 하는 것:

| 토픽 (네임스페이스는 launch마다 다름) | 메시지 타입 | 용도 | 필수 |
|---|---|---|---|
| `…/color/image_raw` | `sensor_msgs/msg/Image` (rgb8/bgr8) | **RGB** (SAM3 입력) | ✅ |
| `…/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` (16UC1, mm) | **depth (color에 정렬)** | ✅ |
| `…/color/camera_info` | `sensor_msgs/msg/CameraInfo` | **K (fx,fy,cx,cy)** | ✅ |
| `…/depth/color/points` | `sensor_msgs/msg/PointCloud2` | 점구름(선택, 직접 안 씀) | ⬜ |
| `…/extrinsics/depth_to_color` | realsense2_camera/Extrinsics | 참고 | ⬜ |

> **가장 중요 — depth는 반드시 color에 aligned:**
> back-project가 "color 마스크 픽셀 + 같은 픽셀의 depth"를 써서, depth 해상도·시점이 color와 **1:1 일치**해야 함.
> → `aligned_depth_to_color` 를 켜야 함 (raw depth는 시점이 달라서 못 씀).

**카메라 호스트에서 실행 예:**
```bash
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    pointcloud.enable:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30
```
(정확한 토픽명은 실행 후 `ros2 topic list | grep camera` 로 확인 — 보통 `/camera/camera/…` 또는 `/camera/…`)

**QoS 주의:** 이미지 토픽은 보통 **SENSOR_DATA = BEST_EFFORT**. 구독할 때 QoS 맞춰야 (파시니처럼).

---

## 4. 파이프라인 코드 변경 (캡처 단계 교체)

**현재:** `scripts/capture_realsense_once.py` → `src/affordance_grasp/io/realsense.py`(pyrealsense2 직접)

**변경:** ROS 구독으로 NPZ 만드는 새 캡처 노드 추가. 해야 할 변환:
```
color image_raw (rgb8/bgr8)  ──cv_bridge──▶  BGR uint8
depth 16UC1 (mm)             ──/1000──────▶  float32 meters
CameraInfo.k[0,2,4,5]        ────────────▶  K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
                             ──저장──▶ NPZ {rgb, depth(m), K}  ← 기존 포맷 그대로
```
→ NPZ 포맷만 맞추면 **run_topdown_grasp.py 등 나머지는 수정 없이 동작.**

**만들 것:** `scripts/capture_from_ros.py` (예시 로직)
```python
# color/depth/camera_info 3개를 message_filters로 시간동기화(ApproximateTimeSync)
# → cv_bridge로 변환 → 1프레임 저장 → np.savez(rgb=bgr, depth=m, K=K)
# QoS: BEST_EFFORT (이미지)
# 토픽명은 인자로 (--color_topic, --depth_topic, --info_topic)
```
그리고 `run_pipeline_interactive.py`의 Stage0(캡처)을 이 ROS 캡처로 교체하는 옵션(`--camera_source ros`) 추가.

---

## 5. 네트워크 / 대역폭 계산

**비압축 스트리밍 (640×480×30fps):**
```
color rgb8 : 640×480×3 = 0.92MB/frame × 30 = 27.6MB/s ≈ 221 Mbps
depth 16bit: 640×480×2 = 0.61MB/frame × 30 = 18.4MB/s ≈ 147 Mbps
합계       : ≈ 368 Mbps  → 기가비트(1Gbps) 유선이면 충분
```
**절약 방법:**
- **파이프라인은 query당 1프레임만 필요** → 30fps 다 안 받아도 됨. 카메라를 낮은 fps(예: 6fps)로 하거나, 구독 시 throttle.
- `image_transport` **compressed**(JPEG/PNG) 쓰면 대역폭 1/5~1/10로 감소.
- WiFi는 비추 (지터/손실 → 프레임 깨짐). **유선 권장.**

**공통:**
- 두 PC **같은 subnet** + **ROS_DOMAIN_ID 일치** (현재 9)
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=0`
- 큰 이미지 = UDP fragmentation → MTU/DDS 버퍼 튜닝 필요할 수 있음 (프레임 깨지면 QoS/버퍼 조정)

---

## 6. 캘리브레이션 (필수)

- `configs/calibration/*.json` 의 **T_base_camera**(카메라↔로봇베이스) + **T_world_base** 필요
- 카메라가 다른 머신에 붙어도, **물리적 위치·자세가 로봇 기준으로 고정**이면 기존 캘리브 유효
- **카메라 위치가 바뀌면 → 재캘리브 필수** (안 그러면 파지가 빗나감)
- 즉 "ROS로 받는다"는 데이터 경로 얘기지, 캘리브는 **물리적 카메라-로봇 관계**라 그대로 적용됨

---

## 7. 로봇/촉각 토픽 (같이 필요, 참고)

| 토픽 | 타입 | 발행처 | QoS |
|---|---|---|---|
| `/franka/joint_position` | Float64MultiArray | Franka PC | — |
| `/franka/target_joint` | Float64MultiArray | 비전PC→Franka | — |
| `/hand/joint_position` | Float32MultiArray | Franka PC | BEST_EFFORT |
| `/hand/target_joint` | Int16MultiArray | 비전PC→핸드 | — |
| `/paxini/ft_r`, `/paxini/ft_l` | Float32MultiArray [4][3] | Franka PC(paxini→nd2) | **BEST_EFFORT** |
| `/move_action` | MoveGroup action | move_group(컨테이너) | — |

---

## 8. 세팅 체크리스트 (새 PC)

**카메라 호스트 PC:**
- [ ] RealSense USB 3.0 연결 + `realsense-viewer` 동작 확인
- [ ] `ros-<distro>-realsense2-camera` 설치
- [ ] `align_depth.enable:=true` 로 launch → `…/aligned_depth_to_color/image_raw` 나오는지 확인
- [ ] ROS_DOMAIN_ID=9, 같은 subnet

**비전/계획 PC:**
- [ ] GPU + cu128 torch (`torch.cuda.is_available()`)
- [ ] conda grasp_fruit + SAM3 (HF 로그인/접근)
- [ ] cv_bridge 설치
- [ ] ros2_humble 컨테이너 (MoveIt) 빌드
- [ ] `configs/paths.yaml` 새 경로로 수정
- [ ] `capture_from_ros.py` 작성 (§4) — color/depth/info 구독 → NPZ
- [ ] 캘리브 JSON (물리 배치 맞으면 복사, 아니면 재캘리브)
- [ ] 네트워크: 카메라 토픽 3개 수신 확인 (`ros2 topic echo …/camera_info --once`)
- [ ] 절전 끄기 (`systemctl mask sleep.target suspend.target`)

**동작 확인 순서:**
1. 카메라 호스트: realsense2_camera launch → 토픽 3개 발행 확인
2. 비전 PC: `ros2 topic list | grep camera` 로 수신 확인
3. `capture_from_ros.py` → NPZ 생성 확인 (rgb/depth/K)
4. 그 NPZ로 `run_topdown_grasp.py --input <npz> --query orange` → PCA 나오면 성공

---

## 9. 요약 (한 장)

> **바꾸는 것:** 카메라 캡처만 pyrealsense2 → ROS 구독(color+aligned_depth+camera_info → NPZ).
> **새 PC 스펙:** GPU 12~16GB(SAM3), 32GB RAM, SSD, 1Gbps 유선.
> **필수 토픽 3개:** `color/image_raw`, `aligned_depth_to_color/image_raw`, `color/camera_info` (depth는 **color에 aligned 필수**, QoS **BEST_EFFORT**).
> **안 바뀌는 것:** SAM3·PCA·IK·로봇 로직, 캘리브(물리 배치 고정이면 그대로).
> **작성 필요 코드:** `capture_from_ros.py` (ROS→NPZ 변환).
