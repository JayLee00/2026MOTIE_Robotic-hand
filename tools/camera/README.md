# tools/camera — front RealSense (D435i)

**카메라 토픽은 Control PC(192.168.0.100)가 발행한다.** 이 PC 에서는 카메라 노드를 띄우지
않는다 — 구독만 한다.

| 토픽 | 쓰는 곳 |
|---|---|
| `/front_cam/front/color/image_raw` | 파지(seq 1) 캡처, 내려놓기(seq 4) 인지 |
| `/front_cam/front/aligned_depth_to_color/image_raw` | 동일 (⚠ `align_depth.enable:=true` 필수) |
| `/front_cam/front/color/camera_info` | 내부 파라미터 K |

## 확인

```bash
tools/camera/check_camera.sh
```

## Control PC 기동 (참고 — 그쪽 담당자가 실행)

2026-08 통합 시 카메라 파라미터가 Control PC 의 `control_pc.launch.py` 에 **영구 반영**되었다.
별도 카메라 launch 가 필요 없다:

```bash
ros2 launch trajectory_receiver control_pc.launch.py require_control:=true   # 카메라 포함
```

카메라만 띄우려면 나머지를 끈다:
```bash
ros2 launch trajectory_receiver control_pc.launch.py \
  state_pub:=false trajectory:=false ee:=false arm_q:=false hand:=false arbiter:=false
```

⚠ **`rs_launch.py` 를 직접 띄우지 말 것.** 기본값으로 뜨면 (a) 네임스페이스가 `/camera/camera`
가 되어 토픽 이름이 어긋나고, (b) TF 루트가 `camera_link` 이라 트윈의 `world → front_camera_link`
static TF 와 연결되지 않으며, (c) depth auto-exposure 가 켜져 **흰색 펄프 트레이에서 depth 가
62~76% 소실**된다(측정치). 통합 초기에 실제로 발생했던 상황이다.

### 이 저장소가 들고 있는 launch 파일

[`.../franka_kistar_bringup/launch/realsense_front.launch.py`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_bringup/launch/realsense_front.launch.py)
— 위 파라미터의 **원본 출처**다(Control PC 쪽에는 이 파일이 없어 파라미터만 옮겨 갔다).
카메라를 이 PC 로 되돌릴 때만 직접 쓴다. 그럴 땐 `ros-humble-realsense2-camera` 설치 + udev 규칙
`tools/ros2/dex_ros/99-realsense-libusb.rules` 가 필요하고, **외부 캘리브 재측정도 필수**다
([docs/MIGRATION.md](../../docs/MIGRATION.md) §5).

### 확정된 카메라 파라미터 (양쪽이 일치해야 하는 계약)

| 항목 | 값 | 이유 |
|---|---|---|
| `camera_namespace` / `camera_name` | `front_cam` / `front` | 소비 토픽 이름 |
| `base_frame_id` | `camera_link` | 드라이버가 앞에 `front` 를 붙여 루트가 `front_camera_link` 가 된다. `front_camera_link` 로 주면 `front_front_camera_link` 가 되어 TF 트리가 끊긴다 |
| `align_depth.enable` | `true` | 없으면 `aligned_depth_to_color` 토픽 자체가 안 생긴다 |
| `depth_module.enable_auto_exposure` | `false` | 광택 트레이에서 IR 포화 → depth 소실 |
| `depth_module.exposure` | `1500` | 스윕으로 찾은 최소 손실 지점 (62% → <1%) |
| `temporal_filter.enable` | `true` | 깜빡이는 실제 반환을 프레임 간 누적 |
| `hole_filling_filter.enable` | `false` | depth 를 날조하고 후처리 스레드를 멈춘다 |
| color/depth profile | `640x480x30` | `WxHxFPS` 문자열이어야 함 (v4.5x 에서 `color_width` 스타일 제거) |
| `clip_distance` | `1.3` | 작업 영역 밖 절단 |

## 캘리브레이션

- 파지: `skill-set/grasp/configs/calibration/extrinsic_20260612_170053.json`
- 내려놓기: `skill-set/place/vision_pipeline/core/extrinsic.py` 에 4x4 하드코딩
  (참고값 사본: `skill-set/place/tf.txt`)

두 값 모두 **카메라↔FR3 베이스의 물리 배치**에 묶여 있다. 카메라·로봇 배치가 바뀌면
둘 다 다시 측정해서 **함께** 갱신해야 한다.
