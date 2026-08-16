# tools/rviz — 시각화

RViz 는 MoveIt 트윈 launch 가 함께 띄운다(`use_rviz:=true`). 별도 실행 경로는 두지 않았다 —
설정 파일이 `franka_kistar_bringup` 패키지 안에 있고 launch 가 그것을 참조하기 때문이다.

## 설정 파일 위치

[`tools/ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_bringup/rviz/`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_bringup/rviz/)

| 파일 | 용도 |
|---|---|
| `fr3_kistar.rviz` | 기본 |
| `fr3_kistar_camera.rviz` | 카메라 디스플레이(FrontRGB/FrontDepth/FrontCloud) 포함 — 트윈 래퍼가 `camera_view:=true` 로 강제 |

## 바꿔 쓰기

```bash
tools/moveit/launch_twin.sh rviz_config:=fr3_kistar.rviz
tools/moveit/launch_twin.sh use_rviz:=false          # 헤드리스
```

## place 단계의 디버그 마커

내려놓기 skill 은 `/place_debug/markers` 로 후보 포즈·포인트클라우드를 발행한다.
RViz 에 MarkerArray 디스플레이를 추가하면 배치 판단 근거를 볼 수 있다.
또한 `grid_service`(:8815)가 캡처 이미지 그리드를 브라우저로 띄운다.
