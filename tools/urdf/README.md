# tools/urdf — 로봇 모델 (URDF / xacro / mesh)

URDF 는 ROS 패키지 안에 있어야 `package://` 참조가 풀리므로, 파일을 여기로 복사하지 않고
**어디에 무엇이 있는지**만 정리한다.

## 소스

| 무엇 | 경로 |
|---|---|
| 듀얼 FR3 + KISTAR 핸드 xacro | [`.../franka_kistar_description/urdf/`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/urdf/) |
| 생성 URDF 스냅샷 | [`.../franka_kistar_description/urdf/generated/`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/urdf/generated/) |
| 메시 (58M) | [`.../franka_kistar_description/meshes/`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/meshes/) |
| FR3 본체 메시·관절한계 | `tools/ros2/fr_ws/src/franka_description/` (xacro 가 `$(find franka_description)` 로 include) |
| 테이블 URDF | `.../franka_kistar_bringup/urdf/generated/{table,ttable}.urdf` |

## place skill 이 별도로 들고 있는 사본 (건드리지 말 것)

내려놓기 skill 의 `hand_fk` 는 매 완성마다 URDF 를 직접 파싱한다. 그 경로는
**place 저장소 기준 상대경로**라 별도 사본이 필요하다:

| 파일 | 경로 |
|---|---|
| 생성 URDF (FK 파싱용) | `skill-set/place/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/urdf/generated/dual_fr3_kistar_v2.urdf` |
| 오른손 시각 메시 | `skill-set/place/dex_ros/.../meshes/fr3_kistar_right/kistar_hand/` |
| PaXini 패드 URDF | `skill-set/place/KISTAR_URDF/.../kistar_hand_right_paxini.urdf` |
| PaXini 팁 메시 | `skill-set/place/KISTAR_URDF/.../01_kistar_hand_stl/paxini_tip_visuals.stl` |

> `dual_fr3_kistar_v2.urdf` 는 dex_ros 에서 **gitignore 된 빌드 산출물**이다. 원본 저장소를
> 새로 클론하면 이 파일이 없어 `igr_service` 가 뜨지 않는다. 그래서 place 번들이 자기 경로에
> 사본을 들고 온다. 로봇 모델을 수정했다면 **두 곳을 함께** 갱신할 것.

## 재생성

```bash
tools/urdf/regenerate.sh
```

트윈 launch 는 생성 URDF 스냅샷(`urdf/generated/*.urdf`)과 사이드카 `*.sha256` 의 해시를
검사하고 어긋나면 **즉시 전체를 shutdown 한다**(`strict_urdf_snapshot` 기본 true).
따라서 xacro/메시/`franka_description` 을 고쳤다면 반드시 재생성해야 한다.

이 래퍼가 하는 일:
1. dex_ros 의 `scripts/regenerate_urdf.sh` 를 올바른 `REPO_ROOT` 로 호출
   (그 스크립트의 REPO_ROOT 자동 계산은 원본 PC 의 디렉토리 깊이를 가정해서 현재 배치에서는
   어긋난다 — 직접 부르면 `input not found`)
2. `dual_fr3_kistar{,_v2}.urdf`, `table.urdf`, `ttable.urdf` + 각 `.sha256` 생성
3. 위 §place 사본을 정본과 동일하게 동기화
4. 사이드카 해시 검증

정상이면 트윈 로그에 `[urdf_snapshot] OK — generated/*.urdf hashes match sidecars.` 가 뜬다.
