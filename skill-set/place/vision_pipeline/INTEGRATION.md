# 통합 시나리오에서의 Place (물체 내려놓기, SEQ 4)

> `물체 파지(1) → 손 안 조작(2) → 물성 추론(3) → **물체 내려놓기(4)**` 체인에서 Place 역할을
> **상시 skill 서버**로 붙인다.
>
> **이 문서는 place 모듈 관점의 설명이다. 실제 기동 절차는 프로젝트 루트의
> [`docs/RUNBOOK.md`](../../../docs/RUNBOOK.md) 를 따른다** — 단일 명령
> `./run_fruit_demo.sh` 가 모델 서비스 · 트윈 · 이 서버 · 4단계를 한 번에 처리한다.
>
> **단독 Place 테스트**는 예전 그대로 [`GOAL_TEST.md`](GOAL_TEST.md) (변경 없음).
>
> ⚠ 구성 변경 이력: 예전에는 3대(Control / Current / 계획 PC) 구성이었고 이 서버는 Current PC
> 도커 안에서 돌았다. **지금은 2대다** — Control PC(192.168.0.100) + 통합 PC(192.168.0.101).
> 순서는 `pipeline/run_pipeline.py` 가 감독하고, 도커는 쓰지 않는다(네이티브 ROS2 Humble).
> 모두 `ROS_DOMAIN_ID=9`.

## 단독(GOAL_TEST) vs 통합의 차이

| | 단독 테스트 (GOAL_TEST.md) | 통합 실행 (체인) |
|---|---|---|
| Place 실행 | `python -m vision_pipeline.run` (parent_pose→child_pose 이동 포함) | `vision_pipeline.skill_server` **상시 서버**. parent/child_pose 이동 **없음** |
| 트리거 | 사람이 명령 1개 실행 | arbiter 의 seq 4 차례. parent 인지는 seq 1(파지) 시작 시 **prewarm** |
| 끝 | T_act→release→retract→parent_pose | 동일. retract 가 parent_pose 도달 시 Place 단계 종료(DONE) |
| move_group(트윈) | 이 PC 에서 실행 | 이 PC 에서 실행 — ⚠ **정확히 1개**만 |

물체는 파지부터 계속 손에 쥐어져 있고(제어권만 넘어옴), 물성 추론이 끝나면 팔이 이미
~child_pose 이므로 Place 는 child_pose 로 **이동하지 않는다** — 그 자리에서 child 캡처 → 인지
→ 배치 → release → parent_pose 복귀.

## 수동 실행 (단계별 검증용)

단일 명령을 쓰면 아래는 자동 처리된다. 손으로 돌릴 때만 참고.

```bash
# 프로젝트 루트에서
source tools/env/setup_env.sh

# 1) 모델 서비스 5종 (상시. Molmo 로딩 ~1분, GPU0 Molmo / GPU1 SAM+AnyPlace+IGR)
bash skill-set/place/vision_pipeline/run_services.sh

# 2) MoveIt 트윈 — 이 PC 에서 정확히 1개만
tools/moveit/launch_twin.sh

# 3) place skill 서버 (상시)
cd skill-set/place
/usr/bin/python3 -u -m vision_pipeline.skill_server scenario=fruit hand_pc=true
#   → `place skill READY ... awaiting Place turn #1` 이면 대기 상태(정상). 과일마다 자동 반복.

# 4) 진행 로그 (선택)
/usr/bin/python3 tools/diagnostics/place_logger.py     # /place/status 구독
```

시스템 python3(3.10)로 실행한다 — `rclpy` + `dual_arm_msgs` + `sequence_client`(Place 4 핸드셰이크)가
필요하고, 이들은 `tools/env/setup_env.sh` 가 소싱하는 kistar_ws 오버레이에 있다.

전제(Control PC 담당): Dual_Arm RT 런타임 · `control_pc.launch require_control:=true`
(arbiter 포함) · PaXini writer(손끝 무접촉 상태에서 tare) · front RealSense
(`align_depth.enable:=true`).

## ONE move_group 규칙 (중요)

`move_group` 은 **정확히 1개**여야 한다. skill_server preflight 가 `/move_action` 서버 1개를
요구하며, 2개면 예전 "모든 arm move 실패" 버그가 재현된다. 통합 PC 의 트윈
(`dual_fr3_kistar_moveit.launch.py` → `..._planning_pc_v2`) 하나만 띄울 것.

`hand_pc=false` 면 오른손 mesh 의존은 빠지지만 v2 URDF + paxini URDF +
`paxini_tip_visuals.stl` 은 무조건 필요하다.

## 외부 캘리브레이션

카메라→베이스 외부 파라미터는 고정 실측 상수이며 `core/extrinsic.py` 에 4x4 로 하드코딩되어
있다(참고 사본: 저장소 루트 `tf.txt`). **카메라 발행자는 Control PC 이고, 카메라를 물리적으로
옮기지 않았다면 이 값은 그대로 유효하다.** 카메라/로봇 배치를 바꿨다면
`T(right_fr3_link0 ← front_cam_optical)` 를 재측정해 (a) bringup 의 static TF/URDF 와
(b) `tf.txt` + `core/extrinsic.py` 를 **둘 다 동일하게** 갱신할 것.

## 그립 인수인계 (Stiffness 3 → Place 4)

물체는 파지(1)부터 계속 쥐어져 있고, 물성 추론(3)이 그 파지를 유지하는 Position target 을
`/hand/right/q_target` 으로 스트리밍한다. 제어권이 넘어오면 직전 소유자는 스트리밍을 멈추고,
`require_control:=true` 에서는 소유자가 아닌 노드의 target 은 무시된다.

그래서 이 서버는 **제어권 획득 직후 직전 단계의 target 을 그대로 이어받아** release 직전까지
20Hz 로 유지한다(`hand_hold_start()`). 로그: `GRIP HANDOVER: holding the previous stage's hand
target [...]`.

인수할 target 을 못 봤으면(또는 30초 이상 오래되었으면) **아무것도 붙잡지 않고** 경고만 남긴다 —
16-DoF 목표를 추측해 발행하는 것은 파지를 다시 명령하는 것이라 더 위험하다. 이때는 수신 노드가
래치한 목표가 그립을 유지한다.

## 안전 / 주의

- 첫 팔 동작은 속도 스케일 낮게, E-stop 손 위에. 배치·하강은 파지물체 충돌 미고려라 여유·저속.
- Release 는 손을 **Voltage 모드**로 전환 → 파지 성립 후 Hand GUI 를 만지지 말 것(폭주 위험).
- **Voltage 는 release 구간에서만 존재한다.** 정상·예외·중단 모든 경로에서 Position + servo OFF
  로 복귀한다([R5] `try/finally` + 프로세스 종료 안전망). 상세: `docs/ARCHITECTURE.md` §7.
  회귀 테스트: `python3 vision_pipeline/test_hand_safety.py`
- PaXini 가 0이거나 없으면 서버가 "tactile triggers DISABLED" 를 크게 로그하고
  T_act(Case 2)에서만 release 한다(조용히 넘어가지 않는다).
- 실패 시 서버는 abort 한다(DONE 없음) → arbiter 가 IDLE 로 회수 → 체인이 거짓 성공으로
  진행하지 않는다.
