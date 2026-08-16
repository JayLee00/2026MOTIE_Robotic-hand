# sequence_client — 시퀀스 제어권 클라이언트 (GPU PC 팀용)

각 시퀀스(1=Pick, 2=Inhand, 3=Stiffness, 4=Place)의 **Start(S)/End(E)** 신호,
**하트비트** 자동 발행, **자동 체이닝**(직전 시퀀스 DONE 대기)을 감싼 Python 라이브러리.

## GPU PC에 설치

복사 단위는 두 패키지: `dual_arm_msgs` + `sequence_client`.

```bash
# ros2 워크스페이스 src/에 두 디렉터리 복사 후
colcon build --packages-select dual_arm_msgs sequence_client
source install/setup.bash
```

## 사용법

```python
import rclpy
from dual_arm_msgs.msg import SequenceState
from sequence_client import SequenceClient

rclpy.init()
client = SequenceClient(SequenceState.SEQ_INHAND)          # client_id = seq_id = 2
client.wait_for_previous_done(SequenceState.SEQ_PICK)      # 직전 DONE 대기 (Pick은 생략)
with client:      # 진입 = Start(S) + 하트비트, 정상 탈출 = End(E) → DONE
    ...  # 실제 동작 (타겟 스트리밍 — docs_dev/ROS2_TOPIC_GUIDE.md §2)
client.shutdown()
```

- 예외로 빠져나오면 **release 하지 않고** 하트비트만 멈춘다 → 3초 후 arbiter가
  자동 회수(IDLE) → 다음 시퀀스는 `PreviousAborted`로 중단 (잘못 진행 방지).
- 체인 전체 재실행 전에는 제어 PC의 arbiter를 재시작할 것 (latched 상태 초기화).
- 추후 오케스트레이터 노드가 자율 체이닝을 대체할 수 있음.

## 예제 실행

```bash
ros2 run sequence_client pick_sequence_example            # 시퀀스 1 단독
ros2 run sequence_client run_sequence 2 --work-sec 3      # 체이닝 포함 범용 러너
```

프로토콜 상세: `docs_dev/USAGE_GUIDE.md` §4, `docs_dev/ROS2_TOPIC_GUIDE.md` §3
