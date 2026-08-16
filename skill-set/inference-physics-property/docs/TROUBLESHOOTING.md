# 트러블슈팅 & 개발 노트 — SHM → ROS2 배포

> SHM 직접제어(데이터 수집 환경)에서 검증된 `deploy.py` 로직을 **Dual_Arm_Hand_Ctrl 의 ROS2
> 토픽**으로 옮겨 배포(deploy)하면서 발생한 문제와 해결, 그 과정에서 추가한 기능을 정리한다.
> 전반적인 사용법은 [README](../README.md) 참고. 여기는 "왜 이런 문제가 났고 어떻게 고쳤는지"에 집중.

핵심 배경 한 줄: **실제 관절 제어(1kHz stiff hold)는 제어 PC 의 C++/RT 가 담당하고, 이 레포의
파이썬 코드는 100Hz 로 목표치(q_target)만 발행하는 상위 계층**이다. 문제 대부분은 이 경계(비-RT
파이썬 + DDS 토픽)에서 생겼다.

---

## 요약 표

| # | 증상 | 원인 | 해결 | 파일 |
|---|------|------|------|------|
| A1 | `FileNotFoundError` 모델 없음 | 모델 경로 절대경로 하드코딩(`/home/yesol/...`) | `_MODEL_DIR = str(_MODELS_DIR)` (레포 상대경로) | `real_deploy_inference_final.py` |
| A2 | `TypeError: num_layers` | repo `model.py` 가 LSTM 체크포인트 구조와 불일치 | 학습본 `model.py` 로 교체(또는 `num_layers` 인자 추가) | `model.py` |
| A3 | `Expected all tensors on same device (cuda/cpu)` | `x` 는 CPU, `mean/std` 는 cuda | `torch.tensor(..., device=self.device)` | `real_deploy_inference_final.py` |
| A4 | 라벨 못 찾음 / colcon 미번들 | 라벨을 `labels/*/` 하위로 이동 | `LABEL_DIR` 지정 + `package_data` 를 `labels/*/*.yaml` | `real_deploy_inference_*.py`, `setup.py` |
| B1 | `샘플 부족 — 추론 불가` | `/paxini/right/ft` 미수신 → valid=0 → 전 샘플 스킵 | paxini_writer 실행(힘 피드백 확보) | (운영: 제어 PC) |
| B2 | `⚠ mN 파일 없음` 경고 | jkin mN side-channel 파일 부재 | 통합모델은 jkin 미사용 → `USE_JKIN` 일 때만 경고 | `real_deploy_inference_final.py` |
| B3 | 힘 임계 계속 미도달 / 추론 품질↓ | 학습(palm-down·100Hz) vs 실행(palm-up·저속) 도메인 불일치 | Path A(실행 조건 맞추기) 또는 Path B(재보정) | 전반 |
| C1 | 손 동작 느리고 끊김(2s 정지) | 명령 QoS RELIABLE + 수신자 BEST_EFFORT → write 블로킹 | q_target 스트림을 BEST_EFFORT + servo/mode 중복발행 제거 | `deploy_ros2.py` (브리지) |
| C2 | 스퀴즈 시 비-엄지 손가락 벌어짐 | hold 기준이 feedback 스냅샷 + 보간, palm-up 부하 | 파지 명령값으로 고정 + 매 tick pin | `deploy.py` |
| C3 | 서보 켤 때 손가락 튐 | q_target 없이 servo 먼저 on → q_tar=0 점프 | 현재 자세를 q_target 으로 먼저 발행 후 servo | `deploy_ros2.py` (브리지) |
| D1 | 토픽 미매칭(로봇 안 움직임) | 구 토픽명(`_r` 접미사) 사용 | 신 규약 `/<side>/` 로 전부 변경 | `deploy_ros2.py`, README |
| **F1** | 추론 힘이 학습보다 낮음(스퀴즈 임계 미달, ~4N) · *초기엔 "루프 느림"으로 오진* | **rate 는 정상**(F5, ~95Hz)·자세 무관(F6). 실제는 **P2P→ROS2 로 힘 경로 변화** — 측정(H1: 배포=ft vs 학습=Σ127) 또는 모션(H2) | **H1 확정**(실측: Σ127 이 ft 의 2~3배) → 브리지 **`/raw` 구독**(=`--paxini raw` **표준 경로**). 남은 임계미달(77~87%)은 힘크기 별도 과제 (F5–F7) | `deploy_ros2.py`, `deploy_ros2_exp*.py` |

---

## A. 실행 / 환경 에러

### A1. 모델 경로 하드코딩 → `FileNotFoundError`

- **증상**: 다른 머신(`/home/cy/motie_ws/...`)에서 실행 시
  `FileNotFoundError: '/home/yesol/.../models/....pth'`.
- **원인**: `UNIFIED_MODELS` 가 `_MODEL_DIR = "/home/yesol/stiffness_deploy_ros2/..."` 절대경로
  기반으로 모델 경로를 만듦 → 다른 머신에 그 경로가 없음.
- **해결**: 이미 정의된 레포 상대경로를 사용.
  ```python
  # real_deploy_inference_final.py
  _MODEL_DIR = str(_MODELS_DIR)   # = Path(__file__).../models  → 어느 머신이든 동작
  ```
- **교훈**: 모델/라벨은 레포 번들 + `Path(__file__)` 기준으로만 참조(절대경로 금지).

### A2. `TypeError: StiffnessRegressor.__init__() got an unexpected keyword argument 'num_layers'`

- **증상**: 새 LSTM 체크포인트 로드 시 위 에러.
- **원인**: repo 의 `model.py` 가 "1층 축소" 구버전이라 `num_layers` 인자를 안 받음. 체크포인트의
  `model_config` 에는 `num_layers`(파일명 `L2`/`L1`)가 들어있음 → `model_cls(**cfg)` 에서 충돌.
- **해결**: **학습에 쓴 `model.py` 로 교체**(가장 확실). 임시로는 `StiffnessRegressor` 에
  `num_layers` 인자 추가 후 `nn.LSTM(num_layers=num_layers, ...)`.
- **주의**: `size mismatch` / `Missing key` 가 나면 구조가 더 어긋난 것 → 학습본 통째 교체 필요.

### A3. CUDA/CPU 디바이스 불일치

- **증상**: 추론 진입 시
  `RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu`.
- **원인**: `x = torch.tensor(s)` 는 CPU 생성인데 정규화(`x = (x - self.mean)/...`)를 이동 전에 수행,
  `self.mean/std` 는 cuda 에 있음(GPU 머신).
- **해결**: 입력을 처음부터 device 에 생성.
  ```python
  x = torch.tensor(s, dtype=torch.float32, device=self.device)
  ```
- **참고**: CPU 전용 머신에선 안 드러나고, GPU 머신에서만 발생.

### A4. 라벨 재구성 + colcon 번들

- **증상**: 라벨을 `labels/general/`, `labels/trial2/` 하위로 옮긴 뒤 로드 실패 가능 / colcon
  설치본에 라벨 미포함.
- **해결**:
  - `LABEL_DIR` 을 실제 사용할 하위폴더로 지정(예: `labels/trial2` 또는 `labels/general`).
  - `setup.py` 의 `package_data` 를 `labels/*.yaml` → `labels/*/*.yaml` 로(서브폴더 포함).

---

## B. 추론이 안 되거나 품질이 떨어짐

### B1. `샘플 부족 — 추론 불가` (가장 흔함)

- **증상**: 스퀴즈는 끝나는데 매번 `[추론 결과] 샘플 부족`. 파지/스퀴즈 힘도 전부 "미도달".
- **원인 사슬**:
  1. `/paxini/right/ft` 미수신 → `Ros2PaxiniBridge.read()` 가 항상 `valid=0` 반환.
  2. 엔진 `add_sample` 이 `if s["valid"] != 1: return` 으로 **모든 프레임 스킵** → 버퍼 0개.
  3. `infer()` 의 `n < MIN_LEN` → `(None,None,None)`.
  4. 덤으로 힘=0 이라 파지/스퀴즈 threshold 도 못 넘음.
- **해결**: **제어 PC 에서 paxini_writer 실행**(우측 손).
  ```bash
  python3 ~/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r
  ros2 topic hz /paxini/right/ft      # ~90Hz 확인
  ```
- **정리**: paxini 는 힘 피드백 + 추론 입력의 **필수 소스**. 없으면 추론 자체가 성립 안 함.

#### B1-b. 더 고약한 변종 — **토픽은 오는데 값이 0 고정** (2026-07-27)

- **증상**: `ros2 topic hz /paxini/right/ft` 는 89Hz 정상, `attach()` 도 통과, 그런데 힘이 계속
  ~0N. `valid=1` 이라 `add_sample` 도 정상 적재되고 추론까지 돌아간다(결과는 매번 거의 동일).
- **원인**: `paxini_writer.py` **미실행**. `/paxini/*` 는 `shm_state_publisher` 가 PaXini **SHM
  영역을 그대로 재발행**하는 구조라, writer 가 SHM 을 채우지 않으면 **0 을 89Hz 로 영원히 재발행**한다.
  → hz·attach·valid 전부 정상으로 보이는 **거짓 정상**. (B1 의 "미수신"과 달리 로그로 안 잡힌다.)
- **판별**: `ros2 topic hz` 로는 불가능. **값 변화**로 재야 한다.
  ```bash
  python3 tools/sensor_update_rate.py --duration 15 --topics /paxini/right/ft /paxini/right/raw
  #  FROZEN + 전 채널 0  → writer 미실행/센서 미동작
  #  특히 /paxini/right/raw(1524ch) 까지 전부 0 이면 접촉 문제가 아니다
  pgrep -af paxini                              # 프로세스 확인
  ros2 topic info /paxini/right/ft --verbose     # 퍼블리셔가 shm_state_publisher 뿐이면 writer 없음
  ```
- **해결**: writer 실행 후 `LIVE` 확인. 경로 주의 — `quick_start.txt` 의 `~/.Dual_Arm_Hand_Ctrl` 는 오타.
  ```bash
  python3 ~/motie_ws/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r
  ```
- **교훈**: **`ros2 topic hz` 는 통신 점검용이고 센서 생존 점검에 쓸 수 없다.** 힘 관련 계측·결론
  (§B3/§F5/§F6) 전에 항상 센서가 `LIVE` 인지 먼저 확인할 것.
  절차는 [UPDATE_RATE_CHECK.md §3.5](UPDATE_RATE_CHECK.md#35-센서-실측-update-rate-값-변화-기준).

### B2. `⚠ mN 파일 없음` 경고

- **증상**: `⚠ [deploy] mN 파일 없음 → SHM raw j_kin 사용(학습과 불일치!). ...`.
- **원인**: 학습은 j_kin(FT)을 mN 보정값(`/tmp/deep_ws_raw_06_hand_j_kin_mN.txt`)으로 썼는데
  그 side-channel 파일이 없어 SHM raw 로 폴백.
- **해결**: 통합모델(방식2/3)은 `USE_JKIN=False` 라 **jkin 자체를 안 씀** → 경고를 `USE_JKIN` 일
  때만 뜨도록 조건화(노이즈 제거). jkin 쓰는 모델을 쓸 땐 side-channel 프로세스를 실행할 것.

### B3. 힘 임계 미도달 + 학습/실행 도메인 불일치 (핵심 리스크)

- **증상**: 데이터 수집 땐 파지/스퀴즈 힘 임계에 잘 도달했는데, 배포에선 계속 미도달.
  ⇒ 기존 데이터셋 기준 추론 성능 저하 우려.
- **원인 3가지**:
  1. **손바닥 방향**: 수집=아래, 배포=위 → 중력·접촉·힘 분포가 통째로 달라짐(파지도 약해짐).
  2. **속도(ROS2)**: 스퀴즈가 느려져 힘-시간 프로파일이 달라짐.
  3. **속도 → 샘플 수** (가장 직접적): 엔진은 스퀴즈 동안 매 tick `add_sample` 후
     `downsample_avg(FACTOR=10)`. 학습 100Hz → 유효 10Hz/많은 스텝, 배포 ~20Hz → 스텝 수가
     수배 짧아짐 → LSTM 입력이 OOD(심하면 `샘플 부족`).
- **로그 함정**: `[squeeze] threshold(10.0) 도달` 문구는 **무조건 출력**(실제 미도달이어도 찍힘).
- **대응**:
  - **Path A (실행을 학습에 맞춤, 기존 데이터셋 유지)**
    1. 가능하면 손바닥 **아래(palm-down)** 로 실행.
    2. 발행률을 ~100Hz 로 회복 + `HAND_SQUEEZE_DURATION`/제어주기를 학습과 일치.
       또는 **`FACTOR = round(실제rate/10)`** 로 맞춰 다운샘플 후 시퀀스(rate·길이·평균창)를 학습과 일치.
    3. 파지/스퀴즈가 학습과 같은 힘·자세에 도달하도록: grip curl 강화, **스퀴즈에 힘-도달 curl 추가**.
  - **Path B (새 조건에 모델을 맞춤, palm-up 필수일 때)**: 실행 조건(palm-up·ROS2 속도)에서
    소량 재수집 → fine-tune / 정규화 통계 재계산. 관절은 **Δjoint 모델(`m3_dO`)** 로 자세 offset 완화.
- **계측**: `deploy_ros2_exp.py` 가 스퀴즈당 **add_sample 수 · 유효 rate · downsample 스텝 · 도달 Fz**
  를 출력 → 학습 스펙과 격차를 수치로 비교(개선 우선순위 결정).

---

## C. 동작 이상 (모션 / 제어)

### C1. 손 동작이 느리고 끊김 (move-stop-move-stop)

- **증상**: 파지/스퀴즈 이동이 느리고 몇 번에 나눠 끊기듯 움직임. 심하면 파지 실패로 물체 놓침.
- **진단**:
  ```bash
  ros2 topic hz /hand/right/q_target
  # average ~17Hz, max 2.7s  ← 100Hz 목표인데 몇 Hz로 폭락 + 수초 정지
  ```
- **원인**: 명령 QoS 가 `RELIABLE + depth=1` 인데 100Hz 로 스트리밍. 수신 노드
  `hand_target_receiver` 는 **BEST_EFFORT** → RELIABLE writer 가 오지 않는 ack 를 기다리며
  `publish()` 가 `max_blocking_time`(≈100ms)까지 블로킹 → 루프가 몇 Hz로 떨어짐. (SHM 시절엔 DDS가
  없어 매끄러웠음.)
- **해결** (전부 `Ros2ShmBridge`):
  1. **q_target(hand·arm) 을 BEST_EFFORT** 스트림 QoS로 (수신자와 매칭 → 블로킹 제거).
  2. **servo/mode 는 RELIABLE 유지**(단발) 하되 **값이 바뀔 때만 발행**(100Hz 중복 제거).
- **결과**: ~100Hz 회복, 부드러운 파지/스퀴즈.

### C2. 스퀴즈 시 엄지 외 손가락이 위치를 벗어남

- **증상**: 스퀴즈 중 비-엄지 손가락이 "고정" 되지 않고 밀려 물체를 놓침. (palm-up 로 바꾼 뒤 심해짐)
- **원인**: 비-엄지 hold 기준을 **feedback 스냅샷(`shm.read()` = 관절 실측)** 으로 잡고 보간까지 함 →
  노이즈/랙/드리프트가 그대로 반영. position 명령 자체가 흔들리진 않지만 기준이 약함 + palm-up 부하.
- **해결** (`deploy.py`, 두 곳):
  1. `move_hand_to_squeeze`: `squeeze_target = list(return_position)` — **명령된 파지 자세**로 고정
     (feedback 스냅샷 대신).
  2. `move_hand_to_target_until_force`: force 판정 대상이 아닌 finger(스퀴즈의 비-엄지)를 **target 에
     즉시 pin + 매 tick 재명령**(`progress=1`, `current[j]=target[j]`).
- **결과**: 비-엄지가 파지 위치에 단단히 고정되어 스퀴즈가 안정. (grip 단계는 force_set=전체라 영향 없음)
- **잔여**: 그래도 밀리면 **하드웨어 토크 한계**(palm-up). `q_target`(명령) vs `joint_states`(실제)를
  echo 로 비교 — 명령은 고정인데 실제가 벌어지면 컨트롤러 강성/게인 문제.

### C3. 서보 켤 때 손가락이 튐

- **증상**: 시작 시 서보 on 순간 손가락이 q_tar=0 으로 점프하며 튐(파지 놓칠 위험).
- **원인**: q_target 없이 `cmd_servo=true` 를 먼저 보내면 수신 노드가 q_tar 초기값 0 을 잡음.
- **해결**: `bridge.safe_hand_servo_on()` — **현재 손 자세를 q_target 으로 1회 먼저 발행 → 정착 →
  servo on**. 두 진입점(deploy_ros2 / deploy_task3_ros2)의 `attach()` 직후 호출.

### C4. "ROS2 실행 환경이 RTOS 가 아니라서 그런가?"

- **정리**: 부분적으로만 맞다. **관절 stiff hold 는 제어 PC 의 C++ 1kHz(RT) 가 SHM 마지막 target 을
  붙잡는다** — 파이썬 쪽이 non-RT 여도 명령값만 고정이면 유지돼야 정상. 그래서:
  - **위치 유지 실패**는 파이썬 non-RT 가 직접 원인 아님(명령값/토크/통신 문제). 단, **C++ 컨트롤러
    자체를 non-RT 머신에서 돌리면** 1kHz 데드라인을 놓쳐 hold/부드러움이 다 나빠짐 → `uname -a` 로
    RT 커널/실행 위치 확인.
  - **끊김/느림**엔 non-RT `time.sleep` 지터가 "일부" 기여하지만, 실측된 초 단위 정지는 C1(QoS)이 주범.

---

## D. 토픽 / 인터페이스

### D1. 토픽 이름 구→신 규약 (`/<side>/`)

실측(`ros2 topic list`, 2026-07-06) 및 `ROS2_TOPIC_GUIDE` Appendix 기준, 구 `_r` 접미사 토픽은
더 이상 발행/구독되지 않는다. 브리지를 신 규약으로 전부 변경:

| 구 | 신 |
|---|---|
| `/hand/q_target_r` | `/hand/right/q_target` |
| `/hand/cmd_servo_r` / `cmd_mode_r` | `/hand/right/cmd_servo` / `cmd_mode` |
| `/franka/q_target_r` | `/franka/right/q_target` |
| `/franka/joint_states` / `/hand/joint_states` | `/franka/right/joint_states` / `/hand/right/joint_states` |
| `/hand/kin_r` | `/hand/right/kin` |
| `/paxini/ft_r` | `/paxini/right/ft` |

### D2. QoS 규칙

- **상태(subscribe)**: BEST_EFFORT (sensor-data). 잘못 RELIABLE 로 구독하면 콜백 안 옴.
- **명령(publish)**: 단발(servo/mode)은 RELIABLE, **고속 스트림(q_target)은 BEST_EFFORT**
  (수신자와 매칭 + write 블로킹 방지 — C1 참고).
- **촉각**: 추론은 `resultant`(합력, `/paxini/right/ft` = 4×3 합)만 사용(`USE_TACTILE=False`)
  → 127포인트 `/paxini/right/raw` 불필요. 브리지가 ft 를 point0 에 실어 sum 재구성해도 동일.

---

## E. 이번 작업에서 추가/변경한 기능 (개발 노트)

| 기능 | 내용 | 파일 |
|------|------|------|
| **Franka 고정 + 물체원위치 제거** | 팔 이동(`move_franka_to`) 및 4단계 원위치 제거 → 손만 안전→파지→스퀴즈 | `deploy.py` |
| **task3 시퀀스** | "이미 파지한 상태에서 스퀴즈만" (Pick→Inhand→**Stiffness**→Place 체인의 3번) | `deploy_task3.py`, `deploy_task3_ros2.py` |
| **시퀀스 제어권 프로토콜** | `SequenceClient(3)`: 직전 Inhand(2) DONE 대기 → 제어권 획득 → 스퀴즈 → End(3=DONE) | `deploy_task3_ros2.py` |
| **과일별 힘 임계값 config** | `fruit_thresholds.yaml` + `set_thresholds_for_fruit(fruit)` (grip/squeeze 과일별) | `deploy.py`, `fruit_thresholds.yaml` |
| **안전 서보-온** | 현재 자세 q_target 선발행 후 servo on (C3) | `deploy_ros2.py` (브리지) |
| **명령 QoS 최적화** | q_target BEST_EFFORT + servo/mode 중복발행 제거 (C1) | `deploy_ros2.py` (브리지) |
| **스퀴즈 비-엄지 고정** | 파지 명령값으로 pin (C2) | `deploy.py` |
| **인터랙티브 메뉴** | 추론 후 [1]재스퀴즈 [2]안전복귀 후 재파지→스퀴즈 [3]안전복귀 후 종료 | `deploy_ros2.py` |
| **통합 추론 엔진(방식2/3)** | 통합 정규화 + 과일 one-hot, 변위 O/X (`m2/m3 × dX/dO`) | `real_deploy_inference_final.py` |
| **계측 도구** | 스퀴즈당 샘플 수·rate·힘 출력(기존 코드 보존, 엔진 wrapper) | `deploy_ros2_exp.py` |

---

## F. [실행 계획] 토픽 수신 rate 저하 · 잔여 끊김 — 계측(deploy_ros2_exp) 후 개선

> C1(QoS 블로킹)으로 **초 단위 정지**는 없앴지만, **파이썬 루프가 목표 100Hz 를 못 채우는 잔여 문제**가
> 남아 있다. 이 절은 그 문제를 정의하고, `deploy_ros2_exp.py` 로 **실측치를 뽑아** 우선순위대로 개선하는
> 계획을 정리한다. (원인 축 → C1·C4, 추론 영향 → B3)

### F1. 문제 정의

- **증상 A — 토픽 받는 속도가 느리다**: 파이썬 모션·계측 루프가 상태 토픽(`/paxini/right/ft`,
  `/hand/right/joint_states`)을 **초당 수십 회밖에** 소비하지 못한다. 상태 QoS 가 `BEST_EFFORT · depth=1`
  이라 소비가 느리면 들어온 샘플 대부분이 **드롭/스테일** → 루프가 실제 반영하는 rate 가 발행 rate
  (paxini ~90Hz)보다 훨씬 낮다.
- **증상 B — 로봇이 끊기듯 움직인다**: setpoint(`q_target`)가 **불균일·저빈도**로 나가 팔/손이 거친 목표
  갱신을 받는다. (C1 이후 잔여분 + C4 의 non-RT `time.sleep` 지터)
- **파급(B3)**: 학습은 **100Hz · `FACTOR=10`**. 배포 루프 rate 가 낮으면 스퀴즈당 **유효 샘플·downsample
  스텝이 수배 짧아져** LSTM 입력이 OOD → 심하면 `샘플 부족 — 추론 불가`.
- **목표**: 유효 루프 rate 를 학습(100Hz)에 **정합**하거나, 최소한 `FACTOR` 로 시퀀스를 맞춰 추론이 학습과
  같은 분포를 보게 한다 + 로봇 모션을 부드럽게.

### F2. 근본 원인 (코드 근거)

모션 루프(`move_hand_to_target_until_force` / `_hold_hand_position` / `move_hand_to`, 전부
`CONTROL_RATE_HZ=100`, `time.sleep(0.01)`)의 **매 tick 작업이 과중**하다:

1. **중복 I/O**: 한 tick 에서 `paxini.read()` 를 **2번**(`normal_forces()` + `engine.add_sample()`),
   `shm.read()` 도 1~2번. `resultant_from_tactile`(4×127×3 합) 등 numpy 재구성까지 매 tick.
2. **GIL 경합**: ROS executor 를 **별도 데몬 스레드**에서 spin. 파이썬 GIL 때문에 구독 콜백과 모션 루프가
   **병렬이 아니라 직렬화** → 소비·발행 둘 다 느려짐.
3. **`time.sleep(0.01)` 누적**: 고정-rate 스케줄러가 아니라 **작업시간 위에 sleep 을 더하는** 구조 →
   실제 주기 = (작업 + 0.01)초 > 10ms → rate < 100Hz. non-RT 지터도 가세(C4).
4. **depth=1 드롭**: 상태 토픽은 최신 1개만 유지 → 느린 소비자는 나머지를 버림(스테일 read).

> C1 은 이 중 "RELIABLE ack 대기로 인한 **초 단위 블로킹**"만 없앴다. 위 1~4 는 그 이후에도 남는
> **구조적 rate 상한**이다 — 그래서 지금 `deploy_ros2_exp` 로 실측해 개선한다.

### F3. 계측 — `deploy_ros2_exp.py` 로 실측치 뽑기

> **실행 절차·결과 기록은 [UPDATE_RATE_CHECK.md](UPDATE_RATE_CHECK.md) 로 분리했다** —
> `./tools/measure_update_rate.sh --label <이름>` 한 줄로 아래 계측 + `ros2 topic hz` 교차확인이
> 함께 `docs/rate_log/<run>/` 에 저장되고 `summary.md`(표·자동판정)가 생성된다. 아래는 그 원리.

기존 코드를 건드리지 않고 엔진만 감싼 `MeasureEngine` 이 **스퀴즈 1회마다** 아래를 출력한다:

```bash
source env.sh
python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py   # 과일 선택 → 파지 → 스퀴즈(→계측)
```
```
[measure] 스퀴즈 계측 (학습 대비 격차 확인용)
  add_sample 호출 = N,  valid(적재) = V
  유효 rate       = R Hz   (수집시간 D s)         ← 루프가 실제로 도는 rate (핵심 지표)
  downsample 스텝 = V//FACTOR   (FACTOR=10, MIN_LEN=10)   ← 학습 대비 시퀀스 길이
  finger별 최대 Fz = [...]   thumb 최대 = F N (스퀴즈 임계 T N)   ← 힘 도달 여부
```

동시에 와이어에서도 교차 확인(원인 분리):
```bash
ros2 topic hz /hand/right/q_target      # setpoint 발행 rate (끊김 진단, 목표 ~100Hz)
ros2 topic hz /paxini/right/ft          # 힘 소스 rate (~90Hz)
ros2 topic hz /hand/right/joint_states  # 상태 소스 rate
```

**비교 기준(학습 스펙)**: rate ≈ 100Hz, `FACTOR=10`, 스텝 = 유효샘플//10 ≥ `MIN_LEN(10)`.
→ 과일 4종 × 각 몇 회씩 돌려 **평균 rate·스텝·Fz** 를 표로 남긴다(**개선 전 baseline**).

### F4. 개선 계획 (우선순위 · 각 단계 후 exp 로 재측정)

| 순 | 개선 | 무엇을 | 기대효과 | 검증(exp) |
|---|------|--------|----------|-----------|
| **0** | baseline 계측 | 과일별 rate·스텝·Fz 기록 (exp `#3`) | 학습과의 격차를 수치화 | F3 |
| **1** | **FACTOR 정합** (exp `#1`) | `FACTOR = round(실측 rate / 10)` 로 다운샘플을 학습(100Hz)과 맞춤 | `샘플 부족`/OOD 직접 완화, **코드 최소·즉효** | 스텝 ≥ MIN_LEN, 학습 길이와 근접 |
| **2** | **루프 경량화** (exp `#4`) | tick 당 `paxini.read()` 1회로 캐시해 공유(`add_sample_arrays` 경로), 미사용 구독(`/hand/right/kin` 등 `USE_JKIN=False` 시) 제거 | 유효 rate ↑, 끊김 완화 | rate 상승 재측정 |
| **3** | **힘-도달 curl** (exp `#2`) | 스퀴즈에 `grip_curl` 처럼 "임계 도달까지 추가 curl" 추가 | thumb Fz 가 임계 도달 → 힘 도메인 학습 정합(B3) | Fz ≥ 임계, 스퀴즈 길이 일관 |
| **4** | (rate 여전히 낮으면) 구조 개선 / Path B | 발행 스트림과 무거운 추론 샘플링 분리, 또는 palm-up·ROS2 조건에서 소량 재수집→fine-tune(B3 Path B) | 근본 rate 확보 or 조건 재정합 | 목표 rate 도달 or 재학습 성능 |

**완료 기준**: (a) 유효 rate·downsample 스텝이 학습 스펙에 근접, (b) thumb Fz 가 임계 도달,
(c) `ros2 topic hz /hand/right/q_target` 가 목표 근처로 안정, (d) 육안상 파지/스퀴즈가 부드럽다.

> 원칙: **한 번에 하나씩** 바꾸고 매번 `deploy_ros2_exp` 로 같은 지표를 재측정해 효과를 격리한다.
> 기존 `deploy_ros2.py` 는 보존하고 실험은 `_exp` 에서.

### F5. 계측 결과 (2026-07-14, `deploy_ros2_exp` baseline) → 계획 재편

과일 4종 × 3회 실측(`docs/result_log/*.txt`). **가정과 결과가 정반대 — 루프 rate 는 이미 정상, 진짜
문제는 힘(force) 도메인.** (추론 강성/등급은 *최종 모델이 아니라* 판단 제외; 아래는 모델-무관 계측치.)

| 과일 | 유효 rate | steps | thumb Fz 최대 | 스퀴즈 임계 | 도달률 |
|---|---|---|---|---|---|
| plum | 94–97Hz | 15 | 2.8–3.3N | 8.0N | ~38% |
| kiwi | 95–97Hz | 15 | 3.3–4.5N | 13.0N | ~30% |
| tomato | 83–97Hz | 15 | 3.2–3.7N | 10.0N | ~35% |
| lemon | 94–98Hz | 15 | 3.6–5.2N | 12.0N | ~37% |

**판정**
- ✅ **루프 rate 정상 (~95Hz ≈ 학습 100Hz)** → 이전 "~20Hz 병목" 가정 **반증**. steps=15 ≥ MIN_LEN(10)
  → 샘플 부족·FACTOR 문제 없음 → **F4 `#4`(루프 경량화)·`#1`(FACTOR 정합) 불필요.**
- ❌ **힘 임계 전 과일·전 회차 미도달** (thumb ~3–5N vs 8–13N, 파지 임계 5–10N 도 미도달). `grip_curl`
  이 켜져 있는데도 미도달 → **현재 조건(palm-up 등)의 힘 상한이 ~5N** 로 보임. 스퀴즈가 힘-정지가 아니라
  **위치-정지**로 끝나 **힘-시간 프로파일이 학습과 불일치(OOD)** → **이게 유일한 실질 문제.**

**재편된 우선순위 (rate → force)**
| 순 | 개선 | 무엇을 | 검증(exp) |
|---|------|--------|-----------|
| 1 | **힘 상한 규명** | (a) 가능하면 **palm-down** 자세로 재측정(중력 도움) (b) 스퀴즈에 **힘-도달 curl**(#2) 추가로 thumb 를 임계까지 더 닫음 | thumb Fz 오르나 / 임계 근접하나 |
| 2 | 상한이 낮으면(도달 불가) → **Path B** | 현재 조건(palm-up·~5N·~95Hz)에서 소량 재수집 → fine-tune·정규화 재계산, **Δjoint 모델(m3_dO)** 로 포즈 offset 완화. 또는 임계를 **달성 가능한 값**으로 낮추고 그 힘수준 데이터로 정합 | 과일 구분되는 강성 분포 |
| 3 | **최종 모델 적용 후 재평가** | (현재는 최종 모델 아님) 최종 모델이 **현재 조건 학습본인지** 확인 후 적용 → 동일 exp 재측정 | 도메인 불일치 해소 여부 |

> **교훈**: 계측 전 "루프가 느리다(~20Hz)"는 가정이 **틀렸다**. 실측하니 rate 는 정상, 병목은 **힘**이었다
> — 개선 노력을 루프/FACTOR 가 아니라 **힘 도메인·모델 정합**에 써야 한다. (측정 먼저, 최적화 나중.)

### F6. palm-down 대조 실험 (2026-07-14, `docs/result_log_palm_down/`) → palm 가설 기각

기존 코드로 **자세만 palm-down 으로** 바꿔 재측정. **결과: 자세는 힘에 영향 없음.**

| 과일 | 스퀴즈 임계 | palm-UP thumb Fz | palm-DOWN thumb Fz |
|---|---|---|---|
| plum | 8N | 2.8–3.3N | 2.6–4.0N |
| kiwi | 13N | 3.3–4.5N | 3.2–3.5N |
| tomato | 10N | 3.2–3.7N | 2.8–3.8N |
| lemon | 12N | 3.6–5.2N | 3.9–4.1N |

- **palm-down ≈ palm-up (~3–4N), 여전히 임계의 30–40%.** → **C2/B3 의 "palm 방향이 힘 저하 원인" 가설 기각.**
  힘 상한 ~4–5N 은 **자세 무관** ⇒ **palm-up 유지에 힘 손해 없음.** (rate·steps·수집시간도 palm-up 과 동일)

**핵심 미해결 질문 — "학습/수집 데이터의 실제 힘은?"**
- `[squeeze] threshold 도달` 로그는 **무조건 출력**(B3 로그 함정). 즉:
  - **(A) 수집도 ~4N 이었다** → 배포 ~4N 은 **in-domain → 힘은 문제 아님** → 개선 대상은 **최종 모델**.
  - **(B) 수집은 실제 8–13N 도달** → **진짜 도메인 갭** → 힘 curl 또는 Path B.
- → **로그가 아니라 데이터로** 수집셋의 실제 Fz(resultant) 분포를 확인해야 (A)/(B) 가 갈린다.

**재편 계획 (분기형)**
| 순 | 확인/개선 | 분기 |
|---|---|---|
| 1 | **학습 데이터의 실제 스퀴즈 Fz 확인** (수집셋 resultant, 또는 학습 정규화 통계) | ~4N → (A) 힘 무문제 → 3 으로 / 8–13N → (B) 갭 확정 → 2 |
| 2 | (B면) **힘-도달 curl** `deploy_ros2_exp_forcecurl.py` 로 Fz 오르나 | 상승 → 코드로 도달 / ~4–5N 정체 → 물리 상한 → **Path B**(현 조건 재학습·정규화 재계산) |
| 3 | **최종 모델(전체 데이터 + 과일 one-hot) 적용 후** 동일 exp 재측정 | 강성이 과일별로 구분·정합되는지 |

> palm-down 실험이 준 값: **자세는 변수 아님.** 남은 원인은 "① 스퀴즈가 힘-정지 아닌 위치-정지(~4N 에서 멈춤)"
> 또는 "② 학습도 원래 ~4N 저력" 둘 중 하나 — **1번(학습 힘 확인)이 분기점**이다.

### F7. 힘 저하의 원인 = 코드/워크스페이스 변화 (P2P → ROS2) — **H1(측정) 확정 · 조치 완료**

사용자 관찰(**하드웨어 불변 + 수집 때 스퀴즈 임계 도달 + palm 무관 ~4N**) → 힘 저하는 물리가 아니라
**P2P 수집 → ROS2 배포로 바뀐 소프트웨어 경로** 문제. 단, 두 메커니즘 중 어느 쪽인지는 **1회 실측으로 갈린다.**

**공통 사실 (코드 확인)**: 추론 힘 = `resultant_from_tactile = nan_to_num(tac).sum(axis=1)` — **합 연산은
양쪽 동일**. 차이는 *무엇을 합하느냐*:
- **학습/수집**: 진짜 127점 tactile 합 `Σ127` (수집코드와 동일).
- **ROS2 배포**: 브리지가 `/paxini/*/ft`(4×3)를 point0 에 실은 `[ft,0,…,0]` 을 합 → **= ft**.
- ⇒ 두 값이 같을 조건은 **`ft == Σ127` 하나뿐.**

**⚠ 정정**: 앞서 "ft ≠ Σ127 이라 이게 근본원인"이라고 단정했으나, **코드는 등가/비등가를 증명하지 못한다.**
`decoded_to_numpy_arrays` 는 ft·tactile 을 **다른 오프셋·다른 scale**(`ft_scale` vs `tactile_scale`)로 따로
파싱한다. 주석상 둘 다 "N"이지만 **`ft == Σ127` 은 D2 의 (검증 안 된) 가정**이었다. → **실측으로 판별 완료(아래 ✅: H1 확정).**

**결정적 테스트 (스퀴즈 중, 같은 프레임에서 thumb 의 ft vs Σ127 비교)**:
```bash
python3 tools/paxini_writer.py --hand r --print-period 0.1   # 출력 = Σ127 (손가락별)
ros2 topic echo /paxini/right/ft                              # ft (deploy 가 쓰는 값)
```

| 실측 결과 | 가설 | 원인 | 수정 |
|---|---|---|---|
| **ft ≪ Σ127** (예 4N vs 10N) | **H1 (측정/출처)** | 배포가 학습과 다른(작은) 양(ft)을 먹임. 물리 힘은 정상, 읽는 값만 틀림 | **브리지를 `/paxini/*/ft` → `/paxini/*/raw`(4×127×3) 구독**으로 → `resultant_from_tactile` 이 학습과 동일한 Σ127 계산. **재학습·힘·자세 불필요.** (대역폭 1524 float×90Hz ≈ 무시) |
| **ft ≈ Σ127** (둘 다 ~4N) | **H2 (모션/실제 힘)** | 읽기값은 정상 → 배포가 **실제로 ~4N만 누름**. 수집(P2P)은 8–13N 눌렀는데 ROS2 명령 경로가 약함 | 모션/제어 경로 점검: q_target(명령) vs joint_states(실제) 대조, 수신 노드 servo 모드/게인, 스퀴즈 curl(→ `deploy_ros2_exp_forcecurl.py`) |

**✅ 실측 결과 (확정 · `--paxini raw` 3채널) → H1 지지**

같은 접촉에서 thumb 최대 Fz 를 두 소스로 비교했다:

| 힘 소스 | thumb Fz 최대 | 스퀴즈 임계 | 도달률 |
|---|---|---|---|
| `/paxini/right/ft` (4×3) | 4.5 ~ 5.3 N | 10 N | 45 ~ 52% |
| **Σ127 (`--paxini raw`)** | **11.40 / 9.90 N** | 14.81 / 11.43 N | **77% / 87%** |

→ 같은 접촉에서 **Σ127 이 ft 의 2~3배** ⇒ **H1(측정 경로) 확정.** 배포가 학습보다 **작은 양(ft)** 을 먹고
있었고, 브리지를 **`/paxini/*/raw` 구독**으로 바꿔 `resultant_from_tactile` 이 학습과 동일한 Σ127 을 계산하도록
하여 해소했다. `resultant = Σ127` 은 A↔B **parity 오차 0** 으로도 검증됨. ⇒ **`--paxini raw` 가 표준 경로.**

> **남은 과제(별도)**: H1 은 확정·조치됐으나 Σ127 도달률이 아직 **77~87%(임계 미달)** 다. 읽는 값이 아니라
> **힘 크기 자체**를 올리는 것은 모션/curl 문제로 별도 과제(→ `deploy_ros2_exp_forcecurl.py`, H2 계열)로 남긴다.

> **정합**: palm-up=palm-down(~4N) → H1 이면 "읽는 값이 자세 무관", H2 이면 "실제 힘이 자세 무관"(모션 한계) — 둘 다 설명 가능했고, **ft vs Σ127 실측이 유일한 판별자였다 → 실측 결과 H1 로 확정.**
>
> **교훈 2**: 주석에 "동등"이라 적힌 최적화(ft ≡ Σ127)를 **검증 없이 신뢰**하지 말 것. 힘이 이상하면
> **모델 입력의 출처부터** 학습과 1:1 대조 + 실측 대조.

---

## 부록 — 자주 쓰는 확인 명령

```bash
# 환경 (모든 터미널/PC 동일)
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0

# 상태 토픽 수신 확인 (이 패키지가 구독하는 것)
ros2 topic hz /paxini/right/ft            # ~90Hz (없으면 B1)
ros2 topic echo --once /hand/right/joint_states

# 명령 발행률 (C1 진단)
ros2 topic hz /hand/right/q_target        # ~100Hz 여야 정상, 낮으면 QoS 블로킹

# 수신자 QoS (C1 해결 방향 결정)
ros2 topic info /hand/right/q_target --verbose   # Subscriber Reliability 확인

# 시퀀스 상태 (task3)
ros2 topic echo /sequence_state --qos-durability transient_local --qos-reliability reliable
```
