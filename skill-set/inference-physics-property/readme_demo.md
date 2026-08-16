# 시나리오1 Task3 실행방법

실제 실행 명령어 부분은 2, 3번에 정리되어 있다.
**다른 PC 로 이 모듈만 떼어가서 실행**하려면 → **[§6 다른 PC 로 이식해서 실행하기](#6-다른-pc-로-이식해서-실행하기-배포-패키지)** (zip 배포판).

| | 기존 과일별 학습모델용 (`deploy_ros2` / `deploy_task3_ros2`) | **현 ecoflex 학습모델용** (`*_demo.py`) |
|---|---|---|
| 힘 임계 | `fruit_thresholds.yaml` 과일별 로드 | **고정** 파지 6.0 N · 스퀴즈 11.0 N (`collect_ros2_new` 규약) |
| 촉각 입력 | `/paxini/right/ft` (4×3 합력) | **`/paxini/right/raw` (4×127×3)** — 127점 분포 필수 |
| 모델 | `real_deploy_inference_final` 강성 등급 | **`models/ecoflex2fruit/Champ_repair_s42.pth`** (67ch, 3속성) |
| 결과 | 강성값 + 등급 | **무게(g) · 크기(mm) · 강성** 3속성 |
| 결과 GUI | `gui/stiffness_gui.py` | **`gui/property_gui.py`** (`/property/result`) |

---

## 1. 구성 파일

### 1.1 주요 코드 — 이 세 개가 데모의 전부!

| 파일 | 역할 |
|---|---|
| [`launch/deploy_ros2_demo.py`](stiffness_deploy_ros2/launch/deploy_ros2_demo.py) |  **(A) 단독/테스트용** — 안전→파지→스퀴즈→추론, 이후 터미널에서 입력 받아 반복여부 결정 |
| [`launch/deploy_task3_ros2_demo.py`](stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py) | **(B) 데모용** — 이미 파지한 상태에서 스퀴즈 1회. topic으로 in-hand에서 2를 받아야 실행. 완료시 3 출력. |
| [`gui/property_gui.py`](stiffness_deploy_ros2/gui/property_gui.py) | **결과 GUI** — `/property/result` 구독, SIZE / STIFFNESS / WEIGHT 를 큰 숫자로 표시 + 강성 LOW·MID·HIGH 배지 + 최근 15개 리스트. tkinter 단독 프로세스라 제어 루프에 영향 없음 |

> **팔(Franka)은 움직이지 않고 손만** 동작한다.
> High/Mid/Low 범위는 `property_gui.py`에서 결정한다.

### 1.2 두 진입점이 끌어다 쓰는 모듈

| 파일 | 역할 |
|---|---|
| `launch/ecoflex_engine.py` | ★ **추론 엔진** — ckpt 로드, 학습 전처리 복제(채널 조립·avgpool 32스텝·포화 repair·정규화), `capture_baseline()` / `add_sample()` / `infer()` |
| `launch/eco_model.py` | 모델 클래스 `MultiTargetRegressorAux` — deep_ws **vendored 사본** |
| `launch/eco_paxini_features.py` | 파생 채널 계산(`resul_curved` · `contact`) — **vendored 사본** |
| `launch/property_result_pub.py` | `/property/result` 발행 + `property_gui.py` **자동 spawn**. `infer()` dict → GUI JSON 필드 매핑(`stif`→`stiffness`, `mass`→`weight`, `size`→`diameter`) |
| `launch/deploy_ros2.py` | ROS2 브리지 `Ros2ShmBridge` (상태 구독 · 명령 발행) — 그대로 재사용 |
| `launch/deploy_ros2_exp_rawft.py` | `Ros2RawPaxiniBridge` — `/paxini/right/raw`(4×127×3) 구독. 힘 판정과 엔진 입력을 이 하나가 담당 |
| `launch/deploy.py` (A) / `launch/deploy_task3.py` (B) | 모션 시퀀스·힘 판정 헬퍼 |
| `launch/*.txt` | 파지 포즈 파일(tomato·lemon·kiwi·plum·ecoflex·pose1~5) + `initial_pose.txt` |

> vendored 사본 덕분에 **deep_ws 없이 자립 실행**된다. deep_ws 원본이 바뀌면 사본을 갱신하고
> [`launch/test_ecoflex_engine_offline.py`](stiffness_deploy_ros2/launch/test_ecoflex_engine_offline.py)

---

## 2. 준비 (제어 PC)

```bash
shm ; rs ; nd                                                    # Ctrl 스택 + shm_state_publisher
python3 ~/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r      # PaXini writer
```

- **`/paxini/right/raw` 발행이 필수**다. 없으면 힘=0 으로 진행되어(경고 후 계속) 추론이 무의미해진다.
  `ft` 만 있고 `raw` 가 없으면 데모는 쓸 수 없다.
- `deploy_task3_ros2_demo` 만 추가로 `sequence_arbiter` 실행 + 제어 PC 가 `require_control:=true` 로 launch 돼 있어야 한다.

```bash
# 발행 확인 — raw 가 와야 데모가 정상 동작 (env.sh 는 §3.2)
source env.sh
ros2 topic hz /paxini/right/raw
```

---

## 3. 실행 (산업부 PC)

데모 기준으로는 3.4(B)만 실행하면 된다. 3.3(A)는 단독 실행용.

### 3.1 설치 (최초 1회)

> 이미 설치돼 있으면 **3.2 로 넘어가도 된다.**

```bash
cd <통합_ws>
rosdep install --from-paths src/stiffness_deploy_ros2 --ignore-src -r -y
pip install -r src/stiffness_deploy_ros2/requirements.txt        # torch 등
```


### 3.2 환경 (`env.sh`)

```bash
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh
```

`env.sh` 는 conda 를 PATH 에서 제거하고 시스템 python3(`/usr/bin/python3`) + ROS2 Humble 을 잡은 뒤
`ROS_DOMAIN_ID=9` · `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` · `ROS_LOCALHOST_ONLY=0` 을 설정한다.


### 3.3 (A) `deploy_ros2_demo.py` — 파지부터 직접, 반복 스퀴즈

task 별(pick-inhand-inference-place) 연결 없이 단일 실행을 하는 경우에 사용한다.


```bash
[터미널1] - moveit 실행
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh
source ~/isaac_ws/dex_soldering/dex_ros/isaac-ros/kistar_ws/install/setup.bash
ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py joint_state_mode:=direct
```

```bash
[터미널2] - 파지-스퀴즈-모델추론 실행
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh
python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py
python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py --no-gui   # GUI 없이(개발용)
```

표준 메시지만 쓰므로 `env.sh` 하나로 충분하다(Dual_Arm 워크스페이스 source 불필요).

**진행 순서** — 포즈 번호 입력 → 안전 위치 → 파지(close 2.5 s, 6.0 N 도달까지)
→ **안정화 0.8 s = 엔진 baseline(학습 ①pre_wait 등가)** → 스퀴즈(11.0 N) → 3속성 추론 → 다음 동작 메뉴.

메뉴는 `[1] 다시 스퀴즈` · `[2] 안전복귀 후 재파지→스퀴즈` · `[3] 안전복귀 후 종료`.

포즈는 과일이 아니라 **포즈 파일** 선택이다(임계값은 과일과 무관하게 고정):

```
실제 학습된 포즈는 pose1~pose5 이며, pose1 무난하다.

포즈 선택  [1] tomato  [2] lemon  [3] kiwi  [4] plum  [5] ecoflex
           [6] pose1  [7] pose2  [8] pose3  [9] pose4  [10] pose5 :
```

정상 출력:

```
[property_gui] 결과 GUI 실행 (pid=...).
[rawft] 힘 출처 = /paxini/right/raw (4×127×3, 진짜 Σ127) — point0 트릭 없음
[pose] 과일 포즈 적용: tomato.txt -> GRIP=[...]
[deploy_ros2_demo] 임계값 고정 — 파지 6.0 N · 스퀴즈 11.0 N (collect_ros2_new 와 동일)
[eco-engine] Champ_repair_s42.pth 로드 — 95,511 파라미터 · device=cuda · 채널 67 [...]
[eco-engine] baseline 80프레임 확보 (①pre_wait 등가)
================================================
  [추론 결과] ecoflex2fruit 3속성
    무게(mass) = 128.4 g
    크기(size) = 61.2 mm
    강성(stif) = 3.417
    사용 프레임 = 214
================================================
```

---

### 3.4 (B) `deploy_task3_ros2_demo.py` — 이미 파지한 상태에서 스퀴즈만 (시퀀스 체인)

task 별(pick-inhand-inference-place)로 연결 하는 경우에 사용한다.

```bash
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh
source ~/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash   # ★ dual_arm_msgs + sequence_client
python3 stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py
```

`run_task3_gui.sh` 와 동일한 두 줄 source 규약이며, 워크스페이스 경로가 다르면 그 경로로 바꾼다.
**실행 전에 물체를 이미 파지**하고 있어야 한다.

**진행 순서** — 과일 번호 입력(**포즈 파일 선택 용도로만 유지**) → Inhand(#2) DONE 대기
→ 제어권 획득 → 파지 확인 hold(1 s, 이 구간이 **baseline**) → 스퀴즈(11.0 N) → 3속성 추론
→ End(제어권 반납). **1회 실행 후 종료**한다.

```
[sequence] 직전 Inhand(#2) DONE 대기...
[sequence] 제어권 획득 → Stiffness(#3) 시작
=================== 1. 파지 상태 확인 ===================
=================== 2. 스퀴즈 모션 ===================
  [추론 결과] ecoflex2fruit 3속성 ...
스퀴즈 시퀀스 완료 — 제어권 반납, 다음 시퀀스(Place)로 이어받음.
```
---

## 4. model, label 관련
### 4.1 `models/` — 체크포인트

```
stiffness_deploy_ros2/models/
├── ecoflex2fruit/              ← ★ 데모가 쓰는 폴더
│   ├── Champ_repair_s42.pth       champ — 현재 사용중!! 데모 고정 사용
│   ├── Anchor_s42.pth             anchor — 후보1, 앵커 강성 추가 출력
│   ├── RC_v2_5_s42.pth            rc — 후보2, resultant(Σ127) 입력 대표
│   ├── gru_anchor_s42.pth         gru — 후보3, 과일 rank 전이 대표
│   ├── sensors.json               패드 기하 — resul_curved·contact 계산에 필수
│   └── README.md                  후보 4종의 출처·역할 판정표
└── *.pth                       기존 배포(강성 등급)용 — 데모는 사용하지 않음
```

> ⚠ **현재 코드 기준 실제 사용 모델은 `gru_anchor_s42.pth`** 다.
> 두 데모 진입점 모두 `EcoflexPropertyEngine(variant="gru")` 로 호출한다
> (`deploy_ros2_demo.py:197`, `deploy_task3_ros2_demo.py:168`). champ 로 되돌리려면
> 그 두 줄을 `variant="champ"` 로 고친다 — 코드 주석의 "Champ 고정" 표기는 옛 설정이다.

- 정규화 통계(`seq_mean`/`seq_std`/`target_norm`)는 **`.pth` 안에 들어 있다** → 별도 통계 파일 불필요.
  이 두 파일에서는 **무시된다.** 다른 후보로 바꾸려면 각 파일의 `variant=` 인자만 고친다
  (`champ` · `anchor` · `rc` · `gru`, 매핑은 `ecoflex_engine.py` 의 `MODEL_FILES`).
- 경로는 `Path(__file__)` 기준이라 **장비별 절대경로 수정이 필요 없다.**

### 4.2 `labels/` — 라벨 2종 (기존 등급 라벨 + 데모용 ecoflex 개체 라벨)

```
stiffness_deploy_ros2/labels/
├── general/{class,name,stiffness}.yaml    ← 기존 배포용 LABEL_DIR 기본값 (데모는 읽지 않음)
├── trial2/{class,name,stiffness}.yaml     ← 기존 배포용 (데모는 읽지 않음)
└── object_labels_oldstif/                 ← ★ 데모가 읽는 폴더 — ecoflex 18개체 실제값
    ├── mass.yaml                             {normalize: {min,max}, utils: {dict: {개체번호: g}}}
    ├── size.yaml                             (동일 구조, mm)
    └── stif.yaml                             (동일 구조 — ⚠ 반드시 oldstif 판)
```

**기존 등급 라벨**(`general/`·`trial2/`)은 옛 과일용 `StiffnessInferenceEngine` 이 강성 등급 경계와
클래스명(soft/mid/hard)을 읽던 곳이다. 데모 엔진은 등급을 내지 않으므로 쓰지 않는다 —
그래도 **지우면 안 된다**(기존 진입점 `deploy_ros2.py` · `deploy_task3_ros2.py` 가 여전히 사용).

**`object_labels_oldstif/`** 는 deep_ws 학습 라벨(`data/ecoflex_new/object_labels_oldstif`)의
repo 내 사본이며, 데모에서 다음처럼 적용된다:

| 시점 | 무엇을 하나 | 코드 |
|---|---|---|
| 시작 시 | 3개 yaml 로드 → 18개체 실제값 표 + normalize 범위 | 두 데모 상단 `LABEL_DIR` → `load_labels()` (`ecoflex_engine.py`) |
| 시작 시 | **정합 가드** — 라벨 normalize 범위 ↔ ckpt `target_norm` 대조. 다른 판(예: 새 stif 라벨)이면 `⚠ oldstif 판이 맞는지 확인!` 경고 | `check_label_norm()` |
| 추론 직후 | **실제값 대조** — 추론 3속성과 가장 가까운 개체 상위 3개를 실제값·Δ와 함께 터미널 출력 | `nearest_specimens()` |

```
  [추론 결과] ecoflex2fruit 3속성
    무게(mass) = 162.1 g ...
  [실제값 대조] 최근접 ecoflex 개체 (labels/object_labels_oldstif)
    1위 ecoflex_18 — mass 160.8 g (Δ  1.3) · size 65.97 mm (Δ1.05) · stif  1.670 (Δ0.143)   [정규화 거리 0.090]
    2위 ecoflex_17 — ...
```

- 최근접 판정 거리 = 3속성을 **학습과 같은 방식**(min–max, stif 는 log₁₀ 공간)으로 0~1
  정규화한 뒤의 L2 — 축 스케일 차이 없이 대등 비교한다.
- **추론값 자체에는 영향이 없다.** 정규화·물리 단위 복원은 전부 ckpt 내장
  `stats`/`target_norm` 으로 하고, 라벨은 "이 값이 어느 개체에 가까운가" 표시 전용이다.
  라벨 로드에 실패해도 경고 후 대조 없이 추론은 그대로 진행된다.
- ⚠ **oldstif 판이어야 하는 이유**: 챔피언 학습·정규화가 이 라벨 기준이다. 다른 판을 넣으면
  대조표가 조용히 틀어지는데, 위의 정합 가드가 시작 시점에 이를 잡아준다
  (champ ckpt 기준 일치값: mass 90.0~162.3 · size 54.54~67.2 · stif 0.67~13.82).

**활용 방법** — GUI 에는 안 나가고 터미널에만 찍히므로 시연에 영향 없이 이렇게 쓸 수 있다:

| 활용 | 방법 |
|---|---|
| ① 실물 일반화 검증 | **알고 쥔 개체**(예: ecoflex_12)를 스퀴즈 → 1위가 그 개체인지 확인. 여러 개체를 반복하면 top1/top3 적중률이 나와 오프라인 평가(deep_ws README3 R7 랭킹)의 **실물판 대조표**가 된다 |
| ② 모델 후보 비교 | 같은 개체·같은 파지에서 `variant=` 만 champ→anchor/rc/gru 로 바꿔 재실행 → 4종의 Δ(추론−실제)를 같은 조건에서 비교. 실물에서 어느 후보가 나은지 즉석 판정 |
| ③ 이상 감지 | 평소 잘 맞던 개체가 갑자기 정규화 거리 큰 값(예: >0.5)으로 벗어나면 센서 드리프트·파지 불량·baseline 오염 신호 — 결과를 믿기 전에 재파지/paxini 수신 상태부터 확인 |
| ④ 미지 물체 감 잡기 | 과일 등 라벨 없는 물체도 "어느 ecoflex 개체에 가까운가"로 대략의 무게·크기·강성 대역을 읽는다 |

숫자 읽는 법: `Δ` = 축별 \|추론−실제\| (물리 단위), `[정규화 거리]` = 3축 종합 근접도
(0 에 가까울수록 근접 — **0.1 안팎이면 사실상 그 개체**로 봐도 된다).


---

## 5. 추가설명

### 5.1 시퀀스 제어권 프로토콜 — (B) 전용 (`deploy_task3_ros2.py` 와 완전히 동일)

arbiter 서비스 호출로 제어권을 주고받고, 상태는 arbiter 가 latched `/sequence_state` 에 게시한다.

| 단계 | 코드 | 실제 동작 |
|---|---|---|
| ① 대기 | `wait_for_previous_done(SEQ_INHAND)` | latched `/sequence_state` 에서 `{seq_id=2, DONE}` 대기 |
| ② 실행 | `with client:` 진입 | `/sequence/request_control` 승인 → arbiter 가 `{3, RUNNING, owner=3}` 게시 + 하트비트 시작 |
| ③ 완료 | `with` 정상 탈출 | `/sequence/release_control` → **arbiter 가 `{3, DONE}` 게시** → Place(#4) 가 이어받음 |

- `/sequence_state` 는 **transient_local(latched)** 이라 이미 지나간 DONE 도 받는다 → 대기 없이 즉시 시작될 수 있다.
- `{2, IDLE}`(직전이 하트비트 타임아웃으로 회수된 실패 상태)이면 `PreviousAborted` 예외로 중단한다 — 잘못 이어받지 않도록.
- 동작 중 예외로 빠져나가면 `end()` 대신 `abort()` → **DONE 없이** 하트비트만 정지, 3초 후 arbiter 가 자동 회수(IDLE).
- 추론 실패(`res is None`)는 예외가 아니므로 **DONE 은 정상적으로 나가고** GUI 에만 에러가 표시된다.

---

### 5.2 임계값 규약

두 데모 모두 `collect_ros2_new.py` 수집 규약에 맞춰 **고정값**을 쓴다 — 과일별 `fruit_thresholds.yaml`
로드를 하지 않는다.

`SQUEEZE_DELTA_N`은 학습 데이터 상에서는 3~5 [N] 내의 범위에서 랜덤화.

| 상수 | 값 | 쓰이는 곳 |
|---|---|---|
| `GRIP_FORCE_THRESHOLD` | 6.0 N | (A) 파지 종료 판정 / **(B) 에서는 미사용** — 스퀴즈 임계 계산의 기준값일 뿐 |
| `SQUEEZE_DELTA_N` | 5.0 N | 추가 압축량 |
| `SQUEEZE_FORCE_THRESHOLD` | **11.0 N** | (A)(B) 공통 — 스퀴즈 정지 판정 |

판정 대상은 `/paxini/right/raw` 의 thumb 합력(진짜 Σ127)이다.

> (B)는 이미 파지한 상태에서 시작하므로(파지는 직전 Inhand 담당) 6.0 N 이 실제 파지 동작을 좌우하지 않는다.
> 값을 바꾸려면 **두 파일 각각**의 상단 상수를 고쳐야 한다 — 상수를 공유하지 않는다.

---

### 5.3 결과 GUI (`gui/property_gui.py`)

두 데모 모두 시작 시 GUI 를 **별도 프로세스로 자동 실행**한다. 실행이 실패해도 배포는 계속되고,
배포가 끝나도 창은 남는다.

```bash
python3 .../deploy_ros2_demo.py --no-gui                    # 자동 실행 끄기
python3 stiffness_deploy_ros2/gui/property_gui.py           # GUI 만 따로 (ROS 연결)
python3 stiffness_deploy_ros2/gui/property_gui.py --demo    # 로봇 없이 레이아웃만 확인
```

- **토픽** `/property/result` (`std_msgs/String`, JSON, **transient_local latched**) — GUI 를 나중에 띄워도 마지막 결과를 받는다.
- **표시** SIZE(mm) · STIFFNESS(N/mm, LOW·MID·HIGH 배지) · WEIGHT(g) + 최근 15개 리스트(강성 내림차순).
- **상태** READY / MEASURING / DONE / error — `phase` 필드로 전환된다.
- **사진** `assets/<sample>.png|jpg` → `assets/sample.png|jpg` 순으로 찾는다. 데모는 샘플명을
  `sample_1`, `sample_2` … 로 자동 채번하므로, 사진을 띄우려면 `gui/assets/sample.png` 를 두면 된다
  (없으면 플레이스홀더 표시 — 동작에는 지장 없음).

---

### 5.4 토픽 인터페이스 (데모 기준, side=right)

**명령 (데모 → Ctrl, publish)**

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/hand/right/q_target` | Float32MultiArray[16] | 손 관절 목표 |
| `/hand/right/cmd_servo` | Bool | 서보 on/off |
| `/hand/right/cmd_mode` | Int32 | 핸드 모드(0=volt, 1=pos, 2=cur) |
| `/franka/right/q_target` | Float64MultiArray[7] | 브리지가 보유하나 **데모는 팔을 움직이지 않음** |

**상태 (Ctrl → 데모, subscribe)**

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/hand/right/joint_states` | JointState | 손 q(count) — `joint_abs`·`joint_delta` 채널 소스 |
| `/hand/right/kin` | Float32MultiArray | 손 kinesthetic(4×3) — `ft` 채널 소스 |
| **`/paxini/right/raw`** | Float32MultiArray | **촉각 원본 4×127×3 — 힘 판정 + `resul_curved`·`contact` 채널** |
| `/franka/right/joint_states` | JointState | 팔 q(고정 확인용) |

**결과 (데모 → GUI, publish)**

| 토픽 | 타입 | QoS |
|---|---|---|
| `/property/result` | std_msgs/String (JSON) | RELIABLE · **transient_local(latched)** |

시퀀스 체인용(B)은 추가로 `/sequence_state`(구독) 와 `/sequence/request_control` ·
`/sequence/release_control`(서비스) · `/sequence/heartbeat`(발행) 을 쓴다.

QoS 규약: 상태 best-effort, 단발 명령(servo/mode) reliable, 고속 스트림(q_target) best-effort
— 상세는 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) §C1.

---

### 5.5 문제 해결

| 증상 | 원인 / 해결 |
|------|------------|
| `상태 토픽 미수신` 후 종료 | `shm_state_publisher_node`(+C++ 컨트롤러) 미실행 / `ROS_DOMAIN_ID` 불일치 |
| `/paxini/right/raw 미수신` 경고 | paxini writer 또는 `shm_state_publisher` 의 raw 발행 미실행. 힘=0 fallback 이라 추론이 무의미 → 반드시 해결 |
| `[eco-engine] 샘플 부족 (n < 16)` | 스퀴즈가 너무 짧거나 유효 프레임 부족. 물체가 실제로 물려 11.0 N 까지 압축되는지, `valid` 플래그가 서는지 확인 |
| `[eco-engine] ⚠ baseline 프레임 0` | 안정화 구간에서 유효 프레임을 못 받음 → Δ채널 영점이 어긋나 결과 부정확. paxini 수신 상태부터 확인 |
| `[eco-engine] 포화 프레임 N개 수리` | **정상 로그**(학습 SAT_MODE=repair 정합). 개수가 과도하면 파지력이 센 것 |
| `ModuleNotFoundError: dual_arm_msgs` | (B) 전용 — Dual_Arm 워크스페이스 `install/setup.bash` 미source (§3.4 참고) |
| `ModuleNotFoundError: torch` | `env.sh` 를 source 하지 않아 conda python 이 잡힘. torch 는 시스템 python 에 `--user` 로 설치돼 있다 |
| GUI 가 안 뜸 | `[property_gui] 스크립트 없음` 로그면 경로 문제. 그 외에는 `--no-gui` 없이 실행했는지, tkinter 설치 여부 확인 |
| 손이 안 움직임 | Ctrl `hand_target_receiver_node` 미실행 또는 명령 QoS 불일치 |
| (B) 가 계속 대기 | 직전 Inhand(#2) 가 DONE 이 아님. `ros2 topic echo /sequence_state --qos-durability transient_local --qos-reliability reliable` 로 확인 |

---

## 6. 다른 PC 로 이식해서 실행하기 (배포 패키지)

강성·무게·크기 추론 모듈만 떼어 다른 PC 에서 돌리기 위한 zip 배포판이다.
**추론 PC(산업부 PC) 쪽에 필요한 파일만** 담겨 있고, 제어 스택(Dual_Arm_Hand_Ctrl ·
paxini writer)은 **포함되지 않는다** — 그건 기존처럼 제어 PC 에서 뜬 채로 ROS2 토픽으로 붙는다.

### 6.1 대상 PC 요구사항

| 항목 | 요구 | 확인 |
|---|---|---|
| OS / ROS2 | Ubuntu 22.04 + **ROS2 Humble**(apt 설치) | `ls /opt/ros/humble` |
| python | **시스템 python3.10** (`/usr/bin/python3`) — conda 금지 | `env.sh` 가 conda 를 PATH 에서 제거 |
| pip | `torch`(CUDA 또는 CPU) · `numpy` · `pyyaml` | `pip install --user -r requirements.txt` |
| GUI | `python3-tk` (tkinter) | 없으면 `--no-gui` 로 추론만 가능 |
| 네트워크 | 제어 PC 와 **같은 LAN**, `ROS_DOMAIN_ID=9` 동일 | `env.sh` 가 설정 |

GPU 는 선택이다. CUDA 가 없으면 자동으로 CPU 추론으로 떨어진다(모델이 10만 파라미터급이라 CPU 로도 충분).

### 6.2 설치 (대상 PC, 최초 1회)

```bash
# 1) 압축 해제 — 어느 경로든 무방(코드가 전부 __file__ 상대경로라 절대경로 수정 불필요)
cd ~
unzip stiffness_predict_demo.zip          # → ~/stiffness_predict_demo/
cd ~/stiffness_predict_demo

# 2) 의존 설치 (env.sh 규약 = 시스템 python + --user)
sudo apt install -y python3-tk            # GUI 용(tkinter)
source env.sh
pip install --user -r requirements.txt    # torch / numpy / pyyaml
#   CUDA 버전을 맞추려면 torch 만 따로:
#   pip install --user torch --index-url https://download.pytorch.org/whl/cu121

# 3) 준비 상태 자체 점검 (로봇·ROS토픽 없이 실행 가능)
source env.sh
python3 check_setup.py
```

`check_setup.py` 는 python 종류 · 패키지 · 번들 파일 · **모델 실물 로드** · 라벨↔ckpt 정합까지
확인하고, 통과하면 아래처럼 끝난다:

```
[eco-engine] gru_anchor_s42.pth 로드 — 109,352 파라미터 · device=cuda · 채널 67 [...]
  [ OK ] 엔진 로드 성공 (variant=gru, device=cuda)
  [ OK ] 라벨 18 개체 로드
[eco-engine] 라벨 normalize = ckpt norm 일치 (oldstif 정합 확인)
결과: (A) deploy_ros2_demo 실행 준비 완료.
```

> `dual_arm_msgs` / `sequence_client` 가 `[WARN]` 로 뜨는 것은 정상이다 —
> **(B) 시퀀스 체인에서만** 필요하다(§6.4).

### 6.3 실행 — (A) 단독 데모 : 파지부터 직접

제어 PC 에서 §2 준비(`shm ; rs ; nd` + paxini writer)가 끝나 `/paxini/right/raw` 가
발행 중이어야 한다. 대상 PC 에서:

```bash
cd ~/stiffness_predict_demo
./run_demo.sh                 # env.sh source + GUI 자동 실행까지 한 번에
./run_demo.sh --no-gui        # GUI 없이(개발용)
```

수동으로 하려면 §3.3 과 동일하다:

```bash
cd ~/stiffness_predict_demo
source env.sh
python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py
```

실행 직전 토픽 수신 확인(권장):

```bash
source env.sh
ros2 topic hz /paxini/right/raw      # 안 오면 추론이 무의미 — 제어 PC 부터 확인
ros2 topic hz /hand/right/joint_states
```

진행·메뉴·출력 해석은 §3.3 과 완전히 동일하다(포즈 번호 입력 → 파지 → 안정화 0.8 s →
스퀴즈 11.0 N → 3속성 추론 → `[1] 다시 스퀴즈 / [2] 재파지 / [3] 종료`).

### 6.4 실행 — (B) 시퀀스 체인 : 이미 파지한 상태에서 스퀴즈만

`dual_arm_msgs` + `sequence_client` 가 **대상 PC 에도** 있어야 한다(이 zip 에는 없음).
Dual_Arm_Hand_Ctrl 워크스페이스를 그 PC 에 두고 빌드한 뒤:

```bash
cd ~/stiffness_predict_demo
./run_task3_demo.sh
#   워크스페이스 경로가 다르면:
DUAL_ARM_WS=/경로/install/setup.bash ./run_task3_demo.sh
```

수동은 §3.4 와 동일:

```bash
source env.sh
source ~/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash
python3 stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py
```

### 6.5 GUI 만 따로 / 로봇 없이 확인

```bash
source env.sh && python3 stiffness_deploy_ros2/gui/property_gui.py     # ROS 연결
python3 stiffness_deploy_ros2/gui/property_gui.py --demo               # 로봇 없이 레이아웃만
```

`/property/result` 가 latched 라서 GUI 를 나중에 띄워도 마지막 결과를 받는다.

### 6.6 zip 에 담긴 것 / 담기지 않은 것

```
stiffness_predict_demo/
├── run_demo.sh              ★ (A) 실행 (env.sh + GUI 자동)
├── run_task3_demo.sh        ★ (B) 실행 (+ Dual_Arm 워크스페이스 source)
├── check_setup.py           ★ 이식 후 자체 점검
├── readme_demo.md           이 문서
├── env.sh · requirements.txt · package.xml · setup.py · setup.cfg · resource/
├── run_deploy_gui.sh · run_task3_gui.sh      기존(과일 강성등급) 배포용
├── docs/TROUBLESHOOTING.md
└── stiffness_deploy_ros2/
    ├── core/       shm_common.py · paxini_shm.py            (상수·타입 정의용)
    ├── launch/     deploy_ros2_demo.py · deploy_task3_ros2_demo.py  ← 진입점 2개
    │               ecoflex_engine.py · eco_model.py · eco_paxini_features.py
    │               property_result_pub.py · deploy.py · deploy_task3.py
    │               deploy_ros2.py · deploy_ros2_exp.py · deploy_ros2_exp_rawft.py
    │               real_deploy_inference*.py · model.py · stiffness_result_pub.py
    │               hand_pose_io.py · test_ecoflex_engine_offline.py
    │               *.txt(포즈 11종) · fruit_thresholds.yaml
    ├── gui/        property_gui.py(★결과 GUI) · stiffness_gui.py · assets/
    ├── models/     ecoflex2fruit/(★4종 ckpt + sensors.json) + 기존 등급모델 *.pth
    └── labels/     object_labels_oldstif/(★18개체) + general/ · trial2/
```

- **포함 안 됨** — 제어 스택(`Dual_Arm_Hand_Ctrl`), `dual_arm_msgs`/`sequence_client`,
  MoveIt 워크스페이스(`kistar_ws`), 수집 데이터(`collect_logs/`), 수집·변환 스크립트
  (`collect_ros2*.py`, `bag_to_*.py`, `verify_parity.py`), 로그(`logs/`).
  데모 추론에는 어느 것도 필요 없다.
- `real_deploy_inference*.py` · `model.py` · `stiffness_result_pub.py` · `deploy_ros2.py` 는
  **데모가 직접 쓰진 않지만 import 사슬에 걸려 있어** 반드시 함께 있어야 한다
  (`deploy_ros2_demo` → `deploy_ros2` → `real_deploy_inference_final` → `model`).
- `models/*.pth`(기존 등급 모델 12 MB)와 `labels/general·trial2` 는 데모가 읽지 않는다.
  용량을 줄이려면 지워도 (A)·(B) 데모는 동작한다 — 단 기존 `deploy_ros2.py` 경로는 못 쓴다.
- `test_ecoflex_engine_offline.py` 는 학습 저장소(`deep_ws`)가 있어야 돌아가는 검증용이라
  이식 PC 에서는 보통 실행하지 않는다.

### 6.7 이식 후 자주 걸리는 것

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: torch` | `env.sh` 를 source 안 했거나 conda python 이 잡힘. `python3 check_setup.py` 로 실행 python 경로 확인 |
| `상태 토픽 미수신` 후 즉시 종료 | 제어 PC 스택 미실행, 또는 `ROS_DOMAIN_ID` 불일치 / 다른 서브넷. 양쪽 다 `ROS_DOMAIN_ID=9`·`ROS_LOCALHOST_ONLY=0` 인지 확인 |
| `/paxini/right/raw 미수신` 경고 | paxini writer 또는 shm_state_publisher 의 raw 발행 미실행. 힘=0 fallback 이라 결과가 무의미 → 반드시 해결 |
| GUI 안 뜸 | `sudo apt install python3-tk`. 그래도 안 되면 `--no-gui` 로 추론만 수행 |
| `ModuleNotFoundError: dual_arm_msgs` | (B) 전용 — Dual_Arm 워크스페이스 source 필요(§6.4) |
| 추론값이 이상함 | §4.2 ④ 처럼 라벨 최근접 정규화 거리를 먼저 본다. 0.5 초과가 계속되면 센서/파지/baseline 문제 — §5.5 표 참고 |

나머지 증상은 §5.5 와 `docs/TROUBLESHOOTING.md` 를 그대로 따르면 된다.