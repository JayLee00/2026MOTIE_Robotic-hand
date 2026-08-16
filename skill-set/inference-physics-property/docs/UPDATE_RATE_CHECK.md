# update rate 확인 — 실행 절차 & 결과 기록

> 이 워크스페이스(`~/motie_ws/stiffness_deploy_ros2`)에서 **update rate(루프/토픽 갱신률)를
> 계측하고 결과를 남기는** 절차. 명령을 순서대로 실행하면 결과가 `docs/rate_log/<run>/` 에
> 자동 저장되고, 아래 [5. 결과 기록](#5-결과-기록-붙여넣기) 표에 붙여넣어 누적 비교한다.
>
> 배경·원인 분석은 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) **§F**(F1 문제정의 → F5/F6 계측결과),
> 사용법 전반은 [README](../README.md) 참고.

---

## 0. 무엇을 왜 재나 (판정 기준)

> ⚠ **`ros2 topic hz` 는 "통신 속도"이고 "센서 update rate" 가 아니다.**
> 퍼블리셔(`shm_state_publisher`)가 SHM 의 **같은 값을 재발행**하면 센서가 완전히 멈춰도
> hz 는 정상(89/200Hz)으로 보인다. 그래서 **값이 실제로 바뀌는 빈도**를 따로 재야 한다
> → [3.5 센서 실측 update rate](#35-센서-실측-update-rate-값-변화-기준). 실제로 이 방법으로
> paxini 촉각이 전 채널 0 고정인 것을 잡아냈다([7.2](#72-센서-갱신-실측-2026-07-27)).

| 지표 | 어디서 | 학습/목표 기준 | 의미 |
|---|---|---|---|
| **센서 갱신율** | `sensor_update_rate.py` 의 `유의미 갱신율` | 0 이 아닐 것 | ★★ 값이 실제로 바뀌는 빈도. 이게 0 이면 아래 지표 전부 무의미 |
| **루프 유효 rate** | `deploy_ros2_exp` 의 `[measure] 유효 rate` | **~100 Hz** | ★핵심. 파이썬 모션 루프가 실제로 도는 속도 = 추론 입력 샘플링 속도 |
| **downsample 스텝** | `[measure] downsample 스텝` | **≥ MIN_LEN(10)** | `valid // FACTOR(10)`. 학습 시퀀스 길이와의 정합 |
| **setpoint 발행률** | `ros2 topic hz /hand/right/q_target` | **~100 Hz** | 낮으면 모션 끊김(§C1 QoS 블로킹) |
| **힘 소스 rate** | `ros2 topic hz /paxini/right/ft` | **~90 Hz** | 없으면 추론 자체가 성립 안 함(§B1) |
| (동반) **thumb 최대 Fz** | `[measure] thumb 최대` | 스퀴즈 임계 도달 | rate 가 정상이어도 여기서 막힌다(§F5/F6) |

기준 상수는 코드에 있다 — [real_deploy_inference_final.py:117-118](../stiffness_deploy_ros2/launch/real_deploy_inference_final.py#L117-L118)
(`FACTOR = 10`, `MIN_LEN = FACTOR`), 제어 주기는 `deploy.py` 의 `CONTROL_RATE_HZ = 100`.

> **주의(로그 함정)**: `[squeeze] threshold(N) 도달` 문구는 **미도달이어도 무조건 출력**된다.
> 힘 도달 여부는 반드시 `[measure] thumb 최대` 숫자로 판단할 것.

---

## 1. 사전 조건 — 제어 PC 스택 + 힘 소스

`update rate` 는 **상대(제어 PC 노드)가 떠 있어야** 의미가 있다. 순서대로:

```bash
# (제어 PC 터미널들 — quick_start.txt 의 축약 명령)
shm     # C++ 컨트롤러 + shm_state_publisher_node
rs      # ROS2 환경
nd      # arm_q_target_receiver_node / hand_target_receiver_node

# 힘 피드백(필수 — 없으면 valid=0 으로 추론 불가, §B1)
#   ※ quick_start.txt 의 `~/.Dual_Arm_Hand_Ctrl` 는 오타 — 아래가 실제 경로
python3 ~/motie_ws/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r
```

> ★ **writer 가 안 떠 있어도 `/paxini/right/ft` 는 89Hz 로 발행된다**(0 을 재발행).
> `ros2 topic hz` 로는 절대 구분되지 않으니, 아래로 **값이 살아있는지**부터 확인할 것:
>
> ```bash
> python3 tools/sensor_update_rate.py --duration 10 --topics /paxini/right/ft
> #   → 판정이 LIVE 여야 진행. FROZEN 이면 writer 부터 살린다 (7.2 참고)
> ```

## 2. 환경 & 토픽 살아있는지 확인

```bash
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh          # ROS2 humble + 시스템 python3 (conda 금지 — env.sh 주석 참고)

ros2 node list                                     # 제어 PC 스택이 보이는지
ros2 topic echo --once /hand/right/joint_states     # 손 상태
ros2 topic echo --once /paxini/right/ft             # 촉각(힘)
ros2 topic hz /paxini/right/ft                      # ~90Hz (Ctrl+C)
```

하나라도 안 오면 → `attach()` 타임아웃으로 배포가 즉시 종료된다. 먼저 1번을 해결.

## 3. 자동 계측 (권장) — 한 커맨드로 실행 + 저장

`tools/measure_update_rate.sh` 가 **rate 모니터 + 계측판 배포 + 결과 저장 + 요약 생성**을 한 번에 한다.

```bash
cd ~/motie_ws/stiffness_deploy_ros2

# (a) baseline — 현재 코드 그대로 계측  (실행 후 과일 번호 입력)
./tools/measure_update_rate.sh --label kiwi_baseline

# (b) 힘-도달 curl 판과 비교할 때
./tools/measure_update_rate.sh --label kiwi_curl --forcecurl

# (c) 배포를 다른 터미널에서 직접 돌릴 때 — rate 모니터만 30s
./tools/measure_update_rate.sh --monitor --duration 30
```

배포는 대화형이다: 과일 번호 입력 → 파지 → 스퀴즈 → `[measure]` 블록 출력 → 메뉴에서
`1`(재스퀴즈)로 **3회 이상 반복** 후 `3`(안전복귀·종료). 종료 시 요약이 자동 출력된다.

저장 결과 (`docs/rate_log/<타임스탬프>_<label>/`):

| 파일 | 내용 |
|---|---|
| `exp_stdout.log` | 배포 전체 로그 — `[measure]` 블록 포함 (**원본 근거**) |
| `hz_<토픽>.log` | `ros2 topic hz` **시계열** (`epoch average rate: …`) — 구간별 재분석 가능 |
| `info_<토픽>.txt` | 토픽 QoS (`--verbose`) — §C1 Reliability 불일치 진단 |
| `meta.txt` | 실행 조건 (git commit·dirty·python·kernel·DOMAIN_ID) |
| `summary.md` | 붙여넣기용 마크다운 표 + 자동 판정 |

감시 토픽을 바꾸려면:

```bash
RATE_TOPICS="/hand/right/q_target /franka/right/q_target /paxini/right/ft" \
  ./tools/measure_update_rate.sh --label wide
```

나중에 재분석 / 여러 run 비교:

```bash
python3 tools/rate_summary.py docs/rate_log/20260727_101500_kiwi_baseline
python3 tools/rate_summary.py docs/rate_log/*        # run 통합 비교표까지
```

## 3.5 센서 실측 update rate (값 변화 기준)

3번의 자동 계측에는 **이미 포함**되어 있다(`sensor_change.log` / `sensor_change.json`).
센서만 따로 확인하려면:

```bash
source env.sh
python3 tools/sensor_update_rate.py --duration 20                       # 기본 4개 토픽
python3 tools/sensor_update_rate.py --duration 20 --topics /paxini/right/raw /paxini/right/ft
python3 tools/sensor_update_rate.py --duration 20 --out docs/rate_log/manual  # CSV/JSON 저장
```

무엇을 보나 — 메시지마다 payload 를 이전 값과 비교해서:

| 항목 | 뜻 |
|---|---|
| **메시지 발행률** | `ros2 topic hz` 와 같은 값 (통신) |
| **★ 유의미 갱신율** | 값이 실제로 바뀐 빈도 = **센서 update rate** (`\|Δ\| ≥ 1e-4`) |
| **판정** | `LIVE`(정상 갱신) / `JITTER`(LSB 만 흔들림=실질 정지) / `FROZEN`(값 고정) |
| 중복(스테일) | 같은 값 재발행 비율. 높으면 오버샘플 |
| 정적 채널 | 측정 내내 안 바뀐 채널 — 어느 손가락/축이 죽었는지 특정 |
| 최소 변화폭 | 양자화 단위 추정 (손 관절=1 count, paxini Fz=0.1N) |

> `/paxini/right/ft` 는 **timestamp·seq 가 없는** `Float32MultiArray(12)` 라서
> ([deploy_ros2.py:229-233](../stiffness_deploy_ros2/launch/deploy_ros2.py#L229-L233))
> 원본 센서 샘플 시각을 알 방법이 없다 — 그래서 **값 변화로 역산**하는 것이 유일한 방법이다.
> 브리지의 `seq` 는 '수신 메시지 카운터'일 뿐이니 센서 갱신 근거로 쓰면 안 된다.
>
> **해석 주의**: 접촉이 없으면 paxini 가 정상적으로 0 고정일 수 있다. 단
> `/paxini/right/raw`(1524ch) 까지 **전 채널 정확히 0** 이면 접촉 문제가 아니라
> **소스(paxini_writer) 미동작**이다 → [7.2](#72-센서-갱신-실측-2026-07-27) 참고.
> 팔은 고정 운용이므로 `/franka/right/joint_states` 의 `JITTER` 는 정상이다.

## 4. 수동 계측 (스크립트 없이 / 부분 확인)

터미널 2개로 나눠서:

```bash
# 터미널 A — 계측판 배포 (로그를 파일로도 남기려면 tee)
cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
python3 -u stiffness_deploy_ros2/launch/deploy_ros2_exp.py | tee docs/rate_log/manual_$(date +%H%M).log

# 터미널 B — 와이어 rate 교차 확인 (A 가 스퀴즈 중일 때 봐야 유효)
source env.sh
ros2 topic hz /hand/right/q_target        # setpoint 발행률, 목표 ~100Hz
ros2 topic hz /paxini/right/ft            # 힘 소스, ~90Hz
ros2 topic hz /hand/right/joint_states    # 상태 소스
ros2 topic info /hand/right/q_target --verbose   # 수신자 Reliability (§C1)
```

> 터미널 B 의 `hz` 는 **모션이 없는 동안 0Hz** 다(`q_target` 은 움직일 때만 발행).
> 반드시 스퀴즈 구간의 값을 읽을 것 — 자동 스크립트는 타임스탬프 시계열로 남기므로 이 문제가 없다.

---

## 5. 결과 기록 (붙여넣기)

> `summary.md` 의 숫자를 아래 표에 옮기고, 원문은 접힌 블록에 붙여 근거를 남긴다.
> **run 디렉토리명**을 같이 적어야 나중에 원본을 찾을 수 있다.

### 5.1 회차 요약 — 루프 rate & 힘

| 날짜 | run 디렉토리 | 과일 | 회차 | 유효 rate(Hz) | steps | thumb Fz(N) | 스퀴즈 임계(N) | 도달률 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-27 | `20260727_010320_kiwi_contact` | **tomato** | 0 | 97.8 | 15 | 4.50 | 10.0 | 45% | ★ 최신·접촉 양호(전 손가락 반응) |
| 2026-07-27 | 〃 | tomato | 1 | 97.8 | 15 | 5.20 | 10.0 | 52% | 〃 |
| 2026-07-27 | 〃 | tomato | 2 | 97.9 | 15 | 4.70 | 10.0 | 47% | 〃 |
| 2026-07-27 | `20260727_005724_kiwi_contact` | kiwi | 0 | 97.8 | 15 | 1.70 | 13.0 | 13% | 접촉 약함·finger3 무반응(대조용) |
| 2026-07-27 | 〃 | kiwi | 1 | 97.7 | 15 | 2.10 | 13.0 | 16% | 〃 |
| 2026-07-27 | 〃 | kiwi | 2 | 97.8 | 15 | 1.50 | 13.0 | 12% | 〃 |

### 5.2 센서 갱신 — 값 변화 기준 (`sensor_change.json`)

| 날짜 | run 디렉토리 | 토픽 | 판정 | 발행률(Hz) | 갱신 전체/활성(Hz) | 정적채널 | 비고 |
|---|---|---|---|---|---|---|---|
| 2026-07-27 | `20260727_010320_kiwi_contact` | `/paxini/right/ft` | ✅ LIVE | 89.5 | 46.2 / **89.7** | **0/12** | ★ 접촉 양호. 접촉구간만 보면 **72.6Hz(81%)** |
| 2026-07-27 | 〃 | `/hand/right/joint_states` | ✅ LIVE | 199.7 | 160.4 / 198.6 | 0/16 | 양자화 1 count |
| 2026-07-27 | 〃 | `/hand/right/kin` | ⚠ FROZEN | 199.7 | 0 / 0 | 12/12 | 통합모델 미사용(USE_JKIN=False) → 무해 |
| 2026-07-27 | 〃 | `/franka/right/joint_states` | ✅ LIVE | 199.7 | 1.1 / 197.3 | 0/7 | 스퀴즈 시 순간 흔들림 74회. 팔 고정이라 정상 |
| 2026-07-27 | `20260727_005724_kiwi_contact` | `/paxini/right/ft` | ✅ LIVE | 89.4 | 13.7 / 45.1 | 3/12 (ch9-11) | 접촉 약함(대조용). 접촉구간 **27.0Hz(29%)** |

### 5.3 와이어 rate — `ros2 topic hz`

| 날짜 | run 디렉토리 | `/hand/right/q_target` | `/paxini/right/ft` | `/hand/right/joint_states` | 비고 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

### 5.4 원문 붙여넣기 — `[measure]` 블록

<details>
<summary>run: ____________________ (과일: ____)</summary>

```
<!-- 여기에 [measure] 블록 / summary.md 원문을 붙여넣으세요.
     예:
[measure] 스퀴즈 계측 (학습 대비 격차 확인용)
  add_sample 호출 = 150,  valid(적재) = 150
  유효 rate       =   94.7 Hz   (수집시간 1.57s)
  downsample 스텝 = 15   (FACTOR=10, MIN_LEN=10)
  finger별 최대 Fz = [4.4, 1.8, 1.2, 0.8]
    thumb 최대 = 4.40 N   (스퀴즈 임계 13.0 N)
-->
```

</details>

<details>
<summary>run: ____________________ (과일: ____)</summary>

```
<!-- 붙여넣기 -->
```

</details>

<details>
<summary>run: ____________________ (과일: ____)</summary>

```
<!-- 붙여넣기 -->
```

</details>

### 5.5 `ros2 topic hz` 원문

<details>
<summary>run: ____________________</summary>

```
<!-- 붙여넣기: average rate / min / max / std dev / window -->
```

</details>

### 5.6 관찰 메모 (자유 서술)

- 육안 모션(파지/스퀴즈)이 부드러웠는지:
- 끊김·정지 구간이 있었는지 (있으면 몇 초, 어느 단계):
- 그 외 이상 로그:

---

## 6. 판정 & 다음 액션

`summary.md` 하단의 자동 판정이 아래 기준을 그대로 적용한다.

| 결과 | 판정 | 다음 액션 |
|---|---|---|
| **센서 갱신율 = 0 (FROZEN)** | ❌ 최우선 | 아래 것들 전부 무의미. 소스부터 살린다 → [7.2](#72-센서-갱신-실측-2026-07-27) |
| 유효 rate **≥ ~85Hz** & steps ≥ MIN_LEN | ✅ rate 정상(학습 100Hz 정합) | rate 는 문제 아님 → **힘/모델**로 이동 (§F5 결론) |
| 유효 rate **< ~85Hz** | ❌ 루프 병목 | §F4 `#2` 루프 경량화(tick당 `paxini.read()` 1회 캐시, 미사용 구독 제거) |
| steps **< MIN_LEN(10)** | ❌ 샘플 부족/OOD | §F4 `#1` `FACTOR = round(실측 rate / 10)` 정합 |
| `q_target` hz 만 낮고 루프 rate 는 정상 | ❌ 발행 경로 문제 | `info --verbose` 로 수신자 Reliability 확인(§C1 QoS) |
| `/paxini/right/ft` 미수신 | ❌ 힘=0 | `paxini_writer.py --hand r` 실행(§B1) |
| rate 정상인데 **thumb Fz 미도달** | ❌ 힘 도메인 | `--forcecurl` 로 Fz 상승 여부 확인 → 정체 시 **Path B**(재수집·fine-tune, §F6) |

> 원칙: **한 번에 하나만** 바꾸고 같은 절차로 재측정해 효과를 격리한다(§F4 각주).

---

## 7. 계측 이력

### 7.1 참고 — 기존 baseline (2026-07-14, `docs/result_log*/`)

이번 계측과 비교할 기준선. **rate 는 이미 정상, 병목은 힘**이라는 것이 당시 결론(§F5).

| 과일 | 유효 rate | steps | thumb Fz 최대 | 스퀴즈 임계 | 도달률 |
|---|---|---|---|---|---|
| plum | 94–97Hz | 15 | 2.8–3.3N | 8.0N | ~38% |
| kiwi | 95–97Hz | 15 | 3.3–4.5N | 13.0N | ~30% |
| tomato | 83–97Hz | 15 | 3.2–3.7N | 10.0N | ~35% |
| lemon | 94–98Hz | 15 | 3.6–5.2N | 12.0N | ~37% |

palm-down 대조 실험(§F6)에서 **자세는 힘에 영향 없음**(~3–4N 동일)으로 확인 → palm 가설 기각.

### 7.2 센서 실측 갱신율 (2026-07-27) — 결론: **센서 83.3Hz, rate 는 문제 아님**

`20260727_010320_kiwi_contact`(★기준, tomato·접촉 양호)와 `20260727_005724_kiwi_contact`
(대조, 접촉 약함) 2 run. 센서 OFF 로 잰 run 들은 무효라 삭제했다(아래 "센서 OFF 사고").

| 토픽 | 발행률 | 갱신 전체/활성 | 판정 | 해석 |
|---|---|---|---|---|
| `/paxini/right/ft` (12ch) | 89.5Hz | 46.2 / **89.7Hz** | ✅ LIVE | **0/12 정적**, 접촉구간 72.6Hz(81%) |
| `/hand/right/joint_states` (16ch) | 199.7Hz | 160.4 / **198.6Hz** | ✅ LIVE | 양자화 1 count |
| `/hand/right/kin` (12ch) | 199.7Hz | 0 / 0Hz | ⚠ FROZEN | 전 채널 0. **통합모델은 `USE_JKIN=False` 로 미사용 → 무해** |
| `/franka/right/joint_states` (7ch) | 199.7Hz | 1.1 / 197.3Hz | ✅ LIVE | 유의미 변화 74회(스퀴즈 순간 흔들림). 팔 고정이라 정상 |

**핵심 — 촉각 값 변화율은 센서 스펙이 아니라 접촉력에 따라 달라진다.**
Fz 양자화가 **0.1N** 이라, 힘이 약하면 양자화 계단을 넘는 프레임이 적어 "갱신율"이 낮게 나온다.
같은 센서인데 접촉 강도만 다른 두 run 을 접촉 구간만 잘라 비교하면:

| run | 최대 \|F\| | 접촉구간(\|F\|≥1N) 갱신율 | 변화 프레임 비율 | 정적 채널 |
|---|---|---|---|---|
| `005724` (약접촉) | 3.2N | 27.0Hz | 29% | 3/12 (finger3 전멸) |
| **`010320` (양호)** | **5.3N** | **72.6Hz** | **81%** | **0/12** |
| (무접촉 구간, 양쪽) | ~0N | 0.1–0.4Hz | 0% | — |

⇒ 접촉이 확실하면 갱신율이 **센서 상한(데이터시트 83.3Hz)** 에 근접한다(관측 72.6Hz = 87%).
약접촉에서 잰 45Hz·27Hz 는 **0.1N 양자화 때문에 낮게 측정된 하한**이었다 — 센서가 느린 게 아니다.

> ⚠ **'활성 89.7Hz' 를 센서 rate 로 읽으면 안 된다.** 변화간격 중앙값(11.14ms)이 곧
> **발행 타이머 tick(11.12ms)** 이라, 그 수치는 센서가 아니라 타이머를 잰 것이다.
> 센서 rate 판정은 **접촉구간 변화 횟수/초**(72.6Hz)로 해야 한다.
> 상세 논거는 [UPDATE_RATE_CONCLUSION.md §5.5](UPDATE_RATE_CONCLUSION.md) 참고.

> **추론 관점**: 엔진은 150 샘플을 `FACTOR=10` 으로 줄여 15 스텝(≈9.8Hz)을 만든다.
> **센서 83.3Hz ≫ 9.8Hz** 이므로 힘 채널 정보량은 충분 — `FACTOR`/rate 는 문제가 아니다
> (§F5 결론 유지). 남은 문제는 **힘의 크기**뿐이다.

| 지표 | 결과 (`010320`) | 판정 |
|---|---|---|
| 루프 유효 rate | 97.8Hz (3회, 편차 0.1) | ✅ 학습 100Hz 정합 |
| downsample steps | 15 (MIN_LEN=10) | ✅ 샘플 충분 |
| 촉각 센서 갱신 | 83.3Hz(소스)·관측 72.6Hz, 전 채널 반응 | ✅ 정상 |
| thumb 최대 Fz | 4.5 / 5.2 / 4.7N (임계 10N) | ❌ **도달률 45–52%** |

**손가락별 반응** (변화횟수 / 최대 \|Fz\|) — `005724` → `010320` 개선:

| | thumb | finger1 | finger2 | finger3 |
|---|---|---|---|---|
| `005724` | 695 / 2.2N | 87 / 0.5N | 144 / 0.7N | **0 / 0.0N** |
| `010320` | 2845 / 5.3N | 1263 / 2.5N | 2120 / 2.5N | **2052 / 1.6N** |

→ **finger3 센서는 정상**이었다(이전 run 은 단순 미접촉). 손가락 4개 모두 힘을 받고 있다.

**남은 문제는 하나 — 힘이 임계의 절반**: thumb 4.8N 평균 vs 임계 10N.
단 2026-07-14 의 tomato baseline(3.2–3.7N, ~35%)보다는 **개선**되었다(4.5–5.2N, ~48%).
→ §F6 의 분기점(**학습셋의 실제 스퀴즈 Fz 확인**)이 그대로 다음 할 일이다.

<details>
<summary>참고 — 센서 OFF 사고 (같은 날 이전 run, 삭제됨)</summary>

`paxini_writer.py` 미실행 상태로 계측해 **thumb Fz 0.0–0.5N** 이 나왔다. `/paxini/*` 는
`shm_state_publisher` 가 PaXini **SHM 영역을 그대로 재발행**하는 구조라, writer 가 SHM 을
채우지 않으면 **0 을 89Hz 로 영원히 재발행**한다 → `ros2 topic hz` 는 89Hz 로 건강하고
`attach()` 도 통과하고 `valid=1` 이라 추론까지 도는 **거짓 정상**. `/paxini/right/raw` 의
**1524/1524 채널이 전부 정확히 0** 인 것으로 확정했다.

```bash
pgrep -af paxini                                # 프로세스 있나
ros2 topic info /paxini/right/ft --verbose      # 퍼블리셔가 shm_state_publisher 뿐이면 writer 없음
python3 ~/motie_ws/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r   # quick_start.txt 경로는 오타
```

**교훈**: `ros2 topic hz` 는 통신 점검용이고 **센서 생존 점검에는 쓸 수 없다.**
계측 전에 항상 `sensor_update_rate.py` 로 `LIVE` 를 확인하고 나서 나머지를 재라.
(→ [TROUBLESHOOTING §B1-b](TROUBLESHOOTING.md))

</details>

---

## 부록 — 파일 위치

| 항목 | 경로 |
|---|---|
| 계측 러너 | [tools/measure_update_rate.sh](../tools/measure_update_rate.sh) |
| **센서 갱신 계측** | [tools/sensor_update_rate.py](../tools/sensor_update_rate.py) — 값 변화 기준 |
| 요약/재분석 | [tools/rate_summary.py](../tools/rate_summary.py) |
| 계측판 배포 | [deploy_ros2_exp.py](../stiffness_deploy_ros2/launch/deploy_ros2_exp.py) |
| 힘-도달 curl 판 | [deploy_ros2_exp_forcecurl.py](../stiffness_deploy_ros2/launch/deploy_ros2_exp_forcecurl.py) |
| 결과 저장 위치 | `docs/rate_log/<타임스탬프>_<label>/` (이번 절차), `docs/result_log*/`(2026-07-14 baseline) |
| 환경 설정 | [env.sh](../env.sh) — conda 금지 이유 포함 |
