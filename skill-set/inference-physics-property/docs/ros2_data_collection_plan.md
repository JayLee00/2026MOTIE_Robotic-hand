# 데이터 수집의 ROS2 전환 — 설계 결정 및 실행 계획

> 상태: 설계 확정 · 구현 착수 전
> 대상 독자: `stiffness_deploy_ros2`(실행 워크스페이스)에서 이어 작업하는 개발자
> 정본(canonical): **이 문서.** 실행 워크스페이스에 자기완결로 둠 — **이 레포만 pull 해도 전부 이어갈 수 있다.** (Gen3에도 동일 사본이 있으나 참조 불필요.)
> 한 줄 요지: **딥러닝 학습 데이터를 "배포 시 모델이 실제로 보게 될 신호를, 배포와 동일한 파이프라인·rate로" 수집하도록, 수집 코드를 배포 코드와 하나로 합친다.**

---

## 1. 배경 / 문제

- **Gen3**: 센서 구동 + 기존 데이터 수집 시퀀스. SHM(공유메모리) 직접 접근으로 동작.
  - 모션: `launch/motion_sequence_A_self.py` (11데모 자동 시퀀스, SHM 직접)
  - 로깅: `core/hdf5_logger_1k.py` (SHM을 명목 1kHz로 읽어 HDF5 24개 데이터셋 저장, 실제 ~150Hz 가변)
- **stiffness_deploy_ros2**: 실제 로봇 배포. `deploy.py`의 모션/힘판정/추론 로직을 **그대로 재사용**하되, SHM I/O만 **ROS2 토픽**으로 치환.

실 로봇 배포는 ROS2를 쓰므로, **데이터 수집도 ROS2 시스템 기준으로** 돌리려 한다. 문제는 "어떻게 바꾸면 딥러닝 학습에 적합한 데이터가 나오는가" 이다.

---

## 2. 핵심 통찰 (모든 결정의 근거)

### 2.1 전송 계층은 이미 어댑터 패턴으로 분리돼 있다

`deploy.py`의 모션/힘판정 로직은 `shm.read()` / `shm.write_partial()` / `paxini.read()` **인터페이스에만 의존**한다. `stiffness_deploy_ros2`의 `Ros2ShmBridge` / `Ros2PaxiniBridge`는 이 인터페이스(attach/read/write_partial/detach)를 **ROS2 토픽으로 구현**한 것뿐이고, `deploy.py`는 한 줄도 바뀌지 않았다.

그리고 `motion_sequence_A_self.py`도 **똑같은 인터페이스**로 SHM을 쓴다.

> ⇒ 수집 시퀀스를 ROS2로 바꾸는 것은 **이미 검증된 브리지를 그대로 꽂는 일**이지, 새로 짜는 일이 아니다.

### 2.2 지금 배포는 이미 train/deploy 불일치로 성능이 깎이고 있다

`stiffness_deploy_ros2/docs/TROUBLESHOOTING.md` 의 F1·B3·F절이 이 문제다.

| 항목 | 학습 데이터 (SHM 로거) | 실제 배포 (ROS2 토픽) |
|---|---|---|
| 힘 소스 | 127점 촉각 합 `Σ127` | `/paxini/right/ft` 재구성 합력 (4×3) |
| 루프 rate | ~100Hz (FACTOR=10 전제) | 실측 ~17–20Hz |
| 결과 | — | 힘 저평가(~4N), 시퀀스 길이 OOD, 심하면 "샘플 부족 — 추론 불가" |

이 불일치의 원인이 정확히 **"학습은 SHM 파이프라인, 배포는 ROS2 파이프라인"** 이다.
따라서 "수집도 ROS2로" 는 단순 정리가 아니라 **문서에 적힌 성능 문제를 직접 고치는 조치**다. 이것이 아래 모든 결정의 근거다.

---

## 3. 결정 요약

### q1. 통합 여부 → **부분 통합** (런타임 코어는 하나로, 오프라인 자산은 Gen3에)

현재 두 레포에 `shm_common.py` · `paxini_shm.py` · `deploy.py` · `deploy_ros2.py` · `real_deploy_inference_*` · `model.py` · pose txt가 **중복 존재하고 이미 드리프트**했다 (Gen3의 `deploy_ros2`는 `real_deploy_inference_old`를, ROS2 패키지는 `real_deploy_inference_final`을 import). 이 상태로 수집까지 얹으면 소스가 3벌로 갈라진다.

관심사로 나눠서 통합한다:

- **로봇에 올라가는 런타임 코어 = 단일 ROS2 패키지 (single source of truth).**
  `shm_common` · `paxini_shm` · ROS2 브리지 · `deploy.py`의 모션 프리미티브(grip/squeeze) · pose IO.
  → `stiffness_deploy_ros2`를 기준 소스로 삼고, **수집을 이 패키지의 새 엔트리포인트로 추가**.
- **오프라인 자산 = Gen3 유지.** `raw_dataset/`, 학습 스크립트, `split_demos_by_marker`/`export_displacement_force`, 실험용 motion_sequence 변종들 — 로봇에 안 올라가므로 여기 남긴다.

이렇게 하면 수집 엔트리포인트와 배포 엔트리포인트가 **같은 모션 프리미티브 + 같은 브리지를 공유** → q2의 train/deploy 일치가 "노력해서 맞추는 것"이 아니라 **구조적으로 보장**된다.

### q2a. 모션 방식 → **ROS2 토픽 (브리지 재사용)**, SHM 직접 방식 폐기

| | SHM 직접 (`motion_sequence_A_self`) | ROS2 토픽 (`deploy_ros2` 브리지) |
|---|---|---|
| train/deploy 일치 | ❌ 배포와 다른 경로 → 지금의 OOD 원인 | ✅ 배포와 동일 신호·지연·QoS |
| 코드 통합 | 모션 프리미티브 중복 유지 | ✅ deploy와 프리미티브 공유 |
| 검증 상태 | — | ✅ QoS/드리프트 이슈 이미 해결됨 |

전제: **rate 일관성.** 관건은 "100Hz 달성"이 아니라 **"수집·배포 rate를 하나로 고정"** 이다. 수집을 ROS2로 옮기면 "수집도 배포도 같은 rate"가 되어 OOD가 사라진다.

### q2b. 로깅 → **rosbag2(mcap) raw + 버전 고정 변환기 → HDF5**

재현성만 놓고 보면 rosbag이 구조적으로 우위다. 이유는 편의가 아니라 **가역성**이다:

- HDF5-at-capture(현재 로거)는 rate·정렬·필드선택·contact 임계를 **캡처 순간 비가역적으로 확정**한다. 나중에 정렬/rate가 틀렸음을 알아도 **같은 물리 스퀴즈를 다시 수집할 수 없어** 데이터가 죽는다. (지금 F절 OOD 디버깅이 정확히 이 상황.)
- rosbag은 raw를 immutable하게 두고 정렬·다운샘플·정규화를 **버전 고정된 변환기로 지연**한다. 전처리 버그를 나중에 발견해도 같은 bag에서 재생성하면 된다. 게다가 `ros2 bag play`로 **로봇 없이 배포 코드에 재생** → 학습 데이터로 배포를 재현하는 최고 수준의 재현성.

**단, 정직한 반대급부:** rosbag의 재현성은 **버전 고정된 결정론적 변환기**가 있어야 성립한다. 없으면 재현성은 raw 계층에만 있다.

> **핵심 원칙:** 재현성이 실제로 깨지는 곳은 컨테이너(rosbag/HDF5)가 아니라, **버전관리 안 된 전처리 변환**(resample rate, FACTOR=10, 정규화 min/max, force_zero 채널)이다. 어느 포맷을 쓰든 이 변환을 데이터와 함께 freeze/version하지 않으면 재현 안 된다.

**최종 구성:** rosbag(불변 raw) = 취득 재현성 + replay, 버전 고정 변환기 → HDF5 = 학습 재현성. 기존 HDF5 학습 파이프라인은 변환기 출력으로 계승.

부수 정리: 지금 `/tmp/gen3_squeeze_on.txt` 파일 + monotonic 마커로 하던 구간 신호를 **토픽으로 발행**(예: `/collect/squeeze_on`, `/collect/demo_marker`)하면 rosbag이 같이 기록 → 파일 IPC 제거.

---

## 4. 기술적 근거 (전처리·정렬·rate)

### 4.1 센서마다 rate가 다른데 정렬(ZOH/보간)이 학습에 이점인가

정렬은 "이점"이라기보다 대부분 **요구조건**이다. 시퀀스 모델(LSTM/Transformer)은 매 timestep마다 모든 채널이 채워진 **직사각 텐서 (T, F)** 를 요구한다. 비동기 다중 rate에서는 정렬 없이는 애초에 이 텐서를 못 만든다. 질문은 "정렬하면 이득이냐"가 아니라 **"어떤 정렬이 가장 덜 해로운가"** 이다.

제대로 정렬하면 생기는 실제 이점:
1. **배칭 가능한 고정 텐서** → GPU 배치 학습.
2. **일관된 Δt 의미** → "force가 N스텝에 걸쳐 상승" 같은 시간 패턴이 데모 간 같은 의미.
3. **교차모달 동시각 학습** → 같은 t의 tactile↔joint 관계.
4. (이 프로젝트 특화) **고정 rate → 시퀀스 길이 안정** → 지금 OOD의 직접 원인 제거.

### 4.2 ZOH vs 보간 — 이 프로젝트에선 **causal ZOH가 정답**

| | ZOH (마지막 값 유지) | 보간 |
|---|---|---|
| 인과성 | ✅ causal — 과거 값만 사용 | ❌ non-causal — t 값에 t 이후 샘플 필요 |
| 배포 재현성 | ✅ **배포 loop가 하는 것과 정확히 동일** | ❌ 배포는 미래 샘플이 없어 재현 불가 |

배포의 `add_sample`은 매 tick "지금까지 받은 최신 토픽 값"을 읽는다 = **causal ZOH**. 보간으로 학습 데이터를 만들면 모델은 배포가 절대 못 만드는 값을 학습 → **train/deploy 갭을 다시 만든다.** 보간은 오프라인 분석·시각화용으로만.

주의: ZOH는 staircase/중복행을 만든다. 특히 속도/미분 피처를 ZOH 데이터에서 뽑으면 "0→튐→0" 인공 패턴이 생기니 유의.

### 4.3 전처리 rate는 무엇에 맞추나 — **토픽 rate가 아니라 배포 loop rate**

데이터 경로:

```
센서 토픽(paxini ~90Hz, kin ~100Hz)
  → 배포 loop가 매 tick 최신값을 ZOH로 읽음
  → per-tick 시퀀스
  → FACTOR=10 다운샘플
  → 모델
```

모델은 토픽 메시지를 직접 먹지 않고 **loop가 ZOH 샘플한 결과**를 먹는다. 그래서:

```
전처리 격자 rate  ≤  배포 loop rate  ≤  토픽 rate
```

- **"토픽 rate보다 빠르게 리샘플 금지"는 유효한 상한**이다 — 위로 올리면 정보가 안 늘고(정보량 상한 = 소스 rate), ZOH 중복행 + 시퀀스 길이 인플레 + 미분 아티팩트만 생긴다.
- 하지만 **더 tight한 target은 "배포 loop rate와 같게 + causal ZOH"** 이다. loop rate가 토픽보다 낮으면(지금 17~20Hz) 전처리도 그 낮은 rate에 맞춰야 한다.
- 다중 센서면 천장 = **가장 느린 필수 센서(paxini ~90Hz)**. 빠른 센서에 맞춰 올리면 느린 센서가 stale. 지터를 감안해 **peak가 아니라 지속(floor) rate 이하**로.

### 4.4 배포 loop 설계에 따라 "토픽 rate에 맞춘다"가 정답이 될 수도

| 배포 loop 설계 | loop rate | 전처리를 무엇에 맞추나 |
|---|---|---|
| **현재: free-running `read; sleep(0.01)`** | 토픽 rate와 무관, 불안정(17~20Hz) | 먼저 loop rate를 **고정/실측**해야 정의됨 (= F절) |
| **개선: 센서 콜백 구동** (paxini 메시지마다 add_sample) | **paxini 토픽 rate로 고정**(안정) | 이 경우 **"토픽 rate에 맞춘다"가 정답** + 지터 소멸 |

### 4.5 rate만으론 부족 — 정책도 같아야

같은 rate라도 정책이 다르면 다른 시퀀스가 나온다. **causal ZOH · 같은 tick 기준 · 같은 FACTOR/평균창**을 세트로 맞춘다.

### 4.6 함정: 레거시 학습 rate 맞추려 업샘플 금지

예전 "1kHz/100Hz" 숫자에 맞추려 90Hz 수집을 업샘플하지 말 것(아티팩트 + OOD 재도입). achievable rate(≈paxini)로 **재수집·재학습**하거나 `FACTOR=round(실측rate/10)`로 시퀀스 결과를 맞춘다. (참고: 예전 로거의 "1kHz"도 실제론 ~150Hz 가변이라 그 목표 자체가 허수였다.) 안전한 방향은 **업샘플이 아니라 다운샘플**이고, FACTOR=10 **평균**은 anti-alias LPF 역할도 겸한다.

### 4.7 채널별 실측 특성 (2026-07-28) — ★ 학습에서 주의할 것

`joint` + `kin` + `paxini` **세 채널을 모두 수집·학습**하기로 결정한 뒤 실측한 값이다.
(도구: `tools/sensor_update_rate.py`, 손 모션 중 14.5s. 자세한 방법론은
[UPDATE_RATE_CONCLUSION.md](UPDATE_RATE_CONCLUSION.md) — "발행률(통신) vs 값 변화율(센서)")

| 채널 | 발행률 | **실제 갱신** | 중복 | 양자화 | 채널 std 범위 | header stamp |
|---|---|---|---|---|---|---|
| **kin** (12) | 200.0Hz | **199.99Hz** | **0%** | 1.0 | **5.6 ~ 1300** | 없음 |
| **joint** (16) | 200.0Hz | **200.00Hz** | **0%** | 1.0 | 10.8 ~ 79.7 | **있음**(200Hz) |
| **paxini `/raw`** (1524) | 89.5Hz | 43.1(전체) / **89.8(활성)** | **51.8%** | 0.1 | 정적 622/1524 | 없음 |

**세 채널 모두 LIVE 확인.** 이전에 `kin` 이 전 채널 0(FROZEN)으로 보였던 것은 **손이 idle**
이던 시점의 관측이며, 손이 움직이면 200Hz 로 전 12채널이 갱신된다(중복 0%).

> ⚠ §4.3 의 "kin ~100Hz" 는 실측 **200Hz** 로, §4.3/4.4 의 "loop 17~20Hz" 는 실측
> **97.9Hz** 로 갱신됐다(TROUBLESHOOTING §F5 에서 이미 반증). 아래 여유 계산이 최신이다.

```
kin/joint 200Hz ─┐
                 ├→ 수집 루프 97.9Hz ─→ FACTOR=10 ─→ 모델 입력 ≈10Hz
paxini    83.3Hz ┘                                   ↑ 세 채널 모두 8배 이상 여유
```

⇒ **어느 채널도 모델 입력 rate 의 병목이 아니다.** 단 시간해상도 상한은 **가장 느린
paxini(센서 스펙 83.3Hz)** 가 정한다(§4.3 의 "천장 = 가장 느린 필수 센서" 원칙 그대로).
100Hz tick 에서 paxini 는 약 1.2 tick 마다 갱신되어 ZOH 중복이 섞이지만, FACTOR=10 으로
10Hz 로 줄이면 그 중복은 사라진다.

#### ★ 학습에서 주의할 것

**① `kin` 은 채널별 스케일 편차가 3자리 — 채널별 정규화가 필수**

채널 std: `5.6 / 1153 / 820 / 2.8 / 257 / 78 / 15 / 1300 / 418 / 8.6 / 384 / 122`.
채널 0·3·9 는 std 3~9 인데 채널 1·7 은 1200~1300 이다. **전체 표준편차 하나로 정규화하면
작은 채널이 소실**된다. `joint`(10~80)·`paxini`(진폭 3) 와 concat 할 때도 스케일 통일 필요.

**② `paxini` 127점은 본질적으로 희소 — 모델 설계에 반영**

| 상태 | 값이 있는 point |
|---|---|
| 이번 측정(약접촉) | 정적 **622/1524 채널**(41%) |
| 실제 수집(접촉 중) | 508점 중 **62점만 활성**(≈12%) — thumb 27 / f1 0 / f2 24 / f3 11 |

접촉면만 켜지므로 정상이지만, 127점 입력 모델은 **이 희소성을 전제**로 설계해야 한다
(접촉 point 마스킹 / pooling). 합력(Σ127)을 **함께** 쓰는 편이 안전한 이유다.

**③ `kin` 소스가 조용히 갈린다 — 가장 큰 리스크**

```python
# real_deploy_inference_final.read_live_sample()
ft = mN파일(/tmp/deep_ws_raw_06_hand_j_kin_mN.txt) 있으면 그 값,  없으면 SHM raw(j_kin)
```

즉 HDF5 의 **`ft` 데이터셋 = kin(j_kin)** 이고, 그 값의 **스케일이 파일 존재 여부로 통째로
바뀐다.** 현재는 mN 파일이 없어 SHM raw 로 수집된다(root attr `raw_hand_j_kin_mN_present=0`).
새로 수집·학습하면 self-consistent 하지만, **나중에 누가 side-channel 을 켜면 배포가 조용히
다른 스케일을 읽어** 추론이 망가진다(§B2 가 "학습과 불일치" 로 경고한 지점).

→ 대응: 소스를 명시적으로 고정하거나, **배포 시 root attr 의 `raw_hand_j_kin_mN_present`
와 실제 소스를 대조해 불일치면 거부**. 지금은 provenance 만 기록되고 강제는 없다.

**④ 정렬 기준: `joint_states` 만 header stamp 를 갖는다**

`kin`·`paxini` 는 timestamp/seq 가 없어 bag 변환 시 정렬 기준이 **q_target tick 뿐**이다.
이것이 parity 정확도의 구조적 한계다(아래).

#### 수집 커맨드와 저장 채널 (추가 코드 불필요)

```bash
python3 stiffness_deploy_ros2/launch/collect_ros2.py --fruit tomato --num-demos N --paxini raw
```

| HDF5 채널 | shape | 정체 | A↔B parity |
|---|---|---|---|
| `joint` | (n, 16) | 손 관절 | 6 counts 차 (시점차 의심) |
| `ft` | (n, 12) | **kin (j_kin)** | 72 차 (시점차 의심) |
| `resultant` | (n, 4, 3) | **Σ127 합력** | **0** ✅ |
| `tactile` | (n, 4, 127, 3) | **127점 원본** | **0** ✅ |

`USE_TACTILE=False` 는 *추론 엔진 버퍼*에만 적용되고 **HDF5 에는 127점이 그대로 저장**된다.
`tactile` 이 있으면 `resultant` 는 재계산 가능하지만 역은 불가 → **둘 다 보관**.
성능 대가 없음: 루프 97.9Hz 유지, bag 세션당 ~24MB(0.75MB/s), 기록 손실 0.3% 이하.

#### parity 현황 (P5) — 남은 과제

- `resultant`·`tactile` **오차 0** → 힘/촉각 경로는 수집=배포가 텐서 수준에서 입증됨.
- `joint`(6) · `ft`=kin(72) 는 **출처가 아니라 샘플 시점 차이**로 보인다
  (A=수집 루프 tick, B=`hand_qtar` tick). 확정하려면 시각 정렬이 필요한데,
  **A 는 `time.monotonic_ns`, B(bag)는 epoch 를 같은 `t_mono_ns` 이름으로 저장**해
  절대시각 짝지음이 불가능하다.
  → 조치: `bag_to_hdf5` 는 epoch 를 `t_ns` 로 이름 분리, `collect_ros2` 는
    monotonic↔epoch 오프셋을 root attr 로 기록.

---

## 5. "둘 다 사용"이 만드는 전제조건 (촉각 소스 = `/raw`)

모델이 **resultant(4×3) + 127점 tactile 둘 다** 소비할 가능성이 높다. 이 답이 상위 제약을 만든다.

- 현재 배포 모델은 `USE_TACTILE=False` — resultant만, 힘 소스는 `/paxini/right/ft`(4×3).
- 학습/수집은 진짜 127점 합 `Σ127`. 이 `ft(4×3) ≠ Σ127` 불일치가 배포 힘 저평가의 유력 원인이고, **이미 준비된 해법이 `deploy_ros2_exp_rawft.py`** — 브리지를 `/paxini/right/ft` → **`/paxini/right/raw`(4×127×3)** 구독으로 바꿔 학습과 동일한 Σ127을 계산.

> ⇒ 모델이 127점을 쓰면, 수집·배포의 촉각 소스는 **둘 다 `/paxini/right/raw`(4×127×3)** 여야 하고, 이는 **제어 PC `shm_state_publisher`가 `/raw`를 1급 토픽으로 발행**해야 성립한다.
>
> 핵심: **ROS2 수집을 `/raw`에서 하는 것 = 배포 힘 갭(H1)을 고치는 것 = 같은 인프라 변경.** 별개 작업이 아니다.

---

## 6. 실행 계획 (Phase 0 → 6)

원칙: **측정 안 된 가정 위에 코드를 쌓지 않는다.** 빌드보다 진단이 먼저.

| 단계 | 무엇을 | 도구(대부분 존재) | 통과 게이트 |
|---|---|---|---|
| **P0. 진단/측정** | ① 토픽 실측 rate ② `/paxini/right/raw` 발행 여부 ③ **ft vs Σ127 실측(H1/H2 판별)** ④ 배포 loop 실제 rate·샘플수 ⑤ 모델 입력 계약(USE_TACTILE/USE_JKIN/FACTOR/force_zero) | `ros2 topic hz`, `deploy_ros2_exp.py`, `deploy_ros2_exp_rawft.py`, `paxini_writer.py`, 체크포인트 확인 | rate·/raw유무·힘경로진단·loop rate·모델계약이 **숫자로 확정** |
| **P1. 계약 확정(ADR)** | 촉각 소스=`/paxini/right/raw`, target rate+정책(causal ZOH@loop rate, ≤paxini floor), 기록 토픽 세트, loop 설계(free-run vs 센서콜백) | 문서 | 한 장짜리 topic/rate/alignment 계약서 |
| **P2. 코드 통합** | 중복·드리프트된 런타임 코드(shm_common·paxini·bridge·deploy 모션프리미티브·pose IO)를 **ROS2 패키지 1벌로**. `real_deploy_inference_*` 갈래 정리. bridge가 `/raw` 구독 가능하게(rawft 브리지 병합) | 통합 리팩터 | 수집·배포가 **같은 bridge·같은 모션 소스** 참조 |
| **P3. 수집 엔트리포인트** | `collect_ros2` 신규 — deploy와 **동일 bridge+동일 grip/squeeze**로 다중 데모 자동 루프. squeeze_on/demo 마커를 **토픽으로 발행**(/tmp 파일 제거) | 신규(모션 재사용) | 모션이 배포와 바이트 수준으로 동일 |
| **P4. 로깅+변환기** | rosbag2(mcap)로 계약 토픽 raw 기록 + **버전 고정 결정론적 변환기**(배포 causal ZOH를 loop rate로 재생 → 기존 HDF5 스키마). provenance(git SHA·QoS·rate·FACTOR·norm) freeze | 신규 | bag→HDF5 재실행 시 동일 산출 |
| **P5. parity 검증 ★** | 같은 수집 세션을 ① 변환기로 텐서화, ② `ros2 bag play`로 배포 add_sample이 만든 텐서 — **둘이 일치**하는지. 신규 ROS2 vs 구 SHM 데이터의 힘·시퀀스길이 분포 비교로 **갭 정량화** | 신규 스크립트 | 두 텐서 ≈ 동일, OOD 해소 확인 |
| **P6. 스케일 수집 + Gen3 정리** | parity 확인 후 실제 데이터셋(과일×반복) 수집. Gen3는 오프라인(학습/분석) 전용으로, SHM 수집 경로는 legacy로 | — | 데이터셋 확보, 관심사 분리 완료 |

### 왜 이 순서인가 (의존성)

- **P0가 P1을 결정한다.** ft≪Σ127(H1)이면 `/raw` 구독만으로 힘 갭이 풀리고 재학습 불필요 → 계약이 간단. ft≈Σ127(H2)이면 모션/제어 경로 문제라 수집 코드로는 못 고치고 별도 트랙. **이 판별 없이 P2 이후를 짜면 헛수고 위험.**
- **P2가 P3의 전제다.** 코드가 1벌이어야 "수집이 배포를 재사용"이 성립한다.
- **P5가 진짜 목표의 증명이다.** "수집=배포"를 말이 아니라 **텐서 일치로 증명**하는 게 이 프로젝트의 핵심 산출물.

---

## 7. 외부 의존(막혀 있는) 결정

1. ~~**`/paxini/right/raw`(4×127×3)를 제어 PC `shm_state_publisher`가 상시 발행할 수 있는가?**~~
   → **해소 (2026-07-28 실측).** `/paxini/right/raw`(1524ch = 4×127×3) 상시 발행 확인,
   `--paxini raw` 로 수집·bag 기록까지 동작(bag 2877 msg, 손실 0.3%), HDF5 에 127점
   `tactile (n,4,127,3)` 저장 + `resultant = Σ127` 일치(A↔B parity 오차 0). §4.7 참고.
   **협의 불필요.**
2. **배포 loop rate를 고정할 수 있는가 / 센서콜백 구동으로 바꿀 것인가.**
   target rate가 여기서 정해진다 (§4.4, F절).
   → 부분 해소: 실측 loop **97.9Hz(3회, 편차 0.1)** 로 이미 안정적이며 학습 100Hz 에 정합
   (기존 "17~20Hz" 가정은 반증). 센서콜백 구동 전환은 여전히 열린 설계 선택.

---

## 8. 하드웨어 유무에 따른 착수점

- **로봇/제어 스택이 떠 있으면** → 바로 **P0 실측**(도구 다 있음). 가장 값진 첫 일.
- **로봇이 없으면** → **P0-⑤(모델 입력 계약 코드 확인)** + **P2 통합 설계**는 오프라인 선행 가능. 나머지 측정은 하드웨어 대기.

---

## 9. 실행 위치 & 워크스페이스 실행 가능성

**결론: `stiffness_deploy_ros2` 워크스페이스 하나로 P0~P5(수집·검증 전 과정)가 닫힌다. Gen3는 런타임에 필요 없다.**

근거 (2026-07-26 확인):
- `stiffness_deploy_ros2/env.sh`가 명시 — Dual_Arm_Hand_Ctrl도 Gen3도 source하지 않고, 같은 `ROS_DOMAIN_ID`(=9)에서 DDS 토픽만 주고받으면 됨. 표준 msg(std/sensor_msgs)만 사용.
- `stiffness_deploy_ros2/stiffness_deploy_ros2/core/`에 `shm_common.py`·`paxini_shm.py` 자체 사본 존재. `deploy.py`의 `from core.*`는 이 패키지 root로 해석됨(`sys.path`에 패키지 root 삽입). 코드 내 `# Gen3`·`/home/prime/...` 문자열은 전부 **주석·docstring일 뿐 실제 import 아님**.
- ⇒ 계획서 §3 q1의 "런타임 코어 = 단일 ROS2 패키지"가 **이 워크스페이스에 사실상 이미 구현돼 있음**. 여기서 수집을 만들면 Gen3 사본이 자동으로 legacy가 됨.

| 단계 | 이 워크스페이스에서? | 상태 |
|---|---|---|
| P0 진단 | ✅ 도구 존재 (`deploy_ros2_exp.py`, `deploy_ros2_exp_rawft.py`, `deploy_ros2_exp_ftcheck.py`) | 로봇/제어스택 up이면 바로 실행 |
| P2 코드 통합 | ✅ **이미 self-contained** | 거의 완료 — 정리만 |
| P3 collect_ros2 | ✅ `launch/`에 추가 + `setup.py` entry_points 한 줄 | net-new (모션 재사용) |
| P4 rosbag+변환기 | ✅ 여기 신규 | **로깅 코드 전무** — 순수 net-new |
| P5 parity 검증 | ✅ 여기 신규 스크립트 | net-new |
| P6 학습/분석 | ⚠️ Gen3(오프라인) | 로봇 불필요, 계획대로 Gen3에 잔류 |

**코드가 아닌 "런타임 전제" 2가지 (워크스페이스 밖, 떠 있어야 하는 프로세스):**
1. 제어 PC 스택(C++ 컨트롤러 + `shm_state_publisher` + arm/hand receiver + `paxini_writer`)이 토픽 발행 중.
2. **★ 결정적 게이트: `/paxini/right/raw`(4×127×3)를 제어 PC가 발행하는가.** 127점 tactile은 이 워크스페이스에서 만들 수 없고 제어 PC `shm_state_publisher`가 발행해야 함 (`deploy_ros2_exp_rawft.py`가 이 전제를 명시). 미발행이면 수집이 `/paxini/right/ft`(4×3)로 제한 → 제어 PC 담당자 협의가 P1 게이트.

---

## 10. 현재 상태 & 다음 단계 (handoff)

> 다른 컴퓨터/새 세션에서 이어갈 때 여기부터 읽으면 됨.

**현재 상태 (2026-07-27):**
- 설계·결정 **확정** (§3). 기술 근거 정리 완료 (§4~5).
- **수집 파이프라인 + 9단계 시퀀스 + 부가기능 코드 구현 완료** — `recording_engine.py`·`collect_ros2.py`(9단계 시퀀스: 팔=MoveIt 이동, 손=파지/스퀴즈)·`moveit_arm_mover.py`(dex_ros MoveIt 충돌회피 팔 이동, Option B)·`bag_to_hdf5.py`(구간별 추출, palm-down 포함)·`verify_parity.py`. 부가: 구간별 flag(`/collect/segment`)·힘 임계 랜덤화·데모 성공 판정(outcome)·그룹명 통일(`{segment}__run`). 상세·사용법은 **`docs/DATA_COLLECTION_HANDOFF.md` §6**. **남음: `ARM_POSES` 실제 값 입력 + 로봇 실측 end-to-end(+P0 `/raw`·move_group 게이트).**
- `stiffness_deploy_ros2` 워크스페이스가 **self-contained임을 확인** (§9). 실행·구현은 전부 거기서.
- 빠른 실행 요약본: `docs/DATA_COLLECTION_HANDOFF.md` (같은 폴더). 이 문서(정본)와 요약본 둘 다 이 워크스페이스에 있어 **Gen3 없이 자기완결**.

**즉시 다음 액션 = P0의 `/raw` 확인** (이 한 번이 "이 워크스페이스만으로 되는가"를 가름):

```bash
cd ~/stiffness_deploy_ros2 && source env.sh
ros2 topic list | grep -E 'paxini|hand|franka'          # /paxini/right/raw 존재?
ros2 topic hz /paxini/right/raw                          # 있으면 rate 확인
python3 stiffness_deploy_ros2/launch/deploy_ros2_exp_rawft.py   # 있으면 Σ127 실측
python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py        # 배포 loop rate·샘플수 실측
```

- `/raw` **보이면** → P0 나머지(loop rate·ft vs Σ127 판별) 실측 후 P3/P4 구현 착수.
- `/raw` **안 보이면** → 유일한 실질 blocker. 제어 PC에 `/raw` 발행 요청 전까지 4×3까지만 수집 가능.

**차단 요인:** ① `/paxini/right/raw` 발행 여부(외부) ② 로봇/제어 스택 가동 여부.

---

## 부록 A. 참조 파일

경로는 이 레포 루트 기준. `.../` 없이 실제 경로.

| 역할 | 경로 |
|---|---|
| 전송 어댑터 인터페이스 | `stiffness_deploy_ros2/core/shm_common.py` (`ShmAccess`) |
| PaXini SHM 리더 | `stiffness_deploy_ros2/core/paxini_shm.py` |
| ROS2 브리지 (기준) | `stiffness_deploy_ros2/launch/deploy_ros2.py` (`Ros2ShmBridge`, `Ros2PaxiniBridge`) |
| `/raw` 구독 브리지 (병합 대상) | `stiffness_deploy_ros2/launch/deploy_ros2_exp_rawft.py` |
| loop rate 계측 | `stiffness_deploy_ros2/launch/deploy_ros2_exp.py` |
| ft vs Σ127 판별 | `stiffness_deploy_ros2/launch/deploy_ros2_exp_ftcheck.py` |
| 모션 프리미티브 | `stiffness_deploy_ros2/launch/deploy.py` (`move_hand_to`, `..._until_force`, `..._squeeze`) |
| 추론 엔진(입력 계약) | `stiffness_deploy_ros2/launch/real_deploy_inference_final.py` |
| 배포 문제해결·개발노트 | `docs/TROUBLESHOOTING.md` (F1·B3·F·H1/H2) |
| (참고, **이 워크스페이스에 없음 · Gen3 전용**) 기존 SHM 로거 | `Gen3/core/hdf5_logger_1k.py` — P4 변환기의 HDF5 스키마 참고용 |
| (참고, **Gen3 전용**) 마커 기반 데모 분할 | `Gen3/core/split_demos_by_marker.py` |
| (참고, **Gen3 전용**) 기존 SHM 모션 시퀀스 | `Gen3/launch/motion_sequence_A_self.py` |

## 부록 B. 용어

- **ZOH (Zero-Order Hold)**: 새 샘플이 올 때까지 직전 값을 유지하는 보간. causal(과거만 사용).
- **causal**: 시각 t의 값을 t 이전 정보만으로 계산. 실시간 배포가 할 수 있는 유일한 방식.
- **Σ127**: 손가락별 127개 촉각 포인트의 합(진짜 합력). 학습/수집의 힘 정의.
- **OOD (Out-Of-Distribution)**: 배포 입력이 학습 분포를 벗어난 상태. rate/시퀀스길이/힘 스케일 불일치로 발생.
- **FACTOR**: 추론 엔진의 다운샘플 계수(=10). 학습이 100Hz 전제라 유효 시퀀스 길이를 좌우.
- **provenance**: 데이터 재현에 필요한 이력(코드 git SHA, QoS, 실측 rate, FACTOR, 정규화 통계 등).
