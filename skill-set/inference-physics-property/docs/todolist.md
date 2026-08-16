# TODO — 남은 과제 (이어서 작업용)

> 2026-07-28 기준. 각 항목은 **배경(왜) → 무엇을(어디를) → 완료 판정** 순으로 적었다.
> 세션이 끊겨도 이 문서만 보고 이어갈 수 있게 파일·라인·실측 근거를 함께 남긴다.
> 관련 문서: [ros2_data_collection_plan.md](ros2_data_collection_plan.md) ·
> [UPDATE_RATE_CONCLUSION.md](UPDATE_RATE_CONCLUSION.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 한눈에 보기

| # | 항목 | 우선 | 규모 | 로봇 필요 | 상태 |
|---|---|---|---|---|---|
| 1 | parity 시계 기준 통일 (`t_mono_ns` monotonic/epoch 혼용) | **P1** | 소 | ✕ | ✅ **완료·검증됨** (2026-07-28) |
| 2 | `kin` 소스 고정 (mN 파일 유무로 스케일 갈림) | **P1** | 소 | ✕ | ✅ **(a)+(b) 완료·검증됨** (2026-07-28) |
| 3 | 파이프라인 완주 검증 (팔 이동 성공 세션으로 3~5단계) | **P2** | 중 | ○ | 🔶 **실전 검증됨**(마커 수정 동작·run=0) · `outcomes.json` 만 남음 |
| 4 | `os._exit` stdout flush 유실 (5개 파일) | P2 | 소 | ✕ | ✅ **완료·검증됨** (2026-07-28) |
| 5 | `deploy_ros2_exp_forcecurl.py` import 순서 버그 | P2 | 소 | ✕ | ✅ **완료·검증됨** · 루프 98Hz 실측은 불필요(아래) |
| 6 | `palm_down` 전용 자세 캡처 (현재 safe 재사용) | P3 | 소 | ○ | ⏸ **보류 — 현행 유지**(2026-07-28 결정) |
| 7 | §F7 힘 경로 결론 반영 (ft vs Σ127 실측 확보됨) | P3 | 소 | ✕ | ✅ 완료 (H1 확정) |
| 8 | bag 기록 토픽 정리 (`/paxini/right/ft` 중복) | P3 | 소 | ✕ | ✅ 검증완료 · **제거 비권고**(절감 1.2%뿐) → 미적용 |
| 9 | parity 판정 허용오차가 절대값(`tol=0.001`) — raw count 채널에 부적합 | P3 | 소 | ✕ | ✅ **완료·검증됨** · `step_k` 2.0 재교정(실전 오탐 해소) |
| 10 | `bag_to_hdf5 --paxini ft` 가 소스 부재 시 조용히 0프레임 | P3 | 소 | ✕ | ✅ **완료·검증됨** (2026-07-28) |
| 11 | 저장 포맷 개편 — session.h5 연속 타임라인(중복 h5 2개 정리) | **P1** | 중 | ✕ | ✅ **완료** — 산출물 3개 확정(session.h5+bag+outcomes.json) |
| 12 | paxini 발행 공백(11.12ms 주기에 20~33ms 구멍) | P3 | 소 | ○ | ✅ **조사 불필요** — 스퀴즈 구간 영향 0%, 자동 감시로 대체 |

---

## 2026-07-28 2차 검증 — 요약

로봇 없이 검증 가능한 항목을 전부 처리했다. **상세 결과·근거는 각 항목 본문**에 있다.

| # | 결과 | 한 줄 |
|---|---|---|
| 9 | ✅ 수정+검증 | 스텝 정규화 판정 도입 → 기존 세션 PASS, 인위 교란 **4/4 FAIL 검지** |
| 4 | ✅ 검증 | `os._exit` 직전 flush 5/5, 파이프·tee 로 출력 유실 없음 |
| 5 | ✅ 검증 | import 순서 수정본이 `real_deploy_inference_final`(USE_JKIN=False) 로드 확인 |
| 3 | 🔴→🔧 | **마커 유실로 `--skip-outcomes` 가 무력화되던 버그 발견·수정**, outcome 경로는 정상 |
| 8 | ✅ 검증 | `ft` 제거는 안전하지만 절감 **1.2% 뿐** → **제거 비권고, 미적용** |
| 2 | ✅ (b) 완료 | 체크포인트 ↔ 배포 kin 소스 provenance 대조 추가(불일치 시 실행 거부) |
| 10 | ✅ 수정+검증 | `--paxini` 소스가 bag 에 없으면 즉시 거부(8케이스 확인). `auto` 폴백은 유지 |

검증 재현: `tools/check_jkin_source_pin.py`(2번) · `tools/check_grip_retry.py`(파지 게이트) ·
`tools/check_parity_timing.py <세션>`(1·9번 불일치를 bag 원본과 bit-exact 대조).

**로봇이 필요한 미결 항목은 없다.** #3 실제 완주는 224840·231155 세션으로 끝났고,
#5 루프 98Hz·#12 paxini 공백은 각각 '불필요' 로 결론냈다(각 항목 참조).
#6(`palm_down` 전용 자세)만 사용자 결정으로 보류 상태다.

---

## 1. parity 시계 기준 통일 — **P1**

### 배경
`verify_parity.py` 가 A(라이브)↔B(bag변환) 를 대조할 때 `joint`(6 counts)·`ft`=kin(72) 차이가
남는데, **이것이 실제 데이터 불일치인지 단순 샘플 시점 차이인지 아직 확정하지 못했다.**
시각으로 짝지어 비교하면 판정되지만, 두 파일이 **같은 이름 `t_mono_ns` 에 다른 시계**를 쓴다:

| 파일 | 값(예) | 시계 |
|---|---|---|
| A (라이브) | 562,100,654,829,586 (≈562,101s) | `time.monotonic_ns()` (부팅 이후) |
| B (bag)   | 1,785,164,064,658,799,918 (≈1.785e9 s) | **epoch** (bag 메시지 시각) |

차이 ≈ 1,784,601,964초 → 절대시각 짝지음이 원리적으로 불가.
`verify_parity.py` 에 시각정렬(`align_by_time`)은 **이미 구현**돼 있고 시계 불일치를 감지해
`※A=monotonic/B=epoch 로 시각정렬 불가` 로 리포트한다 — **이름·오프셋만 맞추면 바로 동작한다.**

### 무엇을
- `bag_to_hdf5.py:249` — `out["t_mono_ns"] = ...` 가 **epoch** 를 담고 있다.
  → `t_ns`(epoch) 로 이름 분리. (`bag_to_hdf5.py:321` 의 `len(data["t_mono_ns"])` 도 함께)
- `recording_engine.py:103` — 라이브는 monotonic 그대로 유지(`t_mono_ns` 의미 정확).
- `collect_ros2.py` — root attr 에 **monotonic↔epoch 오프셋**을 기록
  (예: `t_offset_ns = time.time_ns() - time.monotonic_ns()` 를 세션 시작 시 1회).
- `verify_parity.py` — B 의 `t_ns` + A 의 `t_mono_ns` + 오프셋으로 짝지음(또는 A 를 epoch 로 환산).

### 완료 판정
`verify_parity.py` 출력의 note 가 `t정렬 N/150쌍 Δt≤Xms` 로 뜨고, `joint`/`ft` 오차가
- **≈0** → 시점 차이였음(수집=bag 재현 성립) 또는
- **여전히 큼** → 실제 불일치. 그때 원인(출처·필터)을 별도 추적.

### ✅ 검증 결과 — 2026-07-28 (로봇 불필요, 기존 수집 세션으로 재현)

검증 대상: `collect_logs/collect_tomato_20260728_160837/` (라이브 h5 79KB + bag 25MB).
**로봇 없이 검증 가능** — parity 는 기록·변환 경로만 쓰므로 수집된 세션 하나만 있으면 재현된다.

```bash
cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
S=collect_logs/collect_tomato_20260728_160837
python3 stiffness_deploy_ros2/launch/bag_to_hdf5.py "$S/bag"
python3 stiffness_deploy_ros2/launch/verify_parity.py "$S"/collect_*.h5 "$S/from_bag.h5"
```

**시계 통일은 동작한다 (= 완료 판정 충족).** 라이브 root attr `t_offset_ns=1784601963993731707`
이 기록되고 parity 가 이를 써서 환산·짝지음에 성공:

```
ℹ A t_offset_ns=1784601963993731584 로 monotonic→epoch 환산 후 시각정렬
squeeze_A__run000  101  251  150  lag 0   joint 5  ft 119  resul 0  tacti 0  FAIL  t정렬 101/101쌍 Δt≤0.93ms
squeeze_B__run000  101  251  150  lag 0   joint 0  ft   0  resul 0  tacti 0  PASS  t정렬 101/101쌍 Δt≤0.59ms
```
(`※A=monotonic/B=epoch 로 시각정렬 불가` 메시지는 더 이상 뜨지 않는다. 짝지음률 101/101, Δt<1ms.)

**남은 `joint`/`ft` 차이는 시점 차이 — 실제 불일치가 아니다.** 프레임 단위로 추적한 근거:

| 관측 | 값 | 해석 |
|---|---|---|
| 불일치 프레임 수 | **2 / 101** (index 0, 63) | 나머지 99 프레임은 **bit-exact 일치** → 출처·스케일 동일 |
| 불일치 크기 | joint 5 counts / ft 119 | — |
| B 프레임간 변화 **중앙값** | joint **4** / ft **116** | 불일치가 **1 샘플 스텝 이내** = 샘플 시점 차이 |
| 최근접 매칭 Δt | +0.11ms / −0.23ms | 정렬은 정확. hand 스트림이 ~198Hz(5ms)라 tick 경계에서 다른 메시지가 선택됨 |
| `resultant`/`tactile` | 0 / 101 | paxini(90Hz)는 tick 대비 느려 경계에 덜 걸림 |
| `squeeze_B` 전 채널 | 0 / 101 | 같은 코드·같은 run 에서 완전 일치 → 구조적 결함 아님 |

→ **결론: 수집(A) = bag 재현(B) 성립.** 원인은 리포트의 후보 (1) `q_target tick ≠ add_sample tick 미세차`.
출처·필터·스케일 문제(후보 3·4)는 배제됨(99/101 bit-exact + squeeze_B 완전일치).

부수 확인(같은 실행에서 얻음): bag 변환이 `3 그룹 / 1061 frames`, `arm_q_target=1748`,
`move_palm_down__run000` 559 frames → **팔 이동이 처음으로 성공했고 ★구간 3개가 모두 채워졌다**(→ 3번 과제).
새 파지 게이트도 실측 통과: `grip_reached_fingers=3`, `grip_peak_force_n=15.4N`(임계 10.7N).

---

## 2. `kin` 소스 고정 — **P1** (학습 리스크 중 가장 큼)

### 배경
HDF5 의 **`ft` 데이터셋 = kin(j_kin)** 이고, 그 값의 **스케일이 파일 존재 여부로 통째로 바뀐다**:

```python
# real_deploy_inference_final.py:265, read_live_sample()
RAW_HAND_J_KIN_FILE = Path("/tmp/deep_ws_raw_06_hand_j_kin_mN.txt")
ft = mN파일 있으면 그 값(mN 보정),  없으면 SHM raw(j_kin)
```

현재 mN 파일이 **없어서 SHM raw** 로 수집된다(root attr `raw_hand_j_kin_mN_present=0`).
새로 수집·학습하면 self-consistent 하지만, **나중에 누가 side-channel 을 켜면 배포가 조용히
다른 스케일을 읽어** 추론이 망가진다. `USE_JKIN` 이 True 일 때만 경고가 뜨므로
(§B2) kin 을 학습에 쓰는 새 구성에서는 **경고조차 안 뜰 수 있다.**

### 무엇을
둘 중 하나(또는 둘 다):
- **(a) 소스 명시 고정** — `RAW_HAND_J_KIN_FILE` 를 설정값으로 올리고 기본 비활성,
  수집·배포가 같은 값을 쓰도록 강제.
- **(b) provenance 대조 거부** — 배포 시작 시 체크포인트/데이터셋의
  `raw_hand_j_kin_mN_present` 와 **현재 소스**를 비교해 불일치면 **즉시 종료**.
  (지금은 기록만 하고 강제가 없다)

### 완료 판정
mN 파일을 만들어 두고 배포를 띄웠을 때, 수집 당시와 소스가 다르면 **실행이 거부**되거나
같은 소스로 강제되는 것을 확인.

### ✅ 검증 결과 — 2026-07-28 (로봇 불필요, (a) 채택)

구현은 **(a) 소스 명시 고정**: `USE_MN_SIDE_CHANNEL` 환경변수(기본 `0`)가 소스를 결정하고
(`real_deploy_inference_final.py:270`), `assert_jkin_source_pinned()`(274행)이 시작 시 검증,
`read_live_sample()`(314행)은 **스위치 OFF 면 파일을 아예 읽지 않는다**.

검증 방법: 로봇/SHM 없이 `read_live_sample` 에 SHM raw=111 / mN 파일=999 인 스텁을 넣어
**실제 데이터 경로까지** 확인. 환경변수만 바꿔 6케이스 하위 프로세스 실행.

```bash
cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
python3 tools/check_jkin_source_pin.py      # 6케이스 자동 실행 (아래 표와 같아야 함)
```

| # | 스위치 | mN 파일 | 가드 | `ft` 실제 출처 | 판정 |
|---|---|---|---|---|---|
| 1 | OFF | 없음 | PASS | SHM_raw | 수집 당시(`present=0`)와 일치 ✔ |
| 2 | OFF | **정상** | PASS + 경고 | **SHM_raw** | ★조용한 스케일 변경 차단 확인 ✔ |
| 3 | ON | 없음 | **REFUSED** | — | 학습과 다른 스케일 폴백 금지 ✔ |
| 4 | ON | 만료(5s 전) | **REFUSED** | — | `SIDE_CHANNEL_MAX_AGE_SEC=1.0` 동작 ✔ |
| 5 | ON | 형식오류(12개 아님) | **REFUSED** | — | 파싱 실패도 거부 ✔ |
| 6 | ON | 정상 | PASS | mN_file | 수집이 mN 였던 경우 정상 경로 ✔ |

**2번이 이 과제의 핵심** — "나중에 누가 side-channel 을 켜면 배포가 조용히 다른 스케일을 읽는다"는
원래 위험이 닫혔다(파일이 생겨도 경고만 뜨고 SHM raw 유지).

### ✅ (b) provenance 대조 — 구현·검증 완료 (2026-07-28)

**왜 (a) 만으로는 부족했나**: (a) 는 **배포 쪽 소스를 고정**할 뿐 **수집 당시 소스와 대조하지
않으므로**, 반대 방향 불일치가 조용히 통과했다 —— 어떤 데이터셋을 `USE_MN_SIDE_CHANNEL=1` 로
수집(`raw_hand_j_kin_mN_present=1`)·학습한 뒤 배포를 기본값(OFF)으로 띄우면 가드는 PASS 인데
배포는 SHM raw → **학습과 다른 스케일**. 현 데이터셋(`present=0`)에서는 무해하지만
**mN 로 수집하는 순간 위험해진다.**

**구현** ([real_deploy_inference_final.py](../stiffness_deploy_ros2/launch/real_deploy_inference_final.py)):
- `load_model()` 의 `meta` 에 `jkin_source` / `raw_hand_j_kin_mN_present` 추가
  (`use_joint_delta` 등 기존 입력 규약과 같은 자리 — 체크포인트가 이미 쓰던 방식).
- `current_jkin_source()` / `ckpt_jkin_source(meta)` / `assert_jkin_source_matches_ckpt(meta)` 추가.
- 엔진 `__init__` 에서 체크포인트 로드 **후** 호출(기존 `assert_jkin_source_pinned()` 는
  '배포 쪽 고정'만, 이쪽은 '학습과의 일치'를 본다).
- **라벨 없는 기존 체크포인트는 `SHM_raw` 로 가정** → 기본(OFF) 배포는 무영향, 스위치 ON 만 거부.

**검증** (`python3 tools/check_jkin_source_pin.py` 의 `PROV` 줄):

| 배포 스위치 | 체크포인트 라벨 | 결과 |
|---|---|---|
| OFF | 없음 / `SHM_raw` / `mN_present=0` | PASS |
| OFF | `jkin_source=mN_side_channel` / `mN_present=1` | **REFUSED** ← 원래 열려 있던 구멍 |
| ON | 없음 / `SHM_raw` / `mN_present=0` | **REFUSED** |
| ON | `jkin_source=mN_side_channel` / `mN_present=1` | PASS |

실제 체크포인트(`sota_m3_dX_baseline_s53.pth`)를 학습 스크립트처럼 stamp 해 `load_model` 로
통과시킨 결과도 3/3 정확히 읽혔다(`jkin_source='mN_side_channel'`→mN, `mN_present=1`→mN,
`mN_present=0`→SHM_raw). stamp 안 한 원본은 `None`(라벨없음).

### 남은 한 줄 — 학습 스크립트 쪽
이 저장소에는 학습 스크립트가 없다(`torch.save` 없음). 학습 시 아래 한 줄만 추가하면
배포가 정확히 대조한다(둘 중 아무 이름이나 인식):

```python
ckpt["jkin_source"] = "SHM_raw"          # 또는 "mN_side_channel"
ckpt["raw_hand_j_kin_mN_present"] = 0    # 또는 수집 HDF5 root attr 값을 그대로 전달
```

---

## 3. 파이프라인 완주 검증 — **P2** (정식 수집 전 필수)

### 배경
지금까지의 세션은 **팔 이동이 전부 실패**(move_group 미기동)해서:
- 라이브 HDF5 에 `move_palm_down` 그룹 없음 → ★STAR_SEGMENTS 한 구간이 비어 있음
- `outcomes.json` 없음(판정 프롬프트 전 중단) → `bag_to_hdf5 --skip-outcomes` 검증 불가

> **정정(2026-07-28)**: `--skip-outcomes` 를 막고 있던 것은 `outcomes.json` 이 아니다.
> `bag_to_hdf5` 는 `outcomes.json` 을 **읽지 않고** bag 의 `/collect/demo_outcome` 토픽만 본다.
> 실제 원인은 **`S` 마커 유실**이었다(아래 검증 결과). `outcomes.json` 은 라이브 쪽 sidecar.

### 무엇을
[quick_start.txt](../stiffness_deploy_ros2/quick_start.txt) D-0 의 터미널 A/B 절차대로:
1. 터미널 A: move_group 기동 후 **닫지 말고 유지** (`ros2 node list | grep -c "^/move_group$"` = 1)
2. 터미널 B: 촉각 LIVE 확인 → `collect_ros2.py --fruit tomato --num-demos 1 --paxini raw`
   - `⚠ ... 이동 실패` 경고가 **한 번도 없어야** 함
   - run 끝 판정 프롬프트 입력(→ `outcomes.json` 생성)
3. `bag_to_hdf5.py "$S/bag"` → 4. `verify_parity.py`

### 완료 판정
라이브 HDF5 에 그룹 **3개**(`squeeze_A__run000`, `move_palm_down__run000`, `squeeze_B__run000`)
+ `outcomes.json` 존재 + parity 판정이 나옴(1번 과제가 선행되면 더 정확).

### 🔴 검증 중 발견한 버그 — 마커 유실로 `--skip-outcomes` 가 무력화 (수정 완료)

`collect_tomato_20260728_160837` bag 에 run 시작 마커 `S,0` 이 **없었다**. 그러면
[bag_to_hdf5.py:182](../stiffness_deploy_ros2/launch/bag_to_hdf5.py) `run_of()` 가 **-1** 을 돌려
`outcomes.get(-1)` → 전 그룹 `outcome='unjudged'` 가 되고, **`--skip-outcomes` 가 아무것도
걸러내지 못한다 → 파지 실패 run 이 학습 데이터에 그대로 남는다.**
그룹명은 `max(rid,0)` 이라 `run000` 으로 보여 눈치채기 어렵다(group attr `run=-1` 로만 드러남).
첫 `segment` 라벨(`safe`)도 함께 유실됐다(seg 15개 = 라벨 7 + 빈값 8).

**원인**: 마커 퍼블리셔 QoS 가 durability=VOLATILE 인데 `start_bag()` 후 **고정
`time.sleep(1.0)`** 으로 기다렸다 → `ros2 bag record` 가 구독을 붙이기 전에 발행된 마커는
bag 에 남지 않는다. 실제 `ros2 bag record` 상대로 재현:

| `start_bag` 후 대기 | 기록된 마커 |
|---|---|
| 없음(0s) | `S`·`safe` 유실, `E` 만 — **실측 세션과 같은 패턴** |
| `sleep(1.0)` (예전 코드) | **여전히 유실** (`grip`+`E` 만) |
| `wait_for_recorder()` (수정) | **4/4 전부 기록** ✔ |

**수정**: [collect_ros2.py](../stiffness_deploy_ros2/launch/collect_ros2.py) 에
`MarkerPublisher.wait_for_recorder(topics, timeout=15)` 추가 — 시간이 아니라 기록 대상
`/collect/*` 토픽의 **구독자 수**를 확인한 뒤 진행(미확인 시 경고).

### ✅ outcome 태깅 경로 자체는 정상 (2026-07-28, 로봇 불필요)

기존 bag 에 유실된 `S,0` 마커를 복원해 재변환하니 정상 동작했다:

| 주입 | 결과 |
|---|---|
| `S,0` + `outcome=success` | 3그룹 모두 `run=0`, `outcome='success'` ✔ |
| `S,0` + `grip_fail` + `--skip-outcomes grip_fail` | 3그룹 **전부 제외**, `0 그룹` ✔ |
| (`S` 없음) + `outcome` 만 | `run=-1`, `outcome='unjudged'`, 필터 무동작 ✘ |

### ✅ 파지 게이트 재시도 경로 — 스텁 검증 완료 (2026-07-28, `tools/check_grip_retry.py`)

실측 세션(160837)은 **1회에 성공**(`n=3`)해서 재시도 분기가 하드웨어에서 한 번도 실행되지
않았다. `_grip`/`_peak_forces`/`D.move_hand_to` 를 스텁으로 갈아끼워 호출 순서를 확인:

| 케이스 | grip 호출 | hand_open | 반환 |
|---|---|---|---|
| 1회차 성공 | 1 | 0 | ok ✔ |
| 2회차 성공 | 2 | 1 | ok ✔ |
| 3회차(마지막) 성공 | 3 | 2 | ok ✔ |
| 전부 실패(소진) | 3 | 2 | **not ok** ✔ |

- 마지막 시도 뒤에는 손을 펴지 **않는다**(중단 경로 `grip_fail_abort` 가 처리) ✔
- 경계: 임계 **이상** `GRIP_MIN_FINGERS` 개 → 성공 / 그보다 1개 적으면 실패 ✔
- `min_fingers=4`(전 손가락 요구)는 접촉 없는 손가락이 늘 있어 **항상 실패**한다
  (실측 분포 예 `[11.4, 0.00, 16.5, 2.2]N`) → 4 는 쓰지 말 것.

**`GRIP_MIN_FINGERS` = 3** (2026-07-28 사용자 요청으로 2→3 상향).
근거: 실측 세션 160837 이 `grip_reached_fingers=3`(임계 10.71N, 최대 15.4N) 이라 3개는
달성 가능하다. 다만 §F7 의 다른 실측 분포는 **2개만 도달**했으므로, 물체·자세에 따라
**재시도가 잦아지거나 `grip_fail` 이 늘 수 있다** — 그러면 2 로 되돌리거나
`GRIP_FORCE_RANGE`(현재 7~12N) 하한을 낮추는 쪽을 검토할 것.
테스트는 상수를 읽어 기대값을 만들므로 값을 바꿔도 `tools/check_grip_retry.py` 가 따라온다.

**하드웨어 미검증**: 손 펴기 후 재파지가 물리적으로 다시 잡히는지, 재시도가 잦지 않은지.
(향후 시도별로 파지 자세를 바꿀 자리는 `_grip_with_retry` 의 `attempt` TODO 참고.)

### ✅ 실전 검증 — 2026-07-28 19:33~19:37, 로봇 3 run (마커 수정이 실제로 동작)

세션 3개: `193304`·`193344`(grip_fail) / `193657`(성공).

**마커 유실 수정 확인** — `193657` 기준, 예전 세션(`160837`)과 대조:

| | 예전(고정 sleep) | 지금(`wait_for_recorder`) |
|---|---|---|
| `demo_marker` | 1개 (`E` 만) | **2개 (`S`,`E`)** ✔ |
| `segment` 라벨 | 7개 (`grip` 부터) | **9개 (`safe_start`,`move_grip` 포함)** ✔ |
| `seg` 메시지 | 15 | **18** ✔ |
| 변환 후 group `run` | **-1** | **0** ✔ |

→ `run_of()` 가 정상 매칭되므로 **`--skip-outcomes` 가 이제 실제로 작동한다.**

**grip_fail 중단 경로도 실전 동작**: `193304`·`193344` 는 `grip_fail_abort` 라벨 + bag 에
`demo_outcome='0,grip_fail'` 자동 발행 + 라이브 HDF5 **0 그룹**(스퀴즈 데이터 없음) — 설계대로.

**그 2 run 의 실패 원인은 파지 기준이 아니라 촉각 센서 정지였다**:

| 세션 | raw 프레임 | 값 범위 | 0 아닌 점 | 값 변화 | 판정 |
|---|---|---|---|---|---|
| 193304 | 2130 | 0.00 ~ 0.00 | 0/1524 | **0회** | **FROZEN** |
| 193344 | 2109 | 0.00 ~ 0.00 | 0/1524 | **0회** | **FROZEN** |
| 193657 | 3016 | −0.40 ~ 3.10 | 168/1524 | 11818회 | LIVE |

→ **파지 게이트의 부수 효과(좋은 쪽)**: 촉각이 죽으면 힘이 0 이라 게이트가 걸려 run 이
`grip_fail` 로 중단된다. 예전에는 그대로 palm-up 으로 넘어가 **쓸모없는 데이터를 수집**했다.
`ros2 topic hz` 로는 89Hz 정상으로 보이는 상황(§B1-b false green)을 게이트가 잡아준 셈이다.
※ 수집 전 촉각 LIVE 확인(부록 ①)을 건너뛰지 말 것.

성공 run(`193657`) 의 grip 구간 finger별 최대 Fz = **[11.2, 15.4, 12.3, 7.2] N** (4개 모두 접촉,
3번째가 11.2N) → 임계 7~12N 랜덤에서 `GRIP_MIN_FINGERS=3` 은 충분히 달성 가능.

### 🔴 판정 유실 — stdin 이 비대화형이면 `input()` 이 즉시 EOFError (수정 완료)

`195101` 세션에서 사용자가 Enter 를 눌렀는데도 `outcomes.json`·group attr `outcome`·bag 의
`demo_outcome` 이 **셋 다 없었다.** bag 타임스탬프가 원인을 확정했다:

```
E 마커 발행       19:51:36.759
bag 마지막 메시지  19:51:36.969   → E 이후 0.21초에 종료
```

**0.21초는 사람이 프롬프트를 읽고 Enter 를 누를 시간이 아니다** → `input()` 이 기다리지 않고
즉시 `EOFError` 를 냈다(= stdin 비대화형). 그 예외는 `__main__` 의
`except (SystemExit, KeyboardInterrupt)` 에 걸리지 않아 `main()` 을 빠져나가고, `finally` 만
실행돼 파일은 정상 닫혔다. **33초 로봇 동작의 판정이 통째로 유실**되고, 사용자가 누른 Enter 는
셸로 갔다. 실행 순서상 `pub_outcome`(574행) 전에 죽으므로 bag 에도 안 남는다.

왜 위험한가: 판정이 없으면 `--skip-outcomes` 로 **실패 run 을 학습에서 걸러낼 수 없다**.
게다가 조용히 실패한다(traceback 이 로그에 묻힌다).

**수정** ([collect_ros2.py](../stiffness_deploy_ros2/launch/collect_ros2.py)) — 사용자 요청:
넘어가지 말고 **입력할 때까지 기다린다.**
1. `_prompt()` 신규: stdin 이 대화형이면 `input()`, **아니면 제어 터미널(`/dev/tty`)에서 직접
   읽는다** → 파이프·리다이렉트로 실행해도 프롬프트가 화면에 뜨고 입력을 기다린다.
   (EOF 를 반복 호출로 버티려 하면 무한 루프가 되므로 이 방법이 유일하다.)
   프롬프트도 같은 터미널에 쓰므로 stdout 이 파이프여도 보인다.
2. 제어 터미널이 **아예 없을 때만**(진짜 데몬) `not_judged` 로 기록하고 계속 — 판정을 못 받아도
   `outcomes.json`·group attr·bag 토픽은 남는다(유실 없음).
3. `main()` 시작 시 입력 경로를 미리 확인해 **로봇을 33초 움직이기 전에** 알린다.

> 구현 함정(실측): `open("/dev/tty", "r+")` 는 `BufferedRandom` 이라 seek 가능성을 요구해
> 터미널에서 `OSError: File or stream is not seekable` 로 **실패한다.**
> 읽기(`"r"`)·쓰기(`"w"`)를 **따로** 열어야 한다.

검증:

| stdin | 제어 터미널 | 입력 | 결과 |
|---|---|---|---|
| `/dev/null` | 있음 | `2` | **기다렸다가** `grip_fail` ✔ |
| `/dev/null` | 있음 | Enter | `success` ✔ |
| `/dev/null` | 있음 | `zz`→`4` | 재질문도 tty 로, `discard` ✔ |
| 없음(setsid) | **없음** | — | `not_judged`, 무한루프 없음 ✔ |
| TTY | 있음 | Enter / `2` | `success` / `grip_fail` (회귀 없음) ✔ |

참고 — stdin 을 끊는 것들: 파이프(`| tee`)·리다이렉트(`< /dev/null`)·백그라운드(`&`)·
IDE 태스크/버튼 실행. 이제 그래도 판정은 받을 수 있지만, 로그를 남길 땐
`script -qc "python3 ... collect_ros2.py ..." collect.log` 가 여전히 안전하다.

### 남은 로봇 항목
`outcomes.json` 만 남았다 — 지금까지 4 run 모두 판정이 기록되지 않았다(위 유실 때문).
다음 run 을 **대화형 터미널**에서 돌려 프롬프트에 답하면 이 항목은 끝난다.
(수정 후에는 비대화형이어도 `not_judged` 로 최소한 기록은 남는다.)
(수정 후에는 `[bag] record 구독 확인 (…개 마커 토픽, N초)` 가 로그에 찍혀야 하고,
변환 결과 group attr 이 `run=0` 이어야 한다. `run=-1` 이면 마커가 또 유실된 것.)

---

## 4. `os._exit` stdout flush 유실 — P2

### 배경
`os._exit()` 는 파이썬 정리(=stdout flush)를 건너뛴다. 대화형 터미널은 line-buffered 라
문제없지만, **파이프/리다이렉트하면 마지막 출력이 통째로 유실**된다.
실제로 `collect_ros2.py --help` 가 빈 출력이었고, `capture_pose.py | tee` 로는
**ARM_POSES 블록이 사라진다**(pty 로 실행하면 정상 출력 확인).

해당 파일: `capture_pose.py`, `collect_ros2.py`, `deploy_ros2.py`,
`deploy_task3_ros2.py`, `test_moveit_mover.py`

### 무엇을
`os._exit(...)` 직전에 `sys.stdout.flush(); sys.stderr.flush()` 추가.
(`os._exit` 자체는 rclpy 종료 hang 회피 목적이라 유지)

### 완료 판정
`python3 stiffness_deploy_ros2/launch/capture_pose.py --help | cat` 로 출력이 보이고,
`| tee log.txt` 로 ARM_POSES 블록이 파일에 남는다.

### ✅ 검증 결과 — 2026-07-28 (로봇 불필요)

| 검증 | 결과 |
|---|---|
| 정적점검 (5개 파일) | 실제 `os._exit(` 호출은 파일당 **1곳**, 그 직전에 flush 있음 ✔ |
| 기전 반례 | `print` 후 flush 없이 `os._exit` → 파이프 출력 `""` / flush 추가 → 남음 |
| `collect_ros2.py --help \| cat` | **21줄** (원래 빈 출력이었던 케이스) ✔ |
| `test_moveit_mover.py --help \| cat` | **18줄** ✔ |
| `capture_pose.py … \| tee` | **ARM_POSES 블록 남음** ✔ |

`capture_pose.py` 는 첫 인자가 토픽이므로 가짜 퍼블리셔(`/fake/joint_states`)로 검증했다
(로봇 스택이 내려가 있어도 되고, 실제 상태 토픽에 쓰지 않아 안전).

`deploy_ros2.py`·`deploy_task3_ros2.py` 는 **실행하면 손·팔이 움직이므로 정적점검만** 했다
(로봇이 살아 있을 때 절대 `--help` 로 떠보지 말 것 — 두 파일은 argparse 이전에 배포 루프로 들어간다).

---

## 5. `deploy_ros2_exp_forcecurl.py` import 순서 버그 — P2

### 배경
`import deploy as D`(30행)가 `import deploy_ros2_exp as EXP`(31행) **보다 먼저** 실행돼,
`deploy_ros2` 의 `sys.modules.setdefault` shim 이 무효화된다. 그 결과 **구 추론 엔진**
(`launch.real_deploy_inference_old`)이 로드되어:
- tomato 만 등록된 구 `FRUIT_CONFIG` (kiwi 선택 시 종료)
- `USE_JKIN=True` 경로 → mN 경고 + tick 당 작업 증가로 **루프 58.3Hz**(정상 97.8Hz)
- 정규화·마스킹 임계도 달라 **baseline 과 비교 불가**

검증 근거:
```
(a) import deploy 먼저 → 엔진 = launch.real_deploy_inference_old
(b) shim 먼저        → 엔진 = real_deploy_inference_final
```

### 무엇을
두 import 순서를 바꾼다(`deploy_ros2_exp` 를 먼저 import → shim 설치 후 `deploy` 로드).

### 완료 판정
실행 시 `[통합모델] 선택=m3_dX ...` 가 뜨고 kiwi 선택 가능, 루프 rate 가 ~98Hz.

### ✅ 검증 결과 — 2026-07-28 (로봇 불필요)

판별자는 **`sys.modules["launch.real_deploy_inference_old"]` 가 `real_deploy_inference_final` 과
같은 객체인가**(= `setdefault` shim 이 이겼나). 두 순서를 각각 하위 프로세스로 실행:

| import 순서 | `launch.real_deploy_inference_old` 의 실체 | shim 승리 | `USE_JKIN` |
|---|---|---|---|
| (a) `deploy_ros2_exp` → `deploy` (**현재 코드**) | `real_deploy_inference_final.py` | True | **False** |
| (b) `deploy` → `deploy_ros2_exp` (버그 재현) | `real_deploy_inference_old.py` | False | **True** |

실제 `deploy_ros2_exp_forcecurl` 을 그대로 import 한 결과도 **final / `USE_JKIN=False`** ✔

> **정정**: 배경의 "tomato 만 등록된 구 `FRUIT_CONFIG`(kiwi 선택 시 종료)" 는 부정확하다.
> 두 엔진 **모두** 4과일 키를 갖고 있고 `kiwi`/`lemon`/`plum` 은 `model=None`(tomato 만 실파일).
> 실제 차이는 구 엔진에 `USE_UNIFIED`·`USE_SOTA_ENSEMBLE` 이 **없어서** 통합모델 경로가 아예
> 없다는 것이다 → 결론("구 엔진으로는 kiwi 사용 불가")은 그대로 유효하다.

### 루프 ~98Hz 실측은 **불필요** (2026-07-28)
완료 판정에 적어둔 '루프 rate ~98Hz' 는 버그의 **증상**이었고, 고쳐졌는지는 이미 원인
수준에서 확인됐다(위 표: 엔진 identity + `USE_JKIN`). 게다가
`deploy_ros2_exp_forcecurl.py` 는 **배포측 실험 스크립트**로 `collect_ros2.py` 가 쓰지 않는다
→ 데이터 수집과 무관하다. 그 스크립트를 실제로 돌릴 때 로그의 loop rate 를 눈으로 확인하면
충분하고, 그때 58Hz 가 나오면 import 순서가 되돌아간 것이다(회귀 신호로만 쓴다).

---

## 6. `palm_down` 전용 자세 캡처 — P3

### 배경
`collect_ros2.py` `ARM_POSES["palm_down"]` 이 **safe 와 동일값(placeholder)** 이다.
손목축(joint5)이 palm_up(+1.070) ↔ palm_down(-1.223) 로 2.29 rad 뒤집히므로
★`move_palm_down` 구간은 **실제 모션으로 기록**되지만, 자세 자체는 safe 와 같다.

### 무엇을
`capture_pose.py` 로 전용 palm-down 자세를 캡처해 교체.
(분석 시 그 전 데이터는 `palm_down ≡ safe` 임을 감안)

### 완료 판정
`ARM_POSES["palm_down"]` 이 safe 와 다른 값이고, 4→5단계 이동 경로가 짧아짐.

### ⏸ 보류 — 현행(`palm_down ≡ safe`) 유지 (2026-07-28, 사용자 결정)
당장은 지금 값으로 수집한다. **데이터 해석 시 주의**: `move_palm_down` 구간은 실제 모션으로
기록되지만 도착 자세가 `safe` 와 같으므로, 손목 뒤집힘(joint5 ±) 이 들어간 구간이 아니다.
나중에 전용 자세로 바꾸면 그 전/후 데이터의 `move_palm_down` 은 **분포가 다른 구간**이 된다
(섞어 학습할 때 확인 필요). 바꿀 때 판단 근거는 위 완료 판정 그대로.

---

## 7. §F7 힘 경로 결론 반영 — P3 (근거 이미 확보)

### 배경
§F7 의 분기점(**ft ≪ Σ127 인가 = H1**)에 대한 실측이 나왔다:

| 힘 소스 | thumb Fz 최대 | 스퀴즈 임계 | 도달률 |
|---|---|---|---|
| `/paxini/right/ft` (4×3) | 4.5 ~ 5.3N | 10N | 45~52% |
| **Σ127 (`--paxini raw`)** | **11.40 / 9.90N** | 14.81 / 11.43N | **77% / 87%** |

→ 같은 접촉에서 **Σ127 이 2~3배 크다 ⇒ H1(측정 경로) 지지.**
`resultant = Σ127` 도 A↔B parity 오차 0 으로 검증됐다.

### 무엇을
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) §F7 의 판정 표를 이 수치로 채우고, 결론
(“배포가 학습보다 작은 양(ft)을 먹고 있었다 → `/raw` 구독으로 해소”)을 확정 기록.
`--paxini raw` 가 표준 경로임을 §F1 요약 표에도 반영.

### 완료 판정
§F7 이 "실측 필요" 가 아니라 "H1 확정 / 조치 완료" 로 갱신됨.
(단 여전히 임계 미달(77~87%)이므로 **힘 크기 자체는 별도 과제**로 남긴다)

---

## 8. bag 기록 토픽 정리 — P3

### 배경
1 run(≈32s) bag = **24.3 MiB** (0.75MB/s). 20 runs ≈ 500MB, 과일 4종 ≈ 2GB.
기록 손실은 없다(raw 0.3%, 200Hz 토픽 0%).

| 토픽 | 개수/32s | 비고 |
|---|---|---|
| `/paxini/right/raw` | 2877 | **필수**(127점) |
| `/paxini/right/ft` | 2877 | raw 가 있으면 **중복** |
| `/hand/right/kin` | 6448 | **학습에 사용 → 유지** |
| `/franka/right/q_target` | 0 | 팔 이동 실패 시 0 (성공하면 필요) |

### 무엇을
`--bag-topics` 로 `/paxini/right/ft` 제외 검토. **단 parity 검증이 끝난 뒤에** 바꿀 것
(변환기가 참조할 수 있음).

### ✅ 검토 결과 (코드 확인, 2026-07-28)
- 기본 토픽: [collect_ros2.py:127-134](../stiffness_deploy_ros2/launch/collect_ros2.py) `DEFAULT_BAG_TOPICS` 에
  `/paxini/right/ft`(129행) + `/paxini/right/raw` 둘 다 포함.
- 변환기 소스 선택([bag_to_hdf5.py:91](../stiffness_deploy_ros2/launch/bag_to_hdf5.py)):
  `pax_src = ("raw" if raw ∈ present else "ft") if paxini_pref=="auto" else paxini_pref`.
  → **변환기가 `/paxini/right/ft` 를 참조하는 경우는 `--paxini ft` 또는 `auto`+raw 부재일 때뿐.**
  raw 는 항상 기록되고 §F7 에서 `--paxini raw` 가 표준이 됐으므로 **ft 토픽은 중복 → 제거 안전**.
- **적용(대기)**: parity(1번) 통과 확인 후 `DEFAULT_BAG_TOPICS` 에서 129행의 `"/paxini/right/ft"` 한 줄만 삭제.
  (라이브 브리지 소스 [collect_ros2.py:351](../stiffness_deploy_ros2/launch/collect_ros2.py) 의
  `/paxini/right/ft` 하드코딩은 별개 이슈 — bag 토픽 정리와 무관.)

### 완료 판정
용량이 줄고 parity·변환이 그대로 통과.

### ✅ 실측 검증 — 2026-07-28 · **결론 변경: 제거하지 말 것** (미적용)

원본 bag 에서 `/paxini/right/ft` 만 뺀 사본을 만들어 측정·변환·parity 를 모두 돌렸다.

| 항목 | 결과 |
|---|---|
| 용량 | 24.16 → **23.86 MiB (−1.2%, 0.30 MiB/run)** — 20 runs 해도 ≈6MB |
| 변환 | 3그룹 **전 데이터셋 bit 동일**, `paxini_source=raw` ✔ |
| parity | **OVERALL PASS** (기존과 동일) ✔ |
| 대가 | `--paxini ft` / `auto` 폴백 상실 |

**왜 절감이 작나**: 위 표는 메시지 **개수**만 봤다(둘 다 2877). 실제 payload 는
`ft`=12 float vs `raw`=381 float 이라 **용량은 raw 가 지배**한다. `ft` 는 전체의 1.2% 뿐이다.

→ **제거해도 안전하지만 얻는 게 1.2% 이고 폴백을 잃으므로, 그대로 두는 쪽을 권고한다.**
`DEFAULT_BAG_TOPICS` 는 **수정하지 않았다.** (용량이 진짜 문제가 되면 `raw` 쪽 압축·다운샘플이
답이다 — `ft` 제거는 효과가 없다.)

---

## 9. parity 판정 허용오차 — P3 (1번 검증 중 발견)

### 배경
1번 검증에서 데이터는 정상(99/101 bit-exact, 차이가 1 샘플 스텝 이내)인데도 판정이 **FAIL** 로 뜬다.
`verify_parity.py` 의 `tol=0.001` 이 **절대값**이라, raw count 단위 채널에는 물리적으로 도달 불가능한 기준이다:

| 채널 | 단위 | 신호 자체의 프레임간 변화(중앙값) | 절대 tol |
|---|---|---|---|
| `joint` | encoder counts (≈2267) | **4** | 0.001 |
| `ft`=kin | SHM raw (≈11161) | **116** | 0.001 |
| `resultant`/`tactile` | N (스케일 적용됨) | 작음 | 적절 |

즉 스케일 안 된 두 채널만 구조적으로 FAIL 할 수밖에 없다. **FAIL 이 "불일치"가 아니라
"샘플 시점이 1스텝 달랐다"를 의미하므로, 지금 상태로는 판정을 신뢰할 수 없다**(진짜 회귀를 놓친다).

### 무엇을
채널별 판정 기준을 신호 스케일에 맞춘다. 셋 중 하나:
- **(a) 상대 허용오차** — `|A−B| ≤ tol_rel · max(|B|)` (raw count 채널만).
- **(b) 스텝 정규화(권고)** — `|A−B| ≤ k · median(|diff(B)|)` (k≈1.5).
  "차이가 한 샘플 스텝 이내면 시점 차이" 라는 실제 판정 논리를 그대로 코드화한다.
- **(c) 불일치 프레임 비율** — 프레임 단위 판정 후 `불일치율 ≤ 2%` 면 PASS (지금은 2/101 = 2.0%).

어느 쪽이든 리포트에 **불일치 프레임 수 / 전체**를 함께 출력해야 한다(max 만으로는 위 구분이 불가능).

### 완료 판정
같은 세션(`collect_tomato_20260728_160837`)에서 **OVERALL PASS**, 그리고 한 채널을 인위로
1% 흔든 파일에서는 **FAIL** 이 뜬다(= 판정력 유지 확인).

### ✅ 구현·검증 완료 — 2026-07-28

**구현** ([verify_parity.py](../stiffness_deploy_ros2/launch/verify_parity.py)): `frame_judge()` 가
채널을 셋으로 분류한다.
- `exact` — 전 프레임 `|Δ| ≤ tol` → PASS
- `timing` — tol 초과가 **소수**(`--max-bad-frac 0.05`) **이고** 오차가 신호 자체의 1 샘플
  스텝(`median|diff(B)|`)의 **`--step-k 1.5` 배 이내** → `PASS*`(시점 차)
- `fail` — 그 외 → FAIL

`--exact` 로 `timing` 도 FAIL 처리(bit-exact 회귀 감시). 리포트에 **불일치 프레임 수**를
항상 출력한다(max 만으로는 시점차/실불일치 구분 불가 — 이게 애초의 문제였다).

**판정 결과**: 기존 세션 **OVERALL PASS** — `squeeze_A` = `PASS*`
(joint 2/101프레임 ≤1.25스텝, ft 2/101프레임 ≤0.98스텝), `squeeze_B` = `PASS`(전 채널 0).
`--exact` 로는 FAIL(exit 1).

**판정력 유지 확인** — bag HDF5 를 인위로 교란한 4케이스 **전부 FAIL 검지**:

| 교란 | 불일치 프레임 | 스텝 배수 | 무엇이 잡았나 |
|---|---|---|---|
| joint 전체 ×1.01 | 101/101 | 11.1 | 둘 다 |
| joint 전체 +4 counts(=**1스텝**) | 101/101 | **0.7~2.2** | **프레임 수** (스텝 기준은 통과) |
| joint **3프레임만** +40(=10스텝) | **5/101** | 10.0 | **스텝 배수** (프레임 수는 통과) |
| ft 전체 ×1.01 | 101/101 | 0.9~1.9 | **프레임 수** |

→ **두 기준은 상호보완이다.** 어느 하나만 쓰면 2번(전 프레임 1스텝 밀림) 또는
3번(소수 프레임 큰 오차)을 놓친다. `--step-k`·`--max-bad-frac` 를 함께 유지할 것.

### 🔧 재교정 `step_k` 1.5 → 2.0 (2026-07-28, 신규 세션에서 오탐 발생)

신규 세션 `193657` 이 정상인데 `squeeze_A/ft` 가 **1.72스텝**으로 FAIL 이 떴다(기준 1.5).
임계값을 늘려 통과시키는 대신 **원인부터 확정**했다 — [tools/check_parity_timing.py](../tools/check_parity_timing.py)
로 bag **원본 메시지 스트림**과 bit-exact 대조:

```
joint 불일치 3/144  ✔ A·B 둘 다 원본에 존재(시점 차) Δt -5.91~-1.55ms (≈0.9 메시지)
ft    불일치 3/144  ✔ A·B 둘 다 원본에 존재(시점 차) Δt -5.89~-1.54ms (≈0.9 메시지)
```

**A 값이 원본 스트림에 정확히 1곳 존재하고, 메시지 간격이 5.00ms 인데 Δt 가 ≈1 메시지다.**
joint·ft 가 **같은 프레임에서 같은 Δt** 로 어긋난 것도 같은 스냅샷을 잡았다는 뜻 → 같은 데이터,
다른 시점. 원인: 라이브(A)는 tick 순간 executor 가 아직 처리하지 못한 메시지를 못 보고,
bag(B)은 **도착시각** 기준이라 그 메시지를 포함한다. `_latest()` 는 양쪽 다 zero-order-hold 로
**규칙은 이미 같다** — 이 ≤1 메시지 staleness 는 원리적으로 남는다(코드로 없앨 수 없다).

판정 기준의 실제 결함: `step` 을 **프레임 격자(≈10.2ms)** 에서 재는데 불일치의 실체는
**메시지 1개(5ms)** 라, 힘이 급변하는 프레임에서 중앙값의 ~2배까지 커진다. 실측 교정:

| | 값 |
|---|---|
| 정상 2세션(160837·193657) 불일치 프레임 스텝비 **최대** | **1.72** |
| 교란 4종 중 '스텝비로 잡아야 하는' **최소** | **8.00** (3프레임 +10스텝) |
| 나머지 교란 2종 스텝비 | 0.80~1.98 → **프레임 수 기준**(100%)이 잡음 |

→ `1.72 < k < 8.00` 이면 양쪽을 가른다. **k=2.0** 채택(정상 최대의 1.16배, 교란 최소의 1/4).
재검증: 정상 2세션 PASS · 교란 4종 전부 FAIL · `--exact` FAIL.

> **국소 스텝은 쓰면 안 된다**(시도했다가 폐기): 해당 프레임 주변 `max|diff|` 를 분모로 쓰면
> **교란이 자기 분모를 부풀려** 3프레임 교란의 국소스텝비가 1.00 이 되고 정상(1.14)과
> 구분되지 않는다. 분모는 반드시 전역(median)이어야 한다.

**경계에서 애매하면 허용오차로 다투지 말고 `tools/check_parity_timing.py` 를 쓸 것** — bag 이
있으면 판정이 아니라 사실 확인이 된다. A·B 값을 **둘 다** 원본과 대조하므로
`A 만 원본에 있음 → 변환기 문제` / `B 만 → 라이브 기록 문제` 까지 갈라준다
(from_bag.h5 를 인위로 교란해 "변환기 문제"로 잡히는 것까지 확인).

---

## 10. `bag_to_hdf5 --paxini ft` 의 조용한 열화 — P3 (8번 검증에서 발견)

### 배경
`/paxini/right/ft` 가 없는 bag 에 `--paxini ft` 를 주면 **에러 없이** 스퀴즈 구간이
0프레임이 되고 `move_palm_down` 1그룹만 나온다:

```
[bag2h5]   squeeze_B run-1: 프레임 0 — 건너뜀
[bag2h5] 완료: 1 그룹, 559 frames
```

`--paxini raw` 가 표준이 된 뒤(§F7)로는 잘못된 플래그가 조용히 절반짜리 데이터셋을 만들 수 있다.
(8번을 적용해 `ft` 토픽을 빼면 이 위험이 커지는데, 8번은 미적용으로 결론냈으니 지금은 낮은 우선도.)

### 무엇을
`read_streams()` 에서 요청한 `pax_topic` 이 bag 에 **없으면 즉시 SystemExit**
(현재는 `pax` 스트림이 빈 채로 진행). `auto` 는 지금처럼 폴백 유지.

### 완료 판정
`--paxini ft` 를 소스 없는 bag 에 주면 변환이 **거부**되고, `auto`/`raw` 는 그대로 통과.

### ✅ 구현·검증 완료 — 2026-07-28

구현: [bag_to_hdf5.py](../stiffness_deploy_ros2/launch/bag_to_hdf5.py) `read_streams()` 에서
`pax_topic` 이 bag 에 없으면 `SystemExit` — bag 에 실제로 있는 `/paxini/*` 토픽 목록을 함께 출력한다.
`auto` 는 폴백을 그대로 유지하고, **양쪽 다 없을 때만** 거부한다(그 경우 메시지가 "수집 설정 확인").

검증: 원본 bag 에서 토픽을 빼낸 사본 3종(`ft` 없음 / `raw` 없음 / 둘 다 없음)으로 8케이스.

| bag | `--paxini` | 결과 |
|---|---|---|
| 원본 | (기본 auto) | 3그룹 ✔ |
| 원본 | `ft` | 3그룹 ✔ (토픽이 있으니 통과) |
| `ft` 없음 | auto | **raw 폴백** → 3그룹 ✔ |
| `ft` 없음 | `raw` | 3그룹 ✔ |
| `raw` 없음 | auto | **ft 폴백** → 3그룹 ✔ |
| `raw` 없음 | `ft` | 3그룹 ✔ |
| `ft` 없음 | `ft` | **거부** (exit 1) ✔ ← 예전엔 조용히 1그룹 |
| `raw` 없음 | `raw` | **거부** (exit 1) ✔ |
| paxini 전무 | auto | **거부** (exit 1) ✔ |

> 주의(검증 중 겪음): `--paxini ft` 로 변환하면 세션의 `from_bag.h5` 가 **ft 소스로 덮어써진다**
> (그룹·프레임 수는 같아서 눈에 안 띈다). 표준으로 되돌리려면 옵션 없이 다시 변환할 것 —
> 확인은 root attr `paxini_source` 로.

---

## 11. 저장 포맷 개편 — session.h5 (연속 타임라인) — 2026-07-28

### 배경 (기존 h5 2개의 문제)
| | `collect_<ts>.h5` (A, 라이브) | `from_bag.h5` (B, 구간별) |
|---|---|---|
| 그룹 | `squeeze_A/B` 만 | `squeeze_A/B` + `move_palm_down` |
| 채널 | 손만(joint·kin·resultant·tactile) | + palm_down 에만 `arm_joint`/`arm_q_target` |
| 시계 | `t_mono_ns`(monotonic) | `t_ns`(epoch) |

- **중복** = `squeeze_A/B`(채널 동일, B 가 행 수 2배 — A 는 스퀴즈 모션만, B 는 라벨 창 전체).
- 둘 다 둔 이유는 서로 검증(parity)용이었고 **그건 1·9번에서 끝났다.**
- 결정적 결함: **어느 쪽도 '스퀴즈 중 팔 관절/목표각'을 담지 않는다**(arm_* 은 palm_down 그룹뿐).

### 무엇을 (사용자 결정 2026-07-28)
`session.h5` **하나**로 통합 + bag 유지 + 라이브 h5 폐지. 저장 범위 = 시퀀스 전체 연속.
`FT sensor` = **hand kin(12)** 로 확정 → 데이터셋 이름을 `kin` 으로 정정(예전 `ft`).

```
/data   한 행 = 한 tick(100Hz). run·phase 열로 자유롭게 자른다
   t_ns, run(-1=유휴), phase, squeeze_on, valid, age_ms(채널별 경과)
   arm_joint(7) arm_q_target(7)      rad
   hand_joint(16) hand_q_target(16)  encoder counts
   kin(12)                           SHM raw int16
   paxini_resultant(4,3) paxini_raw(4,127,3)   N
/runs   한 행 = 한 run → **데모 분할 기준표**
   run,row0,row1,t0_ns,t1_ns,outcome_code,
   pose_palm_up,pose_palm_down,pose_hand,           ← 숫자
   grip_thr_n,squeezeA_thr_n,squeezeB_thr_n,grip_reached_fingers,grip_peak_force_n
/codes  phase/outcome/arm_pose/hand_pose 숫자↔이름 + arm_pose_joints(실제 관절각)
```
phase: `0` idle · `1` 파지 · `2` palm-up · `3` 스퀴즈A · `4` palm-down · `5` 스퀴즈B ·
`6` 파지해제 · `7` safe 이동 · `8` grip 이동 · `9` 파지실패중단.

### tick = 100Hz 고정 그리드 (실측 근거, 세션 195101)
| 채널 | 실측 |
|---|---|
| q_target(제어 루프) | **10.15ms = 98.5Hz** → 학습 스펙(`CONTROL_RATE_HZ=100`, `FACTOR=10`)과 일치 |
| joint / kin / franka joint | 195.7Hz (5.00ms) |
| **paxini(가장 느린 실센서)** | **87.5Hz (11.12ms)** → 100Hz 그리드면 1 tick 안에 다 들어옴 |

200Hz 로 올리면 용량 2배인데 paxini 는 정보 증가 0(같은 값 2번). 버려지는 joint/kin 절반은
bag 에 남아 복구 가능. 샘플링은 zero-order-hold(라이브 리더와 동일 규칙).

### ✅ 검증 (세션 195101, 로봇 불필요)
```bash
python3 stiffness_deploy_ros2/launch/bag_to_session.py collect_logs/collect_<fruit>_<ts>
```
- 33.0s → **3305 tick**, 전 채널 생성. **파일 0.5 MiB** (예상 21MB 였는데 paxini raw 가
  대부분 0 + ZOH 반복이라 gzip 이 ~40배로 줄인다 → 용량은 걱정거리가 아니다).
- phase 분포: grip 491 · palm_up 776 · squeeze_A 358 · palm_down 574 · squeeze_B 358 ·
  release 319 · safe 474 · move_grip 170 · idle 22 tick.
- 데모 split 실동작: `/runs` 의 `row0:row1` 로 잘라 스퀴즈A 3.58s·B 3.58s·palm_down 5.74s 확인.
- **`arm_q_target` 이 전 구간 3277 tick 에 존재** — 기존 두 파일로는 못 얻던 값.
- `/runs` 자세·임계값 채움을 합성 `outcomes.json` 으로 확인
  (`pose_palm_up=4(palm_up_tilt_left)`, `pose_hand=3(plum.txt)`, `grip_thr=8.493` …).

### 함께 고친 것
자세·임계값이 **라이브 h5 group attr 에만** 있어서 bag 기반 재구성 시 유실됐다 →
`collect_ros2` 가 `outcomes.json` 에 run 별 자세·임계 3개·파지결과를 함께 기록하도록 변경
(`_run_sequence` 반환을 `(names, auto_outcome, meta)` 로).

### ✅ 라이브 h5 폐지 완료 (2026-07-28) — 산출물 3개로 확정

세션 `224840`(`1q` 로 정상 종료)으로 실데이터 검증이 끝나 정리했다.
**저장되는 것 = `session.h5` + `bag/` + `outcomes.json`.**

| 파일 | 역할 |
|---|---|
| `session.h5` | 학습용. 이 파일 하나로 phase별·데모별 분할이 된다(자립) |
| `bag/` | 원본. 재생성·사후 추적용(시점차 증명에 실제로 필요했다 — 9번) |
| `outcomes.json` | 판정·자세·임계 + **수집 provenance**. 579B |

- `collect_*.h5`(라이브)·`from_bag.h5` 는 **더 안 만든다.** 둘을 뒀던 이유는 서로
  검증(parity)이었고 그건 1·9번에서 끝났다. `--live-h5` 로 되살릴 수 있다.
- `RecordingEngine` 은 **계속 필요**하다 — 스퀴즈 첫 유효 프레임에 `/collect/squeeze_on=1` 을
  발행하는 콜백이 거기 있고 그 마커가 `session.h5` 의 `squeeze_on` 열이 된다.
  그래서 engine 은 살리고 writer 만 `_NullWriter`(파일 안 만들고 프레임 수만 셈)로 바꿨다.
- ★ 함께 막은 구멍 2개:
  1. **물체 이름** — `session.h5` 에 `fruit` attr 이 없어 폴더명으로만 알 수 있었다(폴더를
     옮기면 학습 라벨의 근간이 사라진다) → `fruit`/`session` attr 추가.
  2. **수집 provenance** — `git_sha`·`FACTOR`·`USE_JKIN`·`t_offset_ns`·힘범위가 **라이브 h5 의
     root attr 에만** 있었다. 폐지하면 통째로 사라진다 → `outcomes.json` 의 `session` 블록으로
     남기고 `bag_to_session` 이 `collect_*` 접두사로 `session.h5` root attr 에 옮긴다.
     (검증: `collect_git_sha`·`collect_t_offset_ns`·`collect_FACTOR` … 11개 전달 확인)

### phase 코드 — `safe_end` 분리 (2026-07-28)
`safe_start`/`safe_end` 를 같은 7 로 두면 한 run 에서 `phase==7` 이 2구간으로 나와 시작/끝을
구분할 수 없었다 → **`safe_end=10`** 으로 분리. 결과: idle(0) 만 2구간(S 이전·E 이후)이고
**나머지 phase 는 전부 정확히 1구간**이 된다.

```
0=idle  1=grip  2=move_palm_up  3=squeeze_A  4=move_palm_down
5=squeeze_B  6=release  7=safe_start  8=move_grip  9=grip_fail_abort  10=safe_end
```

### 남은 것
사용자 확인 필요(§본문 '더 필요한 것' 참고): 물체 개체 id, 실패 run 보존 정책.

---

## 12. paxini 발행 공백 (11.12ms 주기에 20~33ms 구멍) — P3, 원인 미확정

### 관측 (세션 `collect_ecoflex_20260728_224840`, 36.3s)
`session.h5` 변환 시 **309 tick 이 갱신 없이 hold** 됐다. bag 도착시각으로 추적:

| 토픽 | 중앙 간격 | 2배 초과 공백 | 최대 |
|---|---|---|---|
| `/franka/right/joint_states` | 5.00ms | **1**회 | 10.0ms |
| `/hand/right/joint_states` | 5.00ms | **1**회 | 10.1ms |
| `/hand/right/kin` | 5.00ms | **1**회 | 10.1ms |
| **`/paxini/right/raw`** | 11.12ms | **10**회 | **33.4ms** |
| **`/paxini/right/ft`** | 11.12ms | **10**회 | **33.4ms** |

### 무엇이 배제됐나 (수집 쪽 결백)
- **수집 시퀀스·이 워크스페이스 코드 아님**: 공백은 paxini 두 토픽에만 생긴다.
  200Hz 토픽 3종은 36초 동안 10ms 한 번씩뿐(= tick 1개 놓침, 무의미).
- **rosbag2 레코더 과부하·CPU 전역 스톨 아님**: paxini 공백 시각이 다른 토픽 공백과
  **0/10 겹친다**. 전역 사건이면 같이 빠져야 한다.
- **토픽별 DDS 전송 유실 아님**: `raw`∩`ft` 공백 시각이 **7/10 겹친다**. 두 토픽은 같은 노드가
  발행하므로, 겹친다는 것은 원인이 **발행 이전(상위)** 에 있다는 뜻이다.

### 왜 '센서가 느린 것' 만으로는 설명되지 않나
paxini 중앙 간격이 **11.12ms = 89.9Hz** 로 발행자의 tactile 타이머(90Hz)와 일치한다.
즉 발행은 **고정 타이머**로 나가고 데이터 변화로 게이팅하지 않는다(§B1-b 의 'FROZEN 인데
89Hz 로 계속 발행' 이 같은 성질). **타이머 발행이면 센서가 느려도 메시지 공백은 안 생긴다**
— 같은 값이 90Hz 로 반복될 뿐이다. 따라서 33.4ms(=tick 3개) 공백은 **그 타이머 콜백이
늦어진 것**이고, 후보는 둘로 좁혀진다:

1. **발행 경로 지연** — tactile 메시지가 1524 float 이라 직렬화·복사 비용이 200Hz 토픽
   (7·16 float)보다 훨씬 크다. 제어 PC 부하에 따라 tactile 타이머만 늦어질 수 있다.
2. **센서/UART 가 SHM 갱신을 늦춰 그 타이머 콜백이 블록** — 콜백이 SHM 읽기에서 대기한다면
   가능. (게이팅은 아니지만 블로킹은 별개다)

→ **"센서 자체 문제" 로 단정할 수 없다.** 다만 **수집 쪽 문제가 아니라는 것은 확실**하고,
원인은 제어 PC 의 tactile 발행 경로에 있다.

### 원인을 가르는 확인 (제어 PC 에서, 아직 안 함)
```bash
# ① 발행 타이머가 늦는가 vs 센서가 안 오는가 — SHM 의 Paxini_seq 증가 간격을 직접 본다
#    (seq 가 규칙적인데 토픽만 공백 → 발행 경로 / seq 자체가 공백 → 센서·UART)
# ② tactile 만 늦는지 확인: 같은 노드의 arm/hand 타이머 지연과 대조
# ③ paxini_writer.py 의 UART 읽기 주기 로그
python3 tools/sensor_update_rate.py --duration 30 --topics /paxini/right/raw
```

### ✅ 조사 불필요 결론 (2026-07-28) — 공백이 **스퀴즈 구간에는 안 걸린다**
phase 별로 촉각 stale(>20ms) 비율을 재보니 힘 신호가 학습에 쓰이는 구간은 깨끗했다:

| phase | 231155 | 224840 |
|---|---|---|
| **스퀴즈 A** | **0.0%** (최대 16ms) | **0.0%** (최대 17ms) |
| **스퀴즈 B** | **0.0%** (최대 20ms) | **0.0%** (최대 13ms) |
| palm-down | 0.3% (최대 25ms) | 0.4% (최대 31ms) |
| palm-up | 0.1% | 0.1% |
| 해제 | 0.3% | 0.0% |

→ 공백은 **팔 이동 구간에만** 몰린다. 제어 PC 가 궤적 스트리밍으로 바쁠 때 tactile 타이머가
밀리는 것으로, 앞의 '발행 경로 지연' 가설과 일치한다. **스퀴즈 데이터는 영향 없다.**

### 자동 감시로 대체 (원인 추적 대신)
`bag_to_session.py` 가 변환할 때마다 ★스퀴즈 구간의 촉각 stale 비율을 출력한다
(`STALE_WARN_MS=20`, paxini 정상 간격 ~11ms 의 약 2샘플):

```
[session]   ✔ squeeze_A: 촉각 stale(>20ms) 0.0% (407 tick 중) · 최대 16ms
```
`✔`=0% · `⚠`=0~5% · `✘`=5% 이상(그때는 "age_ms 로 걸러내거나 재수집 검토" 안내 + 이 항목 참조).
검증: 기준을 12ms 로 낮추면 `⚠ 0.2%`, 5ms 로 낮추면 `✘ 44~49%` 로 세 단계 모두 발동 확인.

→ **이 경고가 뜨지 않는 한 12번은 조사하지 않아도 된다.** 뜨면 그때 아래 절차로 원인을 가른다.

### 영향 (지금 데이터는 쓸 수 있다)
`session.h5` 의 `age_ms` 열에 채널별 '마지막 갱신 후 경과' 가 있으므로, 학습 전처리에서
`age_ms[:, paxini] > 20` 같은 조건으로 hold 구간을 걸러내거나 가중치를 낮출 수 있다.
공백 자체도 세션마다 다르다(224840 은 309 tick hold, 231155 은 51 tick) — 부하 의존적이다.

---

## 부록 — 작업 시작 전 매번 확인 (실패 반복 방지)

이번 세션에서 **같은 함정에 세 번** 걸렸다. 순서대로 확인:

```bash
cd ~/motie_ws/stiffness_deploy_ros2
source env.sh                                   # ★ conda 제거가 핵심 (cd && source 로 쓰지 말 것)
echo "DOMAIN=$ROS_DOMAIN_ID python=$(which python3)"   # 9 / /usr/bin/python3

# ① 촉각이 실제로 갱신되나 (ros2 topic hz 로는 절대 구분 안 됨)
python3 tools/sensor_update_rate.py --duration 12 --topics /paxini/right/ft /paxini/right/raw
#    → 판정 LIVE 여야 진행. FROZEN(전 채널 0) 이면 paxini_writer 부터.

# ② move_group 하나만 떠 있나 (팔 이동 쓸 때)
ros2 action list | grep move_action
ros2 node list | grep -c "^/move_group$"        # 1 이어야 함
```

**교훈 3개**
1. `ros2 topic hz` 는 통신 점검용 — **센서 생존 점검에 쓸 수 없다**(0 을 89Hz 로 재발행).
2. `env.sh` 를 먼저 source 하지 않으면 conda python(3.13)이 잡혀 **rclpy import 가 깨진다**.
3. 로그를 남길 때 `| tee` 금지 → `script -qc "명령" 로그.txt` (4번 과제 참고).
