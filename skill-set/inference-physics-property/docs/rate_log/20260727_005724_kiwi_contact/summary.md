# update rate 계측 요약 — `20260727_005724_kiwi_contact`

## 실행 조건

```
mode=exp
label=kiwi_contact
topics=/hand/right/q_target /paxini/right/ft /hand/right/joint_states
date=2026-07-27T00:57:27+09:00
git_commit=8878e47
git_dirty=6 files
python=/usr/bin/python3 (Python 3.10.12)
kernel=6.8.0-124-generic
ROS_DOMAIN_ID=9 RMW=rmw_fastrtps_cpp
```

## 와이어 rate — `ros2 topic hz`

(idle 구간 제외: 1.0Hz 미만 샘플은 평균에서 제외)

| 토픽 | 유효샘플/전체 | 평균 Hz | 중앙값 | 최소 | 최대 |
|---|---|---|---|---|---|
| `/hand/right/joint_states` | 50/50 | 199.6 | 200.0 | 180.2 | 200.2 |
| `/hand/right/q_target` | 24/24 | 78.9 | 98.0 | 19.8 | 99.0 |
| `/paxini/right/ft` | 50/50 | 89.4 | 89.1 | 86.5 | 90.0 |

## 센서 실측 update rate — 값 변화 기준

`ros2 topic hz`(발행률)와 달리 **값이 실제로 바뀌는 빈도**. 퍼블리셔가 같은 값을
재발행하면 발행률은 정상이어도 센서는 멈춰 있다.

| 토픽 | 판정 | 발행률(Hz) | 갱신(전체 평균) | 갱신(활성) | 중복 | 정적채널 | 비고 |
|---|---|---|---|---|---|---|---|
| `/paxini/right/ft` | LIVE | 89.38 | 13.73 | 45.09 | 84.7% | 3/12 | 양자화 0.1 |
| `/hand/right/joint_states` | LIVE | 199.75 | 161.27 | 198.81 | 19.7% | 0/16 | 양자화 1.0 |
| `/hand/right/kin` | FROZEN | 199.75 | 0.0 | 0.0 | 100.0% | 12/12 | 전 채널 0 고정 |
| `/franka/right/joint_states` | JITTER | 199.75 | 0.0 | 0.0 | 0.0% | 0/7 | max\|Δ\| 중앙 5.25e-06 (LSB) |

- **갱신(전체 평균)** = idle 구간까지 포함한 평균 → 실제보다 낮게 보인다.
- **갱신(활성)** = 1/변화간격 중앙값 → 값이 갱신될 때의 실제 rate. **이쪽을 볼 것.**

## 배포 계측 — `[measure]` (과일=kiwi, 파지임계=10.0N, 스퀴즈임계=13.0N)

| # | 유효 rate(Hz) | 수집(s) | valid/calls | steps (MIN_LEN) | thumb Fz(N) | 임계(N) | 도달률 |
|---|---|---|---|---|---|---|---|
| 0 | 97.8 | 1.52 | 150/150 | 15 (10) | 1.70 | 13.0 | 13% |
| 1 | 97.7 | 1.52 | 150/150 | 15 (10) | 2.10 | 13.0 | 16% |
| 2 | 97.8 | 1.52 | 150/150 | 15 (10) | 1.50 | 13.0 | 12% |

- 유효 rate 평균 **97.8 Hz** (학습 기준 100Hz, n=3)
- thumb 최대 Fz 평균 **1.77 N** (n=3)

## 판정

- ⚠ /hand/right/kin FROZEN — 단 통합모델은 USE_JKIN=False 로 미사용 → 무해
- ✅ 센서 갱신 정상: /paxini/right/ft, /hand/right/joint_states
- ✅ 루프 유효 rate 97.8Hz (학습 100Hz 에 정합)
- ✅ downsample 스텝 최소 15 (MIN_LEN=10) — 샘플 충분
- ❌ thumb Fz 도달률 최저 12% — 힘 미도달 → F6 분기(힘-도달 curl / Path B)
