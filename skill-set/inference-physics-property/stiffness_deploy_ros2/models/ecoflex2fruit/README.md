# ecoflex2fruit 배포 번들 (2026-08-14 · deep_ws/src/ecoflex2fruit README2·README3)

실물 후보 4종 (README3 종합 판정의 축별 대표). `ECO_MODEL=<키>` 로 선택하며
기본값은 champ. 전부 3타깃(mass g / size mm / stif) 동시 출력, 67ch 입력.

| ECO_MODEL | 파일 | 출처 phase | 역할 |
|---|---|---|---|
| `champ` (기본) | `Champ_repair_s42.pth` | readme2/phase8-2_satrepair | 공식 배포 기준점 — 모든 짝지은 검정의 기준 |
| `anchor` | `Anchor_s42.pth` | readme2/phase12_anchor | 차기 챔피언 후보(§12b) — 앵커 경로 강성 추가 출력 |
| `rc` | `RC_v2_5_s42.pth` | readme2/phase6-2_input_recheck | stif·입력 축 대표 — resultant(Σ127) 입력, fold 유의 |
| `gru` | `gru_anchor_s42.pth` | readme2/phase12c_xenc | 전이 축 대표 — 과일 rank 전이 1위(R6) |

- `sensors.json` — paxini 127점 좌표/법선 (변형3·5 계산용 — 학습과 동일 파일 사본).
- 채널 구성(FORCE_CHANNELS 순서 포함)·정규화 통계·라벨 범위·config 스냅샷은 전부
  ckpt 안에 있고, launch/ecoflex_engine.py 가 스냅샷 기반으로 동적 조립한다.
- ※ k-fold(pth 파일명 `_fN_`)·ex10 판은 평가용 부분학습 — 배포에 쓰지 않는다.
  배포 정본 = README2 본판 8시드의 s42 대표.

패리티 검증: launch/test_ecoflex_engine_offline.py — 4종 모두 데모 h5 3개에서
67ch 시퀀스·3타깃 예측이 학습 파이프라인과 일치(상대오차 <1e-3) 확인 (2026-08-14).
원본: deep_ws results/model/readme2/<phase>/<run>_s42.pth (수동 복사)
