# 과일 사진 넣는 곳

`stiffness_gui.py` 가 여기서 사진을 읽습니다. 아래 이름으로 넣으세요 (**JPEG/PNG, 확장자 없어도 됨**).

| 파일명(확장자 무관) | 과일 |
|---|---|
| `plum`   (`plum.jpg`/`plum.png` …)   | 자두 |
| `kiwi`   | 키위 |
| `tomato` | 토마토 |
| `lemon`  | 레몬 |

- **형식: JPEG/PNG.** 확장자가 없어도(`plum` 처럼) PIL 이 내용으로 자동 판별합니다.
  JPEG 는 PIL(python3-pil)로 읽고, PIL 이 없으면 PNG/GIF 만 됩니다
  (conda python 이면 `pip install pillow`).
- **크기**: PIL 이 비율 유지하며 부드럽게 축소하므로 원본 크기 그대로 넣어도 됩니다
  (썸네일 ~110px, 큰 사진 ~360px 로 표시).
- 파일이 없으면 회색 "(사진 없음)" 플레이스홀더가 표시됩니다 — 사진만 넣으면 즉시 대체됩니다.

> 참고: GUI 는 시스템 python3 로 실행되며 tkinter 가 필요합니다: `sudo apt install -y python3-tk`
