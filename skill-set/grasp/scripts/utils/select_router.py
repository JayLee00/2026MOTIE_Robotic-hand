#!/usr/bin/env python3
"""하이브리드 선택 라우터 — VLM(외형 필터) → 규칙(위치 확정).

흐름: SAM3 후보 → [VLM] 외형/맥락으로 subset 좁힘 → [규칙] subset에서 위치(argmin/argmax)로 확정.
  - 위치·크기("왼쪽/가장 큰")  = 파이썬 규칙 (VLM이 못 하는 argmin/argmax, 100% 정확)
  - 외형·맥락("익은/검은 판")   = VLM (한글→영어 정규화)
안전장치: 외형 조건이 없으면 VLM subset을 무시하고 전체에서 규칙 적용
          (순수 위치 지시에서 VLM 오필터로 오답 나는 것 방지).
"""

# ── 위치/크기 키워드 → (지표, min|max) ──────────────────────────────────────────
# 긴 표현을 먼저 매칭하도록 순서 주의 (예: '가장 큰' 이 '큰' 보다 먼저)
# 한 글자 위치어(위/뒤/왼/오른)는 조사('판 위에')와 충돌하므로 명확한 표현만 매칭
SPATIAL_RULES = [
    ("smallest", (["가장 작은", "제일 작은", "smallest", "작은 것"], "area", "min")),
    ("biggest",  (["가장 큰", "제일 큰", "biggest", "largest", "가장 가까운", "closest", "큰 것"], "area", "max")),
    ("left",     (["왼쪽", "좌측", "leftmost", "left"], "cx", "min")),
    ("right",    (["오른쪽", "우측", "rightmost", "right"], "cx", "max")),
    ("top",      (["위쪽", "상단", "맨 위", "제일 위", "뒤쪽", "topmost", "top", "upper"], "cy", "min")),
    ("bottom",   (["아래쪽", "하단", "맨 아래", "앞쪽", "bottommost", "bottom", "lower"], "cy", "max")),
]

# ── 외형/맥락 용어 한글→영어 (VLM 은 영어가 훨씬 정확) ─────────────────────────
GLOSSARY = {
    "잘 익은": "ripe", "익은": "ripe", "덜 익은": "unripe", "안 익은": "unripe",
    "검은 판 위": "on the dark table", "검은 테이블 위": "on the dark table",
    "검은 판": "dark table", "검은색 판": "dark table", "검은 테이블": "dark table",
    "어두운 판": "dark table", "검은": "dark", "어두운": "dark",
    "흰색 트레이": "white tray", "흰 트레이": "white tray", "하얀 트레이": "white tray",
    "트레이": "tray", "판 위": "surface", "판": "board",
    "스티커": "sticker", "라벨": "label", "상처": "bruise", "흠집": "spot",
    "색 진한": "dark colored", "진한": "dark colored", "밝은": "bright", "색": "color",
}
# 외형 조건 존재 판단용 키워드 (한글 원문 + 영어)
APPEARANCE_WORDS = list(GLOSSARY.keys()) + [
    "ripe", "unripe", "dark", "bright", "color", "colour", "sticker", "label",
    "bruise", "spot", "tray", "table", "board", "surface",
]


def spatial_key(instruction: str):
    """위치/크기 키워드 파싱. 반환 (name, attr, op) 또는 None."""
    s = instruction.lower()
    for name, (kws, attr, op) in SPATIAL_RULES:
        if any(kw.lower() in s for kw in kws):
            return name, attr, op
    return None


def has_appearance(instruction: str) -> bool:
    """외형/맥락 조건이 지시에 포함됐나."""
    s = instruction.lower()
    return any(w.lower() in s for w in APPEARANCE_WORDS)


def normalize_ko_en(instruction: str) -> str:
    """외형 용어를 한글→영어로 치환 (긴 표현 먼저)."""
    s = instruction
    for ko in sorted(GLOSSARY, key=len, reverse=True):
        s = s.replace(ko, GLOSSARY[ko])
    return s


def _metric(det, attr: str) -> float:
    x0, y0, x1, y1 = det["box"]
    if attr == "cx":
        return (x0 + x1) / 2.0
    if attr == "cy":
        return (y0 + y1) / 2.0
    return max(0, x1 - x0) * max(0, y1 - y0)   # area


def rule_pick(attr: str, op: str, detections: list, subset: list) -> int:
    """subset 안에서 attr 기준 min|max 인 index 반환."""
    idxs = subset if subset else list(range(len(detections)))
    pairs = [(i, _metric(detections[i], attr)) for i in idxs]
    chooser = min if op == "min" else max
    return chooser(pairs, key=lambda t: t[1])[0]


def hybrid_select(instruction: str, detections: list, overlay_rgb, qwen,
                  base_rgb=None, query: 'str | None' = None) -> dict:
    """SAM3 후보 → VLM(외형 subset) → 규칙(위치 확정).
    detections: [{'index','score','box'}]. qwen: QwenSelector. overlay_rgb: 번호박스 RGB.
    base_rgb: 박스 없는 원본 RGB — 주어지면 per-crop VQA (후보별 개별 판정, 권장).
    query: SAM3 검출 물체명 — crop VQA 에서 '진짜 그 물체인가' 오검출 필터에 사용.
    반환: {'index','subset','source','reason'}."""
    n = len(detections)
    all_idx = list(range(n))
    sk = spatial_key(instruction)
    app = has_appearance(instruction)

    # ── 1단계 VLM: 외형 조건이 있을 때만 subset 좁힘 (없으면 전체 = 순수 위치 보호) ──
    if app:
        # 규칙이 위치를 처리하므로, VLM 입력에서 위치어를 제거한다.
        # (안 그러면 VLM 이 위치를 스스로 (틀리게) 적용해 subset 을 오염시킴)
        import re
        inst = instruction
        for _name, (kws, _a, _o) in SPATIAL_RULES:
            for kw in kws:
                inst = re.sub(re.escape(kw), " ", inst, flags=re.IGNORECASE)
        inst_en = normalize_ko_en(inst).strip()
        if base_rgb is not None:
            # per-crop VQA: 후보별 crop 개별 판정 (번호박스 grounding 오류 회피)
            f = qwen.filter_by_crops(base_rgb, detections, inst_en, obj_name=query)
        else:
            f = qwen.filter_candidates(overlay_rgb, detections, inst_en)
        subset = f["subset"] or all_idx
        if not f["subset"]:
            print("[hybrid] crop-VQA 전부 탈락 → 전체 후보로 fallback")
        vlm_reason = f"[VLM입력='{inst_en}'] {f['reason']}"
    else:
        subset = all_idx
        vlm_reason = "no-appearance(all)"

    # ── 2단계 규칙: 위치어 있으면 subset 내에서 위치 확정 ──
    if sk:
        name, attr, op = sk
        idx = rule_pick(attr, op, detections, subset)
        source = f"rule:{name}"
        reason = f"VLM[{vlm_reason}] subset={subset} → rule {name}({op} {attr})"
    else:
        # 위치어 없음 → subset 에서 score 최고
        idx = max(subset, key=lambda i: detections[i]["score"])
        source = "vlm"
        reason = f"VLM subset={subset}, score-max → #{idx} ({vlm_reason})"

    return {"index": idx, "subset": subset, "source": source, "reason": reason}
