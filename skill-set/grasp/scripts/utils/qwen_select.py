#!/usr/bin/env python3
"""Qwen2-VL 기반 파지 대상 선택기.

SAM3 가 찾은 여러 후보(번호 박스 overlay 이미지 + 검출 리스트)와 자연어 지시를 받아,
Qwen2-VL 이 '몇 번을 잡을지' index 를 고른다. 마스크는 SAM3 것을 그대로 쓴다.

이미지 + 텍스트 둘 다 근거로 사용 → 외형(익은 정도/색/위치) 판단 가능.
JSON 파싱 실패 / 범위 초과 시 score 최고(argmax) 로 fallback 하여 로봇이 멈추지 않게 함.
"""
import json
import re

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"   # 4-bit 로 로드 (아래 load_4bit)

SYSTEM_PROMPT = (
    "You are a robot vision assistant selecting exactly ONE object to grasp.\n"
    "All numbered boxes (#0, #1, ...) mark objects of the SAME type — do NOT rename or "
    "re-classify them (e.g. if they are kiwis, never call one an orange).\n"
    "Image coordinates: x increases to the RIGHT, y increases DOWNWARD, origin at the "
    "top-left corner. Each candidate lists its box center (cx, cy) and area.\n"
    "Resolve SPATIAL words strictly from the numbers:\n"
    "  'left'  -> smallest cx      'right' -> largest cx\n"
    "  'top'/'back'/'far' -> smallest cy      'bottom'/'front'/'near' -> largest cy\n"
    "  'biggest'/'closest'/'largest' -> largest area      'smallest' -> smallest area\n"
    "Resolve APPEARANCE words (ripe, dark, bright, color, spot) by LOOKING at the image.\n"
    'Reply in ENGLISH ONLY, with nothing but a JSON object: '
    '{"index": <int>, "reason": "<short English reason>"}. '
    "No markdown fences, no extra text."
)

# per-crop VQA: 후보 1개씩 crop 을 보여주고 개별 yes/no 판정 (multi-instance grounding 회피)
CROP_SYSTEM = (
    "You are a vision QA assistant for a robot. You see a cropped photo with ONE "
    "candidate object inside a GREEN box (other objects may be partially visible).\n"
    "Judge ONLY the object inside the green box, factually, from the image.\n"
    "First describe the SURFACE the boxed object sits on (its immediate background), "
    "THEN decide if object+surface match the description. Being the right kind of "
    "object is NOT enough — the surface/context must also match.\n"
    'Reply in ENGLISH ONLY, nothing but a JSON object: '
    '{"is_object": true|false, "surface": "<short>", "match": true|false, '
    '"reason": "<short>"}. No markdown fences, no extra text.'
)

# 하이브리드용: 외형/맥락으로 후보를 좁히는 필터 프롬프트 (위치어는 규칙이 처리)
FILTER_SYSTEM = (
    "You filter grasp candidates by APPEARANCE and CONTEXT only "
    "(color, ripeness, the surface it sits on, stickers, spots).\n"
    "IGNORE all position/size words (left, right, top, bottom, biggest, smallest) — "
    "those are handled by a separate rule, not you.\n"
    "Return the indices of ALL candidates whose appearance/context matches the instruction. "
    "If the instruction describes NO appearance/context (only position/size, or just the "
    "object name), return ALL indices.\n"
    'Reply in ENGLISH ONLY, nothing but JSON: {"candidates": [ints], "reason": "<short>"}. '
    "No markdown, no extra text."
)


class QwenSelector:
    """Qwen2-VL 을 1회 로드해 반복 선택에 사용."""

    def __init__(self, model_id: str = DEFAULT_MODEL,
                 min_pixels: int = 256 * 28 * 28,
                 max_pixels: int = 768 * 28 * 28,
                 load_4bit: bool = True):
        import torch
        from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                                  BitsAndBytesConfig)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Qwen] 로딩: {model_id}  ({self.device}, 4bit={load_4bit})")
        # min/max_pixels 로 이미지 토큰 수 제한 → VRAM 절감
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        if load_4bit and self.device == "cuda":
            # nf4 4-bit 양자화 (7B 를 ~6GB 로 → 16GB GPU 에 여유있게)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, quantization_config=bnb,
                device_map="cuda", torch_dtype=torch.bfloat16).eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16).to(self.device).eval()
        print("[Qwen] 로딩 완료.")

    def select(self, overlay_rgb: np.ndarray, detections: list,
               instruction: str) -> dict:
        """overlay_rgb: 번호 박스 그린 RGB(np.uint8).
        detections: [{'index','score','box'}] (index 순서).
        반환: {'index','reason','source'} (source='qwen' | 'fallback')."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        n = len(detections)
        fallback_idx = (int(np.argmax([d["score"] for d in detections]))
                        if n else 0)
        h, w = overlay_rgb.shape[:2]

        # 각 박스의 중심(cx,cy)·면적을 미리 계산해 제공 (약한 모델이 직접 계산 못 함)
        lines = []
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            area = max(0, x1 - x0) * max(0, y1 - y0)
            lines.append(f'  #{d["index"]}: center=({cx},{cy}) area={area} '
                         f'score={d["score"]:.2f}')
        det_lines = "\n".join(lines)
        user_text = (
            f"Image size: {w}x{h} (width x height).\n"
            f"Candidates ({n}):\n{det_lines}\n\n"
            f"Instruction: {instruction}\n\n"
            f'Pick the single best-matching index. '
            f'Return JSON {{"index": int in 0..{n - 1}, "reason": str}}.')

        pil = Image.fromarray(overlay_rgb)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=128,
                                      do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        print(f"[Qwen] raw output: {out!r}")

        idx, reason = self._parse(out, n)
        if idx is None:
            print(f"[Qwen] 파싱 실패/범위초과 → fallback score-max #{fallback_idx}")
            return {"index": fallback_idx,
                    "reason": "fallback(score-max)", "source": "fallback"}
        return {"index": idx, "reason": reason, "source": "qwen"}

    # ── 공용 1회 생성 helper ──────────────────────────────────────────────────
    def _generate(self, rgb: np.ndarray, system_prompt: str, user_text: str,
                  max_new_tokens: int = 96) -> str:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        pil = Image.fromarray(rgb)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        return self.processor.batch_decode(
            gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # ── per-crop VQA: 후보마다 crop 을 잘라 개별 yes/no 판정 ──────────────────
    def filter_by_crops(self, base_rgb: np.ndarray, detections: list,
                        instruction_en: str, obj_name: 'str | None' = None) -> dict:
        """번호박스 전체 이미지 대신 후보별 crop 으로 개별 질문.
        Q1: 진짜 obj_name 인가 (SAM3 오검출 제거)
        Q2: instruction 의 외형/맥락 조건에 맞나
        둘 다 yes 인 index 만 subset 으로. 반환: {'subset':[int], 'reason':str}."""
        import cv2
        h, w = base_rgb.shape[:2]
        obj = obj_name or "target object"
        subset, notes = [], []
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            # box 의 1.5배 + 40px 여유 → 놓인 표면(트레이/테이블)까지 보이게
            half = int(max(x1 - x0, y1 - y0) * 1.5) + 40
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            gx0, gy0 = max(0, cx - half), max(0, cy - half)
            gx1, gy1 = min(w, cx + half), min(h, cy + half)
            crop = np.ascontiguousarray(base_rgb[gy0:gy1, gx0:gx1])
            cv2.rectangle(crop, (x0 - gx0, y0 - gy0), (x1 - gx0, y1 - gy0),
                          (0, 255, 0), 2)
            user_text = (
                f"Q1: Is the object inside the GREEN box a {obj}?\n"
                f"Q2: What SURFACE is the boxed object sitting on "
                f"(e.g. white plastic tray, dark table, wooden desk)?\n"
                f"Q3: Given Q1+Q2, does the boxed object match this "
                f'description: "{instruction_en}"? '
                f"(the surface/context in the description must match Q2)\n"
                'Reply JSON only: {"is_object": true|false, "surface": "<short>", '
                '"match": true|false, "reason": "<short>"}')
            out = self._generate(crop, CROP_SYSTEM, user_text)
            is_obj, match, surface, reason = self._parse_crop(out)
            ok = is_obj and match
            print(f"[Qwen] crop #{d['index']}: is_{obj}={is_obj} "
                  f"surface={surface!r} match={match}  ({reason})")
            if ok:
                subset.append(d["index"])
            notes.append(f"#{d['index']}={'O' if ok else 'X'}")
        return {"subset": subset,
                "reason": f"crop-VQA {' '.join(notes)}"}

    @staticmethod
    def _parse_crop(text: str):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return False, False, "", "parse-fail"
        try:
            obj = json.loads(m.group(0))
            return (bool(obj.get("is_object")), bool(obj.get("match")),
                    str(obj.get("surface", "")), str(obj.get("reason", "")))
        except Exception:
            return False, False, "", "parse-fail"

    # ── 하이브리드용: 외형/맥락으로 후보 subset 필터 ──────────────────────────
    def filter_candidates(self, overlay_rgb: np.ndarray, detections: list,
                          instruction_en: str) -> dict:
        """지시의 외형/맥락 조건에 맞는 index subset 반환 (위치어는 무시).
        조건이 없으면 전체 반환. 반환: {'subset':[int], 'reason':str}."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        n = len(detections)
        all_idx = list(range(n))
        lines = []
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            lines.append(f'  #{d["index"]}: center=({cx},{cy}) score={d["score"]:.2f}')
        user_text = (
            f"Candidates ({n}):\n" + "\n".join(lines) +
            f"\n\nInstruction: {instruction_en}\n"
            f'Return JSON {{"candidates": [subset of 0..{n - 1}], "reason": str}}.')

        pil = Image.fromarray(overlay_rgb)
        messages = [
            {"role": "system", "content": FILTER_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        out = self.processor.batch_decode(
            gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"[Qwen] filter raw: {out!r}")

        subset, reason = self._parse_list(out, n)
        if not subset:
            print("[Qwen] filter 파싱실패/빈값 → 전체 후보 사용")
            return {"subset": all_idx, "reason": "filter-fallback(all)"}
        return {"subset": subset, "reason": reason}

    @staticmethod
    def _parse_list(text: str, n: int):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return [], ""
        try:
            obj = json.loads(m.group(0))
            cand = obj.get("candidates", [])
            out = sorted({int(i) for i in cand if 0 <= int(i) < n})
        except Exception:
            return [], ""
        return out, str(obj.get("reason", ""))

    @staticmethod
    def _parse(text: str, n: int):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, None
        try:
            obj = json.loads(m.group(0))
            idx = int(obj["index"])
        except Exception:
            return None, None
        if not (0 <= idx < n):
            return None, None
        return idx, str(obj.get("reason", ""))

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
        print("[Qwen] 모델 해제 완료.")
