#!/usr/bin/env python3
"""SAM3 진단 — 한 이미지에 여러 프롬프트 × threshold 를 격자로 돌려 검출 비교.

학습 없이(Tier 0) "왜/얼마나 안 잡히는지" 측정하는 용도.
파이프라인(run_pipeline_interactive.Sam3Session)과 동일하게 이미지를 먹이되,
best 1개가 아니라 검출된 **모든 인스턴스**를 색칠하고 score 를 표기한다.

용법 (grasp_fruit conda python 으로 실행):
  ~/miniconda3/envs/grasp_fruit/bin/python scripts/sam3_probe_kiwi.py \
      --input data/raw/kiwi_probe_000.npz \
      --prompts "kiwi" "kiwi fruit" "green kiwi" "brown fuzzy kiwi" "kiwifruit" \
      --thresholds 0.5 0.4 0.3 0.2 \
      --out data/interim/kiwi_probe

출력: data/interim/kiwi_probe/{prompt}__thr{t}.png (overlay) + 콘솔 요약표.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import cv2
import torch
from transformers import Sam3Processor, Sam3Model

# 인스턴스별 색 (BGR)
COLORS = [(0, 0, 220), (0, 200, 0), (220, 120, 0), (0, 200, 220),
          (220, 0, 220), (120, 120, 255), (255, 180, 0), (0, 120, 255)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="npz (키 'rgb', 파이프라인과 동일 BGR 저장)")
    ap.add_argument("--model_id", default="facebook/sam3")
    ap.add_argument("--prompts", nargs="+", default=["kiwi"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.4, 0.3])
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--out", default="data/interim/kiwi_probe")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.input)
    rgb = npz["rgb"]
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    # 파이프라인과 동일: 저장은 BGR → SAM3 입력은 RGB
    image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    canvas_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # overlay 바탕
    h, w = image_rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAM3] 로딩: {args.model_id}  ({device})")
    processor = Sam3Processor.from_pretrained(args.model_id)
    model = Sam3Model.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # vision feature 는 이미지당 1회 (프롬프트 무관 재사용)
    img_in = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        vis = model.get_vision_features(pixel_values=img_in.pixel_values)

    # 요약표: prompt → {thr: count}
    summary = {}

    for prompt in args.prompts:
        txt_in = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_m = model(vision_embeds=vis, **txt_in)  # 프롬프트당 1회

        summary[prompt] = {}
        for thr in args.thresholds:
            res = processor.post_process_instance_segmentation(
                out_m, threshold=thr, mask_threshold=args.mask_threshold,
                target_sizes=[(h, w)])[0]
            masks = res.get("masks", [])
            scores = res.get("scores", [])
            boxes = res.get("boxes", [])
            n = len(masks)
            summary[prompt][thr] = n

            # overlay 그리기
            ov = canvas_bgr.copy()
            for i in range(n):
                m = np.array(masks[i].cpu()).astype(bool)
                color = COLORS[i % len(COLORS)]
                ov[m] = color
            blended = cv2.addWeighted(ov, 0.45, canvas_bgr, 0.55, 0.0)
            for i in range(n):
                x0, y0, x1, y1 = [int(v) for v in boxes[i].cpu().tolist()]
                color = COLORS[i % len(COLORS)]
                cv2.rectangle(blended, (x0, y0), (x1, y1), color, 2)
                cv2.putText(blended, f"{float(scores[i]):.2f}", (x0, max(y0 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            tag = f"{prompt}  thr={thr}  n={n}"
            cv2.putText(blended, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            safe = prompt.replace(" ", "_")
            fn = out / f"{safe}__thr{thr}.png"
            cv2.imwrite(str(fn), blended)
            print(f"  [{prompt!r:22}] thr={thr}  → n={n}  scores={[round(float(s),3) for s in scores]}")

    # 요약표 출력
    print("\n===== 검출 개수 요약 (행=prompt, 열=threshold) =====")
    thrs = args.thresholds
    print(f"{'prompt':24} " + " ".join(f"thr{t:<5}" for t in thrs))
    for prompt in args.prompts:
        row = " ".join(f"{summary[prompt][t]:<8}" for t in thrs)
        print(f"{prompt:24} {row}")
    print(f"\noverlay 저장: {out}/")


if __name__ == "__main__":
    main()
