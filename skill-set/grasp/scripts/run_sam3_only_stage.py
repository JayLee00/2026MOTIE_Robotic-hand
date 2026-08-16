#!/usr/bin/env python3
"""
SAM3 segmentation stage — 텍스트 쿼리로 마스크 생성.

Input:  RGB-D NPZ bundle (keys: rgb, depth, K)  OR  a plain RGB image
Output: binary mask PNG  +  JSON summary

Usage:
    python scripts/run_sam3_only_stage.py \
        --input  data/raw/scene.npz \
        --query  "apple" \
        --output_dir  data/interim/

    python scripts/run_sam3_only_stage.py \
        --image  data/raw/scene.png \
        --query  "orange" \
        --output_dir  data/interim/

Output files:
    {output_dir}/{stem}_mask.png          — binary mask (255=object, 0=background)
    {output_dir}/{stem}_overlay.png       — visualisation overlay
    {output_dir}/{stem}_sam3.json         — SAM3 결과 JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_image(args) -> tuple[np.ndarray, str]:
    if args.input:
        npz = np.load(args.input)
        rgb = npz["rgb"]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb, Path(args.input).stem
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), Path(args.image).stem


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------

def _run_sam3_query(model, processor, image_rgb, vision_embeds,
                    text: str, threshold: float, mask_threshold: float,
                    device) -> Optional[dict]:
    import torch

    text_inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds, **text_inputs)

    h, w = image_rgb.shape[:2]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(h, w)],
    )[0]

    masks  = results.get("masks", [])
    scores = results.get("scores", [])
    boxes  = results.get("boxes", [])
    if len(masks) == 0:
        return None

    best_idx = int(np.argmax([float(s) for s in scores]))
    mask_np = masks[best_idx]
    if hasattr(mask_np, "cpu"):
        mask_np = mask_np.cpu()
    box = boxes[best_idx]
    if hasattr(box, "cpu"):
        box = box.cpu()
    return {
        "mask":          np.array(mask_np).astype(bool),
        "score":         float(scores[best_idx]),
        "box_xyxy":      box.tolist() if hasattr(box, "tolist") else list(box),
        "num_instances": len(masks),
        "all_scores":    [float(s) for s in scores],
    }


def run_sam3(image_rgb: np.ndarray, query: str,
             model_id: str, threshold: float, mask_threshold: float) -> dict:
    import torch
    from transformers import Sam3Processor, Sam3Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  SAM3 loading: {model_id}")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print("  SAM3 loaded")

    img_inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    print(f"  Query: {query!r}")
    result = _run_sam3_query(model, processor, image_rgb,
                             vision_embeds, query, threshold, mask_threshold, device)

    del model
    torch.cuda.empty_cache()

    if result is None:
        print(f"  No result for {query!r}")
        return {"used_query": None, "queries_tried": [query], "mask": None, "score": None}

    print(f"  OK: {query!r}  score={result['score']:.3f}  mask_px={result['mask'].sum()}")
    return {
        "used_query":    query,
        "queries_tried": [query],
        "mask":          result["mask"],
        "score":         result["score"],
        "box_xyxy":      result["box_xyxy"],
        "num_instances": result["num_instances"],
        "all_scores":    result["all_scores"],
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(image_rgb: np.ndarray, query: str, sam3: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = sam3.get("mask")
    if mask is not None and mask.any():
        overlay = canvas.copy()
        overlay[mask] = (0, 0, 220)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0)
        box = sam3.get("box_xyxy")
        if box:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 0), 2)
    score_str = f"{sam3['score']:.3f}" if sam3.get("score") is not None else "N/A"
    lines = [
        f"query : {query}",
        f"score : {score_str}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 80), 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="RGB-D NPZ bundle (keys: rgb, depth, K)")
    src.add_argument("--image", help="Plain RGB/BGR image file (.png/.jpg)")

    p.add_argument("--query", required=True,
                   help="SAM3에 직접 전달할 텍스트 (예: 'apple')")

    p.add_argument("--sam3_model_id",       default="facebook/sam3")
    p.add_argument("--sam3_threshold",      type=float, default=0.5)
    p.add_argument("--sam3_mask_threshold", type=float, default=0.5)

    p.add_argument("--output_dir", default=str(ROOT / "data" / "interim"))

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_rgb, stem = load_image(args)
    print(f"Image: {stem}  shape={image_rgb.shape}")

    print(f"\n[1/1] SAM3  query={args.query!r}")
    sam3 = run_sam3(
        image_rgb=image_rgb,
        query=args.query,
        model_id=args.sam3_model_id,
        threshold=args.sam3_threshold,
        mask_threshold=args.sam3_mask_threshold,
    )

    mask = sam3.get("mask")
    if mask is None or not mask.any():
        print("\n[WARN] No mask produced — exiting with error.")
        sys.exit(1)

    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    print(f"\nSaved mask: {mask_path}")

    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), draw_overlay(image_rgb, args.query, sam3))
    print(f"Saved overlay: {overlay_path}")

    summary = {
        "stem":        stem,
        "query":       args.query,
        "sam3": {
            "model":         args.sam3_model_id,
            "queries_tried": sam3.get("queries_tried", []),
            "used_query":    sam3.get("used_query"),
            "score":         sam3.get("score"),
            "box_xyxy":      sam3.get("box_xyxy"),
            "num_instances": sam3.get("num_instances"),
            "mask_pixels":   int(mask.sum()),
        },
        "outputs": {
            "mask":    str(mask_path),
            "overlay": str(overlay_path),
        },
    }
    json_path = out_dir / f"{stem}_sam3.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON:  {json_path}")

    print(f"\nDone.  mask={mask_path}")


if __name__ == "__main__":
    main()
