"""Molmo2-8B pointing service (run in conda env `molmo`, transformers 4.57.1).

    POST /predict  npz{ image:(H,W,3)u8, prompt:str [, multi:bool] }  ->  npz{ points:(N,2) pixel }

Molmo emits XML point tags with coords normalized to 0-1000; we de-normalize to
the input image's pixels. Single prompts return 1 point; multi (local_place) many.

Run:  conda activate molmo && python vision_pipeline/services/molmo_service.py [--port 8810]
"""
import argparse
import os
import re
import sys

import numpy as np

# place-object is added to sys.path inside main() AFTER the model loads, so the
# repo-root place-object/molmo/ dir can't shadow any `molmo` import in remote code.

MODEL_ID = "allenai/Molmo2-8B"
# Molmo2 emits  coords="<group_id> <idx> XXX YYY <idx> XXX YYY ..."  with X,Y
# zero-padded to 3-4 digits, normalized to 0-1000. Requiring X,Y to be 3-4 digits
# naturally skips the 1-digit group_id/index. (verified on real captures)
_POINT = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
_XY = re.compile(r'(?:x\d*|point_x|x)\s*=\s*"?(\d+(?:\.\d+)?)"?[,\s]+(?:y\d*|point_y|y)\s*=\s*"?(\d+(?:\.\d+)?)"?',
                 re.I)


def parse_points(text, W, H):
    """Return list of (x_px, y_px). Handles the 0-1000 'coords="ID X Y; ..."' form
    and an x=.. y=.. fallback; scales by image size when values look normalized."""
    pts = []
    for m in re.finditer(r'coords\s*=\s*"([^"]+)"', text):
        for _id, x, y in _POINT.findall(m.group(1)):
            pts.append((float(x) / 1000.0 * W, float(y) / 1000.0 * H))
    if not pts:                                  # fallback: x=.. y=.. (normalized 0-100 or 0-1)
        for x, y in _XY.findall(text):
            x, y = float(x), float(y)
            sx, sy = (W, H) if x <= 1.5 else (W / 100.0, H / 100.0)
            pts.append((x * sx, y * sy))
    # keep in-bounds
    return [(x, y) for x, y in pts if 0 <= x <= W and 0 <= y <= H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8810)
    a = ap.parse_args()

    sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd(), "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")]
    import torch  # noqa
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[molmo] loading {MODEL_ID} ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(   # bf16 (~17GB), single GPU (no offload/meta)
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda:0")
    sys.path.append("/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")           # add AFTER model load
    from vision_pipeline.services.rpc import serve
    print("[molmo] ready", flush=True)

    def predict(inp):
        img = Image.fromarray(np.asarray(inp["image"]).astype(np.uint8))
        W, H = img.size
        prompt = str(inp["prompt"])
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pts = parse_points(text, W, H)
        if torch.cuda.is_available():                     # release generation cache for co-resident services
            torch.cuda.empty_cache()
        return {"points": np.array(pts, float).reshape(-1, 2), "raw": np.array(text)}

    serve(predict, a.port, name="molmo")


if __name__ == "__main__":
    main()
