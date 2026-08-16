"""SAM3 segmentation service (run in conda env `sam3`, py3.12). Two paths:

  * Interactive point->mask (SAM2-style PVS / instance interactivity) — table/child.
        POST /predict  npz{ image:(H,W,3)u8, point:(2,) [, label:int=1] }
  * Text/concept -> mask (grounding) — fruit-tray parent (spec 3-5-A).
        POST /predict  npz{ image:(H,W,3)u8, text:str }
  -> npz{ mask:(H,W)u8 }   (best-scoring detection)

Run:  conda activate sam3 && python vision_pipeline/services/sam_service.py [--port 8811]
"""
import argparse
import os
import sys

import numpy as np

# NOTE: do NOT add /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place to sys.path before importing sam3 — the
# repo-root place-object/sam3/ dir would be picked up as a namespace package and
# shadow the pip-installed sam3 (editable meta-path finder). Import sam3 first (below).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8811)
    a = ap.parse_args()

    # strip cwd/'' and repo root so place-object/sam3/ can't shadow the installed sam3
    sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd(), "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")]

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sys.path.append("/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")           # add AFTER sam3 import (see note above)
    from vision_pipeline.services.rpc import serve

    print("[sam3] building model (inst-interactivity) ...", flush=True)
    # bpe_path=None -> auto (pkg assets); checkpoint auto-loaded from HF (facebook/sam3)
    model = build_sam3_image_model(enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    print("[sam3] ready", flush=True)

    def _np(x):     # .float() so bfloat16 (autocast) scores/masks survive .numpy()
        return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    @torch.no_grad()
    def predict(inp):
        img = Image.fromarray(np.asarray(inp["image"]).astype(np.uint8))  # PIL -> correct dims
        W, H = img.size
        # --- text / concept path (grounding) ---
        if "text" in inp:
            text = str(inp["text"])
            with torch.autocast("cuda", dtype=torch.bfloat16):   # thread-local (ThreadingHTTPServer)
                state = processor.set_image(img)
                processor.set_text_prompt(text, state)
            masks = _np(state.get("masks"))
            scores = _np(state.get("scores"))
            if masks is None or len(masks) == 0:
                mask = np.zeros((H, W), np.uint8)                # nothing found
            else:
                best = np.asarray(masks[int(np.argmax(scores))]).squeeze()  # (H,W)
                mask = (best > 0).astype(np.uint8)
        else:
            # --- interactive point path (PVS) ---
            pt = np.asarray(inp["point"], float).reshape(1, 2)   # (1,2) x,y pixels
            label = np.array([int(inp["label"])]) if "label" in inp else np.array([1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(img)
                masks, scores, _ = model.predict_inst(
                    state, point_coords=pt, point_labels=label, multimask_output=True)
            best = np.asarray(masks)[int(np.argmax(np.asarray(scores)))]  # IoU argmax
            mask = (best > 0).astype(np.uint8)
        if torch.cuda.is_available():                            # release cache for co-resident services
            torch.cuda.empty_cache()
        return {"mask": mask}

    serve(predict, a.port, name="sam3")


if __name__ == "__main__":
    main()
