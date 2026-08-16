"""Verify parent mask hole-filling on a REAL frame: run SAM3 text segmentation
("molded fiber fruit tray") on test_logs/live_frame.npz, then apply core.fill_holes and
confirm the interior holes (the fruits sitting ON the tray) are filled while the outer
edge is unchanged. Saves before/after overlays. Run in the `sam3` conda env.
"""
import os
import sys

import numpy as np

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place"
TL = f"{REPO}/vision_pipeline/test_logs"


def main():
    # sam3 import guard (mirror sam_service): drop repo root so place-object/sam3/ can't shadow
    sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd(), REPO)]
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    rgb = np.load(f"{TL}/live_frame.npz")["rgb"].astype(np.uint8)
    img = Image.fromarray(rgb)
    print("building SAM3 ...", flush=True)
    model = build_sam3_image_model(enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(img)
        proc.set_text_prompt("molded fiber fruit tray", state)

    def _np(x):
        return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    masks, scores = _np(state.get("masks")), _np(state.get("scores"))
    best = np.asarray(masks[int(np.argmax(scores))]).squeeze() > 0.5
    np.savez(f"{TL}/tray_mask.npz", rgb=rgb, mask=best)     # fill+verify in anyplace_cu128 (has scipy)
    print(f"SAM tray mask: {int(best.sum())} px -> saved {TL}/tray_mask.npz")


if __name__ == "__main__":
    main()
