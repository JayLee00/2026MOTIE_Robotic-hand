#!/usr/bin/env python
"""SAM3 grounding microservice (runs in the `grasp_fruit` conda env).

The live tracker (`live_bbox_gui.py`) runs in the `ros` env, which does NOT have
`transformers` / SAM3; and the `grasp_fruit` env that DOES have SAM3 has no
`rclpy`. So SAM3 runs here as a tiny TCP service: the GUI sends one frame + a text
query ("orange"), SAM3 detects+segments the fruit, and we return its box + mask.
SAM3 is only called on (re)seed (~130 ms), so the GUI keeps tracking at full rate
with SAM2 in between.

Model is loaded ONCE and kept resident (facebook/sam3, bf16, cuda).

Protocol (TCP, localhost): each message is a 4-byte big-endian length prefix
followed by a pickle payload.
  request : {"query": str, "shape": [H, W, 3], "data": <BGR uint8 bytes>}
  reply   : {"ok": bool, "box": [x0,y0,x1,y1]|None, "score": float,
             "mask_shape": [H, W]|None, "mask_packed": <np.packbits bytes>|None}

Run (grasp_fruit env):
    conda run -n grasp_fruit python scripts/sam3_server.py --port 55003
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import time

import numpy as np


def _recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn):
    hdr = _recv_all(conn, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack(">I", hdr)
    payload = _recv_all(conn, n)
    if payload is None:
        return None
    return pickle.loads(payload)


def _send_msg(conn, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


class Sam3:
    """Resident SAM3 model (mirrors Topdown_Grasp's Sam3Session usage)."""

    def __init__(self, model_id, threshold, mask_threshold):
        import torch
        from transformers import Sam3Processor, Sam3Model

        self.torch = torch
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[sam3-server] loading {model_id} on {self.device} ...", flush=True)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(self.device).eval()
        print("[sam3-server] model ready", flush=True)

    def segment(self, image_rgb, query):
        torch = self.torch
        img_in = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vis = self.model.get_vision_features(pixel_values=img_in.pixel_values)
        txt_in = self.processor(text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(vision_embeds=vis, **txt_in)
        h, w = image_rgb.shape[:2]
        res = self.processor.post_process_instance_segmentation(
            out, threshold=self.threshold, mask_threshold=self.mask_threshold,
            target_sizes=[(h, w)],
        )[0]
        masks = res.get("masks", [])
        scores = res.get("scores", [])
        boxes = res.get("boxes", [])
        if not len(masks):
            return []
        # Return ALL instances (score desc). SAM3 gives the global best, which may
        # be a tray fruit -- the CLIENT filters by the grasp ROI to pick the in-hand
        # one, so it must see every candidate.
        order = np.argsort([float(s) for s in scores])[::-1]
        dets = []
        for i in order[:10]:
            i = int(i)
            dets.append({
                "mask": np.asarray(masks[i].cpu()).astype(bool),
                "score": float(scores[i]),
                "box_xyxy": [float(v) for v in boxes[i].cpu().tolist()],
            })
        return dets


def main(argv=None):
    ap = argparse.ArgumentParser(description="SAM3 grounding TCP microservice")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=55003)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    sam3 = Sam3(args.model_id, args.threshold, args.mask_threshold)

    # Warm up so the first real request is fast.
    try:
        sam3.segment((np.random.rand(480, 640, 3) * 255).astype(np.uint8), "orange")
        print("[sam3-server] warmup done", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[sam3-server] warmup failed (non-fatal): {e}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[sam3-server] listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[sam3-server] client {addr}", flush=True)
        try:
            while True:
                req = _recv_msg(conn)
                if req is None:
                    break
                try:
                    h, w, _ = req["shape"]
                    bgr = np.frombuffer(req["data"], dtype=np.uint8).reshape(h, w, 3)
                    rgb = bgr[:, :, ::-1]
                    t0 = time.time()
                    dets = sam3.segment(np.ascontiguousarray(rgb), req["query"])
                    dt = (time.time() - t0) * 1000
                    out_dets = []
                    for d in dets:
                        m = d["mask"]
                        out_dets.append({
                            "box": d["box_xyxy"], "score": d["score"],
                            "mask_shape": [int(m.shape[0]), int(m.shape[1])],
                            "mask_packed": np.packbits(m).tobytes(),
                        })
                    _send_msg(conn, {"ok": bool(out_dets), "dets": out_dets})
                    print(f"[sam3-server] {req['query']!r} -> {len(out_dets)} det "
                          f"({dt:.0f}ms)", flush=True)
                except Exception as e:  # noqa: BLE001
                    _send_msg(conn, {"ok": False, "dets": [], "error": str(e)})
                    print(f"[sam3-server] request error: {e}", flush=True)
        finally:
            conn.close()
            print("[sam3-server] client disconnected", flush=True)


if __name__ == "__main__":
    sys.exit(main())
