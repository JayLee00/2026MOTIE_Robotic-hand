"""ModelsHTTP — implements the orchestrator's `Models` interface by calling the
per-model HTTP microservices (each in its own conda env) via npz-over-HTTP.

Endpoint contracts (POST /predict, npz in -> npz out):
  molmo      image:(H,W,3)u8, prompt:str, multi:bool   -> points:(N,2) pixel
  sam        image:(H,W,3)u8, point:(2,)                -> mask:(H,W) bool
  igr        partial:(N,3) meters                       -> dense:(P,3) meters, contacts:(4,3)
             mode=hand_pc [, hand_q:(16,)]              -> hand_pc:(N,3) world (PaXini FK)
  anyplace   parent_pcd:(Np,3), child_pcd:(Nc,3)        -> out_tf:(K,4,4)
"""
import numpy as np

from vision_pipeline.services.rpc import post_npz

# Per-call ceiling so a wedged service aborts with a clear error instead of hanging
# post_npz's 600s default. Generous enough for a cold Molmo generation (~30-60s).
MODEL_TIMEOUT = 180

DEFAULT_URLS = {
    "molmo": "http://127.0.0.1:8810/predict",
    "sam": "http://127.0.0.1:8811/predict",
    "igr": "http://127.0.0.1:8816/predict",
    "anyplace": "http://127.0.0.1:8801/predict",
}


class ModelsHTTP:
    def __init__(self, urls=None):
        self.urls = {**DEFAULT_URLS, **(urls or {})}

    def molmo(self, rgb, prompt, multi=False):
        out = post_npz(self.urls["molmo"], timeout=MODEL_TIMEOUT, image=rgb.astype(np.uint8),
                       prompt=np.array(prompt), multi=np.array(bool(multi)))
        return [tuple(p) for p in out["points"]]

    def sam(self, rgb, point):
        out = post_npz(self.urls["sam"], timeout=MODEL_TIMEOUT, image=rgb.astype(np.uint8),
                       point=np.asarray(point, np.float32))
        return out["mask"].astype(bool)

    def sam_text(self, rgb, text):
        """SAM3 text/concept path (grounding) — fruit-tray parent (spec 3-5-A)."""
        out = post_npz(self.urls["sam"], timeout=MODEL_TIMEOUT, image=rgb.astype(np.uint8),
                       text=np.array(str(text)))
        return out["mask"].astype(bool)

    def complete_igr(self, partial_pc, hand_q=None, contact_mode="points"):
        """Act-VH IGR completion. The service computes the fingertip (지두) contacts itself (URDF
        FK, object-independent); we ship only the partial. contact_mode="points" = 4 fingerprint
        CENTRES (default); "cloud" = the 4 fingerprint pad SURFACES as a dense point cloud.
        Returns (dense (P,3), contacts (M,3)) — the contacts for viz/logging."""
        kw = {"partial": np.asarray(partial_pc, np.float32)}
        if hand_q is not None:
            kw["hand_q"] = np.asarray(hand_q, np.float32)
        if contact_mode != "points":
            kw["contact_mode"] = np.array(str(contact_mode))
        out = post_npz(self.urls["igr"], timeout=MODEL_TIMEOUT, **kw)
        return out["dense"], out["contacts"]

    def hand_pc_paxini(self, hand_q=None, num_points=2048):
        """PaXini-URDF FK hand surface cloud in world (step 3-14-A-1), from the igr service."""
        kw = {"mode": np.array("hand_pc"), "num_points": np.array(int(num_points))}
        if hand_q is not None:
            kw["hand_q"] = np.asarray(hand_q, np.float32)
        out = post_npz(self.urls["igr"], timeout=MODEL_TIMEOUT, **kw)
        return out["hand_pc"]

    def place(self, parent_pc, child_pc_zalign):
        out = post_npz(self.urls["anyplace"], timeout=MODEL_TIMEOUT,
                       parent_pcd=np.asarray(parent_pc, np.float32),
                       child_pcd=np.asarray(child_pc_zalign, np.float32))
        return out["out_tf"]

    def denoise(self, pc):
        # Remove the disconnected mask-edge/segmentation-bleed clusters + floaters
        # (DBSCAN keep-largest) while preserving the object surface — see
        # core/pointcloud.remove_outliers / core/outlier_removal.
        from vision_pipeline.core.pointcloud import remove_outliers
        return remove_outliers(pc)
