"""LIVE (streaming) SAM2 video-predictor tracker for the live RGB path.

The bundled ``sam2`` install ships :class:`sam2.sam2_video_predictor.SAM2VideoPredictor`
whose :meth:`init_state` is *offline*: it loads a FIXED frame set from disk into
``inference_state["images"]`` and cannot append live frames. This module instead
drives the predictor PER FRAME (streaming) by maintaining a minimal
``inference_state`` ourselves and feeding each newly captured frame into it,
following the well-known "segment-anything-2-real-time" ``SAM2CameraPredictor``
recipe (github.com/Gy920/segment-anything-2-real-time) ADAPTED to this install's
``track_step`` signature.

We mirror the predictor's own internals rather than re-implementing the model
math:

  * frame preprocessing matches ``sam2.utils.misc._load_img_as_tensor``: resize to
    ``image_size`` x ``image_size``, divide by 255, then normalise with
    mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) -> a ``(3,S,S)`` tensor,
    exactly what ``init_state`` stores in ``inference_state["images"][i]``.
  * per-frame tracking reuses the predictor's own
    :meth:`SAM2VideoPredictor._run_single_frame_inference`, which calls
    ``track_step`` with the correct feature-prep (``current_vision_feats`` /
    ``current_vision_pos_embeds`` / ``feat_sizes`` lists) and manages the compact
    memory-offloaded output dict for us.
  * the cond frame (frame 0) is seeded with point prompts via ``track_step``
    (``is_init_cond_frame=True``, ``run_mem_encoder=True``); subsequent frames run
    with ``is_init_cond_frame=False`` and read the memory bank from
    ``output_dict``.
  * mask logits (``image_size//4`` resolution) are upsampled to the original
    ``(H0, W0)`` with bilinear interpolation (the same op
    ``_get_orig_video_res_output`` uses) and thresholded at 0.

We bound memory/compute by keeping the cond frame plus the last ``N`` (default 7)
non-cond outputs in ``output_dict["non_cond_frame_outputs"]``.

Clean boundary: this module imports nothing from the sim stack (no recorder /
Isaac / sim-side modules) -- only ``numpy`` eagerly, with ``torch`` and ``sam2``
imported lazily so the module stays import-safe without them.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Sam2StreamTracker"]

# ImageNet normalisation used by sam2's load_video_frames / _load_img_as_tensor.
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


class Sam2StreamTracker:
    """Stateful, per-frame SAM2 video-predictor tracker (no fixed video).

    Usage::

        tr = Sam2StreamTracker(ckpt, config, device="cuda")
        tr.load_first_frame(rgb0)             # encode frame 0
        mask0 = tr.add_prompt([(u, v)])       # seed -> bool mask at (H0, W0)
        mask1 = tr.track(rgb1)                # propagate via memory
        ...
        tr.reset()                            # re-seed from a new frame

    All public methods that touch the model run under ``torch.inference_mode``
    and (on cuda) bfloat16 autocast, matching the image-predictor path.
    """

    def __init__(self, ckpt: str, config: str, device: str = "cuda") -> None:
        import torch  # lazy heavy import
        from sam2.build_sam import build_sam2_video_predictor

        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._torch = torch
        # vos_optimized=False -> plain SAM2VideoPredictor (no torch.compile);
        # apply_postprocessing keeps the default hole-filling behaviour.
        self.predictor = build_sam2_video_predictor(config, ckpt, device=device)
        self.predictor.eval()

        # Bound the non-cond memory: cond frame + last N non-cond outputs.
        self.max_non_cond = 7
        self.obj_id = 0

        # Streaming state (populated by load_first_frame / add_prompt / track).
        self._state: Optional[dict] = None
        self._output_dict: Optional[dict] = None
        self._frame_idx = -1
        self._H0 = 0
        self._W0 = 0
        # mean/std tensors created once on the compute device.
        self._mean = torch.tensor(_IMG_MEAN, dtype=torch.float32, device=device)[
            :, None, None
        ]
        self._std = torch.tensor(_IMG_STD, dtype=torch.float32, device=device)[
            :, None, None
        ]

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _autocast(self):
        torch = self._torch
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return _nullcontext()

    def _preprocess(self, frame_rgb: np.ndarray):
        """RGB uint8 (H,W,3) -> normalised (3,S,S) float tensor on device.

        Mirrors sam2.utils.misc._load_img_as_tensor: PIL-equivalent resize to
        (image_size, image_size), /255, then (x - mean) / std.
        """
        torch = self._torch
        S = self.predictor.image_size
        img = np.asarray(frame_rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) RGB frame, got shape {img.shape}")
        # (H,W,3) uint8 -> (3,H,W) float on device.
        t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
        t = t.permute(2, 0, 1).float()
        # Resize to (S, S) with bilinear (PIL.resize default is bilinear-ish; the
        # model is robust to the exact filter). antialias matches torchvision Resize.
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(S, S), mode="bilinear",
            align_corners=False, antialias=True,
        ).squeeze(0)
        t = t / 255.0
        t = (t - self._mean) / self._std
        return t  # (3, S, S)

    def _new_state(self, frame_tensor) -> dict:
        """Build a minimal inference_state for a single live frame (index 0)."""
        torch = self._torch
        device = torch.device(self.device)
        state: dict = {}
        # images grows as we stream; index 0 is the current frame.
        state["images"] = [frame_tensor]
        state["num_frames"] = 1
        state["offload_video_to_cpu"] = False
        state["offload_state_to_cpu"] = False
        state["video_height"] = self._H0
        state["video_width"] = self._W0
        state["device"] = device
        state["storage_device"] = device
        # the predictor's _get_image_feature reads/writes these:
        state["cached_features"] = {}
        state["constants"] = {}
        # per-obj structures (used by _consolidate_* paths we do NOT call, but
        # _get_maskmem_pos_enc uses ["constants"] only). Kept for parity / safety.
        state["point_inputs_per_obj"] = {self.obj_id: {}}
        state["mask_inputs_per_obj"] = {self.obj_id: {}}
        state["obj_id_to_idx"] = {self.obj_id: 0}
        state["obj_idx_to_id"] = {0: self.obj_id}
        state["obj_ids"] = [self.obj_id]
        return state

    def _push_frame(self, frame_tensor):
        """Append a new live frame, advance frame_idx, drop stale cache."""
        self._frame_idx += 1
        idx = self._frame_idx
        # Grow the images list so _get_image_feature(idx) finds the frame.
        imgs = self._state["images"]
        while len(imgs) <= idx:
            imgs.append(None)
        imgs[idx] = frame_tensor
        self._state["num_frames"] = idx + 1
        return idx

    def _points_to_inputs(self, points, labels):
        """(u,v) pixel points -> point_inputs dict in model (image_size) coords.

        Matches add_new_points_or_box: normalise by (W0,H0) then scale by
        image_size, batch dim added.
        """
        torch = self._torch
        pts = torch.tensor(points, dtype=torch.float32, device=self.device)
        if pts.dim() == 1:
            pts = pts[None]
        if labels is None:
            lbl = torch.ones(pts.shape[0], dtype=torch.int32, device=self.device)
        else:
            lbl = torch.tensor(labels, dtype=torch.int32, device=self.device)
        # normalise to [0,1] by original size, then scale by model image_size.
        norm = torch.tensor(
            [self._W0, self._H0], dtype=torch.float32, device=self.device
        )
        pts = (pts / norm) * self.predictor.image_size
        return {"point_coords": pts[None], "point_labels": lbl[None]}  # batch dim

    def _upsample_mask(self, low_res_logits) -> np.ndarray:
        """low-res mask logits -> bool mask at (H0, W0)."""
        torch = self._torch
        m = low_res_logits.to(self.device).float()  # (1,1,h,w)
        m = torch.nn.functional.interpolate(
            m, size=(self._H0, self._W0), mode="bilinear", align_corners=False
        )
        return (m[0, 0] > 0.0).detach().cpu().numpy().astype(bool)

    def _prune_memory(self):
        """Keep cond frame + the last ``max_non_cond`` non-cond outputs, and FREE
        the stale per-frame image tensors.

        Once a frame's features are encoded into the memory bank (cond / non-cond
        outputs + ``cached_features``), its raw ``(3, S, S)`` tensor in
        ``inference_state["images"]`` is never read again -- the memory-attention
        path uses the stored ``maskmem_features``, not the raw images. Without
        this, ``images`` grows by one ~12 MB GPU tensor PER FRAME and the process
        OOMs after ~1 min of live streaming (observed: 11 GB, CUDA OOM). We null
        every image except the one just processed so the CUDA allocator can reuse
        that memory instead of endlessly growing.
        """
        nc = self._output_dict["non_cond_frame_outputs"]
        if len(nc) > self.max_non_cond:
            for k in sorted(nc.keys())[: -self.max_non_cond]:
                nc.pop(k, None)

        imgs = self._state["images"]
        keep = self._frame_idx                     # current frame (already consumed)
        for i in range(len(imgs)):
            if i != keep and imgs[i] is not None:
                imgs[i] = None                     # drop GPU tensor ref -> reusable

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def load_first_frame(self, frame_rgb: np.ndarray) -> None:
        """Encode frame 0 and (re)initialise the streaming state."""
        torch = self._torch
        img = np.asarray(frame_rgb)
        self._H0, self._W0 = int(img.shape[0]), int(img.shape[1])
        with torch.inference_mode(), self._autocast():
            ft = self._preprocess(img)
            self._state = self._new_state(ft)
            self._output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            self._frame_idx = 0
            # Warm the backbone feature cache for frame 0 (as init_state does).
            self.predictor._get_image_feature(self._state, frame_idx=0, batch_size=1)

    def add_prompt(
        self,
        points: Optional[Sequence[Tuple[float, float]]] = None,
        labels: Optional[Sequence[int]] = None,
        box: Optional[Sequence[float]] = None,
        obj_id: int = 0,
    ) -> np.ndarray:
        """Seed the cond frame (frame 0) with a box and/or point prompts; return mask.

        A BOX prompt (e.g. from an upstream SAM3 detection) yields a much cleaner
        initial mask than a lone point, so the tracker starts on the right object
        with a tight boundary. SAM2 encodes a box as two points labelled 2
        (top-left) and 3 (bottom-right); we prepend those and append any foreground
        points after.

        Args:
            points: optional foreground (u, v) pixel points in the ORIGINAL frame.
            labels: optional per-point labels (1=fg, 0=bg); defaults to all-fg.
            box: optional (x0, y0, x1, y1) box in ORIGINAL-frame pixels.
            obj_id: tracked object id (single object; default 0).
        """
        if self._state is None:
            raise RuntimeError("call load_first_frame(...) before add_prompt(...)")
        self.obj_id = int(obj_id)
        torch = self._torch

        pts_list: list = []
        lbl_list: list = []
        if box is not None:
            x0, y0, x1, y1 = (float(v) for v in box)
            pts_list += [[x0, y0], [x1, y1]]
            lbl_list += [2, 3]                    # SAM2 box-corner labels
        if points is not None:
            for p in points:
                pts_list.append([float(p[0]), float(p[1])])
            if labels is None:
                lbl_list += [1] * len(points)
            else:
                lbl_list += [int(v) for v in labels]
        if not pts_list:
            raise ValueError("add_prompt needs at least a box or one point")
        point_inputs = self._points_to_inputs(pts_list, lbl_list)
        with torch.inference_mode(), self._autocast():
            current_out, _ = self.predictor._run_single_frame_inference(
                inference_state=self._state,
                output_dict=self._output_dict,
                frame_idx=0,
                batch_size=1,
                is_init_cond_frame=True,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
        self._output_dict["cond_frame_outputs"][0] = current_out
        return self._upsample_mask(current_out["pred_masks"])

    def track(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Advance one frame; propagate the object via memory; return bool mask."""
        if self._state is None or not self._output_dict["cond_frame_outputs"]:
            raise RuntimeError("call load_first_frame + add_prompt before track(...)")
        torch = self._torch
        with torch.inference_mode(), self._autocast():
            ft = self._preprocess(frame_rgb)
            idx = self._push_frame(ft)
            current_out, _ = self.predictor._run_single_frame_inference(
                inference_state=self._state,
                output_dict=self._output_dict,
                frame_idx=idx,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
        self._output_dict["non_cond_frame_outputs"][idx] = current_out
        self._prune_memory()
        return self._upsample_mask(current_out["pred_masks"])

    def reset(self) -> None:
        """Clear streaming state so a new load_first_frame/add_prompt can re-seed."""
        self._state = None
        self._output_dict = None
        self._frame_idx = -1
        self._H0 = 0
        self._W0 = 0


class _nullcontext:
    """Minimal no-op context manager (CPU path, no autocast)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
