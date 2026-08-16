"""Synthetic streaming-tracker test for :class:`Sam2StreamTracker` (no camera).

Builds a synthetic clip of a bright disk that MOVES across frames, seeds the
LIVE SAM2 video predictor on frame 0, and asserts that the propagated mask
follows the disk -- proving temporal tracking works without any camera.

Skipped when torch / CUDA / the sam2 package / the checkpoint are unavailable,
but RUN (not skipped) on a machine that has them.
"""

import os

import numpy as np
import pytest

from in_hand_tracker.perception.sam2_stream import Sam2StreamTracker

# ckpt/config: same as RealSenseSegmenter (estimator.yaml segmenter.sam2.ckpt).
_SIM_ROOT = os.path.join("/home/chanyoung/isaac_ws", "dex_" + "sodering")
SAM2_CKPT = os.path.join(
    _SIM_ROOT, "viserdex_custom", "sam2_ckpts", "sam2.1_hiera_tiny.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _torch_cuda_ok() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    except Exception:
        return False
    return True


_RUNNABLE = _torch_cuda_ok() and _sam2_available() and os.path.isfile(SAM2_CKPT)
_stream_skip = pytest.mark.skipif(
    not _RUNNABLE,
    reason="streaming SAM2 path needs torch+CUDA, the sam2 package, and the ckpt",
)

# Clip geometry.
_H = _W = 256
_N = 12
_RAD = 28
_CY = 128
_CX0, _CX1 = 80, 300  # disk centre x sweeps left -> right (clamped to frame)


def _disk_frame(cx: int, cy: int, rad: int) -> np.ndarray:
    """RGB uint8 frame: a bright red disk on a dark mottled background."""
    rng = np.random.default_rng(cx * 131 + cy)
    img = (rng.integers(0, 40, size=(_H, _W, 3))).astype(np.uint8)  # dark noise
    yy, xx = np.ogrid[:_H, :_W]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
    img[disk] = np.array([240, 40, 40], dtype=np.uint8)  # bright red object
    return img


def _disk_centers():
    cxs = np.linspace(_CX0, _CX1, _N).astype(int)
    cxs = np.clip(cxs, _RAD + 1, _W - _RAD - 1)
    return [(int(cx), _CY) for cx in cxs]


def _true_disk_mask(cx: int, cy: int, rad: int) -> np.ndarray:
    yy, xx = np.ogrid[:_H, :_W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


@_stream_skip
def test_stream_tracks_moving_disk():
    centers = _disk_centers()
    frames = [_disk_frame(cx, cy, _RAD) for (cx, cy) in centers]

    tr = Sam2StreamTracker(SAM2_CKPT, SAM2_CONFIG, device="cuda")

    # --- frame 0: seed + assert the mask covers the disk ---
    tr.load_first_frame(frames[0])
    cx0, cy0 = centers[0]
    mask0 = tr.add_prompt([(cx0, cy0)])
    assert mask0.shape == (_H, _W)
    true0 = _true_disk_mask(cx0, cy0, _RAD)
    iou0 = _iou(mask0, true0)
    disk_area = float(true0.sum())
    assert iou0 > 0.5, f"seed mask IoU too low: {iou0:.3f}"

    # --- subsequent frames: assert the mask FOLLOWS the moving disk ---
    errs = []
    for i in range(1, _N):
        cx, cy = centers[i]
        mask = tr.track(frames[i])
        assert mask.shape == (_H, _W)
        c = _centroid(mask)
        assert c is not None, f"empty mask at frame {i}"
        err = float(np.hypot(c[0] - cx, c[1] - cy))
        errs.append(err)
        area = float(mask.sum())
        # centroid follows the disk within ~25 px
        assert err < 25.0, f"frame {i}: centroid err {err:.1f}px (disk at {cx},{cy})"
        # mask area stays in a sane range (0.4x .. 3x the true disk area)
        assert 0.4 * disk_area < area < 3.0 * disk_area, (
            f"frame {i}: mask area {area:.0f} out of range (disk {disk_area:.0f})"
        )

    print(
        f"\n[stream-test] seed IoU={iou0:.3f}  "
        f"tracked-centroid err: mean={np.mean(errs):.2f}px "
        f"max={np.max(errs):.2f}px over {len(errs)} frames"
    )


@_stream_skip
def test_stream_reset_reseeds():
    """reset() clears state so a fresh load_first_frame/add_prompt can re-seed."""
    centers = _disk_centers()
    frames = [_disk_frame(cx, cy, _RAD) for (cx, cy) in centers]
    tr = Sam2StreamTracker(SAM2_CKPT, SAM2_CONFIG, device="cuda")

    tr.load_first_frame(frames[0])
    tr.add_prompt([centers[0]])
    _ = tr.track(frames[1])

    tr.reset()
    # After reset, track before re-seed must raise.
    with pytest.raises(RuntimeError):
        tr.track(frames[2])

    # Re-seed on a later frame works.
    tr.load_first_frame(frames[3])
    cx3, cy3 = centers[3]
    mask = tr.add_prompt([(cx3, cy3)])
    assert _iou(mask, _true_disk_mask(cx3, cy3, _RAD)) > 0.5
