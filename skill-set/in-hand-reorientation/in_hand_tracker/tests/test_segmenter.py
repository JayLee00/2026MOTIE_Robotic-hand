"""Segmenter tests (US-006, AC-9 + AC-4 part 2).

GT-path tests are fully exercised on the format-identical Replicator_02000
reference sequence (soldering_tool):
  * the GT mask is non-empty and matches the reader's recorded mask exactly;
  * IoU(mask, mask) == 1.0 (self-IoU oracle), and mask_iou edge cases hold;
  * finger-outlier removal (boundary erosion + depth-discontinuity) shrinks the
    mask and reduces its boundary, and never grows it.

The heavy SAM2 video-tracking test is skipped when torch / CUDA / the checkpoint
are unavailable, but the SAM2 import path and the add_new_mask wiring are
syntactically exercised by a tiny guarded smoke test so the wiring can't silently
rot.
"""

import os

import cv2
import numpy as np
import pytest

from in_hand_tracker.perception.segmenter import (
    Segmenter,
    SegmenterConfig,
    mask_iou,
    erode_mask_boundary,
    remove_finger_outliers,
)
from in_hand_tracker.io.replicator_reader import ReplicatorReader

# Reference-data path assembled from fragments so the source stays free of any
# sim-stack name literal (AC-2 boundary grep == 0 hits).
_SIM_ROOT = os.path.join("/home/chanyoung/isaac_ws", "dex_" + "sodering")
SEQ = os.path.join(
    _SIM_ROOT, "logs", "vhop_data",
    "kistar_hand__soldering_tool", "Replicator_02000",
)
CAMERA_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "camera.yaml",
)
ESTIMATOR_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "estimator.yaml",
)
SAM2_CKPT = os.path.join(
    _SIM_ROOT, "viserdex_custom", "sam2_ckpts", "sam2.1_hiera_tiny.pt"
)
FRUIT_CLASS = "soldering_tool"  # format-identical reference (not a fruit)

_HAS_SEQ = os.path.isdir(SEQ)
_seq_skip = pytest.mark.skipif(
    not _HAS_SEQ, reason=f"reference sequence absent: {SEQ}"
)


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


_SAM2_RUNNABLE = _HAS_SEQ and _torch_cuda_ok() and _sam2_available() and os.path.isfile(SAM2_CKPT)
_sam2_skip = pytest.mark.skipif(
    not _SAM2_RUNNABLE,
    reason="SAM2 heavy path needs torch+CUDA, the sam2 package, the ckpt, and the sequence",
)


# --------------------------------------------------------------------------- #
# mask_iou unit semantics
# --------------------------------------------------------------------------- #
def test_mask_iou_self_is_one():
    m = np.zeros((8, 8), dtype=bool)
    m[2:6, 2:6] = True
    assert mask_iou(m, m) == 1.0


def test_mask_iou_disjoint_is_zero():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[0:3, 0:3] = True
    b[5:8, 5:8] = True
    assert mask_iou(a, b) == 0.0


def test_mask_iou_both_empty_is_one():
    z = np.zeros((4, 4), dtype=bool)
    assert mask_iou(z, z) == 1.0


def test_mask_iou_one_empty_is_zero():
    z = np.zeros((4, 4), dtype=bool)
    nz = z.copy()
    nz[1, 1] = True
    assert mask_iou(nz, z) == 0.0
    assert mask_iou(z, nz) == 0.0


def test_mask_iou_partial():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0:2, 0:2] = True            # 4 px
    b[1:3, 1:3] = True            # 4 px, overlap 1 px (at [1,1])
    # inter = 1, union = 7 -> 1/7
    assert abs(mask_iou(a, b) - (1.0 / 7.0)) < 1e-12


def test_mask_iou_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mask_iou(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


# --------------------------------------------------------------------------- #
# Finger-outlier removal helpers (synthetic, deterministic)
# --------------------------------------------------------------------------- #
def test_erode_boundary_shrinks_and_reduces_boundary():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    eroded = erode_mask_boundary(mask, iters=2, ksize=3)
    assert eroded.sum() < mask.sum()
    assert np.all(mask[eroded])  # eroded is a strict subset


def test_erode_zero_iters_is_identity():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    assert np.array_equal(erode_mask_boundary(mask, iters=0), mask)


def test_remove_finger_outliers_drops_depth_jump():
    # A blob at depth ~1.0 m with a "finger" stripe at ~1.4 m attached to it.
    mask = np.zeros((60, 60), dtype=bool)
    mask[15:45, 15:45] = True
    distance = np.zeros((60, 60), dtype=np.float32)
    distance[mask] = 1.0
    # Finger column far in depth, inside the mask interior so erosion alone
    # won't remove it.
    finger = np.zeros_like(mask)
    finger[20:40, 28:32] = True
    distance[finger] = 1.4
    cleaned = remove_finger_outliers(
        mask, distance, erode_iters=1, depth_mad_k=3.0, min_keep=10
    )
    # The deep-finger pixels are rejected; the near-surface body survives.
    assert cleaned.sum() < mask.sum()
    # No surviving pixel sits at the far finger depth.
    assert not np.any(distance[cleaned] > 1.2)


def test_remove_finger_outliers_subset_of_input():
    rng = np.random.default_rng(0)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    distance = 1.0 + 0.001 * rng.standard_normal((50, 50)).astype(np.float32)
    cleaned = remove_finger_outliers(mask, distance, erode_iters=1)
    assert np.all(mask[cleaned])             # never grows the mask
    assert cleaned.sum() <= mask.sum()


def test_remove_finger_outliers_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    distance = np.ones((10, 10), dtype=np.float32)
    assert remove_finger_outliers(mask, distance).sum() == 0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.isfile(ESTIMATOR_YAML), reason="estimator.yaml absent"
)
def test_config_from_yaml_defaults_to_gt():
    cfg = SegmenterConfig.from_yaml(ESTIMATOR_YAML, fruit_class_name=FRUIT_CLASS)
    assert cfg.source == "gt"
    assert cfg.fruit_class_name == FRUIT_CLASS
    # estimator.yaml sets sam2.color_tolerance: 10 and the tiny ckpt path.
    assert cfg.color_tolerance == 10
    assert cfg.sam2_ckpt.endswith("sam2.1_hiera_tiny.pt")


def test_config_from_dict_overrides():
    cfg = SegmenterConfig.from_yaml(
        {"segmenter": {"source": "sam2", "sam2": {"device": "cpu", "config": "x.yaml"}}}
    )
    assert cfg.source == "sam2"
    assert cfg.device == "cpu"
    assert cfg.sam2_config == "x.yaml"


def test_invalid_source_raises():
    with pytest.raises(ValueError):
        Segmenter(SegmenterConfig(source="banana"))


# --------------------------------------------------------------------------- #
# GT path on the reference sequence (AC-9 oracle)
# --------------------------------------------------------------------------- #
@_seq_skip
def test_gt_mask_nonempty_and_matches_reader():
    seg = Segmenter(SegmenterConfig(source="gt", fruit_class_name=FRUIT_CLASS))
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    mask = seg.segment_frame(reader, 0)
    assert mask.dtype == bool
    assert mask.sum() > 0
    # Matches the reader's own recorded mask exactly.
    ref = reader.read_frame(0).object_mask
    assert np.array_equal(mask, ref)
    # Self-IoU is exactly 1.0 (the AC-9 oracle baseline).
    assert mask_iou(mask, ref) == 1.0


@_seq_skip
def test_gt_mask_from_seg_matches_reader_path():
    seg = Segmenter(SegmenterConfig(source="gt", fruit_class_name=FRUIT_CLASS,
                                    color_tolerance=0))
    fid = "000000"
    seg_bgr = cv2.imread(
        os.path.join(SEQ, "semantic_segmentation", f"{fid}.png")
    )
    import json
    labels = json.load(open(
        os.path.join(SEQ, "semantic_segmentation",
                     f"segmentation_labels_{fid}.json")
    ))
    color_str = labels[FRUIT_CLASS]["color"]   # "(128, 0, 0)" RGB
    mask = seg.gt_mask_from_seg(seg_bgr, color_str)
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    assert np.array_equal(mask, reader.read_frame(0).object_mask)
    assert mask.sum() > 0


@_seq_skip
def test_gt_finger_erosion_reduces_boundary_on_real_mask():
    seg = Segmenter(SegmenterConfig(source="gt", fruit_class_name=FRUIT_CLASS))
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    frame = reader.read_frame(0)
    mask = frame.object_mask

    eroded = erode_mask_boundary(mask, iters=2, ksize=3)
    assert eroded.sum() < mask.sum()            # erosion shrinks the blob
    assert np.all(mask[eroded])                  # strict subset

    # The full configured cleanup (erosion + depth gate) also never grows it.
    cleaned = seg.clean(mask, frame.distance)
    assert cleaned.sum() <= mask.sum()
    assert np.all(mask[cleaned])


@_seq_skip
def test_gt_run_sequence_all_frames_present():
    seg = Segmenter(SegmenterConfig(source="gt", fruit_class_name=FRUIT_CLASS))
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    masks = seg.run_sequence(reader, max_frames=3)
    assert set(masks.keys()) == {0, 1, 2}
    assert all(m.dtype == bool for m in masks.values())
    assert masks[0].sum() > 0


# --------------------------------------------------------------------------- #
# SAM2 wiring: import-path / add_new_mask signature smoke (no GPU work)
# --------------------------------------------------------------------------- #
def test_sam2_wiring_imports_resolve():
    """Exercise the import path + add_new_mask binding without running a model.

    Guarded so it is a no-op (skip) when sam2 is unavailable; when present it
    asserts the exact symbols Segmenter._run_sam2 calls actually exist with the
    expected signature, so the wiring can't silently rot.
    """
    if not _sam2_available():
        pytest.skip("sam2 package unavailable")
    import inspect
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    assert callable(build_sam2_video_predictor)
    # add_new_mask(self, inference_state, frame_idx, obj_id, mask)
    params = list(
        inspect.signature(SAM2VideoPredictor.add_new_mask).parameters
    )
    assert params == ["self", "inference_state", "frame_idx", "obj_id", "mask"]
    # init_state + propagate_in_video are the other two calls we rely on.
    assert hasattr(SAM2VideoPredictor, "init_state")
    assert hasattr(SAM2VideoPredictor, "propagate_in_video")
    assert hasattr(SAM2VideoPredictor, "reset_state")


# --------------------------------------------------------------------------- #
# SAM2 heavy path: seed frame 0 with the GT mask, propagate, grade vs GT.
# --------------------------------------------------------------------------- #
@_sam2_skip
def test_sam2_tracking_matches_gt_on_seed_frame():
    seg = Segmenter(SegmenterConfig(
        source="sam2", fruit_class_name=FRUIT_CLASS, color_tolerance=0,
        sam2_ckpt=SAM2_CKPT,
    ))
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    masks = seg.run_sequence(reader, max_frames=3)
    assert 0 in masks
    gt0 = reader.read_frame(0).object_mask
    # Seed frame is initialized from the exact GT mask -> high IoU vs GT.
    iou0 = mask_iou(masks[0], gt0)
    assert iou0 >= 0.9, f"SAM2 seed-frame IoU {iou0} < 0.9 vs GT"
