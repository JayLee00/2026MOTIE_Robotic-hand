"""Perception subpackage: ROI gating, segmentation, and shape fitting.

Pure numpy / cv2 / open3d / scipy. Optional pinocchio is used only by the FK
ROI path and is imported lazily so the package stays import-safe without it.
NO sim-side (recorder / Isaac) imports.
"""

from .ellipsoid_fit import (
    EllipsoidFitResult,
    fit_ellipsoid_center,
    ellipsoid_residual,
)
from .bbox import (
    Box3D,
    aabb_from_sphere,
    tight_obb,
    conservative_aabb,
    box_from_fit,
    aabb_iou,
    box_iou_voxel,
)

__all__ = [
    "EllipsoidFitResult",
    "fit_ellipsoid_center",
    "ellipsoid_residual",
    "Box3D",
    "aabb_from_sphere",
    "tight_obb",
    "conservative_aabb",
    "box_from_fit",
    "aabb_iou",
    "box_iou_voxel",
]

# ROI gating ships in a sibling module that may land separately; export it when
# present so importing this package never hard-fails on partial trees.
try:  # pragma: no cover - exercised only once roi_gate.py exists
    from .roi_gate import RoiResult, roi_crop
except ImportError:
    pass
else:
    __all__ += ["RoiResult", "roi_crop"]

# Segmentation (GT oracle + SAM2 video). Same defensive pattern: importing this
# package never hard-fails on a partial tree.
try:  # pragma: no cover - exercised only once segmenter.py exists
    from .segmenter import (
        Segmenter,
        SegmenterConfig,
        mask_iou,
        remove_finger_outliers,
        erode_mask_boundary,
    )
except ImportError:
    pass
else:
    __all__ += [
        "Segmenter",
        "SegmenterConfig",
        "mask_iou",
        "remove_finger_outliers",
        "erode_mask_boundary",
    ]

# Live (GT-less, fruit-agnostic) segmenter for the REAL RealSense path. Same
# defensive pattern: torch / sam2 are imported lazily inside the segmenter.
try:  # pragma: no cover - exercised only once realsense_segmenter.py exists
    from .realsense_segmenter import (
        RealSenseSegmenter,
        RealSenseSegmenterConfig,
        hsv_center_blob,
    )
except ImportError:
    pass
else:
    __all__ += [
        "RealSenseSegmenter",
        "RealSenseSegmenterConfig",
        "hsv_center_blob",
    ]
