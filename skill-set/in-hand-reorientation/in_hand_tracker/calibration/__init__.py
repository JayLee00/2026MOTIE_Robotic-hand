"""Calibration subpackage: per-object shape initialization (US-005, AC-5).

From an early, low-occlusion frame's masked world-frame point cloud, fit a
rotation-symmetric ellipsoid (a = b; two radii + a center, axis assumed ~world-z
for M1), classify the shape (spheroid vs. elongated) against the configured
``ca_threshold``, and persist ``(a, c)`` / ``shape_class`` into ``objects.yaml``.

Pure numpy / scipy / yaml. No sim-side (recorder / Isaac) imports.
"""

from .per_object_init import (
    FreeFitResult,
    free_fit_spheroid,
    classify_shape,
    calibrate_object,
    write_object_entry,
)

__all__ = [
    "FreeFitResult",
    "free_fit_spheroid",
    "classify_shape",
    "calibrate_object",
    "write_object_entry",
]
