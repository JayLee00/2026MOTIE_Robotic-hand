"""State estimators (position CV-KF + orientation MEKF).

Self-contained, pure numpy/scipy. No sim-side imports. See ``dev/project_doc.md``
sections 3.3 (estimators) and 3.4 (output policy).
"""

from __future__ import annotations

__all__ = []

try:  # pragma: no cover - import is exercised by the unit tests
    from .orientation_mekf import OrientationMEKF  # noqa: F401

    __all__.append("OrientationMEKF")
except Exception:  # pragma: no cover - keep the package import-safe
    pass

try:  # pragma: no cover - the position KF module may land separately
    from .position_kf import PositionKF  # noqa: F401

    __all__.append("PositionKF")
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - fusion needs both estimators + perception.bbox
    from .fusion import StateEstimator, StepOutput  # noqa: F401

    __all__.append("StateEstimator")
    __all__.append("StepOutput")
except Exception:  # pragma: no cover - keep the package import-safe
    pass
