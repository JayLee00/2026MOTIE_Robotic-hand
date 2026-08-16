"""IO subpackage: dataclasses, the Replicator sequence reader, and deprojection.

Pure numpy / cv2 / open3d / scipy. No sim-side (recorder / Isaac) imports.
"""

from .types import Frame, GtShape, FitResult
from .deproject import (
    intrinsics_from_yaml,
    intrinsics_matrix,
    deproject,
    cam_to_world,
    surface_centroid,
)
from .replicator_reader import ReplicatorReader

__all__ = [
    "Frame",
    "GtShape",
    "FitResult",
    "intrinsics_from_yaml",
    "intrinsics_matrix",
    "deproject",
    "cam_to_world",
    "surface_centroid",
    "ReplicatorReader",
]

# Live RealSense source (REAL path). pyrealsense2 is a RUNTIME dependency, so
# it is imported lazily inside RealSenseSource.start()/capture() and inside
# deproject_zdepth: this module imports cleanly without the hardware present.
from .realsense_source import (
    RealSenseSource,
    deproject_zdepth,
    project_points,
)

__all__ += ["RealSenseSource", "deproject_zdepth", "project_points"]
