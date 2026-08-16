"""in_hand_tracker — standalone in-hand object tracking foundation.

Clean-room package (M1~M2): IO + deprojection foundation. It NEVER imports the
sim-side stack (the recorder / Isaac packages) — all such logic is ported
(reimplemented) in numpy / cv2 / open3d / scipy. torch / sam2 / pinocchio are
optional and provided by the host (``dexsdr``) environment.
"""

__version__ = "0.1.0"

__all__ = ["io"]
