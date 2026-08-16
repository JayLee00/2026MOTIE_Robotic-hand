"""AprilTag detection + tag-bundle pose fusion (C3, doc 3.2 + 4 / 4.1).

Wraps ``pupil_apriltags.Detector`` (the AprilTag3 binding) for the families the
project uses -- ``tag36h11`` (robust / general) and ``tagStandard41h12`` (good
for small tags). Given an RGB-or-gray image, camera intrinsics ``(fx, fy, cx,
cy)`` and a physical tag size, it detects tags and returns, per tag, the pose
``(R, t)`` of the tag in the CAMERA frame.

Two responsibilities:

* **Detection + gating (doc 4.1).** Each raw detection is screened by four
  gates before its pose is trusted:
    - ``decision_margin``  -- decode quality (small/blurry tags reject low values).
    - ``reprojection error`` -- the pose is reprojected onto the tag corners; a
      large pixel residual means an unreliable / flipped solve.
    - ``apparent tag size`` -- the tag's pixel footprint; tiny tags carry too
      little geometry for a stable pose.
    - ``view angle`` -- the angle between the tag's plane normal and the camera
      ray. A too-oblique tag has a numerically poor orientation (the classic
      small-planar-tag flip ambiguity), so we DROP its orientation but may keep
      its translation (``position_only``), exactly as the doc prescribes.

* **Tag bundle (doc 4.1).** A :class:`TagBundle` maps ``tag_id -> fixed
  transform tag->object``. Several small tags on different faces of the (curved)
  fruit thus all report the SAME object pose; :meth:`AprilTagDetector.bundle_pose`
  fuses every orientation-trusted face into one object orientation (chordal
  quaternion mean) and one object center (weighted mean of per-tag centers),
  returning an :class:`ObjPose` carrying the fused quaternion (WXYZ at the data
  boundary), center, a per-tag-count and a ``gate`` confidence flag.

Conventions / boundaries:
  * Quaternions are **WXYZ** at the public boundary (Isaac/USD convention,
    matching ``io.types``). Internally we convert to scipy's XYZW.
  * Pure ``numpy`` / ``scipy`` / ``cv2`` / ``pupil_apriltags``. This module
    imports NOTHING from the sim-side stack (recorder / Isaac / SAM packages) --
    the AC-2 clean boundary. ``pupil_apriltags`` is imported lazily so the
    bundle-fusion math + gating logic remain unit-testable without the native
    library present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "SUPPORTED_FAMILIES",
    "GatingConfig",
    "TagObs",
    "TagBundle",
    "ObjPose",
    "AprilTagDetector",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "average_quaternions_wxyz",
]

# Families this project standardizes on (doc 4.1).
SUPPORTED_FAMILIES = ("tag36h11", "tagStandard41h12")


# --------------------------------------------------------------------------- #
# Quaternion helpers (WXYZ boundary <-> scipy XYZW internal)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_xyzw(q_wxyz: Sequence[float]) -> np.ndarray:
    """Reorder a WXYZ quaternion to scipy's XYZW layout."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float]) -> np.ndarray:
    """Reorder a scipy XYZW quaternion to the WXYZ data-boundary layout."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def average_quaternions_wxyz(
    quats_wxyz: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted chordal-L2 mean of WXYZ quaternions (Markley's eigen method).

    Each quaternion is double-cover-ambiguous (``q`` and ``-q`` are the same
    rotation), so we cannot just average components. We average the outer
    products ``w_i q_i q_i^T`` and take the dominant eigenvector, which is the
    rotation minimizing the sum of weighted chordal distances and is invariant
    to per-quaternion sign. Returns a unit WXYZ quaternion with a non-negative
    scalar part (canonical sign).
    """
    qs = np.asarray(quats_wxyz, dtype=np.float64).reshape(-1, 4)
    if qs.shape[0] == 0:
        raise ValueError("average_quaternions_wxyz: no quaternions given")
    if weights is None:
        w = np.ones(qs.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != qs.shape[0]:
            raise ValueError("weights length must match number of quaternions")
    # Normalize each quaternion (guard against drift).
    norms = np.linalg.norm(qs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    qs = qs / norms

    M = np.zeros((4, 4), dtype=np.float64)
    for wi, qi in zip(w, qs):
        M += wi * np.outer(qi, qi)
    # Dominant eigenvector of the (symmetric) accumulation matrix.
    evals, evecs = np.linalg.eigh(M)
    q_mean = evecs[:, int(np.argmax(evals))]
    q_mean = q_mean / (np.linalg.norm(q_mean) + 1e-12)
    if q_mean[0] < 0.0:  # canonical sign: non-negative scalar (w) part
        q_mean = -q_mean
    return q_mean


# --------------------------------------------------------------------------- #
# Configs / result containers
# --------------------------------------------------------------------------- #
@dataclass
class GatingConfig:
    """Thresholds for the four detection gates (doc 4.1).

    Attributes:
        min_decision_margin: reject decodes below this margin (decode quality).
        max_reproj_error_px: reject poses whose mean corner reprojection error
            (pixels) exceeds this (catches bad / flipped solves).
        min_tag_size_px: reject tags whose apparent size (mean edge length in
            pixels) is below this (too little geometry for a stable pose).
        max_view_angle_deg: above this angle between the tag normal and the
            camera ray the orientation is dropped (too oblique); the translation
            may still be kept (``position_only``).
        keep_position_when_oblique: if True an over-oblique tag yields a
            ``position_only`` observation (t kept, R dropped) instead of being
            discarded entirely.
        max_pose_err: optional cap on the solver's own ``pose_err`` (object-space
            error from pupil_apriltags); ``None`` disables this gate.
    """

    min_decision_margin: float = 30.0
    max_reproj_error_px: float = 5.0
    min_tag_size_px: float = 12.0
    max_view_angle_deg: float = 70.0
    keep_position_when_oblique: bool = True
    max_pose_err: Optional[float] = None


@dataclass
class TagObs:
    """A single gated tag observation (pose of the tag in the CAMERA frame).

    Attributes:
        tag_id: decoded tag id.
        tag_family: family string.
        R: (3, 3) camera<-tag rotation (columns are the tag axes in cam frame).
        t: (3,) tag origin in the camera frame (meters).
        decision_margin: decode quality from the detector.
        reproj_error_px: mean corner reprojection error (pixels).
        tag_size_px: apparent tag size (mean edge length, pixels).
        view_angle_deg: angle between the tag normal and the camera ray.
        pose_err: solver object-space error (``None`` if unavailable).
        center_px: (2,) detection center in image pixels.
        accepted: passed ALL gates (orientation trustworthy).
        position_only: orientation dropped (too oblique) but translation kept.
        reject_reason: short tag describing the first failed gate ("" if none).
    """

    tag_id: int
    tag_family: str
    R: np.ndarray
    t: np.ndarray
    decision_margin: float
    reproj_error_px: float
    tag_size_px: float
    view_angle_deg: float
    pose_err: Optional[float] = None
    center_px: Optional[np.ndarray] = None
    accepted: bool = False
    position_only: bool = False
    reject_reason: str = ""

    def quat_wxyz(self) -> np.ndarray:
        """Tag orientation as a WXYZ quaternion (camera frame)."""
        return quat_xyzw_to_wxyz(Rotation.from_matrix(self.R).as_quat())


@dataclass
class TagBundle:
    """A rigid set of tags sharing one object frame (doc 4.1 tag bundle).

    ``members`` maps ``tag_id -> (R_to, t_to)`` where ``(R_to, t_to)`` is the
    FIXED transform from the tag frame to the OBJECT frame: a point expressed in
    the tag frame maps to the object frame by ``p_obj = R_to @ p_tag + t_to``.
    Equivalently the object pose implied by one tag detection ``(R_ct, t_ct)``
    (camera<-tag) is::

        R_co = R_ct @ R_to.T              # camera<-object
        t_co = t_ct - R_co @ t_to         # object origin in camera frame

    Attributes:
        members: ``tag_id -> (R_to (3,3), t_to (3,))``.
        tag_size: physical edge length (m) shared by the bundle's tags (used by
            the detector if a per-call size is not supplied).
        name: optional label.
    """

    members: Dict[int, Tuple[np.ndarray, np.ndarray]]
    tag_size: Optional[float] = None
    name: str = "bundle"

    def __post_init__(self) -> None:
        norm: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for tid, (R_to, t_to) in self.members.items():
            R = np.asarray(R_to, dtype=np.float64).reshape(3, 3)
            t = np.asarray(t_to, dtype=np.float64).reshape(3)
            norm[int(tid)] = (R, t)
        self.members = norm

    def __contains__(self, tag_id: int) -> bool:
        return int(tag_id) in self.members

    def object_pose_from_tag(
        self, R_ct: np.ndarray, t_ct: np.ndarray, tag_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map a single camera<-tag pose to a camera<-object pose (R_co, t_co)."""
        R_to, t_to = self.members[int(tag_id)]
        R_co = np.asarray(R_ct, dtype=np.float64).reshape(3, 3) @ R_to.T
        t_co = np.asarray(t_ct, dtype=np.float64).reshape(3) - R_co @ t_to
        return R_co, t_co


@dataclass
class ObjPose:
    """Fused object pose from a tag bundle (camera frame).

    Attributes:
        center: (3,) object origin in the camera frame (meters).
        quat_wxyz: (4,) object orientation as a WXYZ quaternion (camera frame),
            or ``None`` when no orientation-trusted tag contributed
            (``gate == "position_only"`` / ``"none"``).
        R: (3, 3) object rotation matrix, or ``None`` when orientation absent.
        n_tags: number of tags fused for orientation.
        n_position_tags: number of tags contributing to the center.
        gate: confidence flag -- ``"high"`` (>=1 orientation-trusted tag),
            ``"position_only"`` (only oblique/position tags -> center but no
            orientation), or ``"none"`` (nothing usable).
        used_ids: tag ids that contributed to the orientation fusion.
        orientation_spread_deg: max pairwise angle (deg) among fused
            orientations -- a consistency diagnostic (0 for a single tag).
    """

    center: Optional[np.ndarray]
    quat_wxyz: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    n_tags: int = 0
    n_position_tags: int = 0
    gate: str = "none"
    used_ids: List[int] = field(default_factory=list)
    orientation_spread_deg: float = 0.0


# --------------------------------------------------------------------------- #
# Geometry helpers (gating math) -- library-free so they are unit-testable
# --------------------------------------------------------------------------- #
def _tag_object_corners(tag_size: float) -> np.ndarray:
    """(4, 3) tag-frame corner coordinates matching apriltag3's POSE convention.

    apriltag3's ``estimate_pose_for_tag_homography`` pairs the detected image
    corner ``p[i]`` with the object corner (s = tag_size/2, z = 0)::

        p[0] <-> (-s, +s)   p[1] <-> (+s, +s)
        p[2] <-> (+s, -s)   p[3] <-> (-s, -s)

    Any other order pairs each detected corner with the WRONG model corner and
    inflates the reprojection error by roughly the tag's pixel size (measured
    live: 27 px error on a 25 px tag with the old (-,-),(+,-),(+,+),(-,+)
    order vs 0.02 px with this one), so the reproj gate rejected every genuine
    detection.
    """
    h = 0.5 * float(tag_size)
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


def reprojection_error_px(
    R_ct: np.ndarray,
    t_ct: np.ndarray,
    corners_px: np.ndarray,
    K: Tuple[float, float, float, float],
    tag_size: float,
) -> float:
    """Mean pixel reprojection error of the tag corners under pose ``(R, t)``.

    Projects the known tag-frame corners with the pinhole intrinsics and
    compares against the detected image corners. A robust, library-free check
    that flags flipped / poorly-conditioned solves (doc 4.1 reprojection gate).
    """
    fx, fy, cx, cy = (float(v) for v in K)
    R = np.asarray(R_ct, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_ct, dtype=np.float64).reshape(3)
    obj = _tag_object_corners(tag_size)                  # (4, 3)
    cam = obj @ R.T + t                                  # (4, 3) camera frame
    z = cam[:, 2]
    if np.any(z <= 1e-9):
        return float("inf")
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    proj = np.stack([u, v], axis=1)
    det = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if det.shape[0] != proj.shape[0]:
        return float("inf")
    return float(np.mean(np.linalg.norm(proj - det, axis=1)))


def apparent_tag_size_px(corners_px: np.ndarray) -> float:
    """Apparent tag footprint = mean of the 4 polygon edge lengths (pixels)."""
    c = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] < 2:
        return 0.0
    edges = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    return float(np.mean(edges))


def view_angle_deg(R_ct: np.ndarray, t_ct: np.ndarray) -> float:
    """Angle (deg) between the tag's plane normal and the camera viewing ray.

    The tag normal in the camera frame is ``R[:, 2]``; the viewing ray points
    from the camera toward the tag center (``t / |t|``). 0 deg = head-on,
    90 deg = edge-on. A large angle is the oblique case where orientation is
    untrustworthy (flip ambiguity), so the orientation gate uses this.
    """
    R = np.asarray(R_ct, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_ct, dtype=np.float64).reshape(3)
    normal = R[:, 2]
    nt = np.linalg.norm(t)
    if nt < 1e-9:
        return 0.0
    ray = t / nt
    # The visible face's outward normal should face the camera (toward -ray);
    # take the acute angle so a flipped normal sign doesn't fake "head-on".
    cos = abs(float(np.dot(normal, ray)))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def gate_detection(
    R_ct: np.ndarray,
    t_ct: np.ndarray,
    corners_px: np.ndarray,
    K: Tuple[float, float, float, float],
    tag_size: float,
    decision_margin: float,
    cfg: GatingConfig,
    pose_err: Optional[float] = None,
) -> Tuple[bool, bool, str, float, float, float]:
    """Apply the four doc-4.1 gates to one detection.

    Returns ``(accepted, position_only, reason, reproj_px, size_px, angle_deg)``:
      * ``accepted``      -- passed every gate (orientation trustworthy).
      * ``position_only`` -- failed ONLY the view-angle gate and the config keeps
        position for oblique tags (orientation dropped, translation kept).
      * ``reason``        -- first failed gate ("" if accepted).

    Gate order: decision_margin -> pose_err -> apparent size -> reprojection ->
    view angle. The first four are hard (reject the whole detection); the
    view-angle gate is soft (may degrade to position-only).
    """
    reproj = reprojection_error_px(R_ct, t_ct, corners_px, K, tag_size)
    size_px = apparent_tag_size_px(corners_px)
    angle = view_angle_deg(R_ct, t_ct)

    if decision_margin < cfg.min_decision_margin:
        return False, False, "low_decision_margin", reproj, size_px, angle
    if cfg.max_pose_err is not None and pose_err is not None and pose_err > cfg.max_pose_err:
        return False, False, "high_pose_err", reproj, size_px, angle
    if size_px < cfg.min_tag_size_px:
        return False, False, "tag_too_small", reproj, size_px, angle
    if reproj > cfg.max_reproj_error_px:
        return False, False, "high_reproj_error", reproj, size_px, angle
    if angle > cfg.max_view_angle_deg:
        if cfg.keep_position_when_oblique:
            return False, True, "oblique_position_only", reproj, size_px, angle
        return False, False, "too_oblique", reproj, size_px, angle
    return True, False, "", reproj, size_px, angle


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class AprilTagDetector:
    """Wrap pupil_apriltags + gating + tag-bundle fusion (C3).

    The native ``pupil_apriltags.Detector`` is created lazily on the first
    :meth:`detect`, so the gating + bundle-fusion logic can be exercised in unit
    tests (via :meth:`gate_obs` / :meth:`bundle_pose`) without the native lib.
    """

    def __init__(
        self,
        families: str = "tag36h11",
        tag_size: float = 0.02,
        gating: Optional[GatingConfig] = None,
        bundle: Optional[TagBundle] = None,
        *,
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        upscale: float = 1.0,
    ) -> None:
        for fam in str(families).split():
            if fam not in SUPPORTED_FAMILIES:
                raise ValueError(
                    f"unsupported tag family {fam!r}; "
                    f"supported: {SUPPORTED_FAMILIES}"
                )
        self.families = str(families)
        self.tag_size = float(tag_size)
        self.gating = gating if gating is not None else GatingConfig()
        self.bundle = bundle
        # Pre-detection upsampling factor. A ~25 px tag (small tag on the fruit
        # at working distance) is at the decoder's limit and washes out under
        # specular highlights -- measured live: raw 640x480 detected 0/60
        # frames while a 2x cubic upscale detected the same tag reliably.
        # Corners/centers are scaled back so gating runs in ORIGINAL pixels.
        self.upscale = float(upscale)
        if self.upscale < 1.0:
            raise ValueError(f"upscale must be >= 1.0, got {self.upscale}")
        self._det_kwargs = dict(
            families=self.families,
            nthreads=int(nthreads),
            quad_decimate=float(quad_decimate),
            quad_sigma=float(quad_sigma),
            refine_edges=int(refine_edges),
            decode_sharpening=float(decode_sharpening),
        )
        self._detector = None  # lazy

    # ----- native detector lifecycle ----- #
    def _ensure_detector(self):
        if self._detector is None:
            from pupil_apriltags import Detector  # lazy import (AC: testable)

            self._detector = Detector(**self._det_kwargs)
        return self._detector

    # ----- gating (pure, testable) ----- #
    def gate_obs(
        self,
        tag_id: int,
        tag_family: str,
        R_ct: np.ndarray,
        t_ct: np.ndarray,
        corners_px: np.ndarray,
        K: Tuple[float, float, float, float],
        tag_size: float,
        decision_margin: float,
        center_px: Optional[np.ndarray] = None,
        pose_err: Optional[float] = None,
    ) -> TagObs:
        """Build a gated :class:`TagObs` from raw pose + decode data.

        Shared by :meth:`detect` and by unit tests that construct synthetic
        detections, so the gating contract is identical in both.
        """
        accepted, position_only, reason, reproj, size_px, angle = gate_detection(
            R_ct, t_ct, corners_px, K, tag_size, decision_margin,
            self.gating, pose_err=pose_err,
        )
        return TagObs(
            tag_id=int(tag_id),
            tag_family=str(tag_family),
            R=np.asarray(R_ct, dtype=np.float64).reshape(3, 3),
            t=np.asarray(t_ct, dtype=np.float64).reshape(3),
            decision_margin=float(decision_margin),
            reproj_error_px=reproj,
            tag_size_px=size_px,
            view_angle_deg=angle,
            pose_err=None if pose_err is None else float(pose_err),
            center_px=None if center_px is None else np.asarray(center_px, dtype=np.float64).reshape(2),
            accepted=accepted,
            position_only=position_only,
            reject_reason=reason,
        )

    # ----- detection ----- #
    def detect(
        self,
        img: np.ndarray,
        K: Tuple[float, float, float, float],
        tag_size: Optional[float] = None,
    ) -> List[TagObs]:
        """Detect tags in ``img`` and return a gated :class:`TagObs` per tag.

        ``img`` may be gray ``(H, W)`` or color ``(H, W, 3)``; color is
        converted to gray. ``K = (fx, fy, cx, cy)``. ``tag_size`` (m) overrides
        the instance default for this call.
        """
        size = self.tag_size if tag_size is None else float(tag_size)
        gray = self._to_gray(img)
        fx, fy, cx, cy = (float(v) for v in K)
        s = self.upscale
        if s != 1.0:
            import cv2  # lazy, like _to_gray

            gray = cv2.resize(
                gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC
            )
        detector = self._ensure_detector()
        dets = detector.detect(
            gray,
            estimate_tag_pose=True,
            # Scaled intrinsics keep the metric pose correct on the upscaled
            # image; corners are divided back below so gating / reprojection
            # run against the ORIGINAL K and pixel coordinates.
            camera_params=(fx * s, fy * s, cx * s, cy * s),
            tag_size=size,
        )
        out: List[TagObs] = []
        for d in dets:
            if d.pose_R is None or d.pose_t is None:
                continue
            R = np.asarray(d.pose_R, dtype=np.float64).reshape(3, 3)
            t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
            # The native solver's planar-pose ambiguity (its "more than one new
            # minima found" stderr message) occasionally yields a REFLECTED
            # frame (det = -1) or NaNs; scipy Rotation.from_matrix throws on
            # those downstream. A proper rotation has det = +1 -- drop anything
            # else at the boundary.
            if (not np.all(np.isfinite(R)) or not np.all(np.isfinite(t))
                    or np.linalg.det(R) < 0.5):
                continue
            out.append(
                self.gate_obs(
                    tag_id=d.tag_id,
                    tag_family=getattr(d, "tag_family", b""),
                    R_ct=R,
                    t_ct=t,
                    corners_px=np.asarray(d.corners, dtype=np.float64) / s,
                    K=(fx, fy, cx, cy),
                    tag_size=size,
                    decision_margin=float(d.decision_margin),
                    center_px=np.asarray(d.center, dtype=np.float64) / s,
                    pose_err=None if d.pose_err is None else float(d.pose_err),
                )
            )
        return out

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            import cv2  # lazy: keep import-time clean / optional in tests

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim != 2:
            raise ValueError(f"expected gray (H,W) or color (H,W,3), got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    # ----- bundle fusion ----- #
    def bundle_pose(
        self,
        detections: Sequence[TagObs],
        bundle: Optional[TagBundle] = None,
    ) -> Optional[ObjPose]:
        """Fuse gated tag observations into one object pose (camera frame).

        Only detections whose ``tag_id`` is a bundle member are used. Each maps
        its camera<-tag pose to a camera<-object pose via the fixed tag->object
        transform. Orientation-trusted tags (``accepted``) drive the fused
        orientation (chordal quaternion mean) AND the center; ``position_only``
        tags (too oblique) contribute ONLY to the center. Returns ``None`` if no
        bundle member is present.

        The ``gate`` flag on the returned :class:`ObjPose` is:
          * ``"high"``          -- >=1 orientation-trusted tag (orientation valid),
          * ``"position_only"`` -- only oblique tags (center valid, no orientation),
          * ``"none"``          -- members present but nothing usable.
        """
        bnd = bundle if bundle is not None else self.bundle
        if bnd is None:
            raise ValueError("bundle_pose: no TagBundle configured or supplied")

        ori_quats: List[np.ndarray] = []          # WXYZ, orientation-trusted
        ori_weights: List[float] = []
        ori_ids: List[int] = []
        ori_Rco: List[np.ndarray] = []
        centers: List[np.ndarray] = []            # camera-frame object centers
        center_weights: List[float] = []
        n_members = 0

        for obs in detections:
            tid = int(obs.tag_id)
            if tid not in bnd:
                continue
            n_members += 1
            R_co, t_co = bnd.object_pose_from_tag(obs.R, obs.t, tid)
            if obs.accepted:
                q = quat_xyzw_to_wxyz(Rotation.from_matrix(R_co).as_quat())
                w = max(float(obs.decision_margin), 1e-6)
                ori_quats.append(q)
                ori_weights.append(w)
                ori_ids.append(tid)
                ori_Rco.append(R_co)
                centers.append(t_co)
                center_weights.append(w)
            elif obs.position_only:
                centers.append(t_co)
                center_weights.append(max(float(obs.decision_margin), 1e-6))
            # fully-rejected tags contribute nothing

        if n_members == 0:
            return None
        if not centers:
            return ObjPose(center=None, gate="none", n_tags=0, n_position_tags=0)

        cw = np.asarray(center_weights, dtype=np.float64)
        center = (np.asarray(centers, dtype=np.float64) * cw[:, None]).sum(0) / cw.sum()

        if ori_quats:
            q_mean = average_quaternions_wxyz(ori_quats, ori_weights)
            R_mean = Rotation.from_quat(quat_wxyz_to_xyzw(q_mean)).as_matrix()
            spread = _max_pairwise_angle_deg(ori_Rco)
            return ObjPose(
                center=center,
                quat_wxyz=q_mean,
                R=R_mean,
                n_tags=len(ori_quats),
                n_position_tags=len(centers),
                gate="high",
                used_ids=ori_ids,
                orientation_spread_deg=spread,
            )
        # Only oblique / position tags contributed.
        return ObjPose(
            center=center,
            quat_wxyz=None,
            R=None,
            n_tags=0,
            n_position_tags=len(centers),
            gate="position_only",
            used_ids=[],
            orientation_spread_deg=0.0,
        )


def _max_pairwise_angle_deg(rotations: Sequence[np.ndarray]) -> float:
    """Max geodesic angle (deg) among a set of rotation matrices (0 if <2)."""
    if len(rotations) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            dR = np.asarray(rotations[i]).T @ np.asarray(rotations[j])
            cos = (np.trace(dR) - 1.0) / 2.0
            cos = max(-1.0, min(1.0, float(cos)))
            worst = max(worst, float(np.degrees(np.arccos(cos))))
    return worst
