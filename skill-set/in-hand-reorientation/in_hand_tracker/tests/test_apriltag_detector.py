"""C3: AprilTag gating + tag-bundle fusion (doc 3.2 + 4 / 4.1).

These tests exercise the GATING logic and the BUNDLE-FUSION math with SYNTHETIC,
constructed detections -- no native AprilTag library and no live image are
required for the core asserts. We build mock "detections" with known
``(R, t, decision_margin, corners)`` and assert:

  * gating REJECTS low decision-margin, too-small, and high-reprojection tags,
  * gating drops orientation but KEEPS position for too-oblique tags,
  * a head-on tag PASSES all gates,
  * two tags on DIFFERENT faces (with known tag->object transforms) FUSE to one
    consistent object pose (orientation + center agree with ground truth),
  * an oblique-only bundle returns ``position_only`` (center, no orientation).

A real-detection smoke test is included but skipped unless ``pupil_apriltags``
is importable (it is, in dexsdr) AND a tag image can be synthesized.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from in_hand_tracker.perception.apriltag_detector import (
    SUPPORTED_FAMILIES,
    AprilTagDetector,
    GatingConfig,
    ObjPose,
    TagBundle,
    TagObs,
    average_quaternions_wxyz,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    apparent_tag_size_px,
    reprojection_error_px,
    view_angle_deg,
)

# A plausible 640x480 pinhole.
K = (600.0, 600.0, 320.0, 240.0)
TAG_SIZE = 0.04  # 4 cm


# --------------------------------------------------------------------------- #
# Synthetic-detection helpers
# --------------------------------------------------------------------------- #
def _tag_corners_obj(tag_size):
    # Must match apriltag3's pose pairing (see _tag_object_corners):
    # p[0]=(-s,+s), p[1]=(+s,+s), p[2]=(+s,-s), p[3]=(-s,-s).
    h = 0.5 * tag_size
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


def _project(R_ct, t_ct, tag_size, K):
    """Project the tag's corners into pixels under camera<-tag pose (R, t)."""
    fx, fy, cx, cy = K
    cam = _tag_corners_obj(tag_size) @ np.asarray(R_ct).T + np.asarray(t_ct)
    z = cam[:, 2]
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def make_obs(
    detector,
    tag_id,
    R_ct,
    t_ct,
    decision_margin=50.0,
    tag_size=TAG_SIZE,
    corners=None,
    pose_err=None,
):
    """Build a gated TagObs from an ideal (R, t) by reprojecting its corners."""
    if corners is None:
        corners = _project(R_ct, t_ct, tag_size, K)
    return detector.gate_obs(
        tag_id=tag_id,
        tag_family="tag36h11",
        R_ct=np.asarray(R_ct, dtype=np.float64),
        t_ct=np.asarray(t_ct, dtype=np.float64),
        corners_px=corners,
        K=K,
        tag_size=tag_size,
        decision_margin=decision_margin,
        center_px=corners.mean(axis=0),
        pose_err=pose_err,
    )


# --------------------------------------------------------------------------- #
# Quaternion boundary helpers
# --------------------------------------------------------------------------- #
def test_quat_wxyz_xyzw_roundtrip():
    q_wxyz = np.array([0.5, 0.5, -0.5, 0.5])
    q_wxyz /= np.linalg.norm(q_wxyz)
    back = quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q_wxyz))
    assert np.allclose(back, q_wxyz)
    # scipy uses XYZW; verify our reorder matches a known rotation.
    R = Rotation.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
    q_back = quat_xyzw_to_wxyz(Rotation.from_matrix(R).as_quat())
    # double-cover: same rotation up to sign.
    assert np.allclose(q_back, q_wxyz) or np.allclose(q_back, -q_wxyz)


def test_average_quaternions_handles_sign_ambiguity():
    q = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    q_wxyz = quat_xyzw_to_wxyz(q.as_quat())
    mean = average_quaternions_wxyz([q_wxyz, -q_wxyz, q_wxyz])
    # mean of a rotation and its sign-flipped copy is the SAME rotation.
    R_mean = Rotation.from_quat(quat_wxyz_to_xyzw(mean)).as_matrix()
    assert np.allclose(R_mean, q.as_matrix(), atol=1e-9)
    assert mean[0] >= 0.0  # canonical sign


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def test_apparent_size_matches_projection():
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.5])
    corners = _project(R, t, TAG_SIZE, K)
    size_px = apparent_tag_size_px(corners)
    # head-on at 0.5 m: edge = fx * tag_size / z.
    expected = K[0] * TAG_SIZE / 0.5
    assert np.isclose(size_px, expected, rtol=1e-6)


def test_reprojection_error_zero_for_consistent_pose():
    R = Rotation.from_euler("y", 15, degrees=True).as_matrix()
    t = np.array([0.02, -0.01, 0.4])
    corners = _project(R, t, TAG_SIZE, K)
    err = reprojection_error_px(R, t, corners, K, TAG_SIZE)
    assert err < 1e-6


def test_reprojection_error_large_for_wrong_pose():
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.4])
    corners = _project(R, t, TAG_SIZE, K)
    # Reproject with a clearly wrong (rotated) pose -> big residual.
    R_bad = Rotation.from_euler("x", 40, degrees=True).as_matrix()
    err = reprojection_error_px(R_bad, t, corners, K, TAG_SIZE)
    assert err > 5.0


def test_view_angle_headon_and_oblique():
    t = np.array([0.0, 0.0, 0.5])
    assert view_angle_deg(np.eye(3), t) < 1.0
    R_obl = Rotation.from_euler("x", 75, degrees=True).as_matrix()
    assert view_angle_deg(R_obl, t) > 70.0


# --------------------------------------------------------------------------- #
# GATING
# --------------------------------------------------------------------------- #
def test_headon_tag_passes_all_gates():
    det = AprilTagDetector(families="tag36h11", tag_size=TAG_SIZE)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.35])  # close + head-on => large, low reproj
    obs = make_obs(det, 0, R, t, decision_margin=80.0)
    assert obs.accepted
    assert not obs.position_only
    assert obs.reject_reason == ""
    assert obs.reproj_error_px < 1e-6
    assert obs.tag_size_px > det.gating.min_tag_size_px


def test_low_decision_margin_rejected():
    det = AprilTagDetector(tag_size=TAG_SIZE)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.35])
    obs = make_obs(det, 0, R, t, decision_margin=5.0)  # below default 30
    assert not obs.accepted
    assert not obs.position_only
    assert obs.reject_reason == "low_decision_margin"


def test_tag_too_small_rejected():
    det = AprilTagDetector(tag_size=TAG_SIZE)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5.0])  # far away -> tiny apparent size
    obs = make_obs(det, 0, R, t, decision_margin=80.0)
    assert not obs.accepted
    assert obs.reject_reason == "tag_too_small"
    assert obs.tag_size_px < det.gating.min_tag_size_px


def test_high_reproj_error_rejected():
    det = AprilTagDetector(tag_size=TAG_SIZE)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.35])
    good_corners = _project(R, t, TAG_SIZE, K)
    # Perturb the detected corners heavily so the pose no longer reprojects.
    bad_corners = good_corners + np.array([15.0, -15.0])
    obs = make_obs(det, 0, R, t, decision_margin=80.0, corners=bad_corners)
    assert not obs.accepted
    assert obs.reject_reason == "high_reproj_error"


def test_oblique_tag_drops_orientation_keeps_position():
    det = AprilTagDetector(tag_size=TAG_SIZE, gating=GatingConfig(max_view_angle_deg=70.0))
    R = Rotation.from_euler("x", 80, degrees=True).as_matrix()  # 80 deg oblique
    t = np.array([0.0, 0.0, 0.30])
    obs = make_obs(det, 0, R, t, decision_margin=80.0)
    assert not obs.accepted          # orientation NOT trusted
    assert obs.position_only         # but position is kept
    assert obs.reject_reason == "oblique_position_only"
    assert obs.view_angle_deg > 70.0


def test_oblique_tag_fully_dropped_when_position_not_kept():
    det = AprilTagDetector(
        tag_size=TAG_SIZE,
        gating=GatingConfig(max_view_angle_deg=70.0, keep_position_when_oblique=False),
    )
    R = Rotation.from_euler("x", 80, degrees=True).as_matrix()
    t = np.array([0.0, 0.0, 0.30])
    obs = make_obs(det, 0, R, t, decision_margin=80.0)
    assert not obs.accepted
    assert not obs.position_only
    assert obs.reject_reason == "too_oblique"


def test_pose_err_gate():
    det = AprilTagDetector(tag_size=TAG_SIZE, gating=GatingConfig(max_pose_err=0.5))
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.35])
    obs = make_obs(det, 0, R, t, decision_margin=80.0, pose_err=1.0)
    assert not obs.accepted
    assert obs.reject_reason == "high_pose_err"


def test_unsupported_family_rejected():
    with pytest.raises(ValueError):
        AprilTagDetector(families="tag25h9")
    # both supported families are accepted.
    for fam in SUPPORTED_FAMILIES:
        AprilTagDetector(families=fam)


# --------------------------------------------------------------------------- #
# BUNDLE FUSION
# --------------------------------------------------------------------------- #
def _build_two_face_bundle():
    """Two tags rigidly attached to one object, on different faces.

    Tag 0: on the +z object face, no rotation, offset +z by 0.03 m.
    Tag 1: on the +x object face, rotated 90 deg about object y, offset +x.
    The tag->object transforms below are the INVERSES of those placements.
    """
    # placement: object<-tag for tag 0 (tag frame == object frame, shifted +z)
    R_to_0 = np.eye(3)
    t_to_0 = np.array([0.0, 0.0, 0.03])
    # tag 1 lies on the +x face: its +z (normal) points along object +x.
    R_to_1 = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    t_to_1 = np.array([0.03, 0.0, 0.0])
    return TagBundle(
        members={0: (R_to_0, t_to_0), 1: (R_to_1, t_to_1)},
        tag_size=TAG_SIZE,
        name="fruit",
    )


def _camera_tag_pose(R_co, t_co, R_to, t_to):
    """Forward: given object pose in cam (R_co, t_co) and tag->obj, get cam<-tag."""
    # p_obj = R_to p_tag + t_to ; p_cam = R_co p_obj + t_co
    R_ct = R_co @ R_to
    t_ct = R_co @ t_to + t_co
    return R_ct, t_ct


def test_bundle_fuses_two_faces_to_consistent_pose():
    bundle = _build_two_face_bundle()
    det = AprilTagDetector(tag_size=TAG_SIZE, bundle=bundle)

    # Ground-truth object pose in the camera frame.
    R_co_gt = Rotation.from_euler("xyz", [10, -20, 35], degrees=True).as_matrix()
    t_co_gt = np.array([0.05, -0.02, 0.40])

    # Render both tags' camera poses from the SAME object pose.
    R_ct0, t_ct0 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[0])
    R_ct1, t_ct1 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[1])

    obs0 = make_obs(det, 0, R_ct0, t_ct0, decision_margin=70.0)
    obs1 = make_obs(det, 1, R_ct1, t_ct1, decision_margin=90.0)
    # Both must pass gating to be orientation-trusted.
    assert obs0.accepted and obs1.accepted

    pose = det.bundle_pose([obs0, obs1])
    assert isinstance(pose, ObjPose)
    assert pose.gate == "high"
    assert pose.n_tags == 2

    # Fused orientation matches GT.
    R_fused = pose.R
    dR = R_co_gt.T @ R_fused
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    assert ang < 1e-4, f"fused orientation off by {ang} deg"

    # Fused center matches the GT object origin in the camera frame.
    assert np.allclose(pose.center, t_co_gt, atol=1e-6)

    # The two faces are consistent (near-zero spread).
    assert pose.orientation_spread_deg < 1e-3

    # Quaternion is WXYZ and unit.
    assert pose.quat_wxyz.shape == (4,)
    assert np.isclose(np.linalg.norm(pose.quat_wxyz), 1.0)
    assert pose.quat_wxyz[0] >= 0.0


def test_bundle_single_tag_pose():
    bundle = _build_two_face_bundle()
    det = AprilTagDetector(tag_size=TAG_SIZE, bundle=bundle)
    R_co_gt = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    t_co_gt = np.array([0.0, 0.0, 0.35])
    R_ct0, t_ct0 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[0])
    obs0 = make_obs(det, 0, R_ct0, t_ct0, decision_margin=80.0)
    pose = det.bundle_pose([obs0])
    assert pose.gate == "high"
    assert pose.n_tags == 1
    assert np.allclose(pose.center, t_co_gt, atol=1e-6)
    assert pose.orientation_spread_deg == 0.0
    dR = R_co_gt.T @ pose.R
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    assert ang < 1e-4


def test_bundle_position_only_when_all_oblique():
    bundle = _build_two_face_bundle()
    det = AprilTagDetector(
        tag_size=TAG_SIZE, bundle=bundle, gating=GatingConfig(max_view_angle_deg=70.0)
    )
    # Build an object pose so both tags are seen very obliquely.
    R_co_gt = Rotation.from_euler("x", 82, degrees=True).as_matrix()
    t_co_gt = np.array([0.0, 0.0, 0.32])
    R_ct0, t_ct0 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[0])
    obs0 = make_obs(det, 0, R_ct0, t_ct0, decision_margin=80.0)
    # Force the second tag oblique too by reusing tag 0's geometry under id 1's slot
    # via a steeply tilted independent placement.
    R_ct1 = Rotation.from_euler("x", 84, degrees=True).as_matrix()
    t_ct1 = np.array([0.01, 0.0, 0.33])
    obs1 = make_obs(det, 1, R_ct1, t_ct1, decision_margin=80.0)

    assert obs0.position_only and obs1.position_only
    pose = det.bundle_pose([obs0, obs1])
    assert pose is not None
    assert pose.gate == "position_only"
    assert pose.quat_wxyz is None and pose.R is None
    assert pose.center is not None
    assert pose.n_position_tags == 2


def test_bundle_returns_none_without_members():
    bundle = TagBundle(members={5: (np.eye(3), np.zeros(3))}, tag_size=TAG_SIZE)
    det = AprilTagDetector(tag_size=TAG_SIZE, bundle=bundle)
    # detection id 99 not in the bundle.
    obs = make_obs(det, 99, np.eye(3), np.array([0.0, 0.0, 0.35]), decision_margin=80.0)
    assert det.bundle_pose([obs]) is None


def test_bundle_ignores_rejected_orientation_but_uses_others():
    bundle = _build_two_face_bundle()
    det = AprilTagDetector(tag_size=TAG_SIZE, bundle=bundle)
    R_co_gt = Rotation.from_euler("xyz", [5, 5, 5], degrees=True).as_matrix()
    t_co_gt = np.array([0.02, 0.0, 0.38])
    R_ct0, t_ct0 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[0])
    R_ct1, t_ct1 = _camera_tag_pose(R_co_gt, t_co_gt, *bundle.members[1])
    obs0 = make_obs(det, 0, R_ct0, t_ct0, decision_margin=80.0)        # good
    obs1 = make_obs(det, 1, R_ct1, t_ct1, decision_margin=2.0)         # low margin -> rejected
    assert obs0.accepted and not obs1.accepted and not obs1.position_only

    pose = det.bundle_pose([obs0, obs1])
    assert pose.gate == "high"
    assert pose.n_tags == 1                     # only the good tag drove orientation
    assert pose.used_ids == [0]
    assert np.allclose(pose.center, t_co_gt, atol=1e-6)


# --------------------------------------------------------------------------- #
# Real-detection smoke test (skipped unless lib + image synthesis available)
# --------------------------------------------------------------------------- #
def _have_pupil_apriltags():
    try:
        import pupil_apriltags  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_pupil_apriltags(), reason="pupil_apriltags not available")
def test_real_detection_smoke():
    """Render a tag36h11 marker via OpenCV's AprilTag dictionary and detect it.

    Skipped automatically if cv2.aruco lacks the AprilTag dictionary. Verifies
    the end-to-end wrapper: image -> detect() -> gated TagObs with a sane pose.
    """
    cv2 = pytest.importorskip("cv2")
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "DICT_APRILTAG_36h11"):
        pytest.skip("cv2.aruco AprilTag dictionary unavailable")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    # Draw tag id 0 as a marker image, place it on a white canvas.
    marker_px = 240
    try:
        marker = aruco.generateImageMarker(dictionary, 0, marker_px)
    except AttributeError:
        marker = aruco.drawMarker(dictionary, 0, marker_px)

    canvas = np.full((480, 640), 255, dtype=np.uint8)
    y0, x0 = (480 - marker_px) // 2, (640 - marker_px) // 2
    canvas[y0:y0 + marker_px, x0:x0 + marker_px] = marker

    # Intrinsics consistent with a tag drawn flat, facing the camera.
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    tag_size = 0.04

    det = AprilTagDetector(
        families="tag36h11", tag_size=tag_size, quad_decimate=1.0,
        gating=GatingConfig(min_decision_margin=10.0, min_tag_size_px=10.0,
                            max_reproj_error_px=20.0),
    )
    obs_list = det.detect(canvas, (fx, fy, cx, cy))
    ids = [o.tag_id for o in obs_list]
    assert 0 in ids, f"tag 0 not detected; got {ids}"
    o = next(o for o in obs_list if o.tag_id == 0)
    # The marker is centered and frontal: t should be roughly on the optical axis
    # and in front of the camera.
    assert o.t[2] > 0.0
    assert o.tag_size_px > 0.0
    # head-on synthetic marker: view angle should be small.
    assert o.view_angle_deg < 30.0
