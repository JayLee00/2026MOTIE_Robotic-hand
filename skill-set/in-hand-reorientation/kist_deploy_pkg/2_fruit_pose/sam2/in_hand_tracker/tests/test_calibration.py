"""US-005 / AC-5: per-object shape calibration on synthetic clouds.

Covers:
  (1) Synthetic *sphere* (a = c): free-fit recovers (a, c) within 10% rel error
      and tags ``"spheroid"``.
  (2) Synthetic *prolate* spheroid (c/a = 1.5): free-fit recovers (a, c) within
      10% rel error and tags ``"elongated"`` (>= ca_threshold 1.2).
  (3) Mild measurement noise: still within 10% rel error.
  (4) Partial (self-occluded) hemisphere-ish cloud: still within tolerance and
      tagged correctly (the early-frame use case).
  (5) Classification boundary against ca_threshold.
  (6) objects.yaml write round-trip writes a, c, shape_class (+ semantic_color).
  (7) Degenerate / tiny clouds return a non-ok result instead of raising.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from in_hand_tracker.calibration import (
    FreeFitResult,
    free_fit_spheroid,
    classify_shape,
    calibrate_object,
    write_object_entry,
)

CA_THRESHOLD = 1.2  # mirrors config/estimator.yaml


# --------------------------------------------------------------------------- #
# Synthetic cloud generators (rotation-symmetric, axis = world-z).
# --------------------------------------------------------------------------- #
def _sample_spheroid(a, c, center, n, seed=0, partial=False, noise=0.0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = np.column_stack([a * d[:, 0], a * d[:, 1], c * d[:, 2]])
    if partial:
        # Keep only the camera-facing-ish hemisphere (self-occlusion analog).
        keep = d[:, 2] > -0.3
        pts = pts[keep]
    pts = pts + np.asarray(center, dtype=np.float64)
    if noise > 0.0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts


def _rel_err(est, gt):
    return abs(float(est) - float(gt)) / abs(float(gt))


# --------------------------------------------------------------------------- #
# (1) Sphere
# --------------------------------------------------------------------------- #
def test_sphere_fit_and_tag():
    a_gt = c_gt = 0.040
    center_gt = np.array([0.10, -0.20, 0.55])
    P = _sample_spheroid(a_gt, c_gt, center_gt, n=4000, seed=1)

    fit = calibrate_object(P, ca_threshold=CA_THRESHOLD)
    assert fit.ok
    assert _rel_err(fit.a, a_gt) < 0.10
    assert _rel_err(fit.c, c_gt) < 0.10
    assert np.linalg.norm(fit.center - center_gt) < 0.10 * a_gt
    assert fit.shape_class == "spheroid"


# --------------------------------------------------------------------------- #
# (2) Prolate spheroid, c/a = 1.5
# --------------------------------------------------------------------------- #
def test_prolate_fit_and_tag():
    a_gt, c_gt = 0.030, 0.045  # c/a = 1.5
    center_gt = np.array([-0.05, 0.12, 0.60])
    P = _sample_spheroid(a_gt, c_gt, center_gt, n=4000, seed=2)

    fit = calibrate_object(P, ca_threshold=CA_THRESHOLD)
    assert fit.ok
    assert _rel_err(fit.a, a_gt) < 0.10
    assert _rel_err(fit.c, c_gt) < 0.10
    assert fit.ca_ratio == pytest.approx(1.5, rel=0.10)
    assert fit.shape_class == "elongated"


# --------------------------------------------------------------------------- #
# (3) Noisy sphere
# --------------------------------------------------------------------------- #
def test_noisy_prolate_within_tolerance():
    a_gt, c_gt = 0.030, 0.045
    center_gt = np.array([0.0, 0.0, 0.50])
    P = _sample_spheroid(a_gt, c_gt, center_gt, n=4000, seed=3, noise=0.0008)

    fit = calibrate_object(P, ca_threshold=CA_THRESHOLD)
    assert fit.ok
    assert _rel_err(fit.a, a_gt) < 0.10
    assert _rel_err(fit.c, c_gt) < 0.10
    assert fit.shape_class == "elongated"


# --------------------------------------------------------------------------- #
# (4) Partial (self-occluded) sphere -- the early-frame use case
# --------------------------------------------------------------------------- #
def test_partial_sphere_within_tolerance():
    a_gt = c_gt = 0.040
    center_gt = np.array([0.10, -0.20, 0.55])
    P = _sample_spheroid(a_gt, c_gt, center_gt, n=8000, seed=4, partial=True)
    assert P.shape[0] < 8000  # confirm some surface was removed

    fit = calibrate_object(P, ca_threshold=CA_THRESHOLD)
    assert fit.ok
    assert _rel_err(fit.a, a_gt) < 0.10
    assert _rel_err(fit.c, c_gt) < 0.10
    assert fit.shape_class == "spheroid"


# --------------------------------------------------------------------------- #
# (5) Classification boundary
# --------------------------------------------------------------------------- #
def test_classify_shape_boundary():
    # Round (ratio < threshold) -> spheroid.
    assert classify_shape(0.04, 0.044, ca_threshold=1.2) == "spheroid"
    # At/above threshold -> elongated (oblate or prolate, axis-agnostic).
    assert classify_shape(0.04, 0.060, ca_threshold=1.2) == "elongated"
    assert classify_shape(0.060, 0.04, ca_threshold=1.2) == "elongated"
    # Exact sphere is spheroid.
    assert classify_shape(0.04, 0.04, ca_threshold=1.2) == "spheroid"


# --------------------------------------------------------------------------- #
# (6) objects.yaml write round-trip
# --------------------------------------------------------------------------- #
def test_write_object_entry_roundtrip(tmp_path):
    yaml_path = os.path.join(str(tmp_path), "objects.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"objects": {}}, f)

    a_gt, c_gt = 0.030, 0.045
    P = _sample_spheroid(a_gt, c_gt, [0.0, 0.0, 0.5], n=4000, seed=5)
    fit = calibrate_object(P, ca_threshold=CA_THRESHOLD)
    assert fit.ok

    cfg = write_object_entry(
        yaml_path, "apple", fit, semantic_color="(128, 0, 0)"
    )
    assert "apple" in cfg["objects"]

    with open(yaml_path) as f:
        reloaded = yaml.safe_load(f)
    entry = reloaded["objects"]["apple"]
    assert _rel_err(entry["a"], a_gt) < 0.10
    assert _rel_err(entry["c"], c_gt) < 0.10
    assert entry["shape_class"] == "elongated"
    assert entry["semantic_color"] == "(128, 0, 0)"

    # A second object merges without clobbering the first.
    P2 = _sample_spheroid(0.04, 0.04, [0.0, 0.0, 0.5], n=4000, seed=6)
    fit2 = calibrate_object(P2, ca_threshold=CA_THRESHOLD)
    write_object_entry(yaml_path, "orange", fit2)
    with open(yaml_path) as f:
        reloaded2 = yaml.safe_load(f)
    assert set(reloaded2["objects"]) == {"apple", "orange"}
    assert reloaded2["objects"]["orange"]["shape_class"] == "spheroid"


# --------------------------------------------------------------------------- #
# (7) Degenerate clouds
# --------------------------------------------------------------------------- #
def test_degenerate_cloud_returns_not_ok():
    fit = calibrate_object(np.zeros((3, 3)), ca_threshold=CA_THRESHOLD)
    assert isinstance(fit, FreeFitResult)
    assert not fit.ok
    assert fit.a is None and fit.c is None

    # Too few points also raises at the primitive level.
    with pytest.raises(ValueError):
        free_fit_spheroid(np.zeros((4, 3)))
