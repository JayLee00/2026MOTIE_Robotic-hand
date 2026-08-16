"""Primitive-fit shape completion (env-agnostic, numpy + scipy) — the `complete=sphere`
FALLBACK to the default Act-VH IGR completion (services/igr_service.py).

Why a geometric primitive: a feed-forward PCN/ShapeNet completer (e.g. AdaPoinTr, since
removed) UNDER-SIZES a real object from a small single-view partial cap (~0.7x) and ignores
appended contact points (both verified empirically). A closed geometric primitive can't
under-size — fitting a sphere/ellipsoid to a ~1/3 cap yields the FULL object at the right
extent — and the fingertip CONTACT points (FK) drop in as extra high-confidence surface samples.

This is the lightweight in-repo route recommended by the model-search for fruit /
convex objects (the SOTA productionised equivalent is EMS superquadric fitting,
github.com/bmlklwx/EMS-superquadric_fitting, MIT — adopt that for general shapes).
"""
import numpy as np


def _fit_sphere(P, w=None):
    """Algebraic least-squares sphere fit -> (center(3,), radius)."""
    P = np.asarray(P, float)
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    if w is not None:
        s = np.sqrt(w)[:, None]
        A, b = A * s, b * s[:, 0]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    C = x[:3]
    return C, float(np.sqrt(max(x[3] + C @ C, 1e-9)))


def fit_sphere_robust(partial, contacts=None, iters=3, inlier_pct=85, contact_w=10.0):
    """Outlier-robust sphere fit. Contacts (FK fingertips, on the object surface incl. the
    occluded back) are added as heavily-weighted points so they constrain the fit."""
    P = np.asarray(partial, float)
    w = np.ones(len(P))
    if contacts is not None and len(contacts):
        P = np.vstack([P, np.asarray(contacts, float)])
        w = np.concatenate([w, np.full(len(contacts), contact_w)])
    C, R = _fit_sphere(P, w)
    for _ in range(iters):                              # reweight onto inliers (drop stragglers)
        res = np.abs(np.linalg.norm(P - C, axis=1) - R)
        keep = res <= np.percentile(res, inlier_pct)
        if keep.sum() < 4:
            break
        C, R = _fit_sphere(P[keep], w[keep])
    return C, R


def sample_sphere(C, R, n=16384):
    u = np.random.default_rng(0).standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return (np.asarray(C, float) + R * u).astype(np.float32)


def complete_sphere(partial, contacts=None, n=16384):
    """Full-object cloud from a partial cap via a robust sphere fit (correct size).
    Ideal for fruit / near-spherical graspables; uses FK contact points if given."""
    C, R = fit_sphere_robust(partial, contacts)
    return sample_sphere(C, R, n)
