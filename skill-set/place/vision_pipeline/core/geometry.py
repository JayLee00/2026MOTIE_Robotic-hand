"""Placement geometry for the place pipeline (env-agnostic, numpy only).

Frame conventions (see vision_pipeline_design.md §2/§7):
- everything is expressed in the gravity-aligned `world` frame (z = up), meters.
- `n_palm` = the KISTAR palm normal in world (live `right_palm` +x axis).
- AnyPlace returns T_pred as a RELATIVE transform applied on the LEFT of the
  object's current pose: final_obj = T_pred @ current_obj.

Pipeline (steps 3-15, 3-16, 3-18, 3-19, 4):
    T_zalign         = align_palm_down(child_pc, n_palm)      # palm +normal -> world -z
    child_pc_zalign  = apply(T_zalign, child_pc)              # AnyPlace child input
    T_pred (K,4,4)   = AnyPlace(parent_pc, child_pc_zalign)   # relative, multimodal
    k*               = rank_upright(T_pred)                    # §9-A pick least-tilted
    T_act            = T_pred[k*] @ T_zalign                  # world, left-mult
    EE_target        = T_act @ EE_current                     # rigid grasp
"""
import numpy as np

WORLD_DOWN = np.array([0.0, 0.0, -1.0])


def farthest_points(points, k):
    """Farthest-point sampling: indices of k mutually-spread points (greedy max-min
    distance). Used for the fruit scenario to keep only the k most-separated tray holes
    (step 3-17-A-2). Returns all indices if there are <= k points."""
    P = np.asarray(points, float).reshape(len(points), -1)
    n = len(P)
    if n <= k:
        return list(range(n))
    idx = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]   # start: farthest from centroid
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return idx


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def rotation_align(a, b):
    """Minimal rotation matrix (3x3) sending unit-ish vector a onto b (Rodrigues)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:                      # already aligned
        return np.eye(3)
    if c < -1 + 1e-9:                     # opposite: 180 deg about any axis ⟂ a
        axis = _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        K = _skew(axis)
        return np.eye(3) + 2.0 * (K @ K)  # Rodrigues at theta=pi
    K = _skew(v)
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def homog(R=None, t=None):
    """Build a 4x4 from rotation (3x3) and/or translation (3,)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def apply(T, pc):
    """Apply 4x4 transform T to an (N,3) point cloud -> (N,3)."""
    pc = np.asarray(pc, float)
    return pc @ T[:3, :3].T + T[:3, 3]


def align_palm_down(child_pc, n_palm, pivot=None):
    """T_zalign (4x4, world): rotate so the palm normal points to world -z.

    Rotation pivots about `pivot` (default = child_pc centroid) so the cloud
    stays put in place; AnyPlace mean-centers internally so the pivot only
    affects bookkeeping, not the prediction.
    """
    R = rotation_align(n_palm, WORLD_DOWN)
    c = np.asarray(pivot if pivot is not None else np.asarray(child_pc, float).mean(0), float)
    return homog(t=c) @ homog(R=R) @ homog(t=-c)


def compose_T_act(T_pred, T_zalign):
    """Final placement action in world (left-mult on current object/EE pose)."""
    return np.asarray(T_pred, float) @ np.asarray(T_zalign, float)


def ee_target(T_act, ee_current):
    """World-frame EE goal: object is rigidly grasped, so EE moves by T_act."""
    return np.asarray(T_act, float) @ np.asarray(ee_current, float)


def rank_upright(transforms):
    """§9-A: among candidate transforms (K,4,4), pick the one whose rotation tilts
    the palm-down axis least from world vertical.

    child_pc_zalign has its palm normal along world -z; after a candidate's
    rotation R the palm axis becomes R @ (-z), and cos(R@(-z), -z) = R[2,2].
    Returns (best_index, scores) with scores = R[:,2,2] (higher = more upright).
    """
    T = np.asarray(transforms, float)
    if T.ndim == 2:
        T = T[None]
    scores = T[:, 2, 2]
    return int(np.argmax(scores)), scores


def local_crop_size(child_pc_com, n_palm):
    """Max diameter of the cloud on the plane perpendicular to n_palm (meters).

    Used as AnyPlace's parent local-crop extent (step 3-12).
    """
    pc = np.asarray(child_pc_com, float)
    n = _unit(n_palm)
    proj = pc - np.outer(pc @ n, n)          # drop the n component -> on-plane coords
    proj = proj - proj.mean(0)
    r = np.linalg.norm(proj, axis=1)
    return float(2.0 * r.max())              # diameter ~ 2 * max radius


def crop_region(parent_pc_full, center, size, axis=None, margin=0.0):
    """Crop parent_pc_full to a cube of side `size` around `center` (step 3-17).

    If `axis` (palm normal) is given the crop is a square prism aligned to that
    axis with a little extra depth along it (placement tolerance); else an
    axis-aligned cube. Returns (M,3).

    `margin` (m) expands the crop on EVERY side beyond `size`: the in-plane half-extent
    (radius) becomes size/2 + margin, and the along-axis half-depth grows with it. `size`
    is just the grasped-object footprint (local_crop_size); the bare crop gives AnyPlace too
    little surrounding parent context, so a margin widens it (see orchestrator CROP_MARGIN).
    """
    pc = np.asarray(parent_pc_full, float)
    c = np.asarray(center, float)
    h = 0.5 * size + float(margin)
    if axis is None:
        m = np.all(np.abs(pc - c) <= h, axis=1)
        return pc[m]
    n = _unit(axis)
    d = (pc - c) @ n                          # signed distance along axis
    rad = pc - c - np.outer(d, n)             # in-plane offset
    in_plane = np.linalg.norm(rad, axis=1) <= h
    in_depth = np.abs(d) <= (h + 0.05)        # +5cm tolerance along placement axis
    return pc[in_plane & in_depth]


def contact_project(placed_obj, parent_pts, xy_radius=0.006):
    """Rest a placed object ON the parent surface — resolve AnyPlace interpenetration by lifting
    the object straight up (world +z, anti-gravity) by the MINIMAL Δz that removes all
    penetration below the parent (first contact from above).

    AnyPlace is a learned pose regressor with no non-penetration constraint, so its selected pose
    drives the object INTO the surface. For each object point, the parent's local surface height
    is the max z of parent points within `xy_radius` in the xy plane; the object must sit at or
    above it, so Δz = max over object points of (local surface height − object z), clamped ≥ 0.
    Object points with NO parent beneath them (e.g. over an empty tray hole) impose no
    constraint, so the object still nestles into recesses. Returns (dz, lifted_obj); apply the
    same +z·dz translation to T_act / ee_target (a pure vertical shift keeps the orientation)."""
    from scipy.spatial import cKDTree
    obj = np.asarray(placed_obj, float)
    par = np.asarray(parent_pts, float)
    if len(obj) == 0 or len(par) == 0:
        return 0.0, obj
    tree = cKDTree(par[:, :2])
    dz = 0.0
    for i, nb in enumerate(tree.query_ball_point(obj[:, :2], r=xy_radius)):
        if nb:
            dz = max(dz, float(par[nb, 2].max() - obj[i, 2]))
    lifted = obj.copy()
    lifted[:, 2] += dz
    return dz, lifted


def thicken(pc, thickness, step=0.005, axis=WORLD_DOWN):
    """Extrude a (thin, surface-only) cloud along `axis` to give it SOLID thickness.

    parent_pc_k straight out of crop_region is just the observed TOP surface of the tray, so
    AnyPlace can slide that thin sheet into the gap between the grasped object and the hand
    instead of resting the object ON it. Filling `thickness` m of points DOWNWARD (default the
    gravity direction, world -z) — one layer every `step` m — turns the sheet into a slab the
    object must sit on top of. Returns the original points plus the fill (M*(1+n_layers), 3).
    """
    pc = np.asarray(pc, float)
    if len(pc) == 0 or thickness <= 0:
        return pc
    n = _unit(axis)
    offs = np.arange(step, thickness + 1e-9, step)          # step, 2*step, ... thickness
    layers = [pc] + [pc + o * n for o in offs]              # surface + downward fill
    return np.concatenate(layers, axis=0)


if __name__ == "__main__":  # self-test
    rng = np.random.default_rng(0)
    # palm pointing +x; T_zalign must send it to -z.
    n_palm = np.array([1.0, 0.0, 0.0])
    child = rng.normal([0.3, 0.0, 0.5], 0.03, size=(500, 3))
    Tz = align_palm_down(child, n_palm)
    assert np.allclose(Tz[:3, :3] @ n_palm, WORLD_DOWN, atol=1e-6), "palm not aligned to -z"
    # pivot invariance: centroid stays put
    assert np.allclose(apply(Tz, child).mean(0), child.mean(0), atol=1e-6)
    # ranking: an identity rotation (upright) must beat a 90deg tilt
    T_up = homog(R=np.eye(3))
    T_tilt = homog(R=rotation_align([0, 0, 1], [1, 0, 0]))  # z->x, so [2,2]=0
    k, sc = rank_upright(np.stack([T_tilt, T_up]))
    assert k == 1 and sc[1] > sc[0], (k, sc)
    # T_act / ee_target compose without error
    Tact = compose_T_act(T_up, Tz)
    _ = ee_target(Tact, homog(R=np.eye(3), t=[0.3, 0, 0.5]))
    # local_crop_size of a flat disk of radius 0.05 in the plane ⟂ n_palm ~ 0.10
    th = np.linspace(0, 2 * np.pi, 200)
    disk = np.stack([np.zeros_like(th), 0.05 * np.cos(th), 0.05 * np.sin(th)], 1) + [0.3, 0, 0.5]
    lcs = local_crop_size(disk, n_palm)
    assert abs(lcs - 0.10) < 0.01, lcs
    print("geometry self-test OK  (local_crop_size=%.3f)" % lcs)
