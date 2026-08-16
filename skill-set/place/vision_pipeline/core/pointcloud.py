"""RGB-D -> world point cloud helpers (env-agnostic, numpy only).

Camera is the fixed front_cam; depth is aligned to color (align_depth.enable),
so a color pixel (u,v) maps to depth[v,u] with color intrinsics K. Optical frame
convention: z forward, x right, y down. `T_world_cam` = world<-camera_optical
(from TF; tf.txt provides right_fr3_link0<-cam, TF composes to world).
"""
import numpy as np

from .geometry import apply


def fill_holes(mask):
    """Fill INTERIOR holes of a boolean mask — background regions fully enclosed by the
    mask (e.g. another object placed ON the parent occludes its middle) — while leaving
    the outer boundary/EDGE untouched. Interior-only: a hole that touches the image border
    is NOT filled. (scipy.ndimage.binary_fill_holes.)"""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(np.asarray(mask, bool))


def backproject(depth_m, K, T_world_cam, mask=None, zmin=0.05, zmax=1.3):
    """Back-project a depth image to a world-frame (N,3) cloud.

    depth_m : (H,W) float meters (0/NaN = invalid)
    K       : (3,3) color intrinsics
    mask    : optional (H,W) bool; if given, only those pixels are projected
    """
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = np.mgrid[0:H, 0:W]
    z = depth_m.astype(np.float64)
    valid = np.isfinite(z) & (z > zmin) & (z < zmax)
    if mask is not None:
        valid &= mask.astype(bool)
    u, v, z = uu[valid], vv[valid], z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam = np.stack([x, y, z], axis=1)
    return apply(T_world_cam, cam)


def backproject_pixel(uv, depth_m, K, T_world_cam, win=5):
    """Back-project a single (x=u, y=v) pixel to a world point, using the median
    valid depth in a (2*win+1) window for robustness. Returns (3,) or None."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    H, W = depth_m.shape
    y0, y1 = max(0, v - win), min(H, v + win + 1)
    x0, x1 = max(0, u - win), min(W, u + win + 1)
    patch = depth_m[y0:y1, x0:x1].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 1.3)]
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
    return apply(T_world_cam, cam[None])[0]


def axis_from(T, axis=(1.0, 0.0, 0.0)):
    """World direction of a local axis given world<-frame transform T (e.g. palm +x)."""
    return T[:3, :3] @ np.asarray(axis, float)


def remove_outliers(pc, eps=0.005, min_samples=10):
    """Outlier removal = DBSCAN keep-largest-cluster (connectivity). Drops the
    disconnected mask-edge/segmentation-bleed clusters + floaters (which sit in a
    separate spatial cluster from the object) while PRESERVING the whole object surface.
    Statistical/radius density removal was rejected: the bleed is a DENSE cluster whose
    local density matches the surface, so density methods either miss it or shave the
    surface boundary (see core/outlier_removal.py + the outlier_compare study). Returns
    the kept (M,3) subset."""
    from vision_pipeline.core.outlier_removal import clean
    return clean(np.asarray(pc, float), eps=eps, min_samples=min_samples)


if __name__ == "__main__":  # self-test
    # synthetic: a planar patch at z=0.6m in front of an identity-pose camera
    H, W = 48, 64
    K = np.array([[60, 0, W / 2], [0, 60, H / 2], [0, 0, 1]], float)
    depth = np.full((H, W), 0.6)
    T = np.eye(4)  # camera == world (optical z forward)
    pc = backproject(depth, K, T)
    assert pc.shape[0] == H * W and np.allclose(pc[:, 2], 0.6), "z should be 0.6"
    # principal-point pixel -> on optical axis (x=y=0)
    p = backproject_pixel((W / 2, H / 2), depth, K, T)
    assert np.allclose(p, [0, 0, 0.6], atol=1e-6), p
    print("pointcloud self-test OK  (N=%d)" % pc.shape[0])
