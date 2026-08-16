"""Point-cloud outlier removal that drops ONLY points far from the main body while
PRESERVING the object surface — unlike statistical outlier removal (SOR), which is
boundary-biased and shaves the object's own edge.

Each function returns a boolean keep-mask (True = keep). The real failure mode on our
RGB-D partials is a dense DISCONNECTED cluster (segmentation bleed onto a neighbouring
surface) whose LOCAL density matches the object surface — so density/count methods
(SOR, ROR, LOF) can't separate it from the surface boundary. CONNECTIVITY methods
(DBSCAN keep-largest, connected-components) use the spatial GAP and both remove the far
cluster and keep the whole surface.
"""
import numpy as np


def sor(pts, nb=20, std_ratio=2.0):
    """Statistical Outlier Removal (open3d) — baseline; boundary-biased (shaves surface)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    _, idx = pc.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
    m = np.zeros(len(pts), bool); m[np.asarray(idx)] = True
    return m


def ror(pts, radius=0.005, min_pts=16):
    """Radius Outlier Removal (open3d) — drops points with < min_pts neighbours in radius.
    Removes truly isolated points; a dense far cluster survives (it has neighbours)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    _, idx = pc.remove_radius_outlier(nb_points=min_pts, radius=radius)
    m = np.zeros(len(pts), bool); m[np.asarray(idx)] = True
    return m


def dbscan(pts, eps=0.005, min_samples=10):
    """DBSCAN keep-largest-cluster — THE canonical outlier removal. Drops far
    disconnected clusters + floaters via the spatial gap, keeps the whole connected
    body (surface preserved). Prefers open3d, falls back to sklearn, then a scipy
    connected-components equivalent — so it runs in the ROS container (no open3d) too."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < min_samples + 1:
        return np.ones(n, bool)
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        labels = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_samples))
    except Exception:                                              # noqa: BLE001
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        except Exception:                                          # noqa: BLE001
            from scipy.spatial import cKDTree                       # connected-components fallback
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            pairs = cKDTree(pts).query_pairs(eps, output_type="ndarray")
            if len(pairs) == 0:
                return np.ones(n, bool)
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, labels = connected_components(g, directed=False)
            return labels == np.bincount(labels).argmax()
    if labels.max() < 0:
        return np.ones(n, bool)
    return labels == np.bincount(labels[labels >= 0]).argmax()


def clean(pts, eps=0.005, min_samples=10):
    """Production convenience: return the DBSCAN keep-largest subset of `pts`."""
    pts = np.asarray(pts, float)
    return pts[dbscan(pts, eps=eps, min_samples=min_samples)]


def connected(pts, radius=0.002):
    """Largest connected component of the radius graph (scipy) — connectivity twin of
    DBSCAN; keeps the biggest spatially-connected blob, drops far clusters."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(pts)
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(n, bool)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lbl = connected_components(g, directed=False)
    keep = np.bincount(lbl).argmax()
    return lbl == keep


def lof(pts, n_neighbors=20, contamination="auto"):
    """Local Outlier Factor (sklearn) — flags points whose local density deviates from
    their neighbours'. A uniformly-dense far cluster reads as normal, so it is missed."""
    from sklearn.neighbors import LocalOutlierFactor
    yhat = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(pts)
    return yhat == 1


def isoforest(pts, contamination=0.01, seed=0):
    """Isolation Forest (sklearn) — isolates points in sparse regions of the xyz box."""
    from sklearn.ensemble import IsolationForest
    yhat = IsolationForest(contamination=contamination, random_state=seed).fit_predict(pts)
    return yhat == 1


METHODS = {"sor": sor, "ror": ror, "dbscan": dbscan, "connected": connected,
           "lof": lof, "isoforest": isoforest}
