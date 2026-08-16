"""Act-VH IGR shape completion — the reconstruction core shared by the offline
test (tools/actvh_test/run_igr.py) and the live pipeline service (services/igr_service.py).

Given a partial surface cloud + fingertip-pad contact points (world meters), it
optimizes a latent code of the pretrained IGR auto-decoder to fit those on-surface
samples, extracts the level-0 surface via marching cubes, and returns a completed
point cloud in the SAME world frame/scale. See run_igr.py for the porting notes
(torch 2.7 / cu128, RTX 5090 sm_120).

The functions below are transcribed 1:1 from the IGR repo + Act-VH offline pipeline;
`complete()` runs one reconstruction from a preloaded network (the net is expensive
to build, so callers load it once via load_network and reuse it).
"""
import os
import sys

import numpy as np
import torch
import trimesh
from skimage import measure

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/visuo-haptic-shape-completion"
IGR_CODE = os.path.join(REPO, "IGR", "code")
sys.path.insert(0, IGR_CODE)
from model.network import ImplicitNet, gradient  # noqa: E402  (pure PyTorch)

CKPT = os.path.join(
    REPO, "IGR", "exps", "no_ycb", "2021_07_14_12_58_49",
    "checkpoints", "ModelParameters", "latest.pth",
)

# --- config, from IGR/code/shapespace/shape_completion_setup_offline.conf ---
LATENT_SIZE = 256
D_IN = 3
NET_KW = dict(dims=[512] * 8, skip_in=[4], geometric_init=True, radius_init=1, beta=100)
GLOBAL_SIGMA = 1.8
LOCAL_SIGMA = 0.01
GRAD_LAMBDA = 0.1        # eikonal
NORMALS_LAMBDA = 1.0
LATENT_LAMBDA = 1e-3


def load_network(device):
    net = ImplicitNet(d_in=LATENT_SIZE + D_IN, **NET_KW).to(device)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    net.load_state_dict({k.replace("module.", ""): v
                         for k, v in sd["model_state_dict"].items()})
    net.eval()
    return net


def estimate_normals(points, center):
    """Local open3d normals, oriented outward relative to the cloud center (which
    lies inside the convex object), so even sparse occluded-back contacts get a
    correct outward normal. Does not use the true radius."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    n = np.asarray(pc.normals)
    out = points - center
    dots = np.einsum("ij,ij->i", n, out)
    bad = np.abs(dots) < 1e-9
    n[bad] = out[bad]
    n[np.einsum("ij,ij->i", n, out) < 0] *= -1.0
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n.astype(np.float32)


def denoise(points):
    """Outlier removal on the visual partial cloud = DBSCAN keep-largest-cluster
    (connectivity; removes disconnected clusters/floaters, preserves the surface).
    Contacts are trusted haptic points and are NOT denoised."""
    from vision_pipeline.core.outlier_removal import clean
    return clean(np.asarray(points, float)).astype(points.dtype)


def expand_contacts(contacts, normals, k=30, radius=0.006, seed=12345):
    """Expand each contact into a small on-surface tangent-plane patch (Act-VH touch
    patches) so a handful of contacts carry enough loss weight to hold the occluded
    back side in."""
    rng = np.random.default_rng(seed)
    pts, nrm = [], []
    for c, n in zip(contacts, normals):
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, a); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        rr = radius * np.sqrt(rng.random(k))
        th = rng.random(k) * 2 * np.pi
        disk = np.outer(rr * np.cos(th), u) + np.outer(rr * np.sin(th), v)
        pts.append(c + disk)
        nrm.append(np.tile(n, (k, 1)))
    pts = np.vstack([contacts, np.vstack(pts)]).astype(np.float32)
    nrm = np.vstack([normals, np.vstack(nrm)]).astype(np.float32)
    return pts, nrm


def normalize(points):
    """center on centroid, scale so the largest bbox side maps to length 2
    (preprocess_one_file.preprocess). Returns normalized pts + center + scale."""
    center = points.mean(axis=0)
    p = points - center
    diff = float(np.max(np.max(p, axis=0) - np.min(p, axis=0)))
    scale = 2.0 / diff
    return (p * scale).astype(np.float32), center.astype(np.float32), scale


def sample_nonsurface(pts):
    n = pts.shape[0]
    local = pts + torch.randn_like(pts) * LOCAL_SIGMA
    g = n // 32 if n // 32 > 0 else 1
    glob = torch.rand(g, 3, device=pts.device) * (GLOBAL_SIGMA * 2) - GLOBAL_SIGMA
    return torch.cat([local, glob], dim=0)


def adjust_lr(base_lr, opt, it):
    lr = base_lr * (0.1 ** (it // 400))
    for pg in opt.param_groups:
        pg["lr"] = lr


def optimize_latent(net, pts_partial, nrm_partial, pts_contact, nrm_contact,
                    device, iterations, lr, n_partial):
    """Optimize a fresh latent to fit surface pts+normals; every iter uses a random
    partial subset plus ALL contacts so occluded-back constraints are always on."""
    latent = torch.ones(LATENT_SIZE, device=device).normal_(0, 1.0 / LATENT_SIZE)
    latent.requires_grad_(True)
    opt = torch.optim.Adam([latent], lr=lr)

    np_pts = pts_partial.shape[0]
    k = min(n_partial, np_pts)
    for it in range(iterations):
        idx = torch.randperm(np_pts, device=device)[:k]
        pts = torch.cat([pts_partial[idx], pts_contact], dim=0)
        nrm = torch.cat([nrm_partial[idx], nrm_contact], dim=0)

        sample = sample_nonsurface(pts)

        lat_s = latent.expand(pts.shape[0], -1)
        surf = torch.cat([lat_s, pts], dim=1)
        lat_n = latent.expand(sample.shape[0], -1)
        nonsurf = torch.cat([lat_n, sample], dim=1)
        surf.requires_grad_()
        nonsurf.requires_grad_()

        surf_pred = net(surf)
        nonsurf_pred = net(nonsurf)
        surf_grad = gradient(surf, surf_pred)
        nonsurf_grad = gradient(nonsurf, nonsurf_pred)

        surface_loss = surf_pred.abs().mean()
        grad_loss = ((nonsurf_grad.norm(2, dim=-1) - 1) ** 2).mean()
        normals_loss = (surf_grad - nrm).abs().norm(2, dim=1).mean()
        latent_loss = latent.abs().mean()
        loss = (surface_loss + LATENT_LAMBDA * latent_loss
                + NORMALS_LAMBDA * normals_loss + GRAD_LAMBDA * grad_loss)

        adjust_lr(lr, opt, it)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return latent.detach()


def marching_cubes(net, latent, device, resolution, grid_lim=1.2, chunk=200000):
    """Evaluate the SDF on a uniform [-lim,lim]^3 grid, extract the level-0 mesh
    (largest connected component) in the normalized frame."""
    xs = np.linspace(-grid_lim, grid_lim, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    grid_t = torch.from_numpy(grid).to(device)

    vals = []
    lat = latent.unsqueeze(0)
    with torch.no_grad():
        for i in range(0, grid_t.shape[0], chunk):
            g = grid_t[i:i + chunk]
            inp = torch.cat([lat.expand(g.shape[0], -1), g], dim=1)
            vals.append(net(inp).squeeze(-1).cpu().numpy())
    vol = np.concatenate(vals).reshape(resolution, resolution, resolution)

    if vol.min() > 0 or vol.max() < 0:
        return None

    spacing = (2.0 * grid_lim) / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(
        vol.astype(np.float64), level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts + np.array([-grid_lim, -grid_lim, -grid_lim], dtype=np.float64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    comps = mesh.split(only_watertight=False)
    if len(comps) > 0:
        mesh = max(comps, key=lambda c: c.area)
    return mesh


def complete(net, partial, contacts, device, seed=0, iterations=800, lr=5e-3,
             resolution=128, n_partial_per_iter=1000, n_surface_pts=8000,
             denoise_partial=True, expand_contact_patches=True):
    """One IGR reconstruction. partial (N,3), contacts (M,3) world meters -> completed
    cloud (n_surface_pts,3) world meters, or None if no surface was extracted.
    denoise_partial=True applies DBSCAN outlier removal (pass a pre-cleaned cloud + False
    to skip it). expand_contact_patches=True synthesizes a small tangent disk around each
    contact (for sparse point contacts); pass False when `contacts` is ALREADY a dense
    surface cloud (e.g. the PaXini fingerprint pad surfaces) so it's used verbatim."""
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    partial = np.asarray(partial, np.float32).reshape(-1, 3)
    contacts = np.asarray(contacts, np.float32).reshape(-1, 3) if contacts is not None \
        else np.zeros((0, 3), np.float32)

    if denoise_partial:
        partial = denoise(partial)

    merged = np.vstack([partial, contacts]) if len(contacts) else partial
    center = merged.mean(axis=0).astype(np.float32)
    merged_n = estimate_normals(merged, center)
    nrm_partial = merged_n[:len(partial)]
    if len(contacts):
        nrm_contact = merged_n[len(partial):]
        if expand_contact_patches:
            contacts_e, nrm_contact_e = expand_contacts(contacts, nrm_contact)
        else:                                          # dense cloud already -> use verbatim
            contacts_e = contacts.astype(np.float32)
            nrm_contact_e = nrm_contact.astype(np.float32)
    else:
        contacts_e = np.zeros((0, 3), np.float32)
        nrm_contact_e = np.zeros((0, 3), np.float32)

    _, center_nrm, scale = normalize(merged)
    tp = torch.from_numpy((partial - center_nrm) * scale).to(device)
    tc = torch.from_numpy((contacts_e - center_nrm) * scale).to(device)
    tnp = torch.from_numpy(nrm_partial).to(device)
    tnc = torch.from_numpy(nrm_contact_e).to(device)

    latent = optimize_latent(net, tp, tnp, tc, tnc, device,
                             iterations, lr, n_partial_per_iter)
    mesh = marching_cubes(net, latent, device, resolution)
    if mesh is None or mesh.vertices.shape[0] == 0:
        return None
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_surface_pts, seed=seed)
    except TypeError:
        pts, _ = trimesh.sample.sample_surface(mesh, n_surface_pts)
    world = (np.asarray(pts, np.float64) / scale) + center_nrm
    return world.astype(np.float32)
