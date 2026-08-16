"""Per-run debug artifacts for the place pipeline (container-safe: numpy + PIL +
matplotlib(Agg) + pure-python PLY). Every test run dumps images, Molmo/SAM overlays,
point clouds (.ply), AnyPlace candidate placements, local_crop_size, and a raw
debug.npz so a failed Execute can be diagnosed offline.

Entry point: save_run(out_dir, R, log). R is the orchestrator's intermediates dict;
every save is guarded so a viz error never breaks the pipeline. .ply files open in
MeshLab / CloudCompare / open3d; colours: parent=gray, region=green, placed child=red.
"""
import os

import numpy as np

# MoveIt error codes (moveit_msgs/MoveItErrorCodes) for readable Execute diagnosis.
MOVEIT_ERR = {
    1: "SUCCESS", 99999: "FAILURE", -1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
    -4: "CONTROL_FAILED", -6: "TIMED_OUT", -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED", -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME", -21: "FRAME_TRANSFORM_FAILURE",
    -31: "NO_IK_SOLUTION",
}


def moveit_error_name(code):
    if code is True or code == 1:
        return "SUCCESS"
    try:
        return MOVEIT_ERR.get(int(code), f"code={code}")
    except (ValueError, TypeError):
        return f"code={code}"


def _sub(pc, n):
    pc = np.asarray(pc, float)
    if len(pc) <= n:
        return pc
    idx = np.linspace(0, len(pc) - 1, n).astype(int)
    return pc[idx]


def _apply(T, pc):
    pc = np.asarray(pc, float)
    return pc @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]


# ---- writers --------------------------------------------------------------------
def write_ply(path, *clouds):
    """clouds: (pc(N,3), rgb(3,) or (N,3)) tuples -> one coloured ascii .ply."""
    pts, cols = [], []
    for pc, rgb in clouds:
        pc = np.asarray(pc, float)
        if len(pc) == 0:
            continue
        rgb = np.asarray(rgb, np.uint8)
        rgb = np.tile(rgb, (len(pc), 1)) if rgb.ndim == 1 else rgb
        pts.append(pc)
        cols.append(rgb)
    if not pts:
        return
    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        lines = ("%.5f %.5f %.5f %d %d %d" % (p[0], p[1], p[2], c[0], c[1], c[2])
                 for p, c in zip(pts, cols))
        f.write("\n".join(lines) + "\n")


def _img(rgb):
    from PIL import Image
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def render_points(rgb, pts, color=(255, 40, 40), sel=None, sel_color=(60, 220, 255), r=5):
    """Return rgb (uint8 array) with points circled; point index `sel` in sel_color."""
    from PIL import ImageDraw
    im = _img(rgb)
    d = ImageDraw.Draw(im)
    for i, p in enumerate(pts):
        c = sel_color if (sel is not None and i == sel) else color
        x, y = float(p[0]), float(p[1])
        d.ellipse([x - r, y - r, x + r, y + r], outline=c, width=2)
        d.line([x - r - 3, y, x + r + 3, y], fill=c, width=1)
        d.line([x, y - r - 3, x, y + r + 3], fill=c, width=1)
        d.text((x + r + 2, y - r - 2), str(i), fill=c)
    return np.asarray(im)


def render_mask(rgb, mask, color=(40, 120, 255), alpha=0.5):
    im = np.asarray(_img(rgb)).astype(np.float32)
    m = np.asarray(mask, bool)
    im[m] = (1 - alpha) * im[m] + alpha * np.array(color, np.float32)
    return im.astype(np.uint8)


def render_depth(depth):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    d = np.asarray(depth, float)
    v = np.isfinite(d) & (d > 0)
    lo, hi = (float(d[v].min()), float(d[v].max())) if v.any() else (0.0, 1.0)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    n[~v] = 0.0
    return (cm.turbo(n)[..., :3] * 255).astype(np.uint8)


def save_overlay_points(path, rgb, pts, color=(255, 40, 40), r=5):
    _img(render_points(rgb, pts, color, r=r)).save(path)


def save_overlay_mask(path, rgb, mask, color=(40, 120, 255), alpha=0.5):
    _img(render_mask(rgb, mask, color, alpha)).save(path)


def save_depth(path, depth):
    _img(render_depth(depth)).save(path)


def save_scatter3d(path, clouds, title=""):
    """clouds: list of (pc, color01, label). Rendered 3D preview (Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    allp = []
    for pc, color, label in clouds:
        pc = _sub(pc, 3000)
        if len(pc) == 0:
            continue
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=[color], label=label, depthshade=False)
        allp.append(pc)
    if allp:
        P = np.concatenate(allp)
        c = P.mean(0)
        rng = float(np.abs(P - c).max()) + 1e-3
        for setlim, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(c[ci] - rng, c[ci] + rng)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (up)")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=22, azim=-70)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---- main -----------------------------------------------------------------------
def save_run(out_dir, R, log=print):
    """Dump every artifact we can from R. Never raises."""
    os.makedirs(out_dir, exist_ok=True)

    def step(name, fn):
        try:
            fn()
        except Exception as e:                                              # noqa: BLE001
            log(f"[debug] {name} FAILED ({type(e).__name__}: {e})")

    g = R.get
    # 1) raw RGB / depth
    if g("rgb_parent") is not None:
        step("parent_rgb", lambda: _img(g("rgb_parent")).save(f"{out_dir}/00_parent_rgb.png"))
        step("parent_depth", lambda: save_depth(f"{out_dir}/00_parent_depth.png", g("depth_parent")))
    if g("rgb_child") is not None:
        step("child_rgb", lambda: _img(g("rgb_child")).save(f"{out_dir}/01_child_rgb.png"))
        step("child_depth", lambda: save_depth(f"{out_dir}/01_child_depth.png", g("depth_child")))

    # 2) Molmo overlays (parent place, child grasp, local-place holes)
    if g("pt_parent") is not None:
        step("molmo_parent", lambda: save_overlay_points(
            f"{out_dir}/10_molmo_parent_place.png", g("rgb_parent"), [g("pt_parent")]))
    if g("pt_child") is not None:
        step("molmo_grasp", lambda: save_overlay_points(
            f"{out_dir}/11_molmo_child_grasp.png", g("rgb_child"), [g("pt_child")], color=(40, 220, 40)))
    if g("holes") is not None and len(g("holes")):
        step("molmo_holes", lambda: save_overlay_points(
            f"{out_dir}/12_molmo_localplace_holes.png", g("rgb_parent"), g("holes"), color=(255, 200, 0)))

    # 3) SAM masks
    if g("mask_parent") is not None:
        step("sam_parent", lambda: save_overlay_mask(
            f"{out_dir}/20_sam_parent.png", g("rgb_parent"), g("mask_parent")))
    if g("mask_child") is not None:
        step("sam_child", lambda: save_overlay_mask(
            f"{out_dir}/21_sam_child.png", g("rgb_child"), g("mask_child"), color=(255, 80, 200)))

    # 4) point clouds (.ply)
    GRAY, RED, GREEN, BLUE, ORANGE = ((180, 180, 180), (230, 40, 40),
                                      (40, 200, 40), (60, 120, 255), (255, 150, 0))
    SILVER = (180, 180, 195)
    for key, fn, col in [("parent_pc_full", "30_parent_pc_full", GRAY),
                         ("child_pc_i", "31_child_pc_i", ORANGE),
                         ("child_pc_refined", "31b_child_pc_refined", ORANGE),
                         ("child_pc_com", "32_child_pc_com", BLUE),
                         ("hand_pc", "32b_hand_pc", SILVER),
                         ("child_pc", "33_child_pc_fused", ORANGE),
                         ("child_pc_zalign", "34_child_pc_zalign", RED)]:
        if g(key) is not None:
            step(fn, lambda k=key, f=fn, c=col: write_ply(f"{out_dir}/{f}.ply", (g(k), c)))

    # 4b) each cropped parent region
    regions = g("regions") or []
    if regions:
        rd = f"{out_dir}/35_regions"
        os.makedirs(rd, exist_ok=True)
        for i, r in enumerate(regions):
            step(f"region_{i}", lambda i=i, r=r: write_ply(f"{rd}/region_{i:02d}.ply", (r, GREEN)))

    # 5) AnyPlace candidates: top-K by upright score, parent(gray)+region(green)+placed child(red)
    cand = g("candidates")
    child_z = g("child_pc_zalign")
    if cand is not None and child_z is not None and len(cand):
        cand = np.asarray(cand)
        scores = np.asarray(g("scores")) if g("scores") is not None else cand[:, 2, 2]
        creg = g("cand_region")
        order = np.argsort(-scores)
        k_top = int(min(len(order), max(len(regions), 1)))
        cd = f"{out_dir}/40_candidates"
        os.makedirs(cd, exist_ok=True)
        parent_sub = _sub(g("parent_pc_full"), 25000) if g("parent_pc_full") is not None else np.empty((0, 3))
        childz_sub = _sub(child_z, 5000)
        for rank, ci in enumerate(order[:k_top]):
            T = cand[ci]
            placed = _apply(T, childz_sub)
            reg_i = int(creg[ci]) if creg is not None else -1
            reg_pc = regions[reg_i] if 0 <= reg_i < len(regions) else np.empty((0, 3))
            base = f"{cd}/cand_rank{rank:02d}_score{scores[ci]:.3f}_reg{reg_i}"
            step(f"cand_ply_{rank}", lambda T=T, placed=placed, reg_pc=reg_pc, base=base: write_ply(
                base + ".ply", (parent_sub, GRAY), (_sub(reg_pc, 8000), GREEN), (placed, RED)))
            if rank < 6:  # a few rendered previews (cheaper than one-per-candidate)
                step(f"cand_png_{rank}", lambda placed=placed, base=base, reg_pc=reg_pc: save_scatter3d(
                    base + ".png",
                    [(parent_sub, (0.6, 0.6, 0.6), "parent"),
                     (_sub(reg_pc, 4000), (0.1, 0.7, 0.1), "region"),
                     (placed, (0.9, 0.1, 0.1), "placed child")],
                    title=os.path.basename(base)))
        # headline: full scene + top-1 placed object
        step("scene_top1", lambda: write_ply(
            f"{out_dir}/41_scene_plus_top1.ply",
            (_sub(g("parent_pc_full"), 60000) if g("parent_pc_full") is not None else np.empty((0, 3)), GRAY),
            (_apply(cand[order[0]], child_z), RED)))

    # 6) raw arrays for offline re-analysis
    step("npz", lambda: _save_npz(f"{out_dir}/debug.npz", R))
    # 7) human summary
    step("info", lambda: _save_info(f"{out_dir}/info.txt", R))
    log(f"[debug] artifacts saved -> {out_dir}")


def _save_npz(path, R):
    blob = {}
    for k, v in R.items():
        if v is None:
            continue
        if k == "regions":
            blob["regions_count"] = np.array(len(v))
            for i, r in enumerate(v):
                blob[f"region_{i:02d}"] = np.asarray(r, np.float32)
        elif isinstance(v, (list, tuple)):
            try:
                blob[k] = np.asarray(v)
            except Exception:                                              # noqa: BLE001
                pass
        elif isinstance(v, np.ndarray):
            blob[k] = v.astype(np.float32) if v.dtype == np.float64 else v
        elif isinstance(v, (int, float, bool, str)):
            blob[k] = np.array(v)
    np.savez(path, **blob)


def _save_info(path, R):
    g = R.get
    lines = ["=== place-pipeline run ==="]
    lines.append(f"scenario={g('scenario')}")
    if g("local_crop") is not None:
        lines.append(f"local_crop_size = {float(g('local_crop')):.4f} m")
    lines.append(f"regions (cropped parent_pc) = {len(g('regions') or [])}")
    if g("candidates") is not None:
        lines.append(f"AnyPlace candidates = {len(g('candidates'))}")
    if g("scores") is not None:
        sc = np.asarray(g("scores"))
        lines.append(f"upright scores: max={sc.max():.3f} min={sc.min():.3f} "
                     f"top1_idx={int(np.argmax(sc))}")
    if g("n_palm") is not None:
        lines.append(f"n_palm (world) = {np.round(g('n_palm'), 3).tolist()}")
    for key in ("T_zalign", "T_pred", "T_act", "ee_current", "ee_target"):
        if g(key) is not None:
            lines.append(f"\n{key} =\n{np.asarray(g(key)).round(4)}")
    if g("ee_target") is not None:
        lines.append(f"\nEE_target xyz = {np.asarray(g('ee_target'))[:3, 3].round(4).tolist()}")
    code = g("move_code")
    if code is not None:
        lines.append(f"\nExecute: executed={g('executed')}  MoveIt={moveit_error_name(code)} ({code})")
    with open(path, "w") as f:
        f.write("\n".join(str(x) for x in lines) + "\n")
