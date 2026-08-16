"""URDF forward kinematics for the dual FR3 + KISTAR (v2 generated URDF).

Given the 7 right-arm joints (a franka_pose.yaml pose) + 16 right-hand joints (a
kistar_pose *.txt q_meas), returns the WORLD 4x4 pose of every right-side link — used
to (a) place the fingertip CONTACT points on the grasped object and (b) render the
kistar hand mesh at the grasp. world == base (identity static TF); base->right_fr3_link0
carries the 45deg arm offset (in the URDF).
"""
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (host or /work)
URDF = os.path.join(REPO, "dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/"
                          "urdf/generated/dual_fr3_kistar_v2.urdf")
MESH_PKG = os.path.join(REPO, "dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description")
HAND_JOINTS = [f"right_{f}_joint_{j}" for f in ("thumb", "index", "middle", "ring") for j in range(4)]
FINGERTIPS = ["right_thumb_3_tip", "right_index_3_tip", "right_middle_3_tip", "right_ring_3_tip"]


def _R(rpy):
    r, p, y = rpy
    cx, cy, cz = np.cos([r, p, y]); sx, sy, sz = np.sin([r, p, y])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(rpy, xyz):
    M = np.eye(4); M[:3, :3] = _R(rpy); M[:3, 3] = xyz; return M


def _axis_rot(axis, th):
    a = np.asarray(axis, float); a = a / max(np.linalg.norm(a), 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    M = np.eye(4); M[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K); return M


def parse_urdf(path=URDF):
    """-> (joints: child->(parent,rpy,xyz,axis,type,jname), visuals: link->(mesh_path,rpy,xyz,scale))."""
    s = open(path).read()
    joints = {}
    for m in re.finditer(r'<joint name="([^"]+)"\s+type="([^"]+)"[^>]*>(.*?)</joint>', s, re.S):
        name, typ, body = m.group(1), m.group(2), m.group(3)
        ch = re.search(r'<child link="([^"]+)"', body); pa = re.search(r'<parent link="([^"]+)"', body)
        if not (ch and pa):
            continue
        org = re.search(r'<origin([^/]*)/>', body); rpy = [0, 0, 0]; xyz = [0, 0, 0]
        if org:
            r = re.search(r'rpy="([^"]+)"', org.group(1)); x = re.search(r'xyz="([^"]+)"', org.group(1))
            if r: rpy = [float(v) for v in r.group(1).split()]
            if x: xyz = [float(v) for v in x.group(1).split()]
        ax = re.search(r'<axis xyz="([^"]+)"', body)
        axis = [float(v) for v in ax.group(1).split()] if ax else [0, 0, 1]
        joints[ch.group(1)] = (pa.group(1), rpy, xyz, axis, typ, name)

    visuals = {}
    for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', s, re.S):
        link, body = m.group(1), m.group(2)
        vis = re.search(r'<visual>(.*?)</visual>', body, re.S)
        if not vis:
            continue
        mesh = re.search(r'<mesh filename="([^"]+)"(?:\s+scale="([^"]+)")?', vis.group(1))
        if not mesh:
            continue
        org = re.search(r'<origin([^/]*)/>', vis.group(1)); rpy = [0, 0, 0]; xyz = [0, 0, 0]
        if org:
            r = re.search(r'rpy="([^"]+)"', org.group(1)); x = re.search(r'xyz="([^"]+)"', org.group(1))
            if r: rpy = [float(v) for v in r.group(1).split()]
            if x: xyz = [float(v) for v in x.group(1).split()]
        path_m = mesh.group(1).replace("package://franka_kistar_description", MESH_PKG)
        scale = [float(v) for v in mesh.group(2).split()] if mesh.group(2) else [1, 1, 1]
        visuals[link] = (path_m, rpy, xyz, scale)
    return joints, visuals


def fk(joint_values, path=URDF):
    """joint_values: {joint_name: angle rad}. -> {link: 4x4 world pose}."""
    joints, _ = parse_urdf(path)
    cache = {}

    def pose(link):
        if link in cache:
            return cache[link]
        if link not in joints:                       # root (base)
            cache[link] = np.eye(4); return cache[link]
        pa, rpy, xyz, axis, typ, jn = joints[link]
        T = _T(rpy, xyz)
        if typ in ("revolute", "continuous", "prismatic"):
            T = T @ _axis_rot(axis, float(joint_values.get(jn, 0.0)))
        cache[link] = pose(pa) @ T
        return cache[link]

    links = set(joints) | {v[0] for v in joints.values()}
    return {l: pose(l) for l in links}


def load_hand_q(txt):
    for line in open(txt):
        if line.startswith("q_meas:"):
            return [float(v) for v in line.split(":", 1)[1].split()]
    raise ValueError(f"no q_meas in {txt}")


def joint_values(arm_pose="child_pose", hand_txt=os.path.join(REPO, "kistar_pose/tmp_pose.txt")):
    import yaml
    arm = yaml.safe_load(open(os.path.join(REPO, "franka_pose.yaml")))["poses"][arm_pose]["joints"]
    q = load_hand_q(hand_txt)
    jv = dict(arm)
    jv.update({n: q[i] for i, n in enumerate(HAND_JOINTS)})
    return jv


# ---- PaXini fingertip (지두) fingerprint contact points ----------------------------
# The contact point is the PaXini M2826 fingertip tactile sensor's fingerprint centre
# (지두의 지문). The PaXini-equipped hand URDF mounts a `paxini_tip_visuals` mesh on each
# finger's distal phalanx `_3_link` via a fixed joint (zero offset) + a constant visual
# origin. The sensing PAD is the +x (palmar) face of that mesh; its centre is a FIXED
# point in the `_3_link` frame — so contacts come from FK ONLY (joint encoders), fully
# object-INDEPENDENT and pose-independent. The finger chains are identical between this
# PaXini URDF and the v2 FK URDF (verified: same joint origins/axes, only link names
# differ), so we place the fixed pad centre with the v2 FK `right_{finger}_3_link` pose.
# All 4 fingers, no filtering. (The inter-segment `pad_N` meshes are NOT the fingertip.)
FINGERS = ("thumb", "index", "middle", "ring")
PAXINI_URDF = os.path.join(REPO, "KISTAR_URDF/robots/hands/kistar_hand/kistar_hand_right_paxini.urdf")
_PAXINI_TIP_LINK = "right_hand_index_3_tip_link"     # any finger — same mesh file for all
_paxini_cache = None


def _paxini_pad(pax_urdf=PAXINI_URDF):
    """(pad_raw (3,), {finger: Vo 4x4}) — the PaXini sensing-pad centre in the sensor
    MESH frame (a fixed sensor point = centre of the +z contact face; verified: grasping
    fingers all touch at raw z≈z_max), plus each finger's tip visual origin (which places
    the mesh in `_3_link`; the thumb mounts the sensor differently from the other three)."""
    global _paxini_cache
    if _paxini_cache is not None:
        return _paxini_cache
    import trimesh
    _, visuals = parse_urdf(pax_urdf)
    path, _, _, scale = visuals[_PAXINI_TIP_LINK]
    if not os.path.isabs(path):                                      # URDF mesh path is relative
        path = os.path.join(os.path.dirname(pax_urdf), path)
    raw = np.asarray(trimesh.load(path, force="mesh", process=False).vertices, float) \
        * np.asarray(scale, float)
    face = raw[raw[:, 2] > raw[:, 2].max() - 0.003]                  # +z contact face
    pad_raw = face.mean(axis=0)
    Vo = {}                                                          # per-finger visual origin (rpy,xyz)
    for f in FINGERS:
        _, rpy, xyz, _ = visuals[f"right_hand_{f}_3_tip_link"]
        Vo[f] = _T(rpy, xyz)
    _paxini_cache = (pad_raw, Vo)
    return _paxini_cache


def fingertip_paxini_contacts(jv, path=URDF):
    """The 4 PaXini fingertip (지두) fingerprint-pad CENTRES in WORLD frame — pure FK,
    object-independent. jv: {joint_name: rad} (see joint_values). The fixed sensor pad
    centre is placed in each finger's `_3_link` frame via that finger's tip visual origin,
    then to world by the v2 FK `_3_link` pose. Returns (contacts (4,3), info)."""
    P = fk(jv, path)
    pad_raw, Vo = _paxini_pad()
    pts, info = [], []
    for f in FINGERS:
        link = f"right_{f}_3_link"
        if link not in P:
            continue
        p_link = Vo[f][:3, :3] @ pad_raw + Vo[f][:3, 3]              # pad centre in _3_link frame
        w = P[link][:3, :3] @ p_link + P[link][:3, 3]               # -> world
        pts.append(w)
        info.append((f, link, w))
    return np.array(pts, float), info


_paxini_face_cache = None


def _paxini_pad_face(per_finger=256, pax_urdf=PAXINI_URDF):
    """(face (K,3), {finger: Vo}) — the PaXini sensing-pad +z contact FACE vertices in the
    sensor MESH frame (the fingerprint SURFACE, i.e. the sensing pad only — NOT the whole tip
    mesh), subsampled to `per_finger` points, plus each finger's tip visual origin. Same face
    for all four fingers (one mesh file); only Vo/FK differ per finger."""
    global _paxini_face_cache
    if _paxini_face_cache is not None and _paxini_face_cache[0] == per_finger:
        return _paxini_face_cache[1], _paxini_face_cache[2]
    import trimesh
    _, visuals = parse_urdf(pax_urdf)
    path, _, _, scale = visuals[_PAXINI_TIP_LINK]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(pax_urdf), path)
    raw = np.asarray(trimesh.load(path, force="mesh", process=False).vertices, float) \
        * np.asarray(scale, float)
    face = raw[raw[:, 2] > raw[:, 2].max() - 0.003]                  # +z contact face (fingerprint)
    if per_finger and len(face) > per_finger:                       # uniform subsample
        face = face[np.random.default_rng(0).choice(len(face), per_finger, replace=False)]
    Vo = {}
    for f in FINGERS:
        _, rpy, xyz, _ = visuals[f"right_hand_{f}_3_tip_link"]
        Vo[f] = _T(rpy, xyz)
    _paxini_face_cache = (per_finger, face, Vo)
    return face, Vo


def fingertip_paxini_contact_cloud(jv, per_finger=256, path=URDF):
    """The 4 PaXini fingertip fingerprint PAD SURFACES as a WORLD point cloud — NOT 4 centres.
    The +z contact face of each finger's PaXini tip mesh (subsampled to `per_finger` pts) is
    posed by pure FK (object-independent). Returns (contacts (M,3), per_finger [(finger, cloud
    (Kf,3), centre (3,))]) — the per-finger split lets callers anchor a cloud to a known
    contact centre (e.g. a saved run's live-FK centres)."""
    P = fk(jv, path)
    face, Vo = _paxini_pad_face(per_finger)
    clouds, per = [], []
    for f in FINGERS:
        link = f"right_{f}_3_link"
        if link not in P:
            continue
        p_link = (Vo[f][:3, :3] @ face.T).T + Vo[f][:3, 3]          # face pts in _3_link frame
        w = (P[link][:3, :3] @ p_link.T).T + P[link][:3, 3]         # -> world
        clouds.append(w)
        per.append((f, w, w.mean(0)))
    contacts = np.concatenate(clouds, 0) if clouds else np.zeros((0, 3))
    return contacts.astype(float), per


def hand_point_cloud(jv, n=2048, path=URDF):
    """World-frame surface point cloud of the right KISTAR hand (w/ PaXini tips) at joint
    config jv — pure FK (step 3-14-A-1). SAMPLES THE MESH SURFACES (area-weighted, not just
    vertices) so the density is controllable: `n` points are distributed across the v2 hand-link
    visual meshes + the 4 PaXini fingertip meshes by surface area, then posed to world. This
    lets hand_pc be made as dense as child_pc_com (pass n=len(child_pc_com)). Needs trimesh."""
    import trimesh
    _, visuals = parse_urdf(path)
    P = fk(jv, path)
    meshes = []                                                      # (scaled local mesh, mesh->world 4x4)
    # right-hand v2 links (kistar hand base + finger segments + pads); exclude arm/left
    for link, (mpath, rpy, xyz, scale) in visuals.items():
        if not link.startswith("right_") or "fr3" in link or link not in P:
            continue
        try:
            m = trimesh.load(mpath, force="mesh", process=False)
        except Exception:                                            # noqa: BLE001
            continue
        m = m.copy()
        m.apply_scale(np.asarray(scale, float))
        meshes.append((m, P[link] @ _T(rpy, xyz)))                   # link pose @ visual origin
    # the 4 PaXini fingertip meshes (paxini URDF visual origin on _3_link)
    pax0 = trimesh.load(_paxini_mesh_path(), force="mesh", process=False)
    _, pvis = parse_urdf(PAXINI_URDF)
    for f in FINGERS:
        _, rpy, xyz, _ = pvis[f"right_hand_{f}_3_tip_link"]
        m = pax0.copy()
        m.apply_scale(0.001)
        meshes.append((m, P[f"right_{f}_3_link"] @ _T(rpy, xyz)))
    if not meshes:
        return np.zeros((0, 3), np.float32)
    areas = np.array([max(float(m.area), 1e-9) for m, _ in meshes])
    alloc = np.maximum(1, np.round(n * areas / areas.sum()).astype(int))  # points per mesh ~ area
    rng = np.random.default_rng(0)
    pieces = []
    for (m, Wo), k in zip(meshes, alloc):
        p, _ = trimesh.sample.sample_surface(m, int(k))              # k points on the surface
        pieces.append((Wo[:3, :3] @ np.asarray(p, float).T).T + Wo[:3, 3])  # -> world
    pts = np.vstack(pieces)
    if len(pts) > n:                                                 # trim to exactly n
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts.astype(np.float32)


def _paxini_mesh_path():
    _, pvis = parse_urdf(PAXINI_URDF)
    p = pvis[_PAXINI_TIP_LINK][0]
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(PAXINI_URDF), p)


if __name__ == "__main__":
    import numpy as np
    jv = joint_values()
    P = fk(jv)
    tips = np.array([P[f][:3, 3] for f in FINGERTIPS])
    print("fingertip world positions:\n", np.round(tips, 3))
    print("fingertip centroid:", np.round(tips.mean(0), 3).tolist())

    pts, info = fingertip_paxini_contacts(jv)
    print("\nPaXini fingertip (지두) fingerprint centres — object-independent (FK only):")
    for f, link, p in info:
        print(f"  {f:7s} {link:20s} centre={np.round(p, 4).tolist()}")
    print(f"{len(pts)} contact points")
