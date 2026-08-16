#!/usr/bin/env python3
"""Paxini 촉각 3D 렌더러 — kist-vtdp-wrapper/tools/viz_demo.py 의 Scene3D 를 실시간용으로 이식.

원본은 HDF5 → MP4 오프라인 렌더러였다. 측정해 보니 병목은 Open3D 가 아니라
matplotlib 전체 재그리기(31~51ms)였고, Open3D 렌더 자체는 5.6ms(179fps)로 실시간에 충분하다.
그래서 Scene3D 는 거의 그대로 가져오고, 합성만 matplotlib → cv2 로 바꿨다.

  손가락 4개의 지문 CAD 를 2x2 로 배치하고, 각 127 탁셀을 힘 크기로 색칠한다.
  가장 센 탁셀 2개에는 힘 방향 화살표를 그린다.

입력: (4, 127, 3) float — 손가락 x 탁셀 x (x,y,z). x,y=전단, z=압력. 단위 0.1N.
출력: (H, W, 3) uint8 RGB 이미지.

⚠️ 주의 (원본 문서에서 확인된 사실):
  - paxini 의 4개 블록이 어느 손가락인지는 **미확인**이다 (UART 가 스트림 순서로만 준다).
    그래서 엄지/검지 같은 라벨을 붙이지 않고 part0~3 으로만 표시한다.
  - 06_hand_j_kin 의 축 순서는 (Fz, Tx, Ty) 다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
STL = ASSETS / "fingertip-PX6AX-GEN3-DP-M2826-Omega.stl"
TAXEL_CSV = ASSETS / "taxel_m2826_127.csv"

N_PART, N_TAXEL = 4, 127

# viz_demo.py 의 색/상수를 그대로 유지 (같은 그림처럼 보이게)
CMAP_COLORS = ["#12111c", "#3b1f5e", "#8e2f7a", "#d95f5f", "#f4a24c", "#ffe9a8"]
BG, FG, GRID = "#0e1116", "#e8eaf0", "#2a2f3a"
PART_COLORS = ["#5ee0c8", "#f4a24c", "#c86bd8", "#7aa8ff"]   # part0~3
KIN_TRIPLE = ("Fz", "Tx", "Ty")        # 06_hand_j_kin 축 순서 (원본 문서 확인)
# kin 쪽 손가락 대응은 확인됨. (촉각 4블록의 손가락 대응은 미확인이라 part0~3 으로만 쓴다)
KIN_FINGER_LABEL = ("T", "I", "M", "R")
KIN_FZ_SCALE = 1000.0                  # 0번 채널에만 곱해 스케일 맞춤

# 오프라인 원본은 에피소드 전체의 99.9 백분위로 vmax 를 잡았다(스트리밍 불가).
# 실측 분포(레몬 Demo_14): p99=0.49, p99.9=1.33, max=2.82 → 고정 1.3 이 무난.
VMAX_DEFAULT = 1.3
ARROW_MIN_FRAC = 0.05                  # |F| < 0.05*vmax 면 화살표 생략
TOP_K = 2                              # 파트당 화살표 개수


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def make_cmap():
    """matplotlib 없이 쓰도록 LUT(256,3) 로 미리 굽는다."""
    stops = np.array([_hex_rgb(c) for c in CMAP_COLORS])       # (6,3)
    x = np.linspace(0, 1, len(stops))
    xs = np.linspace(0, 1, 256)
    return np.stack([np.interp(xs, x, stops[:, c]) for c in range(3)], axis=1)


CMAP_LUT = make_cmap()


def load_taxel_xyz() -> np.ndarray:
    """127 탁셀의 XYZ[mm]. STL 과 같은 좌표계라 별도 정합이 필요 없다."""
    rows = []
    with open(TAXEL_CSV) as f:
        next(f)                                                 # 헤더
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append([float(v) for v in ln.split(",")[:3]])
    pts = np.asarray(rows, dtype=np.float64)
    assert pts.shape == (N_TAXEL, 3), f"탁셀 좌표 shape {pts.shape} != (127,3)"
    return pts


def _rot_from_z(v: np.ndarray) -> np.ndarray:
    """+z 를 단위벡터 v 로 보내는 회전행렬 (Rodrigues). viz_demo.py 원본 그대로."""
    z = np.array([0.0, 0.0, 1.0])
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(z, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(np.arccos(c)) * K + (1 - c) * (K @ K)


class Scene3D:
    """지문 CAD 4개 + 탁셀 점군 + 힘 화살표를 오프스크린 렌더 (Open3D).

    생성에 ~0.9초(STL 로드) 걸리므로 시작 시 1회만. render() 는 ~5.6ms.
    """

    def __init__(self, width=1020, height=760, top_k=TOP_K):
        import open3d as o3d                                    # 무거워서 지연 임포트
        self.o3d = o3d
        self.top_k = top_k
        self.w, self.h = width, height

        base = o3d.io.read_triangle_mesh(str(STL))
        if not base.has_triangle_normals():
            base.compute_vertex_normals()
        taxel = load_taxel_xyz()

        bb = base.get_axis_aligned_bounding_box()
        span = bb.get_extent()
        # 2x2 배치 — 원본과 동일한 간격 계수
        self.offsets = [np.array([(p % 2 - 0.5) * span[0] * 1.22,
                                  (0.5 - p // 2) * span[1] * 1.12, 0.0])
                        for p in range(N_PART)]
        self.pts = [taxel + self.offsets[p] for p in range(N_PART)]

        self.rend = o3d.visualization.rendering.OffscreenRenderer(width, height)
        sc = self.rend.scene
        sc.set_background([0.055, 0.067, 0.086, 1.0])

        m_mesh = o3d.visualization.rendering.MaterialRecord()
        m_mesh.shader = "defaultLit"
        m_mesh.base_color = (0.20, 0.22, 0.28, 1.0)
        m_mesh.base_roughness = 0.85
        self.m_pts = o3d.visualization.rendering.MaterialRecord()
        self.m_pts.shader = "defaultUnlit"
        self.m_pts.point_size = 11.0
        self.m_arw = o3d.visualization.rendering.MaterialRecord()
        self.m_arw.shader = "defaultLit"
        self.m_arw.base_color = (0.93, 0.91, 0.66, 1.0)

        for p in range(N_PART):
            m = o3d.geometry.TriangleMesh(base)
            m.translate(self.offsets[p])
            sc.add_geometry(f"mesh{p}", m, m_mesh)

        allp = np.concatenate(self.pts, axis=0)
        center = allp.mean(axis=0)
        ext = allp.max(axis=0) - allp.min(axis=0)
        dist = float(max(ext[0], ext[1])) * 1.12 / (2 * np.tan(np.radians(38) / 2))
        eye = center + np.array([0.0, 0.0, dist])
        self.rend.setup_camera(38.0, center, eye, np.array([0.0, 1.0, 0.0]))
        fwd = (center - eye) / np.linalg.norm(center - eye)
        sc.scene.set_sun_light(list(-fwd), [1.0, 1.0, 1.0], 78000)

    def render(self, frame: np.ndarray, vmax: float = VMAX_DEFAULT) -> np.ndarray:
        """frame: (4,127,3) 힘벡터 → RGB uint8 (h,w,3)."""
        o3d = self.o3d
        vec = np.asarray(frame, dtype=np.float64).reshape(N_PART, N_TAXEL, 3)
        scal = np.linalg.norm(vec, axis=-1)                     # (4,127) 크기
        vmax = max(float(vmax), 1e-6)

        for p in range(N_PART):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pts[p])
            idx = np.clip(scal[p] / vmax, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(CMAP_LUT[(idx * 255).astype(int)])
            self.rend.scene.remove_geometry(f"pts{p}")
            self.rend.scene.add_geometry(f"pts{p}", pcd, self.m_pts)

            self.rend.scene.remove_geometry(f"arw{p}")
            acc = o3d.geometry.TriangleMesh()
            for i in np.argsort(scal[p])[::-1][:self.top_k]:
                f = vec[p, i]
                n = float(np.linalg.norm(f))
                if n < ARROW_MIN_FRAC * vmax:
                    continue
                length = 4.0 + 14.0 * min(n / vmax, 1.0)        # mm
                a = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.45, cone_radius=1.0,
                    cylinder_height=length * 0.72, cone_height=length * 0.28)
                a.rotate(_rot_from_z(f / n), center=(0, 0, 0))
                a.translate(self.pts[p][i])
                acc += a
            if len(acc.vertices):
                acc.compute_vertex_normals()
                self.rend.scene.add_geometry(f"arw{p}", acc, self.m_arw)

        return np.asarray(self.rend.render_to_image())
