#!/usr/bin/env python3
"""과일 CAD 메시(OBJ) 생성 — FoundationPose 의 model-based 입력용.

FoundationPose 는 model-based 모드에서 텍스처 있는 3D 메시를 요구한다. 오렌지처럼
기성 CAD 가 없는 물체는 회전타원체로 근사한다. 실측 지름은 기존 파이프라인의
/fruit/size([a,b,c] m) 에서 읽어오면 된다.

    python3 make_fruit_mesh.py --diameter 0.07 -o orange.obj
    python3 make_fruit_mesh.py --abc 0.072,0.070,0.068 -o orange.obj   # 축별 지름

주의(중요): 구(球)에 가까운 물체는 **회전이 기하학적으로 관측 불가능**하다. 깊이만으로는
어떤 자세든 같은 점군이 나오므로, orientation 은 표면 텍스처(꼭지·반점·색 얼룩)로만
결정된다. 균일한 텍스처를 입히면 FoundationPose 도 회전을 못 잡는다 —
그래서 여기서는 기본으로 **비대칭 텍스처를 UV 로 구워 넣는다**(--texture none 으로 끌 수 있음).
실제 그 오렌지의 회전을 제대로 잡으려면 model-free(참조영상 16장) 쪽이 맞다.
"""
from __future__ import annotations

import argparse
import math
import os


def uv_sphere(a: float, b: float, c: float, n_lat: int, n_lon: int):
    """반지름 (a,b,c) 회전타원체의 UV 구면 메시. (verts, uvs, normals, faces) 반환.

    faces 는 1-based (v/vt/vn) 인덱스 삼각형 목록.
    """
    verts, uvs, norms = [], [], []
    for i in range(n_lat + 1):                     # 위도: 0=북극 .. n_lat=남극
        theta = math.pi * i / n_lat
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(n_lon + 1):                 # 경도: 이음매를 위해 +1 (UV seam)
            phi = 2.0 * math.pi * j / n_lon
            sp, cp = math.sin(phi), math.cos(phi)
            x, y, z = st * cp, st * sp, ct
            verts.append((a * x, b * y, c * z))
            # 타원체 법선 = (x/a^2, y/b^2, z/c^2) 정규화
            nx, ny, nz = x / (a * a), y / (b * b), z / (c * c)
            ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            norms.append((nx / ln, ny / ln, nz / ln))
            uvs.append((j / n_lon, 1.0 - i / n_lat))

    faces = []
    row = n_lon + 1
    for i in range(n_lat):
        for j in range(n_lon):
            v00 = i * row + j + 1                  # OBJ 는 1-based
            v01 = v00 + 1
            v10 = v00 + row
            v11 = v10 + 1
            if i != 0:                             # 북극은 삼각형 1개
                faces.append((v00, v10, v11))
            if i != n_lat - 1:                     # 남극도 마찬가지
                faces.append((v00, v11, v01))
    return verts, uvs, norms, faces


def write_texture(path: str, size: int, kind: str):
    """오렌지 껍질 느낌 + 비대칭 마커 텍스처를 PNG 로 굽는다.

    회전 관측을 위해 일부러 비대칭 패턴(꼭지 자국 + 반점)을 넣는다.
    균일한 주황 텍스처면 회전이 절대 결정되지 않는다.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    u = np.linspace(0, 1, size, dtype=np.float32)[None, :]
    v = np.linspace(0, 1, size, dtype=np.float32)[:, None]

    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 0.95
    base[..., 1] = 0.55
    base[..., 2] = 0.10

    # 껍질 오돌토돌 (고주파 노이즈)
    grain = rng.normal(0.0, 0.035, (size, size, 1)).astype(np.float32)
    base += grain

    if kind == "asym":
        # 꼭지(초록 원반) — 북극 근처 한 점. 이게 회전 기준이 된다.
        du = np.minimum(np.abs(u - 0.25), 1.0 - np.abs(u - 0.25))
        d = np.sqrt((du * 2.0) ** 2 + (v - 0.88) ** 2)
        stem = np.clip(1.0 - d / 0.10, 0.0, 1.0)[..., None]
        base = base * (1 - stem) + stem * np.array([0.25, 0.45, 0.15], dtype=np.float32)

        # 반점 몇 개 — 경도 방향 비대칭성 부여
        for cu, cv, r in [(0.62, 0.42, 0.055), (0.10, 0.55, 0.04), (0.80, 0.70, 0.03)]:
            du = np.minimum(np.abs(u - cu), 1.0 - np.abs(u - cu))
            d = np.sqrt((du * 2.0) ** 2 + (v - cv) ** 2)
            m = np.clip(1.0 - d / r, 0.0, 1.0)[..., None]
            base = base * (1 - m * 0.8) + m * 0.8 * np.array([0.72, 0.36, 0.06],
                                                             dtype=np.float32)

    img = np.clip(base * 255.0, 0, 255).astype("uint8")
    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser(description="과일 회전타원체 CAD 메시(OBJ) 생성")
    ap.add_argument("-o", "--out", default="orange.obj")
    ap.add_argument("--diameter", type=float, default=0.070,
                    help="구 지름 [m] (기본 0.070 = 오렌지)")
    ap.add_argument("--abc", default=None,
                    help="축별 지름 'a,b,c' [m] — 주면 --diameter 무시")
    ap.add_argument("--lat", type=int, default=64)
    ap.add_argument("--lon", type=int, default=128)
    ap.add_argument("--texture", choices=["asym", "plain", "none"], default="asym",
                    help="asym=비대칭 패턴(회전 관측 가능), plain=균일, none=텍스처 없음")
    ap.add_argument("--tex-size", type=int, default=1024)
    a = ap.parse_args()

    if a.abc:
        da, db, dc = (float(x) for x in a.abc.split(","))
    else:
        da = db = dc = a.diameter
    ra, rb, rc = da / 2, db / 2, dc / 2

    verts, uvs, norms, faces = uv_sphere(ra, rb, rc, a.lat, a.lon)

    out = os.path.abspath(a.out)
    stem = os.path.splitext(out)[0]
    base = os.path.basename(stem)
    tex_path = f"{stem}.png"
    mtl_path = f"{stem}.mtl"

    if a.texture != "none":
        write_texture(tex_path, a.tex_size, a.texture)
        with open(mtl_path, "w") as f:
            f.write(f"newmtl {base}\n")
            f.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.050 0.050 0.050\n")
            f.write("Ns 20.0\nd 1.0\nillum 2\n")
            f.write(f"map_Kd {os.path.basename(tex_path)}\n")

    with open(out, "w") as f:
        f.write(f"# 과일 근사 메시 — 지름 {da:.3f} x {db:.3f} x {dc:.3f} m\n")
        if a.texture != "none":
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for uu, vv in uvs:
            f.write(f"vt {uu:.6f} {vv:.6f}\n")
        for nx, ny, nz in norms:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        if a.texture != "none":
            f.write(f"usemtl {base}\n")
        for v0, v1, v2 in faces:
            f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v2}/{v2}/{v2}\n")

    print(f"메시  : {out}  (v={len(verts)}, f={len(faces)})")
    if a.texture != "none":
        print(f"재질  : {mtl_path}")
        print(f"텍스처: {tex_path}  ({a.texture})")
    if a.texture == "plain":
        print("⚠ 균일 텍스처 = 회전 관측 불가. 자세 추정용이면 --texture asym 을 쓰세요.")


if __name__ == "__main__":
    main()
