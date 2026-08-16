#!/usr/bin/env python3
"""스캔한 메시를 FoundationPose 입력으로 다듬는다 (진단 + 단위보정 + 변환).

아이폰 스캐너 앱 결과물은 그대로 쓰면 대개 세 군데서 걸린다:

  1) 단위     — ARKit 은 m 지만 앱에 따라 cm/mm 로 뽑는다. FoundationPose 는 **m** 다.
                7cm 오렌지를 mm 로 내보내면 extents=[70,70,70] → 70미터짜리 물체로 인식.
  2) 텍스처   — STL 은 형상만 있다. 텍스처가 없으면 FoundationPose 가 메시를
                균일 회색([128,128,128], Utils.py:124)으로 렌더해서 RGB 정합이
                아무 정보도 못 준다. 둥근 과일은 이때 회전이 전혀 안 잡힌다.
  3) 폴리곤 수 — 스캔은 수십만 face 가 흔하다. 렌더가 느려진다.

사용:
    # 진단만
    python3 prepare_mesh.py scan.glb

    # 실측 지름 72mm 에 맞춰 변환
    python3 prepare_mesh.py scan.glb -o assets/orange.obj --target-diameter 0.072
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

# trimesh 가 읽는 포맷 (usdz 는 없음 → GLB/OBJ 로 내보내야 한다)
GOOD = {".glb", ".gltf", ".obj", ".ply", ".dae", ".off", ".3mf"}
GEOM_ONLY = {".stl"}


def texture_image(m: trimesh.Trimesh):
    """메시에서 텍스처 이미지를 꺼낸다. GLB 는 PBRMaterial.baseColorTexture 에 들어간다.

    ★ FoundationPose 는 `mesh.visual.material.image` 만 본다(Utils.py:109). GLB 를
    그대로 넘기면 PBRMaterial 이라 .image 가 None 이고 거기서 AttributeError 로 죽는다.
    그래서 이 스크립트가 OBJ(SimpleMaterial) 로 바꿔주는 것이다.
    """
    mat = getattr(getattr(m, "visual", None), "material", None)
    if mat is None:
        return None
    return getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)


def usable_texture(img) -> tuple[bool, str]:
    """텍스처가 회전 단서로 쓸 만한지 판정.

    ★ .mtl 이나 이미지 파일이 없으면 trimesh 는 조용히 **2x2 단색 placeholder** 를
    끼워 넣는다. 그러면 '텍스처 있음' 으로 보이지만 실제로는 균일색이라 회색 렌더와
    다를 게 없다. 크기와 색 다양성을 같이 봐야 이 함정을 잡는다.
    """
    if img is None:
        return False, "없음"
    try:
        a = np.array(img.convert("RGB"))
    except Exception:                                            # noqa: BLE001
        return False, "읽기 실패"
    if min(a.shape[:2]) < 8:
        return False, f"placeholder ({a.shape[1]}x{a.shape[0]}) — .mtl/이미지 누락"
    if len(np.unique(a.reshape(-1, 3), axis=0)) < 8:
        return False, f"단색 ({a.shape[1]}x{a.shape[0]}) — 무늬 없음"
    return True, f"{a.shape[1]}x{a.shape[0]}"


def describe(m: trimesh.Trimesh) -> dict:
    """텍스처/색 유무와 크기를 판정."""
    has_tex = isinstance(getattr(m, "visual", None), trimesh.visual.texture.TextureVisuals)
    tex_img = texture_image(m) if has_tex else None
    tex_ok, tex_note = usable_texture(tex_img)
    has_vcol = False
    if not has_tex:
        try:
            vc = m.visual.vertex_colors
            # trimesh 는 색이 없어도 기본 회색을 채워 넣는다 → 실제로 변하는지 본다
            has_vcol = vc is not None and len(np.unique(vc[:, :3], axis=0)) > 1
        except Exception:                                        # noqa: BLE001
            has_vcol = False
    return {"has_tex": tex_ok, "tex_note": tex_note,
            "has_vcol": has_vcol, "extents": m.extents,
            "nv": len(m.vertices), "nf": len(m.faces)}


def guess_unit(extents: np.ndarray) -> tuple[str, float]:
    """가장 긴 축 길이로 단위를 추정해 (이름, m 로 가는 배율) 반환.

    과일(대략 2~30cm)을 가정한 임계값이다. m/cm/mm 로 각각 0.02~0.3 / 2~30 / 20~300
    이 나오므로 아래처럼 갈라도 겹치지 않는다. 어디까지나 추정이니 확실히 하려면
    --target-diameter 로 실측을 주는 게 맞다.
    """
    d = float(np.max(extents))
    if d < 0.5:
        return "m", 1.0
    if d < 50:
        return "cm", 1e-2
    return "mm", 1e-3


def check_obj_sidecars(path: str) -> None:
    """OBJ 는 3개가 한 벌이다: .obj → (mtllib) → .mtl → (map_Kd) → 이미지.

    어느 하나라도 빠지면 텍스처 없이 로드되므로, 뭐가 없는지 짚어준다.
    스캐너 앱에서 .obj 만 복사해 오는 실수가 흔하다.
    """
    d = os.path.dirname(os.path.abspath(path))
    mtl_names = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("mtllib"):
                mtl_names += line.split()[1:]
            elif line.startswith(("v ", "f ")) and mtl_names:
                break                      # 헤더만 보면 충분
    if not mtl_names:
        print("  ⚠ .obj 안에 mtllib 줄이 없습니다 → 재질 파일이 아예 연결 안 됨(텍스처 없음)")
        return
    for name in mtl_names:
        mtl = os.path.join(d, name)
        if not os.path.isfile(mtl):
            print(f"  ⚠ 재질 파일이 없습니다: {name}  (.obj 옆에 같이 두세요)")
            continue
        maps = []
        with open(mtl, "r", errors="ignore") as f:
            for line in f:
                if line.lower().startswith(("map_kd", "map_ka")):
                    maps += line.split()[1:]
        if not maps:
            print(f"  ⚠ {name} 에 map_Kd(텍스처 이미지) 줄이 없습니다 → 색 없음")
        for img in maps:
            p = os.path.join(d, img)
            print(f"  {'✓' if os.path.isfile(p) else '⚠ 없음:'} 텍스처 이미지 {img}")


def main():
    ap = argparse.ArgumentParser(description="스캔 메시 → FoundationPose 입력 변환")
    ap.add_argument("src", help="스캔 파일 (.glb/.obj/.ply/.stl …)")
    ap.add_argument("-o", "--out", default=None, help="출력 .obj (없으면 진단만)")
    ap.add_argument("--target-diameter", type=float, default=None,
                    help="실측 지름 [m] 으로 강제 스케일 (예: 0.072). 가장 확실하다")
    ap.add_argument("--scale", type=float, default=None,
                    help="배율 직접 지정 (mm→m 이면 0.001). --target-diameter 와 배타")
    ap.add_argument("--target-extents", default=None,
                    help="축별 실측 크기 'a,b,c' [m] (긴축부터). 스캔이 실제 비율을 "
                         "못 잡았을 때 축마다 다른 배율로 맞춘다 (예: 0.070,0.055,0.055)")
    ap.add_argument("--max-faces", type=int, default=50000,
                    help="이보다 많으면 단순화 (0=끔)")
    a = ap.parse_args()

    ext = os.path.splitext(a.src)[1].lower()
    if ext not in GOOD | GEOM_ONLY:
        print(f"✗ trimesh 가 못 읽는 포맷: {ext}")
        print(f"  읽을 수 있는 것: {', '.join(sorted(GOOD | GEOM_ONLY))}")
        if ext == ".usdz":
            print("  → 아이폰 USDZ 는 지원 안 됨. 앱에서 GLB 나 OBJ 로 내보내세요.")
        sys.exit(1)

    if ext == ".obj":
        check_obj_sidecars(a.src)

    m = trimesh.load(a.src, force="mesh")
    info = describe(m)
    unit, to_m = guess_unit(info["extents"])

    print(f"입력      : {a.src}")
    print(f"  정점/면 : {info['nv']:,} / {info['nf']:,}")
    print(f"  크기    : {np.round(info['extents'], 4).tolist()}  → 단위 추정 {unit}")
    print(f"  텍스처  : {('있음 ' + info['tex_note']) if info['has_tex'] else ('✗ ' + info['tex_note'])}")
    print(f"  정점색  : {'있음' if info['has_vcol'] else '없음'}")

    if not info["has_tex"] and not info["has_vcol"]:
        print("")
        print("  ⚠ 색 정보가 전혀 없습니다. FoundationPose 는 이런 메시를 균일 회색으로")
        print("    렌더하므로 RGB 정합이 회전을 전혀 못 잡습니다. 둥근 과일이면 사실상")
        print("    위치만 쓰게 됩니다. 스캐너 앱에서 **텍스처 포함 GLB/OBJ** 로 다시")
        print("    내보내세요 (STL 은 형상 전용이라 안 됩니다).")

    if a.out is None:
        print("\n(-o 를 주면 변환해서 저장합니다)")
        return

    # ── 스케일 ────────────────────────────────────────────────────────────
    if sum(x is not None for x in (a.target_diameter, a.scale, a.target_extents)) > 1:
        print("✗ --target-diameter / --scale / --target-extents 중 하나만 쓰세요")
        sys.exit(1)
    if a.target_extents:
        # 축마다 다른 배율. 메시의 주축(PCA)이 아니라 bounding box 축을 긴 순으로
        # 정렬해 대응시킨다 — 스캔이 실제 비율을 못 잡았을 때 쓴다.
        want = np.array(sorted((float(x) for x in a.target_extents.split(",")),
                               reverse=True))
        cur = np.asarray(m.extents, dtype=float)
        order = np.argsort(cur)[::-1]              # 긴 축부터
        s_vec = np.ones(3)
        s_vec[order] = want / cur[order]
        print(f"\n스케일    : 축별 {np.round(cur,4).tolist()} → "
              f"{np.round(cur*s_vec,4).tolist()} m  (배율 {np.round(s_vec,4).tolist()})")
        if s_vec.max() / s_vec.min() > 1.15:
            print("  ⚠ 축별 배율 차가 큽니다 = 스캔 형상이 실물과 다릅니다.")
            print("    깊이 정합에는 실측을 맞추는 쪽이 유리하지만, 텍스처가 늘어납니다.")
        m.apply_transform(np.diag([*s_vec, 1.0]))
        s = 1.0
    elif a.target_diameter:
        cur = float(np.max(m.extents))
        s = a.target_diameter / cur
        print(f"\n스케일    : 최대축 {cur:.4f} → {a.target_diameter:.4f} m  (×{s:.6g})")
    elif a.scale:
        s = a.scale
        print(f"\n스케일    : ×{s:.6g} (직접 지정)")
    else:
        s = to_m
        print(f"\n스케일    : 단위 추정 {unit} → m  (×{s:.6g})")
        print("  ※ 추정입니다. 실측 지름을 아신다면 --target-diameter 를 쓰세요.")
    if s != 1.0:
        m.apply_scale(s)

    # ── 중심 정렬 (FoundationPose 도 내부에서 하지만 파일도 맞춰둔다) ──────
    m.apply_translation(-m.bounding_box.centroid)

    # ── 단순화 ────────────────────────────────────────────────────────────
    if a.max_faces and len(m.faces) > a.max_faces:
        before = len(m.faces)
        try:
            m = m.simplify_quadric_decimation(a.max_faces)
            print(f"단순화    : {before:,} → {len(m.faces):,} face")
        except Exception as e:                                   # noqa: BLE001
            print(f"단순화 실패(무시): {e}")

    # ── PBR(GLB) → SimpleMaterial 로 강등 ────────────────────────────────
    # FoundationPose 가 material.image 만 읽으므로 여기서 맞춰준다.
    img = texture_image(m)
    if img is not None and getattr(m.visual.material, "image", None) is None:
        m.visual.material = trimesh.visual.material.SimpleMaterial(image=img)
        print("재질      : PBRMaterial → SimpleMaterial (FoundationPose 가 .image 만 읽음)")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    m.export(out)          # .obj 로 내보내면 텍스처는 .mtl + 이미지로 같이 나온다
    final = describe(m)
    print(f"\n저장      : {out}")
    print(f"  크기    : {np.round(final['extents'], 4).tolist()} m")
    print(f"  면      : {final['nf']:,}")
    print(f"  텍스처  : {'유지됨' if final['has_tex'] else ('정점색' if final['has_vcol'] else '없음')}")
    print(f"\n실행:  bash foundation_pose/run_foundation_pose.sh")
    print(f"또는 돌아가는 중이면:")
    print(f"  ros2 topic pub --once /fruit/reset std_msgs/String '{{data: \"{out}\"}}'")


if __name__ == "__main__":
    main()
