# fr3_kistar_left meshes

Meshes copied verbatim from
`/home/cy/isaac_ws/dex-urdf/robots/assembly/fr3_kistar_left/meshes/`
(dex-urdf project).

Layout:
- `visual/` — 8 DAE files (FR3 link0..7 visual meshes)
- `collision/` — 8 STL files (FR3 link0..7 collision meshes)
- `kistar_hand/` — 17 STL files (left-hand bracket + base + fingers + pads + tips)
- `reference/fr3_kistar_left_simplify.urdf` — collision-simplified URDF kept for reference

License follows the upstream dex-urdf project. Do not edit these files in
place; if dex-urdf updates upstream, re-copy the meshes (and re-run the
transpiler at `.omc/scripts/transpile_dex_urdf_to_xacro.py`).
