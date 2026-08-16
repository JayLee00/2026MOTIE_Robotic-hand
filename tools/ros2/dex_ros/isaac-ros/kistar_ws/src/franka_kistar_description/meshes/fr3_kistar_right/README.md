# fr3_kistar_right meshes

Meshes copied verbatim from
`/home/cy/isaac_ws/dex-urdf/robots/assembly/fr3_kistar_right/meshes/`
(dex-urdf project).

Layout:
- `visual/` — 8 DAE files (FR3 link0..7 visual meshes)
- `collision/` — 8 STL files (FR3 link0..7 collision meshes)
- `kistar_hand/` — 17 STL files (right-hand bracket + base + fingers + pads + tips)
- `reference/fr3_franka_right_simplify.urdf` — collision-simplified URDF kept for reference (note: filename is `fr3_franka_*`, not `fr3_kistar_*`, in upstream)

License follows the upstream dex-urdf project. Do not edit these files in
place; if dex-urdf updates upstream, re-copy the meshes (and re-run the
transpiler at `.omc/scripts/transpile_dex_urdf_to_xacro.py`).
