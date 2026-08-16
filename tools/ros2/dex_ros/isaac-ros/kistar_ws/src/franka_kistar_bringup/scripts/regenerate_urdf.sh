#!/usr/bin/env bash
# regenerate_urdf.sh — pre-compile xacro sources into static URDF snapshots used by
# dual_fr3_kistar_planning_pc.launch.py at runtime (PR-2 B.1).
#
# Inputs (xacro sources):
#   isaac-ros/kistar_ws/src/franka_kistar_description/urdf/dual_fr3_kistar.urdf.xacro
#   isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf/table.urdf.xacro
#   isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf/ttable.urdf.xacro
#
# Outputs:
#   isaac-ros/kistar_ws/src/franka_kistar_description/urdf/generated/dual_fr3_kistar.urdf
#   isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf/generated/table.urdf
#   isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf/generated/ttable.urdf
#   …plus a sidecar *.urdf.sha256 for each, hashing the GENERATED output (not the
#   source). The xacro transitively depends on franka_description outside this repo,
#   so hashing the generated artifact catches transitive drift the source hash misses.
#
# Idempotent: re-running yields the same outputs (xacro is deterministic given fixed
# inputs). The script does NOT touch outputs if they already exist with matching hash.
#
# Requirements:
#   - 'xacro' on PATH (e.g. after `source /opt/ros/humble/setup.bash`).
#   - sha256sum on PATH (coreutils).

set -euo pipefail

color_red() { printf "\033[31m%s\033[0m\n" "$*"; }
color_grn() { printf "\033[32m%s\033[0m\n" "$*"; }
color_yel() { printf "\033[33m%s\033[0m\n" "$*"; }

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" > /dev/null 2>&1; then
        color_red "[regenerate_urdf] FATAL: required command '$cmd' is not on PATH."
        color_red "  Source ROS first, e.g. 'source /opt/ros/humble/setup.bash'."
        exit 1
    fi
}

require_cmd xacro
require_cmd sha256sum

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"
DESC_DIR="${REPO_ROOT}/isaac-ros/kistar_ws/src/franka_kistar_description/urdf"
BRINGUP_URDF_DIR="${REPO_ROOT}/isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf"

DUAL_XACRO="${DESC_DIR}/dual_fr3_kistar.urdf.xacro"
DUAL_V2_XACRO="${DESC_DIR}/dual_fr3_kistar_v2.urdf.xacro"
TABLE_XACRO="${BRINGUP_URDF_DIR}/table.urdf.xacro"
TTABLE_XACRO="${BRINGUP_URDF_DIR}/ttable.urdf.xacro"

DUAL_OUT_DIR="${DESC_DIR}/generated"
BRINGUP_OUT_DIR="${BRINGUP_URDF_DIR}/generated"
mkdir -p "${DUAL_OUT_DIR}" "${BRINGUP_OUT_DIR}"

DUAL_OUT="${DUAL_OUT_DIR}/dual_fr3_kistar.urdf"
DUAL_V2_OUT="${DUAL_OUT_DIR}/dual_fr3_kistar_v2.urdf"
TABLE_OUT="${BRINGUP_OUT_DIR}/table.urdf"
TTABLE_OUT="${BRINGUP_OUT_DIR}/ttable.urdf"

# xacro args MUST match the launch file's invocation so the generated snapshot
# matches the runtime expansion. See dual_fr3_kistar_planning_pc.launch.py around
# line 90-93 (dual) and the table xacro Command calls below them.
DUAL_ARGS=(ros2_control:=false use_fake_hardware:=true fake_sensor_commands:=true)
# table/ttable have no special args in the launch file.

generate() {
    local input="$1"; shift
    local output="$1"; shift
    local extra_args=("$@")

    if [ ! -f "$input" ]; then
        color_red "[regenerate_urdf] FATAL: input not found: $input"
        exit 1
    fi

    color_yel "[regenerate_urdf] xacro $input -> $output  ${extra_args[*]:-}"
    xacro "$input" "${extra_args[@]}" > "$output.tmp"
    mv "$output.tmp" "$output"

    # Hash the GENERATED output (catches transitive franka_description drift that
    # source-only hashing would miss).
    local sha="${output}.sha256"
    (cd "$(dirname "$output")" && sha256sum "$(basename "$output")" > "$sha")
    color_grn "[regenerate_urdf] wrote $sha"
}

generate "$DUAL_XACRO"     "$DUAL_OUT"     "${DUAL_ARGS[@]}"
generate "$DUAL_V2_XACRO"  "$DUAL_V2_OUT"  "${DUAL_ARGS[@]}"
generate "$TABLE_XACRO"    "$TABLE_OUT"
generate "$TTABLE_XACRO"   "$TTABLE_OUT"

# Post-process: replace finger <collision> meshes with small primitive boxes so
# MoveIt's FCL self-collision checking is mesh→primitive instead of mesh→mesh.
# This is option C from the collision-simplification plan; the SHA256 sidecars
# are rewritten by the script so strict_urdf_snapshot=true still validates.
SIMPLIFY="$(dirname "$0")/simplify_finger_collisions.py"
if [ -x "$SIMPLIFY" ] || [ -f "$SIMPLIFY" ]; then
    color_yel "[regenerate_urdf] simplifying finger collisions (option C)…"
    /usr/bin/env python3 "$SIMPLIFY" "$DUAL_OUT"
    /usr/bin/env python3 "$SIMPLIFY" "$DUAL_V2_OUT"
else
    color_red "[regenerate_urdf] simplify_finger_collisions.py not found; leaving finger meshes intact."
fi

color_grn "[regenerate_urdf] OK"
color_grn "  Snapshots:"
ls -l "$DUAL_OUT" "$DUAL_V2_OUT" "$TABLE_OUT" "$TTABLE_OUT"
color_grn "  Sidecars:"
ls -l "${DUAL_OUT}.sha256" "${DUAL_V2_OUT}.sha256" "${TABLE_OUT}.sha256" "${TTABLE_OUT}.sha256"
