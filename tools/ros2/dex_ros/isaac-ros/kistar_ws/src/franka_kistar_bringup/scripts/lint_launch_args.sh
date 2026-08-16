#!/usr/bin/env bash
# lint_launch_args.sh — CI + pre-commit lint for plural-form discipline.
#
# Asserts that the legacy plural launch arg 'joint_states_mode' appears in
# DeclareLaunchArgument only ONCE in the planning PC launch file, and that the
# single occurrence is on the stowaway-sentinel line carrying the marker
# __plural_rejected_sentinel__.
#
# Other uses of the literal string (header docstring example, validator code that
# reads the LaunchConfiguration, description strings) are NOT counted because they
# are needed for the rejection mechanism itself.
#
# Self-test mode: pass --self-test to verify the script fails when the sentinel
# is removed from a synthesized copy of the launch file.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"
LAUNCH_FILE="${LAUNCH_FILE:-${REPO_ROOT}/isaac-ros/kistar_ws/src/franka_kistar_bringup/launch/deprecated/dual_fr3_kistar_planning_pc.launch.py}"

SENTINEL='__plural_rejected_sentinel__'
PLURAL='joint_states_mode'

color_red() { printf "\033[31m%s\033[0m\n" "$*"; }
color_grn() { printf "\033[32m%s\033[0m\n" "$*"; }

check_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        color_red "[lint_launch_args] FAIL: launch file not found: $file"
        return 1
    fi

    # Count DeclareLaunchArgument occurrences of the plural literal (the only
    # mode in which the launch system would treat it as an actual arg name).
    local decl_count
    decl_count=$(grep -cE "DeclareLaunchArgument\\(" "$file" | head -1 || true)
    local plural_decl_lines
    plural_decl_lines=$(grep -nE "DeclareLaunchArgument\\(" -A 1 "$file" \
        | grep -E "\"${PLURAL}\"|'${PLURAL}'" || true)

    local plural_decl_count
    plural_decl_count=$(printf "%s\n" "$plural_decl_lines" | grep -cE "${PLURAL}" || true)

    local sentinel_match
    sentinel_match=$(grep -E "\"${PLURAL}\"" "$file" | grep -E "${SENTINEL}" || true)
    local sentinel_count
    sentinel_count=$(printf "%s" "$sentinel_match" | grep -cE "${PLURAL}" || true)
    # Default to numeric 0 if grep set empty under set -e
    plural_decl_count=${plural_decl_count:-0}
    sentinel_count=${sentinel_count:-0}

    if [ "$plural_decl_count" -ne 1 ]; then
        color_red "[lint_launch_args] FAIL: expected exactly 1 DeclareLaunchArgument occurrence of '${PLURAL}'; got ${plural_decl_count}"
        printf "%s\n" "$plural_decl_lines"
        return 1
    fi
    if [ "$sentinel_count" -ne 1 ]; then
        color_red "[lint_launch_args] FAIL: the '${PLURAL}' line MUST carry the sentinel '${SENTINEL}'; got sentinel-matches=${sentinel_count}"
        printf "%s\n" "$sentinel_match"
        return 1
    fi

    # Also assert no stray LaunchConfiguration uses of the plural OUTSIDE the validator.
    # The validator function _validate_args is the only legitimate reader.
    local lc_lines
    lc_lines=$(grep -nE "LaunchConfiguration\\(\"${PLURAL}\"\\)" "$file" || true)
    local lc_outside_validator
    lc_outside_validator=$(printf "%s\n" "$lc_lines" \
        | grep -v "_validate_args" || true)
    # Re-check: the lc_lines themselves do not say _validate_args; they say:
    #   line:    plural = LaunchConfiguration("joint_states_mode").perform(context)
    # which is INSIDE _validate_args. We allow any number of LaunchConfiguration
    # plural reads provided they are inside _validate_args; verify by checking
    # the file slice around each match contains the function name.
    local total_lc
    total_lc=$(printf "%s\n" "$lc_lines" | grep -cE "LaunchConfiguration" || true)
    total_lc=${total_lc:-0}
    if [ "$total_lc" -gt 1 ]; then
        color_red "[lint_launch_args] FAIL: more than one LaunchConfiguration(\"${PLURAL}\") read; only the validator may read it."
        printf "%s\n" "$lc_lines"
        return 1
    fi
    if [ "$total_lc" -eq 1 ]; then
        # Verify the surrounding context (10 lines back) contains _validate_args.
        local lineno
        lineno=$(printf "%s" "$lc_lines" | head -1 | cut -d: -f1)
        local context_start=$((lineno - 10))
        if [ "$context_start" -lt 1 ]; then context_start=1; fi
        if ! sed -n "${context_start},${lineno}p" "$file" | grep -q "_validate_args"; then
            color_red "[lint_launch_args] FAIL: LaunchConfiguration(\"${PLURAL}\") read is NOT inside _validate_args."
            printf "%s\n" "$lc_lines"
            return 1
        fi
    fi

    color_grn "[lint_launch_args] OK: ${LAUNCH_FILE}"
    return 0
}

run_self_test() {
    SELF_TEST_TMPDIR=$(mktemp -d)
    trap 'rm -rf "${SELF_TEST_TMPDIR:-}"' EXIT

    local synth="${SELF_TEST_TMPDIR}/synthesized_launch.py"
    # Synthesize a launch.py copy WITHOUT the sentinel comment — should FAIL.
    sed "s|${SENTINEL}|removed_sentinel|" "$LAUNCH_FILE" > "$synth"
    if LAUNCH_FILE="$synth" check_file "$synth" > /dev/null 2>&1; then
        color_red "[lint_launch_args] SELF-TEST FAIL: synthesized launch WITHOUT sentinel still passed."
        return 1
    fi
    color_grn "[lint_launch_args] SELF-TEST OK: synthesized launch without sentinel correctly fails."
    return 0
}

main() {
    case "${1:-}" in
        --self-test)
            check_file "$LAUNCH_FILE"
            run_self_test
            ;;
        *)
            check_file "$LAUNCH_FILE"
            ;;
    esac
}

main "$@"
