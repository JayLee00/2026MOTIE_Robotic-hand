#!/usr/bin/env python3
"""check_rviz_camera_variant.py — drift guard for the camera rviz variant.

fr3_kistar_camera.rviz is supposed to be fr3_kistar.rviz PLUS three extra
displays (FrontRGB, FrontDepth, FrontCloud) that visualize the front camera.
Every other line must stay byte-for-byte identical in structure to the base
file. This script strips the three known camera displays (and any panel
entries that reference them) out of the variant and asserts the remainder
deep-equals the base file's parsed YAML.

Run this after any edit to fr3_kistar.rviz (dev tool; not installed via
CMake, not part of the ROS package runtime).

Usage:
    python3 scripts/check_rviz_camera_variant.py
    python3 scripts/check_rviz_camera_variant.py --base path/to/base.rviz --variant path/to/variant.rviz
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

CAMERA_DISPLAY_NAMES = {"FrontRGB", "FrontDepth", "FrontCloud"}

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE = PACKAGE_ROOT / "rviz" / "fr3_kistar.rviz"
DEFAULT_VARIANT = PACKAGE_ROOT / "rviz" / "fr3_kistar_camera.rviz"


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _flatten(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten(v)
    else:
        yield obj


def strip_camera_displays(config):
    """Return a deep copy of config with the known camera displays, and any
    panel entries referencing them, removed."""
    config = copy.deepcopy(config)

    displays = config.get("Visualization Manager", {}).get("Displays", [])
    config["Visualization Manager"]["Displays"] = [
        d for d in displays if d.get("Name") not in CAMERA_DISPLAY_NAMES
    ]

    panels = config.get("Panels", [])
    config["Panels"] = [
        p for p in panels
        if not any(str(v) in CAMERA_DISPLAY_NAMES for v in _flatten(p))
    ]

    return config


def diff_keys(a, b, path=""):
    """Yield human-readable descriptions of the differences between a and b."""
    numeric = (int, float)
    if type(a) is not type(b) and not (isinstance(a, numeric) and isinstance(b, numeric)):
        yield f"{path or '<root>'}: type mismatch {type(a).__name__} vs {type(b).__name__}"
        return
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            p = f"{path}.{key}" if path else key
            if key not in a:
                yield f"{p}: missing from variant (after stripping camera displays), present in base"
            elif key not in b:
                yield f"{p}: present in variant (after stripping camera displays), missing from base"
            else:
                yield from diff_keys(a[key], b[key], p)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}: list length mismatch {len(a)} (variant-stripped) vs {len(b)} (base)"
            return
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff_keys(x, y, f"{path}[{i}]")
    else:
        if a != b:
            yield f"{path}: {a!r} (variant-stripped) != {b!r} (base)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="path to fr3_kistar.rviz")
    parser.add_argument("--variant", type=Path, default=DEFAULT_VARIANT, help="path to fr3_kistar_camera.rviz")
    args = parser.parse_args()

    base = load_yaml(args.base)
    variant = load_yaml(args.variant)

    stripped_variant = strip_camera_displays(variant)

    diffs = list(diff_keys(stripped_variant, base))
    if diffs:
        print(
            "DRIFT DETECTED: regenerate fr3_kistar_camera.rviz from fr3_kistar.rviz — bases have diverged",
            file=sys.stderr,
        )
        print("Differing keys:", file=sys.stderr)
        for d in diffs[:50]:
            print(f"  - {d}", file=sys.stderr)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs) - 50} more", file=sys.stderr)
        sys.exit(1)

    print("OK: fr3_kistar_camera.rviz == fr3_kistar.rviz + camera displays")
    sys.exit(0)


if __name__ == "__main__":
    main()
