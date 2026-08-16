#!/usr/bin/env python3
"""simplify_finger_collisions.py — replace mesh-based <collision> for finger links
with small primitive boxes so MoveIt's FCL collision-checking is mesh→primitive
instead of mesh→mesh. Operates on a generated URDF in place (also updates the
sha256 sidecar if one exists).

Heuristic: any link whose name matches the per-finger naming pattern
(left_/right_ × thumb/index/middle/ring × …) gets its <collision> rewritten as
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry><box size="0.025 0.025 0.04"/></geometry>
  </collision>

Default box size is intentionally a slight over-approximation of a finger
segment (~25 × 25 × 40 mm). Override with --box.

The original mesh-based <visual> is kept intact so RViz still shows the high-
fidelity hand model. Only <collision> is rewritten.
"""
import argparse
import hashlib
import os
import re
import sys
from xml.etree import ElementTree as ET


FINGER_TOKENS = ("_thumb_", "_index_", "_middle_", "_ring_")
SIDE_PREFIXES = ("left_", "right_")


def is_finger_link(name: str) -> bool:
    if not any(name.startswith(p) for p in SIDE_PREFIXES):
        return False
    return any(tok in name for tok in FINGER_TOKENS)


def simplify(urdf_path: str, box_size: str):
    with open(urdf_path, "r") as f:
        text = f.read()

    box_repl = (
        "<collision>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        "      <geometry>\n"
        f"        <box size=\"{box_size}\"/>\n"
        "      </geometry>\n"
        "    </collision>"
    )

    pattern = re.compile(
        r'(<link name="([^"]+)">)(.*?)(</link>)', re.DOTALL
    )

    rewritten = 0
    def repl(m):
        nonlocal rewritten
        link_open, name, body, link_close = m.group(1), m.group(2), m.group(3), m.group(4)
        if not is_finger_link(name):
            return m.group(0)
        # NB: package:// URLs contain '/', so we exclude '>' (not '/') in the
        # attribute matcher. Mesh tag is self-closing ("<mesh … />").
        new_body, n = re.subn(
            r'<collision>\s*(?:<origin\s+[^>]*?/>\s*)?<geometry>\s*<mesh\s+[^>]*?/>\s*</geometry>\s*</collision>',
            box_repl,
            body,
            flags=re.DOTALL,
        )
        if n > 0:
            rewritten += 1
        return link_open + new_body + link_close

    new_text, _ = pattern.subn(repl, text)

    if rewritten == 0:
        print(f"[simplify_finger_collisions] no finger collisions found in {urdf_path}")
        return

    with open(urdf_path, "w") as f:
        f.write(new_text)

    sidecar = urdf_path + ".sha256"
    if os.path.isfile(sidecar):
        h = hashlib.sha256()
        with open(urdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        base = os.path.basename(urdf_path)
        with open(sidecar, "w") as f:
            f.write(f"{h.hexdigest()}  {base}\n")

    print(f"[simplify_finger_collisions] rewrote {rewritten} finger collision blocks in {urdf_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urdf", help="Path to generated URDF (will be modified in place)")
    p.add_argument("--box", default="0.025 0.025 0.04",
                   help='Box "x y z" size in metres (default: 0.025 0.025 0.04)')
    args = p.parse_args()
    simplify(args.urdf, args.box)


if __name__ == "__main__":
    main()
