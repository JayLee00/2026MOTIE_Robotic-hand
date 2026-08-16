"""AC-2: the package never references or imports dex_sodering/isaaclab/omni.*."""

import os
import re
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Match the forbidden module names as import targets / dotted references.
FORBIDDEN = re.compile(r"\b(dex_sodering|isaaclab|omni\.)")


def _source_files():
    for dirpath, _dirs, files in os.walk(PKG_ROOT):
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def test_no_forbidden_references_in_source():
    hits = []
    for path in _source_files():
        # Skip this test file itself (it legitimately names them in regex).
        if os.path.basename(path) == "test_no_leak.py":
            continue
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                # ignore prose mentions in comments? No: the AC is a hard grep
                # of source == 0 hits. The other modules only mention the names
                # inside docstrings, which WOULD be a hit, so we forbid any.
                if FORBIDDEN.search(line):
                    hits.append(f"{path}:{lineno}: {line.rstrip()}")
    assert not hits, "forbidden references found:\n" + "\n".join(hits)


def test_importing_package_pulls_no_forbidden_modules():
    # Fresh import and assert no forbidden top-level module got loaded.
    for name in list(sys.modules):
        if name == "in_hand_tracker" or name.startswith("in_hand_tracker."):
            del sys.modules[name]
    import in_hand_tracker  # noqa: F401
    import in_hand_tracker.io  # noqa: F401

    leaked = [m for m in sys.modules
              if m == "dex_sodering" or m.startswith("dex_sodering.")
              or m == "isaaclab" or m.startswith("isaaclab.")
              or m == "omni" or m.startswith("omni.")]
    assert not leaked, f"forbidden modules imported: {leaked}"
