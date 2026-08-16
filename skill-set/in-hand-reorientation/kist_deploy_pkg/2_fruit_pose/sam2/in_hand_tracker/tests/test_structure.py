"""AC: the file tree exists and ``import in_hand_tracker`` works."""

import importlib
import os

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_FILES = [
    "__init__.py",
    "pyproject.toml",
    "config/camera.yaml",
    "config/objects.yaml",
    "config/estimator.yaml",
    "io/__init__.py",
    "io/types.py",
    "io/replicator_reader.py",
    "io/deproject.py",
    "tests/test_structure.py",
    "tests/test_no_leak.py",
    "tests/test_io.py",
]


def test_file_tree_exists():
    missing = [rel for rel in EXPECTED_FILES
               if not os.path.isfile(os.path.join(PKG_ROOT, rel))]
    assert not missing, f"missing files: {missing}"


def test_package_importable():
    mod = importlib.import_module("in_hand_tracker")
    assert hasattr(mod, "__version__")
    io = importlib.import_module("in_hand_tracker.io")
    for name in ("Frame", "GtShape", "FitResult", "ReplicatorReader",
                 "deproject", "cam_to_world", "surface_centroid"):
        assert hasattr(io, name), f"in_hand_tracker.io missing {name}"
