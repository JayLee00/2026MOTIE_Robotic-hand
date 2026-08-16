"""pytest bootstrap for franka_kistar_bringup tests.

Each test loads its target module via importlib.util.spec_from_file_location, so no
helpers live here; this file only puts the package's launch/ and scripts/ directories
on sys.path for any future tests that prefer name-based imports.

Run from the repo root with:
    PYTHONPATH=$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages \
        python3 -m pytest isaac-ros/kistar_ws/src/franka_kistar_bringup/test/ \
            --import-mode=importlib
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)

for path in (os.path.join(PKG_ROOT, "launch"), os.path.join(PKG_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
