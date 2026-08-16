"""Live monitor for the place pipeline. Two sinks, both best-effort:

  * RViz  — staged point clouds / EE pose via the ROS backend's viz_* markers
            (published on /place_debug/markers, auto-shown by the fr3_kistar.rviz
            MarkerArray display — no manual Add).
  * grid  — pipeline images pushed to the grid web service (services/grid_service.py
            :8815) which tiles them into the A|B grid a browser auto-refreshes.

NullMonitor (the default) makes every call a no-op, so mock/offline runs and the pure
flow are unaffected. All calls swallow errors — viz must never break the pipeline.
"""
import urllib.request

import numpy as np

from vision_pipeline import debug
from vision_pipeline.services.rpc import pack


class NullMonitor:
    def cloud(self, *a, **k):
        pass

    def alpha(self, *a, **k):
        pass

    def ee(self, *a, **k):
        pass

    def img_raw(self, *a, **k):
        pass

    def img_depth(self, *a, **k):
        pass

    def img_points(self, *a, **k):
        pass

    def img_mask(self, *a, **k):
        pass

    def hold(self, *a, **k):
        pass


class Monitor:
    def __init__(self, backend=None, grid_url=None, log=print):
        self.b = backend                       # ROS backend for viz_* markers
        self.grid_url = grid_url               # e.g. http://127.0.0.1:8815
        self.log = log

    # ---- RViz point clouds (staged) -----------------------------------------
    def cloud(self, ns, pts, color, alpha=1.0, size=0.005):
        self._viz("viz_points", ns, pts, color, alpha, size)

    def alpha(self, ns, a):
        self._viz("viz_alpha", ns, a)

    def ee(self, T):
        self._viz("viz_ee", T)

    def hold(self, sec):
        self._viz("viz_hold", sec)

    def _viz(self, fn, *a):
        f = getattr(self.b, fn, None)
        if f is None:
            return
        try:
            f(*a)
        except Exception as e:                 # noqa: BLE001
            self.log(f"[viz] {fn} failed ({type(e).__name__}: {e})")

    # ---- grid images (best-effort HTTP) -------------------------------------
    def img_raw(self, cell, rgb):
        self._push(cell, np.asarray(rgb, np.uint8))

    def img_depth(self, cell, depth):
        self._push(cell, debug.render_depth(depth))

    def img_points(self, cell, rgb, pts, color, sel=None):
        self._push(cell, debug.render_points(rgb, pts, color, sel=sel))

    def img_mask(self, cell, rgb, mask, color):
        self._push(cell, debug.render_mask(rgb, mask, color))

    def _push(self, cell, img):
        if not self.grid_url:
            return
        try:                                   # raw POST; grid replies plain text, don't unpack it
            blob = pack(name=np.array(cell), image=np.asarray(img, np.uint8))
            req = urllib.request.Request(self.grid_url + "/put", data=blob,
                                         headers={"Content-Type": "application/octet-stream"})
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:                      # noqa: BLE001
            pass
