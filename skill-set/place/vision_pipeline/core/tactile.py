"""PaXini fingertip force monitor — Release/Retract trigger logic (spec: 2nd goal).

The KISTAR hand carries four PaXini M2826-Omega fingertip sensors (thumb, index,
middle, ring). `/paxini/right/ft` streams the per-finger resultant force
[Fx, Fy, Fz] in Newtons at ~90 Hz; the writer tares to zero at start-up, so a
finger with NO contact reads ~0 and a grasping finger reads a nonzero (mostly Fz)
force.

Two triggers drive the place-then-release sequence:

  * disturbance (Release, Case 1): during the descent, a fingertip force departs
    from its *normal* range — the object touched the tray, a fingertip hit a
    neighbouring object, or an overshoot pressed the object into the surface.
    -> stop descending, release here.

    The "normal range" is estimated **online**, not from a single frozen
    baseline: a rolling window tracks the per-finger force, and the reference
    centre + robust spread are recomputed every sample from that window
    (EXCLUDING the newest `guard` samples so an incipient disturbance can't
    contaminate its own reference). Because the reference follows the slow
    gravity/inertia drift of the decelerating descent, only an ABRUPT departure
    fires — this is object-agnostic (we watch the CHANGE from "this grasp right
    now", never an absolute force, so a heavy grasp and a light grasp behave the
    same). The window is seeded from the grasp baseline captured at the waypoint,
    so the detector is armed from the first descent sample.

  * no-load (Retract): during the release ascent, ALL fingertips drop below
    `noload_thresh` — the object has left the hand. -> stop ascending, retract.
    This one is absolute (an unloaded, tared sensor reads ~0).

Both use a debounce (N consecutive qualifying samples) so a single noisy frame
never fires. Thresholds are constructor args so the orchestrator can tune them.

A liveness check (`is_grasp_live`) guards against the silent-failure mode where
the PaXini writer isn't running / was started after the grasp: a truly grasping
hand cannot read ~0 N on every finger, so that reading means the tactile signal
is dead and the triggers must NOT be trusted (the orchestrator drops to a loud,
explicit degrade instead of pretending everything is fine).

Pure numpy — no ROS — so the trigger logic is unit-testable offline.
"""
from collections import deque

import numpy as np


class ForceMonitor:
    def __init__(self, k=5.0, abs_floor=0.4, noload_thresh=0.6,
                 dist_debounce=3, noload_debounce=5,
                 win=40, guard=4, min_ref=12, live_floor=0.3):
        self.k = float(k)
        self.abs_floor = float(abs_floor)              # N — floor so quiet-grasp micro-sigma can't over-trigger
        self.noload_thresh = float(noload_thresh)      # N — all fingers below this => object gone
        self.dist_n = int(dist_debounce)
        self.noload_n = int(noload_debounce)
        self.win = int(win)                            # rolling-window length (samples)
        self.guard = int(guard)                        # newest samples excluded from the reference
        self.min_ref = int(min_ref)                    # min reference samples before the detector arms
        self.live_floor = float(live_floor)            # N — a real grasp's max |F| must exceed this
        self._buf = deque(maxlen=self.win)             # recent (4,3) force samples
        self._dist_count = 0
        self._noload_count = 0
        self._grasp_force = np.zeros(4)                # per-finger |F| of the primed baseline
        self._null = False

    # ── construction / priming ────────────────────────────────────────────────
    @classmethod
    def null(cls):
        """A monitor whose triggers never fire — used when PaXini is unavailable / not
        trustworthy so the descent runs pure-Case-2 (to T_act) and the release ascent runs
        to the waypoint. Degrades safely (and the orchestrator logs it LOUDLY)."""
        m = cls()
        m._null = True
        return m

    def prime(self, samples):
        """Seed the rolling window + record grasp liveness from the waypoint baseline
        window, shape (M,4,3) [N]. Arms the adaptive detector from the first descent sample."""
        s = np.asarray(samples, float).reshape(-1, 4, 3)
        for f in s[-self.win:]:
            self._buf.append(f)
        self._grasp_force = np.linalg.norm(np.median(s, axis=0), axis=1)   # (4,) per-finger |median F|
        return self

    def grasp_force(self):
        """Per-finger |F| (N) of the primed grasp baseline (zeros if never primed)."""
        return self._grasp_force.copy()

    def is_grasp_live(self):
        """True iff the primed baseline looks like a real grasp (some finger clearly loaded).
        A hand actually holding an object cannot read ~0 on every finger; if it does, the
        PaXini stream is dead / mis-tared and the triggers are not trustworthy."""
        return bool(np.max(self._grasp_force) >= self.live_floor)

    def reset_noload(self):
        self._noload_count = 0

    def reset_disturbance(self):
        self._dist_count = 0

    # ── triggers ──────────────────────────────────────────────────────────────
    def disturbance(self, ft):
        """True once some fingertip departs from its online normal range by more than
        max(abs_floor, k·robust_sigma) for `dist_debounce` consecutive samples. The
        reference is the rolling window minus its newest `guard` samples."""
        if self._null or ft is None:
            return False
        f = np.asarray(ft, float).reshape(4, 3)
        if not np.all(np.isfinite(f)):                 # a corrupt frame must never trigger
            return False
        self._buf.append(f)
        ref = list(self._buf)[:-self.guard] if self.guard > 0 else list(self._buf)
        if len(ref) < self.min_ref:                    # warmup: not enough history to judge "normal"
            return False
        w = np.stack(ref)                              # (Lref,4,3)
        center = np.median(w, axis=0)                  # (4,3) robust reference centre
        dev_ref = np.linalg.norm(w - center, axis=2)   # (Lref,4) per-finger deviation magnitude
        med = np.median(dev_ref, axis=0)               # (4,)
        sigma = 1.4826 * np.median(np.abs(dev_ref - med), axis=0)   # (4,) robust std
        cur_dev = np.linalg.norm(f - center, axis=1)   # (4,) current deviation from the reference
        thr = np.maximum(self.abs_floor, self.k * sigma)           # (4,)
        if np.any(cur_dev > thr):
            self._dist_count += 1
        else:
            self._dist_count = 0
        return self._dist_count >= self.dist_n

    def noload(self, ft):
        """True once ALL fingertips read below noload_thresh for `noload_debounce`
        consecutive samples (the object has left the hand)."""
        if self._null or ft is None:
            return False
        f = np.asarray(ft, float).reshape(4, 3)
        if not np.all(np.isfinite(f)):
            return False
        mag = np.linalg.norm(f, axis=1)                # (4,)
        if np.all(mag < self.noload_thresh):
            self._noload_count += 1
        else:
            self._noload_count = 0
        return self._noload_count >= self.noload_n
