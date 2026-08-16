"""Offline unit tests for the PaXini release/retract trigger logic (pure numpy, no ROS).

Run:  python3 -m vision_pipeline.tests.test_tactile
"""
import numpy as np

from vision_pipeline.core.tactile import ForceMonitor


def _grasp(fz=(1.6, 0.1, 1.5, 1.4), noise=0.03, rng=None):
    """One (4,3) sample of a steady grasp: per-finger Fz + small gaussian noise on all axes."""
    rng = rng or np.random.default_rng(0)
    f = np.zeros((4, 3))
    f[:, 2] = np.asarray(fz)
    return f + rng.normal(0, noise, (4, 3))


def _mk(**kw):
    m = ForceMonitor(k=5.0, abs_floor=0.4, noload_thresh=0.6,
                     dist_debounce=3, noload_debounce=5,
                     win=40, guard=4, min_ref=12, live_floor=0.3, **kw)
    rng = np.random.default_rng(1)
    m.prime([_grasp(rng=rng) for _ in range(12)])
    return m, rng


PASS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")


def test_liveness():
    live, _ = _mk()
    check("grasp is live", live.is_grasp_live())
    dead = ForceMonitor()
    dead.prime([np.zeros((4, 3)) + np.random.default_rng(2).normal(0, 0.02, (4, 3)) for _ in range(12)])
    check("all-zero grasp NOT live (silent-degrade guard)", not dead.is_grasp_live())


def test_no_false_disturbance_on_steady_grasp():
    m, rng = _mk()
    fired = any(m.disturbance(_grasp(rng=rng)) for _ in range(200))
    check("steady grasp never fires disturbance (200 samples)", not fired)


def test_no_false_disturbance_on_slow_drift():
    # gravity/inertia drift during the decelerating descent: Fz ramps smoothly by ~0.5 N over
    # the descent — the adaptive reference must track it and NOT fire.
    m, rng = _mk()
    fired = False
    for i in range(200):
        drift = 0.5 * i / 200.0
        s = _grasp(fz=(1.6 + drift, 0.1 + drift, 1.5 + drift, 1.4 + drift), rng=rng)
        fired = fired or m.disturbance(s)
    check("smooth drift never fires disturbance (adaptive reference tracks it)", not fired)


def test_disturbance_fires_on_collision():
    m, rng = _mk()
    for _ in range(30):                                   # settle the window on steady grasp
        m.disturbance(_grasp(rng=rng))
    # abrupt +1.2 N step on the index finger (a fingertip hits a neighbouring object)
    fired_at = None
    for i in range(20):
        s = _grasp(rng=rng)
        s[1, 2] += 1.2
        if m.disturbance(s):
            fired_at = i
            break
    check("collision (+1.2 N step) fires disturbance", fired_at is not None)
    check("fires within debounce (<=4 samples)", fired_at is not None and fired_at <= 4)


def test_disturbance_debounce_ignores_single_spike():
    m, rng = _mk()
    for _ in range(30):
        m.disturbance(_grasp(rng=rng))
    s = _grasp(rng=rng); s[2, 0] += 1.5                   # single 1-frame spike
    one = m.disturbance(s)
    quiet = any(m.disturbance(_grasp(rng=rng)) for _ in range(10))
    check("single-frame spike does NOT fire (debounce)", not one and not quiet)


def test_noload_fires_when_object_gone():
    m, _ = _mk()
    # still loaded -> no
    loaded = any(m.noload(_grasp()) for _ in range(3))
    check("loaded grasp is not no-load", not loaded)
    # object gone -> all fingers ~0
    z = np.zeros((4, 3))
    fired = [m.noload(z + np.random.default_rng(3).normal(0, 0.02, (4, 3))) for _ in range(6)]
    check("all-fingers-~0 fires no-load after debounce", fired[-1] and not fired[0])


def test_none_and_nan_never_fire():
    m, _ = _mk()
    check("None never fires disturbance", not any(m.disturbance(None) for _ in range(10)))
    check("None never fires no-load", not any(m.noload(None) for _ in range(10)))
    nan = np.full((4, 3), np.nan)
    check("NaN frame never fires disturbance", not m.disturbance(nan))
    check("NaN frame never fires no-load", not m.noload(nan))


def test_null_monitor_never_fires():
    n = ForceMonitor.null()
    check("null: disturbance never fires", not any(n.disturbance(_grasp()) for _ in range(50)))
    check("null: no-load never fires", not any(n.noload(np.zeros((4, 3))) for _ in range(50)))


def test_duty_clamp():
    from vision_pipeline.backends.ros_backend import _safe_duty16, RELEASE_DUTY_ABS_MAX
    out = _safe_duty16([9999, -9999, float("nan"), float("inf")] + [60] * 12)
    check("clamps huge +", out[0] == RELEASE_DUTY_ABS_MAX)
    check("clamps huge -", out[1] == -RELEASE_DUTY_ABS_MAX)
    check("NaN -> 0", out[2] == 0.0)
    check("Inf -> 0", out[3] == 0.0)
    check("in-range +60 passes", out[4] == 60.0)
    try:
        _safe_duty16([0] * 15)
        check("wrong length raises", False)
    except ValueError:
        check("wrong length raises", True)


if __name__ == "__main__":
    for t in [test_liveness, test_no_false_disturbance_on_steady_grasp,
              test_no_false_disturbance_on_slow_drift, test_disturbance_fires_on_collision,
              test_disturbance_debounce_ignores_single_spike, test_noload_fires_when_object_gone,
              test_none_and_nan_never_fire, test_null_monitor_never_fires, test_duty_clamp]:
        print(f"\n{t.__name__}:")
        t()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    raise SystemExit(0 if all(PASS) else 1)
