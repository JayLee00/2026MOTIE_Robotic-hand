"""Hand-safety regression checks (no ROS / robot / GPU — mock backend only).

    python3 vision_pipeline/test_hand_safety.py

Covers the two rules that keep the KISTAR hand from running away between stages:

1. [R5] — the Voltage->Position + servo-OFF restore runs on EVERY exit path out of the
   release/retract block, including an exception raised between R1 and R5. The skill server is
   long-lived (it catches per-run exceptions and waits for the next fruit), so a leak here would
   NOT be caught by the process-exit safety net in ros_backend, and the next stage's Position
   counts would be reinterpreted as raw duty.
2. Grip handover — the place turn takes over the Position target the previous stage (Stiffness)
   was streaming, holds it while the arm works, and stops holding BEFORE the Voltage switch.
   A target is never fabricated, and Position counts never go through the Voltage duty clamp.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_pipeline.backends import ros_backend as RB         # noqa: E402
from vision_pipeline.backends.mock import MockBackend          # noqa: E402
from vision_pipeline.orchestrator import PlacePipeline         # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


class BoomMonitor:
    """Monitor whose reset_noload() explodes — an unguarded call between R1 and R5."""

    def reset_noload(self):
        raise RuntimeError("injected failure between R1 and R5")

    def noload(self, *a, **k):
        return False


class OkMonitor:
    def reset_noload(self):
        pass

    def noload(self, *a, **k):
        return True


def make_pipe(b):
    return PlacePipeline(b, models=None, log=lambda *_a: None, debug_dir=None,
                         monitor=None, use_hand_pc=False)


def main():
    print("\n[1] R5 runs even when the release/retract block raises")
    b = MockBackend()
    ctx = {"monitor": BoomMonitor(), "wp": np.eye(4), "stop_reason": "reached"}
    raised = None
    try:
        make_pipe(b)._release_and_retract(ctx, {})
    except Exception as e:                                                # noqa: BLE001
        raised = e
    names = [c[0] for c in b.calls]
    check("exception propagates", raised is not None, f"{type(raised).__name__}: {raised}")
    check("hand_safe_shutdown still ran", "hand_safe_shutdown" in names, f"calls={names}")
    check("shutdown after release",
          names.index("hand_release") < names.index("hand_safe_shutdown"))

    print("\n[2] normal path also reaches R5")
    b2 = MockBackend()
    make_pipe(b2)._release_and_retract(
        {"monitor": OkMonitor(), "wp": np.eye(4), "stop_reason": "reached"}, {})
    n2 = [c[0] for c in b2.calls]
    check("hand_safe_shutdown is last", n2[-1] == "hand_safe_shutdown", f"calls={n2}")

    print("\n[3] grip handover: take over -> hold -> stop before Voltage")
    b3 = MockBackend()
    b3.handover_vec = [1234.0] * 16          # what the previous stage was streaming (counts)
    held = b3.hand_hold_start()
    check("took over the previous target verbatim", held == [1234.0] * 16, f"held[:2]={held[:2]}")
    check("hold active", b3.holding is not None)
    b3.hand_release_sequence([50.0] * 16)
    check("hold stopped before the Voltage switch", b3.holding is None,
          f"hold_calls={[c[0] for c in b3.hold_calls]}")

    print("\n[4] no target to take over -> hold nothing (never fabricate)")
    b4 = MockBackend()
    b4.handover_vec = None
    check("hand_hold_start -> None", b4.hand_hold_start() is None)
    check("nothing held", b4.holding is None)

    print("\n[5] Position counts never go through the Voltage duty clamp")
    counts = [1234.0, -987.0] + [0.0] * 14
    check("_safe_counts16 is lossless", RB._safe_counts16(counts) == counts,
          f"-> {RB._safe_counts16(counts)[:2]}")
    check("_safe_duty16 still clamps", RB._safe_duty16(counts)[0] == RB.RELEASE_DUTY_ABS_MAX,
          f"-> {RB._safe_duty16(counts)[:2]}")
    check("wrong length rejected", RB._safe_counts16([1.0] * 15) is None)
    check("non-finite rejected", RB._safe_counts16([float("nan")] + [0.0] * 15) is None)

    print("\n" + ("HAND SAFETY TEST OK" if not FAILED else f"FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
