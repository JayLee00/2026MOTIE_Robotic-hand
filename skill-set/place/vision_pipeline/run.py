"""Single-command entry for the place pipeline (vision_pipeline_design.md §0).

    python -m vision_pipeline.run scenario=fruit hand_pc=true

Wires the real ROS2 backend + HTTP model services + PlacePipeline. Run inside the
dex_ros Humble container (ROS_DOMAIN_ID=9) with the digital-twin up and all five
model services running (see run_services.sh / README.md).

Emits a real-time, timestamped `[+SS.s]` log of every step (run with `python3 -u`
so it streams through a pipe/tee instead of being lost on Ctrl+C).

For offline dry-runs without the robot use `backend=mock`.
"""
import os
import sys
import time
import traceback
import urllib.request

from vision_pipeline.orchestrator import PlacePipeline
from vision_pipeline.models_client import ModelsHTTP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every service we expect up before a real run (name -> /health port). The camera is not
# a service here — it streams from the 산업부 PC over ROS2 (the ROS backend subscribes).
HEALTH_PORTS = {"anyplace": 8801, "molmo": 8810,
                "sam": 8811, "igr": 8816, "grid": 8815}


def parse_kv(argv):
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def make_logger(log_path=None):
    """Timestamped logger: prints `[HH:MM:SS +SS.s] msg` to stdout AND, if log_path is
    given, appends the same line to a per-run timeline file (line-buffered)."""
    t0 = time.time()
    fh = None
    if log_path:
        try:
            fh = open(log_path, "w", buffering=1)
        except Exception:                                                  # noqa: BLE001
            fh = None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.1f}s] {msg}"
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")
            fh.flush()

    return log


def preflight(log):
    """GET /health on every service so an unreachable one shows up now (from the
    container, over --network host) instead of hanging a model call later."""
    for name, port in HEALTH_PORTS.items():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                state = "UP" if r.status == 200 else f"BAD({r.status})"
        except Exception as e:                                              # noqa: BLE001
            state = f"DOWN ({type(e).__name__})"
        log(f"  service {name:9s} :{port}  {state}")


def main():
    a = parse_kv(sys.argv[1:])
    scenario = a.get("scenario", "table")
    is_mock = a.get("backend") == "mock"
    hold = float(a.get("hold", 0 if is_mock else 300))                     # RViz keep-alive seconds

    # Per-run folder (timestamp without the year) holds BOTH the artifacts (images/clouds/
    # info.txt via debug.save_run) AND a timeline.log of when each step ran.
    run_dir = os.path.join(REPO, "vision_pipeline", "test_logs",
                           "run_" + time.strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = make_logger(os.path.join(run_dir, "timeline.log"))
    log(f"start: scenario={scenario} backend={a.get('backend', 'ros')}")
    log(f"artifacts + timeline -> {run_dir}")

    if is_mock:
        from vision_pipeline.backends.mock import MockBackend, MockModels
        backend, models, monitor = MockBackend(), MockModels(), None
    else:
        log("preflight: model + grid service /health")
        preflight(log)
        log("connecting ROS backend (MoveGroup /move_action, wait <=10s) ...")
        from vision_pipeline.backends.ros_backend import RosBackend
        from vision_pipeline.monitor import Monitor
        backend, models = RosBackend(), ModelsHTTP()
        monitor = Monitor(backend=backend, grid_url="http://127.0.0.1:8815", log=log)
        log("ROS backend ready")
        log("preflight: MoveGroup (exactly one /move_action server)")
        try:
            backend.moveit_ready()
            log("  MoveGroup OK (single move_group)")
        except Exception as e:                                             # noqa: BLE001
            log("MOVEGROUP PREFLIGHT FAILED (arm NOT moved):\n" + str(e))
            return                                                         # fail fast, don't move
        log("preflight: camera streams (color + aligned_depth + camera_info)")
        try:
            backend.camera_ready()
            log("  camera OK (all 3 streams live)")
        except Exception as e:                                             # noqa: BLE001
            log("CAMERA PREFLIGHT FAILED (arm NOT moved):\n" + str(e))
            return                                                         # fail fast, don't move

    R = None
    try:
        use_hand_pc = a.get("hand_pc", "true").lower() in ("true", "1", "yes", "on")
        R = PlacePipeline(backend, models, log=log, debug_dir=run_dir, monitor=monitor,
                          complete_method=a.get("complete", "igr"),
                          use_hand_pc=use_hand_pc).run(scenario=scenario)
    except Exception:                                                      # noqa: BLE001
        log("PIPELINE FAILED:\n" + traceback.format_exc())

    if R and R.get("T_act") is not None:
        log(f"=== DONE scenario={scenario} ===")
        log(f"executed={R.get('executed')} move_code={R.get('move_code')} "
            f"upright_score={round(R.get('upright_score', 0), 3)} "
            f"EE_target_xyz={R['ee_target'][:3, 3].round(4).tolist()}")
        log(f"artifacts: {run_dir}  (images/*.png, clouds/*.ply, 40_candidates/, info.txt, timeline.log)")
        print("T_act:\n", R["T_act"].round(4), flush=True)

    # keep the RViz result alive (staged /place_debug/markers persist; re-publish so a
    # late RViz still shows them). Also holds even on partial failure so you can inspect.
    if monitor is not None and hold > 0:
        monitor.hold(hold)


if __name__ == "__main__":
    main()
