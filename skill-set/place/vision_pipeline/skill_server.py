"""Persistent '물체 내려놓기' (Place) skill server for the MOTIE integration scenario.

ONE command on the Current PC starts an always-on node that plays the Place role
(SEQ_PLACE = 4, the last step) of the shared sequence protocol arbitrated on the Control
PC. It never moves the arm to parent_pose / child_pose (unlike the standalone test in
GOAL_TEST.md); instead it slots into the long-horizon chain:

    Pick(1) → Inhand(2) → Stiffness(3) → **Place(4)**

Per fruit it does two things:

  * PREWARM (Phase A): the moment the run starts (seq PICK goes RUNNING) it captures the
    parent/destination RGB-D and does all parent-only perception (tray cloud, holes,
    waypoint height). This overlaps the grasp/in-hand/stiffness stages, which run on the
    other PCs — so by Place's turn only the child-dependent work is left. No arm motion.

  * EXECUTE (Phase B): at Place's turn it `wait_for_previous_done(STIFFNESS)`, acquires
    control (SequenceClient(4) → request_control + 2 Hz heartbeat), then — with the object
    already gripped and the arm already ~child_pose after stiffness — captures the child in
    place (NO child_pose move), clears the left arm, perceives + places, then release &
    retract to parent_pose. Normal exit releases control → `/sequence_state{4, DONE}`, which
    the orchestrator's arbiter observes cross-PC. A failure aborts (no DONE), so the chain
    never advances on a false success.

Then it loops for the next fruit. The heavy models stay resident (they are separate HTTP
services), so nothing reloads between fruits.

Integration wiring (see INTEGRATION.md):
  * The digital twin (move_group) runs ONLY on the 산업부 PC — this server must NOT launch
    one on the Current PC (that duplicate `/move_action` was the earlier all-moves-fail bug).
    The MoveGroup preflight asserts exactly one move_group exists.
  * Run inside the dex_ros Humble container with the kistar_ws overlay sourced (so
    `dual_arm_msgs` + `sequence_client` import) and the five model services + front camera up.

    python3 -u -m vision_pipeline.skill_server scenario=fruit hand_pc=true
"""
import os
import sys
import time
import traceback

from vision_pipeline.orchestrator import PlacePipeline
from vision_pipeline.models_client import ModelsHTTP
from vision_pipeline.run import HEALTH_PORTS, make_logger, parse_kv, preflight

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_sequence():
    """Import the sequence-protocol bits from the sourced kistar_ws overlay, with a loud,
    actionable message if the workspace isn't on the path (the #1 setup mistake)."""
    try:
        import rclpy                                                    # noqa: F401
        from rclpy.node import Node                                     # noqa: F401
        from rclpy.qos import DurabilityPolicy, QoSProfile              # noqa: F401
        from dual_arm_msgs.msg import SequenceState
        from sequence_client.client import (
            SequenceClient, PreviousAborted, ControlDenied, SequenceError)
    except ImportError as e:                                            # noqa: BLE001
        raise SystemExit(
            "cannot import the sequence protocol (dual_arm_msgs / sequence_client): "
            f"{type(e).__name__}: {e}\n"
            "Source the kistar_ws overlay inside the container first, e.g.\n"
            "  source /work/dex_ros/isaac-ros/kistar_ws/install/setup.bash\n"
            "and build it if needed (docker/build_ws.sh). These packages carry SequenceState + "
            "SequenceClient, the Place(4) handshake with the Control PC arbiter.")
    return SequenceState, SequenceClient, PreviousAborted, ControlDenied, SequenceError


class PlaceSkillServer:
    """Long-lived Place(4) skill node: watch the sequence, prewarm on grasp-start, execute on
    Place's turn, loop. Reuses PlacePipeline.prewarm()/execute() (no parent/child_pose moves)."""

    def __init__(self, scenario, use_hand_pc, log):
        self.scenario = scenario
        self.log = log
        (self.SequenceState, self.SequenceClient, self.PreviousAborted,
         self.ControlDenied, self.SequenceError) = _import_sequence()

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        self.rclpy = rclpy

        from vision_pipeline.backends.ros_backend import RosBackend      # inits rclpy + nodes
        from vision_pipeline.monitor import Monitor
        from std_msgs.msg import String
        self._String = String
        self.b = RosBackend()
        self.m = ModelsHTTP()
        self.monitor = Monitor(backend=self.b, grid_url="http://127.0.0.1:8815", log=log)
        # Milestone status feed: published so a 산업부 logger (place_logger.py in motie_ws) can
        # follow the place progress cross-PC (domain 9). std_msgs/String — no custom msg needed.
        self._status_pub = self.b.node.create_publisher(String, "/place/status", 10)
        # latched watcher on the arbiter's lifecycle bus (own node; spun in the wait loops).
        LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._watch = Node("place_skill_seq_watch")
        self._watch.create_subscription(
            self.SequenceState, "/sequence_state", self._on_seq, LATCHED)
        self._seq = None
        self.pipe = PlacePipeline(self.b, self.m, log=log, debug_dir=None, monitor=self.monitor,
                                  use_hand_pc=use_hand_pc, status=self.publish_status)

    def publish_status(self, msg):
        """Pipeline milestone -> /place/status (for the 산업부 place_logger) + the server's own log."""
        try:
            self._status_pub.publish(self._String(data=str(msg)))
        except Exception:                                                  # noqa: BLE001
            pass
        self.log("[status] " + str(msg))

    def _on_seq(self, msg):
        self._seq = msg

    # ── preflight ─────────────────────────────────────────────────────────────
    def preflight(self):
        """Fail FAST + clearly before we ever touch the arm: model services, exactly one
        move_group (on the 산업부 PC — not a duplicate here), and the live camera streams."""
        self.log("preflight: model + grid service /health")
        preflight(self.log)
        self.log("preflight: MoveGroup (exactly one /move_action server — the 산업부 twin)")
        self.b.moveit_ready()
        self.log("  MoveGroup OK (single move_group)")
        self.log("preflight: camera streams (color + aligned_depth + camera_info)")
        self.b.camera_ready()
        self.log("  camera OK (all 3 streams live)")
        self.log("preflight: sequence bus (/sequence_state, latched)")
        self._spin(1.0)
        self.log("  arbiter state: " + (self._fmt(self._seq) if self._seq is not None
                                        else "none yet (arbiter idle or not started) — will wait"))

    def _fmt(self, st):
        return f"seq_id={st.seq_id} state={st.state} owner={st.owner}"

    def _spin(self, t=0.2):
        self.rclpy.spin_once(self._watch, timeout_sec=t)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run_forever(self):
        self.log(f"place skill READY (scenario={self.scenario}). Serving Place(4) on each run; "
                 "Ctrl+C to stop.")
        n = 0
        while self.rclpy.ok():
            n += 1
            self.log(f"──── awaiting Place turn #{n} ────")
            try:
                self._serve_one()
            except KeyboardInterrupt:
                raise
            except Exception:                                          # noqa: BLE001
                self.log("SERVE ITERATION FAILED (continuing to next run):\n"
                         + traceback.format_exc())
                time.sleep(1.0)

    def _serve_one(self):
        SS = self.SequenceState
        R = None

        # Phase A — prewarm parent as soon as the run starts (seq PICK RUNNING). If we join a
        # run already past stiffness, skip straight to execute (its fallback perceives parent).
        self.log("waiting for a run to start (seq PICK RUNNING) to prewarm parent ...")
        self.publish_status("대기 중: 시나리오 시작(물체 잡기) 감지 대기")
        while self.rclpy.ok():
            self._spin(0.2)
            st = self._seq
            if st is None:
                continue
            if st.seq_id == SS.SEQ_PICK and st.state == SS.RUNNING:
                self.log("grasp started -> PREWARM parent (overlaps grasp/in-hand/stiffness)")
                try:
                    R = self.pipe.prewarm(self.scenario)
                    self.log("prewarm done (parent cloud + holes cached)")
                except Exception as e:                                 # noqa: BLE001
                    self.log(f"prewarm failed ({type(e).__name__}: {e}) -> perceive at execute")
                    R = None
                break
            if st.seq_id >= SS.SEQ_STIFFNESS and st.state == SS.DONE:
                self.log("joined mid-run (stiffness already DONE) -> execute without prewarm")
                break

        # Phase B — wait our turn, acquire control, execute, release (= DONE).
        seqcli = self.SequenceClient(SS.SEQ_PLACE)
        try:
            self.log("waiting for stiffness(3) DONE ...")
            self.publish_status("물성 추론(3) 완료 대기 중 (내 차례 대기)")
            seqcli.wait_for_previous_done(SS.SEQ_STIFFNESS)
        except self.PreviousAborted as e:
            self.log(f"stiffness aborted ({e}) -> chain won't advance; skip this run")
            seqcli.shutdown()
            return
        except self.SequenceError as e:
            self.log(f"sequence wait error ({type(e).__name__}: {e}) -> skip this run")
            seqcli.shutdown()
            return

        try:
            with seqcli:                    # request_control(4)+heartbeat; normal exit=release=DONE
                self.log("=== PLACE turn: control acquired (seq 4) ===")
                # GRIP HANDOVER — the object is already gripped and Stiffness(3) was streaming the
                # Position target that holds it. Control is ours now, and under
                # require_control:=true only the owner's targets are honoured, so WE take that
                # exact target over and keep it alive until the release path switches to Voltage.
                held = self.b.hand_hold_start()
                if held is None:
                    self.log("⚠ GRIP HANDOVER: no hand target seen on /hand/right/q_target — "
                             "holding nothing (the receiver keeps its latched target). A target is "
                             "never fabricated: a guessed 16-DoF pose would re-command the grip.")
                    self.publish_status("경고: 그립 target 인수인계 실패 — 수신 노드가 래치한 목표에 의존")
                else:
                    self.log(f"GRIP HANDOVER: holding the previous stage's hand target {held}")
                    self.publish_status("그립 인수인계 완료 - 직전 단계(물성 추론)의 hand target 유지 중")
                R = R if R is not None else {"scenario": self.scenario, "mode": "integration"}
                self.pipe.execute(self.scenario, R)
                self.log("place complete -> releasing control (/sequence_state{4, DONE})")
        except self.ControlDenied as e:
            self.log(f"control denied ({e}) — another owner holds it; skipping this run")
        except Exception:                                             # noqa: BLE001
            # __exit__ already aborted (heartbeat stopped, NO release) so no false DONE is emitted
            # and the arbiter reclaims to IDLE -> the chain sees Place failed, not done.
            self.log("PLACE FAILED (aborted, no DONE — arbiter will reclaim):\n"
                     + traceback.format_exc())
        finally:
            # Backstop: the release path and hand_safe_shutdown already stop the hold, but a run
            # that fails BEFORE release must not leave this thread republishing a stale target
            # after we have given up control.
            try:
                self.b.hand_hold_stop()
            except Exception:                                         # noqa: BLE001
                pass
            seqcli.shutdown()


def main():
    a = parse_kv(sys.argv[1:])
    scenario = a.get("scenario", "fruit")
    use_hand_pc = a.get("hand_pc", "true").lower() in ("true", "1", "yes", "on")
    log = make_logger()
    log(f"place skill server: scenario={scenario} hand_pc={use_hand_pc}")
    try:
        server = PlaceSkillServer(scenario, use_hand_pc, log)
    except SystemExit as e:
        log(str(e))
        return
    try:
        server.preflight()
    except Exception as e:                                             # noqa: BLE001
        log("PREFLIGHT FAILED (server NOT started, arm NOT moved):\n"
            + f"{type(e).__name__}: {e}")
        return
    try:
        server.run_forever()
    except KeyboardInterrupt:
        log("shutting down (Ctrl+C)")


if __name__ == "__main__":
    main()
