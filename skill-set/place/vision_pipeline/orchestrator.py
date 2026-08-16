"""Place-pipeline orchestrator (Current PC). Steps 1-4 / 3-1..3-23 as a backend-agnostic
state machine. The flow branches by `scenario`:
  * table — parent = Molmo(place point) -> SAM3 interactive; single placement region.
  * fruit — parent = SAM3 TEXT prompt (tray); child hand-fused; many holes -> 5 farthest
            candidate regions; pick by -z cosine; IK-fail -> retry next candidate.
Common: child = Molmo(grasp) -> SAM3 -> raw child_pc_i -> DBSCAN child_pc_refined ->
Act-VH IGR completion -> local_crop_size -> T_zalign -> AnyPlace -> T_act -> via T_preplace,
a collision-free waypoint (parent z_max +30cm) + decelerating descent.

Prompts come from scenario_molmo_prompt.txt (vision_pipeline/prompts.py). Robot I/O
(MoveIt/TF/RGB-D) is behind `Backend`; perception models behind `Models` (HTTP services)
— so the flow + frame math are testable offline with mocks (backends/mock.py, test_flow.py);
the real ROS2 backend (backends/ros_backend.py) is a drop-in.

    pipe = PlacePipeline(backend, models, pose_lib="franka_pose.yaml")
    result = pipe.run(scenario="table")   # -> dict incl T_act, ee_target
"""
import os

import numpy as np
import yaml

from vision_pipeline.core import geometry as G
from vision_pipeline.core import pointcloud as PC
from vision_pipeline import prompts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Placement: a candidate is "good enough" to execute when its best pose's upright cosine
# (rank_upright R[2,2], = cos to world -z) reaches this. Tried random points until one hits it.
COS_MIN = 0.85

# Parent crop fed to AnyPlace (step 3-17). `local_crop` is just the grasped-object footprint;
# a small margin widens the crop so AnyPlace sees a bit more surrounding parent context (tray
# rim / neighbouring holes). Kept MODEST on purpose: AnyPlace normalizes by the parent-crop
# bbox extent, so a LARGE margin shrinks the child in its unit box and degraded orientation in
# earlier sweeps (the "all 20 cos >= 0.8" upright state was with a tight crop). Penetration is
# still handled by the contact projection below, not by the crop. THICKNESS stays 0 (the
# downward extrude hurt orientation). Sweep both via tools/crop_thickness_sweep.py.
CROP_MARGIN = 0.02         # m — added on every side (half-extent = 0.5*local_crop + CROP_MARGIN)
CROP_THICKNESS = 0.0

# Contact projection (step 3-20b). AnyPlace is a learned pose regressor with NO non-penetration
# constraint (nothing in its loss/inference avoids collision), so its selected pose drives the
# object INTO the surface. After picking the placement we RESOLVE that geometrically: lift the
# placed object straight up (world +z, anti-gravity) to first contact with the REAL parent
# surface (geometry.contact_project), so it rests on the tray (and still nestles into empty
# holes). A pure vertical shift keeps the T_act orientation.

# rgb_parent tone reduction (step 3-1). The live parent frame is over-bright so Molmo can't
# read the tray-hole texture/contrast. This is NOT a per-RGB brightness change: measured against
# the user's target 00_parent_rgb_changed.png, the CHROMA is preserved (YCrCb Cr/Cb, LAB a/b
# differ ~0.7) and only LUMINANCE (Y) is remapped by a tone curve — shadows/mids ×~0.70, with a
# soft highlight rolloff. Stored as that curve's (in-luma -> out-luma) control points; applied to
# Y only (chroma untouched). Reproduces the target to the 8-bit floor (MAE ~0.7 vs 5.7 for 0.9o-7).
_TONE_X = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255],
                   np.float32)
_TONE_Y = np.array([1, 11.2, 22.5, 34, 45, 56.7, 68.2, 80.1, 92.6, 105, 118.1, 132, 146.9, 162.9,
                    181.1, 204.3, 252.1], np.float32)
_TONE_LUT = np.clip(np.interp(np.arange(256), _TONE_X, _TONE_Y), 0, 255).astype(np.uint8)


def reduce_brightness(rgb):
    """Darken rgb_parent to match the user's target — a LUMINANCE tone curve (chroma preserved),
    NOT a per-RGB brightness change: RGB->YCrCb, remap Y via _TONE_LUT, keep Cr/Cb, ->RGB."""
    import cv2
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = _TONE_LUT[ycc[:, :, 0]]
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


# INVARIANT guard: the hand is grasping the object, so the FK fingertip (지두) contacts must
# lie on/near the camera-observed object partial. If the median contact->partial gap exceeds
# this, the object cloud is wrong — usually BAD OBJECT DEPTH (specular/shiny object → RealSense
# holes → the mask back-projects to the background), occasionally a bad extrinsic or mis-seg.
# Abort with that diagnosis instead of completing/placing from an inconsistent object cloud.
CONTACT_GAP_MAX = 0.12  # m

# RViz marker colours (r,g,b 0-255) for the staged reveal.
VC = {"parent": (170, 170, 170), "child_i": (255, 140, 40), "child_com": (60, 120, 255),
      "child_zalign": (230, 40, 40), "region": (40, 220, 40), "placed": (255, 240, 40),
      "hole": (255, 200, 0)}

# LEFT arm "clear" pose (7 joints) — fired async right after the child RGB-D capture (the
# earliest safe point: the left+right arms share one MoveGroup /move_action server, so
# firing it during the child move would preempt that move) so it clears while the pipeline
# keeps running. The right-arm pre-placement pose (spec 'T_preplace') lives in franka_pose.yaml.
LEFT_ARM_POSE = [-0.1977, -1.2730, -0.4984, -2.3548, 0.2991, 0.9482, -0.8933]
# Descent time budget (s): the collision-free descent to T_act decelerates over ~this long
# regardless of how far it descends (velocity scales with distance) — see retime_decel.
DESCENT_TIME = 4.0

# ── Release & Retract (2nd goal) ─────────────────────────────────────────────
# After the placement descent reaches T_act (Case 2) OR is stopped early by a PaXini
# disturbance (Case 1), switch the KISTAR hand to Voltage mode, weakly open the fingers, and
# slowly ascend along the descent direction until the object leaves the hand (PaXini no-load),
# then retract linearly to the collision-free waypoint and home to parent_pose.
#
# Weak-open duty (16-DoF order thumb_0..3,index_0..3,middle_0..3,ring_0..3): +60 PWM duty on
# the 11 flexion joints the user specified (thumb_2/3, {index,middle,ring}_1/2/3), 0 on the
# abduction j0's + thumb_0/1. In Voltage mode the raw duty passes straight through to the
# firmware (verified — no per-joint sign flip). On this hand, POSITIVE duty = the GUI-slider
# "open/extend" direction (hardware-confirmed by the operator), so +60 gently OPENS these
# joints. Kept tiny on purpose ("아주 살살"); the backend hard-caps every published duty to
# RELEASE_DUTY_ABS_MAX so a mistuned constant can never drive the hand hard. Firmware range ±2100.
RELEASE_DUTY = [0, 0, 60, 60,   0, 60, 60, 60,
                0, 60, 60, 60,   0, 60, 60, 60]
ASCENT_SPEED = 0.02        # m/s — slow release ascent (Cartesian)
RETRACT_SPEED = 0.08       # m/s — normal-speed linear retract to the waypoint

# Tactile trigger thresholds (exposed here for live tuning). The grasp baseline captured at
# the waypoint SEEDS an online rolling reference; a disturbance = some fingertip force
# deviating from that *live* reference by > max(TAC_DIST_FLOOR, TAC_DIST_K * robust_sigma);
# no-load = ALL fingers below TAC_NOLOAD_N. Debounced (N consecutive samples). The adaptive
# reference (window/guard/min-ref) tracks the descent's own gravity/inertia drift, so
# only ABRUPT force changes (a collision / overshoot) fire it. See core/tactile.ForceMonitor.
TAC_BASELINE_S = 0.4       # s — grasp-baseline window collected at the waypoint (seeds the reference)
TAC_DIST_K = 4.0           # disturbance = dev > max(floor, k*sigma). Lowered 5.0->4.0 for a MODEST
                           # sensitivity bump (catch a lighter / earlier touch); robust MAD sigma
                           # keeps 4*sigma a solid margin over normal grasp noise.
TAC_DIST_FLOOR = 0.35      # N — absolute floor. Lowered 0.4->0.35: the adaptive reference TRACKS the
                           # ~0.36 N inertial bias, so the clean-descent RESIDUAL is ~0.1 N and 0.35
                           # still clears it while catching lighter contact. (If the descent ever
                           # stops spuriously at the START — highest velocity — nudge back toward 0.4.)
TAC_DIST_DEBOUNCE = 2      # consecutive samples over threshold (was 3). ~22ms at 90Hz -> faster stop
                           # on contact; two consecutive white-noise samples over threshold is unlikely.
TAC_NOLOAD_N = 0.6         # N — all fingers below this = no-load (object gone)
TAC_NOLOAD_DEBOUNCE = 5    # consecutive no-load samples
TAC_WIN = 40               # rolling-window length for the online reference (samples)
TAC_GUARD = 4              # newest samples excluded from the reference (so onset can't self-mask)
TAC_MIN_REF = 12           # min reference samples before the disturbance trigger arms
TAC_LIVE_FLOOR = 0.3       # N — a real grasp's max per-finger |F| must exceed this (liveness)


class PlacePipeline:
    def __init__(self, backend, models, pose_lib=None, log=print, debug_dir=None, monitor=None,
                 complete_method="igr", use_hand_pc=True, status=None):
        self.b = backend
        self.m = models
        self.log = log
        self.status_cb = status                    # optional: publish short milestones (integration)
        self.debug_dir = debug_dir
        self.complete_method = complete_method     # "igr" (Act-VH) | "sphere"
        self.use_hand_pc = use_hand_pc             # fuse the PaXini hand cloud into child_pc (arg)
        from vision_pipeline.monitor import NullMonitor
        self.mon = monitor or NullMonitor()
        pose_lib = pose_lib or os.path.join(REPO, "franka_pose.yaml")
        with open(pose_lib) as f:
            self.poses = (yaml.safe_load(f) or {}).get("poses", {})

    def _joints(self, name):
        if name not in self.poses:
            raise KeyError(f"pose '{name}' not in franka_pose.yaml ({list(self.poses)})")
        return self.poses[name]["joints"]

    def _status(self, msg):
        """Emit a short, human-readable milestone to the integration status feed (e.g. the
        산업부 place_logger). No-op in the standalone test where no sink is wired; the verbose
        self.log timeline is separate and unaffected."""
        if self.status_cb:
            try:
                self.status_cb(msg)
            except Exception:                                              # noqa: BLE001
                pass

    def _tip_contacts(self):
        """Fingertip-tip contact points via TF (used by the sphere fallback). The IGR
        path instead uses fingertip-PAD contacts computed service-side."""
        try:
            c = self.b.fingertip_points()
            return c if (c is not None and len(c)) else None
        except Exception as e:                                             # noqa: BLE001
            self.log(f"[3-10b] fingertip contacts unavailable ({type(e).__name__}: {e})")
            return None

    def _hand_q(self):
        """Live KISTAR hand joints (16, rad) for the FK contacts + hand cloud. None ->
        the service falls back to the recorded grasp (kistar_pose/tmp_pose.txt)."""
        try:
            q = self.b.hand_q()
            return q if (q is not None and len(q) >= 16) else None
        except Exception as e:                                             # noqa: BLE001
            self.log(f"[hand] live hand_q unavailable ({type(e).__name__}: {e}) -> recorded grasp")
            return None

    def _build_force_monitor(self):
        """Capture a PaXini grasp-baseline at the (stable) waypoint and build the trigger
        monitor. If PaXini is unavailable OR reads ~0 while grasping (writer not running /
        started after the grasp), the triggers CANNOT be trusted: degrade to a null monitor
        (descent runs pure Case-2 to T_act; release ascends to the waypoint) but say so
        LOUDLY — never silently pretend the tactile triggers are armed."""
        from vision_pipeline.core.tactile import ForceMonitor
        try:
            samples = self.b.collect_paxini(duration=TAC_BASELINE_S, min_samples=8)
        except Exception as e:                                             # noqa: BLE001
            self.log(f"[R0] ⚠ PaXini baseline unavailable ({type(e).__name__}: {e}) "
                     "-> tactile triggers DISABLED (release only at T_act, Case 2).")
            return ForceMonitor.null()
        if samples is None or len(samples) < 3:
            self.log("[R0] ⚠ PaXini stream silent -> tactile triggers DISABLED (release only "
                     "at T_act, Case 2). Is `paxini_writer.py --hand r` running?")
            return ForceMonitor.null()
        mon = ForceMonitor(
            k=TAC_DIST_K, abs_floor=TAC_DIST_FLOOR, noload_thresh=TAC_NOLOAD_N,
            dist_debounce=TAC_DIST_DEBOUNCE, noload_debounce=TAC_NOLOAD_DEBOUNCE,
            win=TAC_WIN, guard=TAC_GUARD, min_ref=TAC_MIN_REF, live_floor=TAC_LIVE_FLOOR)
        mon.prime(samples)
        gf = mon.grasp_force()
        if not mon.is_grasp_live():
            # A grasping hand cannot read ~0 on every finger. This means the tactile signal is
            # dead / mis-tared -> disable the triggers rather than fire a false no-load on the
            # first ascent sample (which would retract while still holding the object).
            self.log(f"[R0] ⚠ PaXini reads ~0 while grasping (|F| per finger = "
                     f"{np.round(gf, 2).tolist()} N) -> tactile signal not trustworthy; triggers "
                     "DISABLED. Run `paxini_writer.py --hand r` (tares on start) BEFORE grasping.")
            return ForceMonitor.null()
        self.log(f"[R0] tactile baseline: |F| per finger = {np.round(gf, 2).tolist()} N "
                 "(triggers armed: adaptive disturbance + no-load)")
        return mon

    def _project_place(self, T_pred, T_zalign, obj_zalign, parent_pc_full):
        """T_act after CONTACT PROJECTION (step 3-20b): AnyPlace's pose drives the object into the
        surface (no non-penetration term), so lift the placed object straight up to first contact
        with the real parent surface. A pure +z shift keeps orientation. Returns (T_act, dz)."""
        T_act = G.compose_T_act(T_pred, T_zalign)
        dz = 0.0
        if obj_zalign is not None and parent_pc_full is not None and len(parent_pc_full):
            try:
                dz, _ = G.contact_project(G.apply(T_pred, obj_zalign), parent_pc_full)
                if dz > 0:
                    lift = np.eye(4); lift[2, 3] = dz
                    T_act = lift @ T_act
            except Exception as e:                                          # noqa: BLE001
                self.log(f"[3-20b] contact projection skipped ({type(e).__name__}: {e})")
        return T_act, dz

    def _place_execute(self, T_pred, T_zalign, ee_cur, z_safe, moveit_error_name,
                       obj_zalign=None, parent_pc_full=None):
        """Execute one placement candidate: move to T_preplace, reach the target x,y at z_safe
        with the T_act orientation (collision-free waypoint), capture the PaXini grasp baseline,
        then a decelerating straight-line descent to T_act that is ABORTED early if a tactile
        disturbance is detected. Returns (code, executed_bool, ctx); IK/plan failure at the
        waypoint or the descent -> (code, False, None) so the caller can try another point. On a
        placed candidate ctx carries what the release/retract phase needs (monitor, eeg, wp)."""
        T_act, dz = self._project_place(T_pred, T_zalign, obj_zalign, parent_pc_full)
        if dz > 0:
            self.log(f"[3-20b] contact projection: lift {dz * 100:.1f}cm so the object RESTS on "
                     f"the parent (AnyPlace has no non-penetration term)")
        eeg = G.ee_target(T_act, ee_cur)
        self.log("[3-21] -> move T_preplace")
        self.b.move_to_joints(self._joints("T_preplace"))
        wp = eeg.copy()
        wp[2, 3] = max(float(eeg[2, 3]), z_safe)                            # lift to the safe height
        code_wp = self.b.move_to_ee_pose(wp)
        self.log(f"[3-22] waypoint z_safe={z_safe:.3f} -> {moveit_error_name(code_wp)}")
        if not (code_wp is True or code_wp == 1):
            return code_wp, False, None
        monitor = self._build_force_monitor()                              # baseline at the stable grasp
        descend_m = float(np.linalg.norm(wp[:3, 3] - eeg[:3, 3]))
        self.log(f"[3-22] monitored descend -> T_act ({descend_m * 100:.1f}cm, decel "
                 f"~{DESCENT_TIME:.0f}s, PaXini disturbance-watched)")
        self._status("배치 하강 시작 (T_act까지, PaXini 접촉 감시)")
        reason = self.b.descend_decel_monitored(eeg, monitor.disturbance, DESCENT_TIME)
        self.log(f"[3-22] descend stop reason: {reason}")
        self._status("하강 정지: " + {"disturbance": "접촉 감지(조기 정지)", "reached": "T_act 도달",
                                    "failed": "계획/실행 실패"}.get(reason, reason))
        if reason == "failed":
            return -1, False, None                                         # no path/exec -> try next point
        ctx = {"monitor": monitor, "eeg": eeg, "wp": wp, "z_safe": z_safe, "stop_reason": reason}
        return 1, True, ctx

    def _release_and_retract(self, ctx, R):
        """2nd goal: after the placement descent has stopped (Case 1 disturbance or Case 2
        arrival), release the grasped object and retract. Release = safe Position->Voltage
        switch + weak-open duty + slow ascent along the descent direction toward the
        collision-free waypoint, watching PaXini for no-load; the instant the object leaves
        (no-load) -> retract linearly to the waypoint at normal speed, then home to parent_pose.
        If no-load is never seen the ascent reaches the waypoint (object almost never still held
        there) and we retract anyway."""
        monitor, wp = ctx["monitor"], ctx["wp"]
        R["stop_reason"] = ctx["stop_reason"]

        # From here the hand may be in VOLTAGE mode. EVERY exit path — normal, exception, or
        # KeyboardInterrupt — must go through [R5] (back to Position + servo OFF), so the whole
        # block is a try/finally. Without it an escape between R1 and R5 would leave the hand in
        # Voltage; the skill server keeps running (it catches per-run exceptions and waits for the
        # next fruit), so the process-exit safety net in ros_backend would never fire and the NEXT
        # stage's Position counts would be read as raw duty -> runaway.
        try:
            # [R1] Release motion: Voltage mode + weakly open the fingers (safe switch sequence).
            self._status("물체 놓기(release) 시작 - 손가락 약하게 개방")
            self.log(f"[R1] release: Voltage-mode weak-open, duty={RELEASE_DUTY}")
            try:
                self.b.hand_release_sequence(RELEASE_DUTY)
            except Exception as e:                                         # noqa: BLE001
                self.log(f"[R1] hand release command failed ({type(e).__name__}: {e})")

            # [R2] Slow ascent (reverse of the descent) toward the waypoint, watching no-load.
            monitor.reset_noload()
            self.log(f"[R2] slow ascent -> waypoint ({ASCENT_SPEED} m/s), watching PaXini no-load")
            try:
                rr = self.b.ascend_slow_monitored(wp, monitor.noload, ASCENT_SPEED)
            except Exception as e:                                         # noqa: BLE001
                rr = "failed"
                self.log(f"[R2] ascent failed ({type(e).__name__}: {e})")
            R["release_reason"] = rr
            self.log("[R2] ascent stop: " + rr + (
                " — object released (no-load detected)" if rr == "noload"
                else " — reached waypoint still loaded; retracting anyway" if rr == "reached"
                else " — ascent planning/exec failed"))
            self._status("물체 이탈 확인됨 (no-load)" if rr == "noload"
                         else "waypoint 도달 (이탈 미확인) - 그대로 복귀" if rr == "reached"
                         else "상승 실패 - 복귀 시도")

            # [R3] Retract motion: lift STRAIGHT UP to the collision-free waypoint FIRST, so the
            # home move starts from a high, clear pose instead of a wide OMPL swing out of the tray
            # clutter. move_ee_linear is collision-checked + timed from the planned path length.
            lifted = True
            if rr != "reached":
                self.log(f"[R3] retract: linear lift -> waypoint ({RETRACT_SPEED} m/s)")
                try:
                    lifted = self.b.move_ee_linear(wp, RETRACT_SPEED)
                except Exception as e:                                     # noqa: BLE001
                    lifted = False
                    self.log(f"[R3] linear retract failed ({type(e).__name__}: {e})")
                if not lifted:
                    self.log("[R3] ⚠ clearance lift FAILED (no straight-line path up) — arm still "
                             "LOW in the clutter; homing SLOWLY from here. Keep E-stop ready.")

            # [R4] home to parent_pose. Lifting to the waypoint FIRST (R3) means this starts from a
            # high, clear pose so the free OMPL joint path is far less likely to swing wide/low.
            self._status("retract: parent_pose(홈) 복귀" + ("" if lifted else " (상승 실패)"))
            self.log("[R4] retract -> parent_pose")
            try:
                self.b.move_to_joints(self._joints("parent_pose"))
            except Exception as e:                                         # noqa: BLE001
                self.log(f"[R4] parent_pose move failed ({type(e).__name__}: {e})")
        finally:
            # [R5] leave the hand SAFE for the NEXT run: zero the Voltage drive, return to POSITION
            # mode + servo OFF. If a run ends in Voltage mode, the next run's position targets
            # (counts) are read as raw duty -> the hand runs away. servo-OFF-first switch.
            try:
                self.b.hand_safe_shutdown()
            except Exception as e:                                         # noqa: BLE001
                self.log(f"[R5] hand safe-shutdown failed ({type(e).__name__}: {e})")
        R["retract_done"] = True
        self._status("물체 내려놓기 완료 (hand: Position 모드 + 서보 OFF)")
        self.log("[R5] release + retract complete (hand -> Position mode, servo OFF)")

    def _save(self, R):
        """Dump artifacts even on failure (shared by the test run + the integration skill)."""
        if self.debug_dir:
            try:
                from vision_pipeline import debug
                debug.save_run(self.debug_dir, R, log=self.log)
            except Exception as e:                                # noqa: BLE001
                self.log(f"[debug] save_run failed ({type(e).__name__}: {e})")

    def run(self, scenario):
        """Standalone TEST entry (GOAL_TEST.md): parent_pose -> child_pose -> place -> release."""
        R = {"scenario": scenario, "mode": "test"}                # collected intermediates
        try:
            self._run(scenario, R)
            return R
        finally:
            self._save(R)

    def _run(self, scenario, R):
        """Test flow: parent_pose -> (capture parent, perceive parent while the arm moves to
        child_pose) -> capture child -> perceive + place -> release & retract."""
        assert scenario in ("table", "fruit"), scenario
        self.log("[1] move -> parent_pose")
        assert self.b.move_to_joints(self._joints("parent_pose")), "parent_pose move failed"
        self._capture_parent(R)
        self.log("[3-2] fire async move -> child_pose")
        child_tok = self.b.start_move_to_joints(self._joints("child_pose"))
        self._perceive_parent(scenario, R)                        # overlaps the child move
        self.log("[3-7] wait child_pose arrival + capture")
        assert self.b.wait_move(child_tok), "child_pose move failed"
        self._capture_child(R)
        left_tok = self._fire_left_clear()
        executed, place_ctx = self._perceive_child_and_place(scenario, R, left_tok)
        if executed and place_ctx is not None:
            self._release_and_retract(place_ctx, R)

    # ── integration skill entry points (no parent_pose / child_pose moves) ───────────────────
    def prewarm(self, scenario):
        """Integration PHASE A (prewarm): parent(destination)-only perception, fired the moment the
        grasp stage starts so it overlaps grasp/in-hand/stiffness. No arm motion. Returns R which is
        handed to execute(). See INTEGRATION.md."""
        assert scenario in ("table", "fruit"), scenario
        R = {"scenario": scenario, "mode": "integration"}
        self.log("[P-A] prewarm: capture parent + parent perception (no arm move)")
        self._status("parent(내려놓을 곳) 사전 인지(prewarm) 시작")
        self._capture_parent(R)
        self._perceive_parent(scenario, R)
        R["prewarmed"] = True
        self._status("parent 인지 완료 (트레이 클라우드 + 배치 후보 확보)")
        return R

    def execute(self, scenario, R):
        """Integration PHASE B: at Place's turn the arm is already ~child_pose (post-stiffness), so
        there is NO child_pose move — capture the child in place, clear the left arm, perceive +
        place + release + retract to parent_pose. R comes from prewarm()."""
        try:
            self._status("물체 내려놓기 실행 시작 (Place 차례)")
            if not R.get("prewarmed"):
                self.log("[P-A] execute: no prewarm cached -> perceiving parent now (slower)")
                self._status("parent 사전 인지 없음 -> 지금 인지 (다소 느림)")
                self._capture_parent(R)
                self._perceive_parent(scenario, R)
            self.log("[3-7] capture child in place (no child_pose move — arm presented post-stiffness)")
            self._capture_child(R)
            left_tok = self._fire_left_clear()
            executed, place_ctx = self._perceive_child_and_place(scenario, R, left_tok)
            if executed and place_ctx is not None:
                self._release_and_retract(place_ctx, R)
            return R
        finally:
            self._save(R)

    # ── perception building blocks (shared by _run + prewarm/execute) ────────────────────────
    def _capture_parent(self, R):
        """[3-1] Capture the parent (destination) RGB-D + darken the COLOR (the live frame is
        over-bright, so Molmo can't read the tray-hole texture/contrast) — depth/K untouched, so
        back-projection is unaffected. camera->world uses the offline extrinsic (fixed camera)."""
        rgb_p, depth_p, K = self.b.capture_rgbd()
        rgb_p = reduce_brightness(rgb_p)
        T_wc = self.b.tf("camera")                                # world<-cam_optical (tf.txt chain)
        R.update(rgb_parent=rgb_p, depth_parent=depth_p, K=K, T_world_cam=T_wc)
        self.mon.img_raw("a11", rgb_p)
        self.mon.img_depth("a12", depth_p)

    def _perceive_parent(self, scenario, R):
        """[3-5/3-6 + 3-17] All parent-only perception: segment the tray -> parent_pc_full, then the
        placement candidate points (fruit: Molmo local_place holes; table: the single place point)
        and the collision-free waypoint height. No robot motion + no child dependency -> the
        integration skill runs this as PREWARM (overlapping grasp/in-hand/stiffness)."""
        rgb_p, depth_p, K, T_wc = R["rgb_parent"], R["depth_parent"], R["K"], R["T_world_cam"]

        # table: Molmo(place) point -> SAM3 interactive (PVS). fruit: SAM3 TEXT prompt (fallback
        # chain of tray names). [3-6] fills INTERIOR holes so the occluded surface is included.
        def _parent_pc(mask):
            return self.m.denoise(PC.backproject(depth_p, K, T_wc, mask=PC.fill_holes(mask)))
        if scenario == "table":
            self.log("[3-5-B-1] parent molmo (place point)")
            pts_parent = self.m.molmo(rgb_p, prompts.place_prompt("table"))
            assert pts_parent, "parent molmo returned no point (place prompt)"
            pt_parent = pts_parent[0]                                       # pixel
            self.mon.img_points("a13", rgb_p, [pt_parent], (255, 40, 40))
            self.log("[3-5-B-2] parent sam (interactive point->mask)")
            mask_p = PC.fill_holes(self.m.sam(rgb_p, pt_parent))
            parent_pc_full = self.m.denoise(PC.backproject(depth_p, K, T_wc, mask=mask_p))
            p_parent_w = PC.backproject_pixel(pt_parent, depth_p, K, T_wc)  # world place point
        else:                                                              # fruit
            # SAM3 text grounding is prompt-sensitive per scene, so try a FALLBACK CHAIN of tray
            # prompts until one yields a non-empty tray cloud (a single prompt returns an EMPTY
            # mask on some trays -> parent_pc_full empty -> the pipeline used to crash at z_max).
            pt_parent, p_parent_w, mask_p, parent_pc_full = None, None, None, None
            for tp in prompts.SAM_FRUIT_TRAY_PROMPTS:
                m = np.asarray(self.m.sam_text(rgb_p, tp))
                if m.sum() == 0:
                    self.log(f"[3-5-A] tray prompt '{tp}' -> empty mask, next")
                    continue
                pc = _parent_pc(m)
                if len(pc):
                    self.log(f"[3-5-A] tray prompt '{tp}' -> {int(m.sum())} px, {len(pc)} pts")
                    mask_p, parent_pc_full = PC.fill_holes(m), pc
                    break
                self.log(f"[3-5-A] tray prompt '{tp}' -> mask has no valid depth, next")
            if parent_pc_full is None or len(parent_pc_full) == 0:
                raise RuntimeError(
                    f"parent tray segmentation empty: SAM3 grounded no tray for any of "
                    f"{prompts.SAM_FRUIT_TRAY_PROMPTS}. Is the tray in view, and does one of these "
                    f"prompts name it? (add a matching one to prompts.SAM_FRUIT_TRAY_PROMPTS)")
        self.mon.img_mask("a14", rgb_p, mask_p, (40, 120, 255))
        R.update(pt_parent=pt_parent, mask_parent=mask_p,
                 parent_pc_full=parent_pc_full, p_parent_w=p_parent_w)
        self.mon.cloud("1_parent_full", parent_pc_full, VC["parent"], 1.0, 0.005)

        # [3-17] parent placement candidate POINTS (parent-only). fruit: ALL Molmo(local_place)
        # holes; table: the single place point. + collision-free waypoint height from the tray top.
        if scenario == "table":
            holes = [pt_parent]
        else:
            self.log("[3-17-A-1] molmo (empty tray holes, multi)")
            holes = self.m.molmo(rgb_p, prompts.local_place_prompt(), multi=True)
            self.mon.img_points("b11", rgb_p, holes, VC["hole"])
        z_max = float(np.asarray(parent_pc_full)[:, 2].max())
        R.update(holes=holes, parent_pc_full_z_max=z_max, z_safe=z_max + 0.30,
                 descent_time=DESCENT_TIME)

    def _capture_child(self, R):
        """[3-7] Capture the child (grasped-object) RGB-D. In the test run the arm was just moved to
        child_pose; in the integration skill it is already ~child_pose after stiffness."""
        self._status("RGB-D(child) 캡처 시작")
        rgb_c, depth_c, K2 = self.b.capture_rgbd()
        R.update(rgb_child=rgb_c, depth_child=depth_c, K_child=K2)
        self.mon.img_raw("a21", rgb_c)
        self.mon.img_depth("a22", depth_c)

    def _fire_left_clear(self):
        """[3-7b] Fire the LEFT arm to its clear pose (async). The left+right arms share ONE
        MoveGroup /move_action server, so the returned token must be WAITED before any placement
        move (else the right-arm goal preempts it mid-motion). It runs during perception (~free)."""
        self.log("[3-7b] fire async LEFT arm -> clear pose")
        try:
            return self.b.start_move_left_arm(LEFT_ARM_POSE)
        except Exception as e:                                             # noqa: BLE001
            self.log(f"[3-7b] left-arm move skipped ({type(e).__name__}: {e})")
            return None

    def _perceive_child_and_place(self, scenario, R, left_tok):
        """[3-8..3-23] Child perception (already captured into R) -> completion -> AnyPlace ->
        contact-projected placement. Returns (executed, place_ctx); the caller runs release/retract.
        Shared by the test run + the integration skill."""
        from vision_pipeline.debug import moveit_error_name
        parent_pc_full, holes = R["parent_pc_full"], R["holes"]
        rgb_p, depth_p, K, T_wc = R["rgb_parent"], R["depth_parent"], R["K"], R["T_world_cam"]
        rgb_c, depth_c, K2 = R["rgb_child"], R["depth_child"], R["K_child"]
        # placement candidate points (parent-only, cheap, deterministic — rebuilt from R["holes"])
        if scenario == "table":
            pts_src = [(R["pt_parent"], R["p_parent_w"])]
        else:
            pts_src = [(h, PC.backproject_pixel(h, depth_p, K, T_wc)) for h in holes]
        pts_src = [(h, c) for h, c in pts_src if c is not None]
        assert pts_src, "no valid placement point (check Molmo points / depth)"

        # [3-8/3-9/3-10] child perception: Molmo(grasp) point -> SAM3 interactive -> raw cloud
        self._status("child(물체) 인지 시작 (파지점 검출 + 분할 + 형상 완성)")
        self.log("[3-8] child molmo (grasp point)")
        pts_child = self.m.molmo(rgb_c, prompts.grasp_prompt())     # FIXED prompt (no grasp arg)
        assert pts_child, "child molmo returned no point (grasp prompt)"
        pt_child = pts_child[0]
        self.mon.img_points("a23", rgb_c, [pt_child], (40, 220, 40))
        self.log("[3-9] child sam (interactive point->mask)")
        mask_c = self.m.sam(rgb_c, pt_child)
        self.mon.img_mask("a24", rgb_c, mask_c, (255, 80, 200))
        child_pc_i = PC.backproject(depth_c, K2, T_wc, mask=mask_c)         # [3-10] RAW (no denoise)
        R.update(pt_child=pt_child, mask_child=mask_c, child_pc_i=child_pc_i)
        self.mon.cloud("2_child_i", child_pc_i, VC["child_i"], 1.0, 0.004)

        # [3-11] DBSCAN outlier removal -> child_pc_refined (removes disconnected floaters,
        # keeps the object surface — see core/outlier_removal).
        self.log("[3-11] DBSCAN refine -> child_pc_refined")
        child_pc_refined = self.m.denoise(child_pc_i)
        R["child_pc_refined"] = child_pc_refined
        hand_q = self._hand_q()     # live kistar hand joints -> FK contacts (3-12) + hand cloud (3-14)

        # [3-12] completion. Act-VH IGR: fits the refined partial + the 4 fingertip (지두)
        # fingerprint-pad contact centres (incl. the camera-occluded back) into a watertight
        # surface; contacts are FK from the LIVE grasp joints (object-independent).
        # `complete=sphere` (geometric fit + fingertip-tip contacts) is the fallback.
        self.log(f"[3-12] complete ({self.complete_method})")
        contact = None
        if self.complete_method == "igr":
            child_pc_com, contact = self.m.complete_igr(child_pc_refined, hand_q=hand_q,
                                                        contact_mode="cloud")
        else:
            from vision_pipeline.core.completion import complete_sphere
            contact = self._tip_contacts()
            child_pc_com = complete_sphere(child_pc_refined, contacts=contact)
        if contact is not None and len(contact):
            R["contact_points"] = contact
            self.mon.cloud("2b_contacts", contact, (255, 0, 255), 1.0, 0.014)
            # [3-12b] grasp-consistency guard. The hand grasps the object, so the FK contacts
            # must sit on/near the observed object partial. A large gap almost always means the
            # OBJECT DEPTH is bad: a shiny/specular object (e.g. an orange) returns no IR, so the
            # depth in its mask is holes/BACKGROUND — the mask then back-projects to the table/
            # tray behind it (~0.3–0.5 m away), not the object. We tell that apart from a genuine
            # extrinsic error by comparing the observed partial distance to the grasp distance:
            # observed >> expected ⇒ background bleed (bad depth), not the extrinsic.
            cam = T_wc[:3, 3]
            gap = float(np.median(np.linalg.norm(
                child_pc_refined[None] - contact[:, None], axis=2).min(axis=1)))
            d_exp = float(np.linalg.norm(contact.mean(0) - cam))         # camera->grasp (true object)
            d_obs = float(np.linalg.norm(child_pc_refined.mean(0) - cam))  # camera->observed partial
            R.update(contact_gap=gap, obj_depth_expected=d_exp, obj_depth_observed=d_obs)
            self.log(f"[3-12b] grasp gap = {gap * 100:.1f} cm | object depth observed "
                     f"{d_obs * 100:.0f} cm vs grasp-expected {d_exp * 100:.0f} cm")
            if gap > CONTACT_GAP_MAX:
                why = ("the observed object is FARTHER than the grasp -> object depth is invalid "
                       "(specular/shiny surface: RealSense returns holes, so the mask back-projects "
                       "to the BACKGROUND behind the object). Check the object region in the depth "
                       "image; fix the depth (matte/textured object, RealSense laser power, or "
                       "disable hole-filling). The extrinsic is NOT necessarily wrong."
                       if d_obs > d_exp + 0.10 else
                       "the object is not where the grasp is -> camera extrinsic miscalibrated "
                       "(recalibrate camera->right_fr3_link0, §9-C/E) or the wrong object was segmented.")
                raise RuntimeError(
                    f"grasped-object partial is {gap * 100:.0f} cm from the FK fingertip contacts "
                    f"(hand grasps it -> expect < {CONTACT_GAP_MAX * 100:.0f} cm). Observed object "
                    f"depth {d_obs * 100:.0f} cm vs grasp-expected {d_exp * 100:.0f} cm: {why}")
        self.mon.cloud("3_child_com", child_pc_com, VC["child_com"], 1.0, 0.004)
        self.mon.alpha("2_child_i", 0.25)

        # [3-13] palm normal + local crop size (max on-plane diameter, plane ⊥ palm height axis)
        self.log("[3-13] palm normal (tf right_palm) + local_crop_size")
        n_palm = PC.axis_from(self.b.tf("right_palm"), axis=(1, 0, 0))      # live right_palm +x (§9-B)
        local_crop = G.local_crop_size(child_pc_com, n_palm)

        # [3-14] hand-cloud fusion — gated by the `hand_pc` run arg (NOT the scenario). When on,
        # fuse the PaXini-URDF-FK hand cloud (live grasp joints) into child_pc, save it to the
        # run log, and visualize it in RViz. Contact-based completion is unaffected either way.
        child_pc = child_pc_com
        if self.use_hand_pc:
            self.log("[3-14] hand_pc=true: PaXini hand cloud (live-joint FK) + fuse")
            try:
                # match child_pc_com density (was a sparse 2048) so the fused child cloud fed
                # to AnyPlace isn't hand-under-represented.
                hand_pc = self.m.hand_pc_paxini(hand_q=hand_q, num_points=len(child_pc_com))
                child_pc = np.concatenate([child_pc_com, hand_pc], axis=0)
                R["hand_pc"] = hand_pc
                self.mon.cloud("2c_hand_pc", hand_pc, (180, 180, 195), 1.0, 0.004)
            except Exception as e:                                         # noqa: BLE001
                self.log(f"[3-14] hand fusion skipped ({type(e).__name__}: {e}) — object-only")
        R.update(n_palm=n_palm, child_pc_com=child_pc_com, child_pc=child_pc, local_crop=local_crop)

        # [3-15/3-16] gravity-align (palm -> world -z)
        T_zalign = G.align_palm_down(child_pc, n_palm)
        child_pc_zalign = G.apply(T_zalign, child_pc)
        obj_zalign = child_pc_zalign[:len(child_pc_com)]     # OBJECT part (no hand) — for contact projection
        R.update(T_zalign=T_zalign, child_pc_zalign=child_pc_zalign)
        self.mon.cloud("4_child_zalign", child_pc_zalign, VC["child_zalign"], 1.0, 0.004)
        self.mon.alpha("3_child_com", 0.25)

        # ee_current = the EE pose NOW (observation). Rigid grasp -> for every candidate
        # EE_target = T_act @ EE_current holds; capture once (arm unmoved since the child capture).
        ee_cur = self.b.tf("right_fr3_link8")
        z_safe = R["z_safe"]                                                # collision-free waypoint height (prewarmed)
        R["ee_current"] = ee_cur

        # left-arm clear must FINISH before any placement move (left+right share ONE MoveGroup
        # server; else the right-arm goal preempts it mid-motion). ~free: it ran during perception.
        if left_tok is not None:
            self.log("[4] wait for left-arm clear to finish (shared move server)")
            try:
                self.b.wait_move(left_tok)
            except Exception as e:                                         # noqa: BLE001
                self.log(f"[4] left-arm wait skipped ({type(e).__name__}: {e})")

        # [3-18..3-23] RANDOM-POINT loop: pick a random hole, crop parent_pc there, run AnyPlace,
        # take its best (max upright-cos) pose. If cos >= COS_MIN try to place it; on IK/plan
        # failure try ANOTHER random point. Repeat until one places or all points are exhausted;
        # then place the highest-cos candidate seen (best-first, skipping ones whose IK failed).
        self._status("AnyPlace 배치 자세 추론 시작")
        order = list(range(len(pts_src)))
        np.random.default_rng().shuffle(order)                             # random order, no repeats
        tried = []                                                         # {cos,T_pred,region,center,hole,ik_failed}
        code, executed, chosen, place_ctx = None, False, None, None
        for n, i in enumerate(order):
            h, c = pts_src[i]
            region = G.thicken(G.crop_region(parent_pc_full, c, local_crop, axis=G.WORLD_DOWN,
                                              margin=CROP_MARGIN), CROP_THICKNESS)
            if len(region) < 16:                                           # empty crop -> skip
                self.log(f"[3-17] point {n + 1}/{len(order)}: empty crop -> skip")
                continue
            self.log(f"[3-18] point {n + 1}/{len(order)}: anyplace ({len(region)} pts)")
            try:
                cand = np.asarray(self.m.place(region, child_pc_zalign))   # (K,4,4)
            except Exception as e:                                         # noqa: BLE001 (skip on service error)
                self.log(f"[3-18] anyplace failed ({type(e).__name__}: {str(e)[:120]}) -> next point")
                continue
            k, sc = G.rank_upright(cand)                                    # best pose of this crop
            rec = {"cos": float(sc[k]), "T_pred": cand[k], "region": region,
                   "center": c, "hole": h, "ik_failed": False}
            tried.append(rec)
            self.log(f"[3-19] point {n + 1}: best cos={rec['cos']:.3f} "
                     f"({'>=' if rec['cos'] >= COS_MIN else '<'} {COS_MIN})")
            if rec["cos"] >= COS_MIN:
                code, ok, ctx = self._place_execute(rec["T_pred"], T_zalign, ee_cur, z_safe,
                                                    moveit_error_name, obj_zalign, parent_pc_full)
                if ok:
                    executed, chosen, place_ctx = True, rec, ctx
                    break
                rec["ik_failed"] = True
                self.log("[3-23-A] cos ok but IK/plan failed -> another random point")

        # fallback: no point reached COS_MIN with a working IK -> place the highest-cos candidate,
        # but ONLY among palm-DOWN ones (cos>0). The selection targets the pose most aligned with
        # gravity (rank_upright = signed cos to world -z); a palm-UP pose (cos<=0) is never placed
        # — it means the hand is below the object and would collide, so we'd rather fail than place
        # it (with the shared MoveGroup + avoid_collisions this also can't move through the table).
        palm_down = [r for r in tried if not r["ik_failed"] and r["cos"] > 0]
        if not executed and palm_down:
            self.log(f"[3-20] none reached cos>={COS_MIN}+IK; placing best palm-down of "
                     f"{len(palm_down)} (of {len(tried)} tried)")
            for rec in sorted(palm_down, key=lambda r: -r["cos"]):
                code, ok, ctx = self._place_execute(rec["T_pred"], T_zalign, ee_cur, z_safe,
                                                    moveit_error_name, obj_zalign, parent_pc_full)
                if ok:
                    executed, chosen, place_ctx = True, rec, ctx
                    break
        elif not executed:
            self.log(f"[3-20] no palm-down (cos>0) placement among {len(tried)} candidate(s) "
                     f"— NOT placing (would be palm-up/colliding)")

        # record + RViz the chosen (or, if none placed, the best-cos) candidate. `candidates` is
        # one best-pose per tried point (1:1 with regions), for the debug 40_candidates grid.
        R.update(regions=[t["region"] for t in tried], centers=[t["center"] for t in tried],
                 scores=np.array([t["cos"] for t in tried]),
                 candidates=(np.array([t["T_pred"] for t in tried]) if tried
                             else np.zeros((0, 4, 4))),
                 cand_region=np.arange(len(tried)))
        show = chosen or (max(tried, key=lambda r: r["cos"]) if tried else None)
        if show is not None:
            R["k_top"] = next((idx for idx, t in enumerate(tried) if t is show), 0)
            T_act, dz = self._project_place(show["T_pred"], T_zalign, obj_zalign, parent_pc_full)
            eeg = G.ee_target(T_act, ee_cur)
            R.update(T_pred=show["T_pred"], T_act=T_act, ee_target=eeg, upright_score=show["cos"],
                     contact_lift=dz)
            self.mon.alpha("1_parent_full", 0.2)
            self.mon.cloud("5_region_selected", show["region"], VC["region"], 1.0, 0.006)
            if scenario == "fruit" and show["hole"] in holes:
                self.mon.img_points("b21", rgb_p, holes, VC["hole"], sel=holes.index(show["hole"]))
            # placed cloud AFTER contact projection (+z lift) — what actually gets executed
            self.mon.cloud("6_placed_top1", G.apply(show["T_pred"], child_pc_zalign) + [0, 0, dz],
                           VC["placed"], 1.0, 0.004)
            self.mon.ee(eeg)
        R["move_code"] = code
        R["executed"] = executed
        self.log(f"[4] execute -> {moveit_error_name(code)} (executed={executed})")

        # Return to the caller (_run test / execute integration), which runs release & retract
        # only if a placement actually executed (descent reached T_act or disturbance-stopped).
        return executed, place_ctx
