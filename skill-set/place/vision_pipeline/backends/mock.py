"""Mock Backend + Models for offline testing of the orchestrator flow + frame math.
No ROS/robot/GPU. Generates geometrically-consistent synthetic data so
PlacePipeline.run() completes and yields a sane T_act / ee_target."""
import numpy as np

W, H = 64, 48
K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])


class MockBackend:
    """Camera == world (identity optical). Parent is a planar patch at z=0.6m."""

    def __init__(self):
        self.calls = []
        # Grip handover (Stiffness->Place). Kept OUT of `calls` on purpose: test_flow.py asserts
        # the exact motion-call sequence, and the hold is orthogonal bookkeeping.
        self.hold_calls = []
        self.handover_vec = [100.0] * 16      # pretend the previous stage streamed this target
        self.holding = None

    def move_to_joints(self, joints):
        self.calls.append(("move_joints", tuple(sorted(joints))))
        return True

    def start_move_to_joints(self, joints):
        self.calls.append(("start_move", tuple(sorted(joints))))
        return {"joints": joints}

    def wait_move(self, token):
        self.calls.append(("wait_move",))
        return True

    def move_to_ee_pose(self, T):
        self.calls.append(("move_ee", np.round(T[:3, 3], 3).tolist()))
        return True

    def start_move_left_arm(self, values):
        self.calls.append(("left_arm", [round(float(v), 3) for v in values]))
        return {"left": list(values)}

    def descend_decel(self, T, duration=4.0):
        self.calls.append(("descend_decel", np.round(T[:3, 3], 3).tolist(), duration))
        return True

    # ---- release/retract (2nd goal) — record calls; no real sensing/motion ----
    def paxini_ft(self):
        return np.array([[0, 0, 3.0], [0, 0, 2.5], [0, 0, 2.8], [0, 0, 2.6]])   # grasp holding

    def collect_paxini(self, duration=0.4, min_samples=8):
        return np.tile(self.paxini_ft(), (max(int(min_samples), 8), 1, 1))

    def descend_decel_monitored(self, T, should_stop, duration=4.0):
        self.calls.append(("descend_monitored", np.round(T[:3, 3], 3).tolist()))
        return "reached"                                    # mock: no disturbance -> Case 2

    def ascend_slow_monitored(self, T, should_stop, speed=0.02):
        self.calls.append(("ascend",))
        return "noload"                                     # mock: object released -> linear lift + retract

    def move_ee_linear(self, T, speed=0.08):
        self.calls.append(("move_ee_linear", np.round(T[:3, 3], 3).tolist()))
        return True

    def hand_grip_handover(self):
        return list(self.handover_vec) if self.handover_vec else None

    def hand_hold_start(self):
        v = self.hand_grip_handover()
        self.holding = v
        self.hold_calls.append(("hold_start", v))
        return v

    def hand_hold_stop(self, timeout=1.0):
        self.holding = None
        self.hold_calls.append(("hold_stop",))

    def hand_release_sequence(self, duty16, settle=0.3):
        self.hand_hold_stop()                  # mirrors RosBackend: counts stop before Voltage
        self.calls.append(("hand_release", list(duty16)))

    def hand_relax(self):
        self.calls.append(("hand_relax",))

    def hand_safe_shutdown(self):
        self.hand_hold_stop()
        self.calls.append(("hand_safe_shutdown",))

    def capture_rgbd(self):
        rgb = np.zeros((H, W, 3), np.uint8)
        depth = np.zeros((H, W), np.float32)
        depth[8:40, 12:52] = 0.6          # a dense planar patch (rest invalid)
        return rgb, depth, K

    def tf(self, frame):
        if frame == "right_palm":
            return np.eye(4)              # palm +x == world +x
        if frame == "right_fr3_link8":
            return _T(t=[0.40, 0.0, 0.50])
        if frame == "hand_root":
            return _T(t=[0.42, 0.0, 0.46])
        return np.eye(4)                 # camera == world

    def hand_q(self):
        return np.zeros(16)

    def fingertip_points(self):
        c = np.array([0.42, 0.0, 0.46])                # around the mock object
        return c + np.array([[.03, 0, .02], [-.03, 0, .02], [.02, .03, -.02], [.02, -.03, -.02]])


class MockModels:
    def molmo(self, rgb, prompt, multi=False):
        c = (W / 2, H / 2)
        return [c, (W / 2 + 6, H / 2 + 4)] if multi else [c]

    def sam(self, rgb, point):
        m = np.zeros((H, W), bool)
        m[8:40, 12:52] = True            # matches the planar patch
        return m

    def sam_text(self, rgb, text):       # fruit-tray text/concept path -> same mock mask
        return self.sam(rgb, None)

    def denoise(self, pc):
        return pc

    def complete(self, partial_pc):
        # small spherical object ~5cm diameter centered on the partial cloud
        rng = np.random.default_rng(0)
        v = rng.normal(size=(512, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * 0.025 + np.asarray(partial_pc, float).mean(0)

    def complete_igr(self, partial_pc, hand_q=None, contact_mode="points"):
        # IGR path: same mock sphere + 4 mock fingertip-pad contact centres (FK-only)
        dense = self.complete(partial_pc)
        c = np.asarray(partial_pc, float).mean(0)
        contacts = c + np.array([[0.024, 0, 0], [-0.024, 0, 0.005],
                                 [0, 0.024, 0.004], [0, -0.024, 0.004]])
        return dense, contacts

    def hand_pc_paxini(self, hand_q=None, num_points=2048):
        rng = np.random.default_rng(1)
        return rng.normal([0.42, 0.0, 0.46], 0.02, size=(int(num_points), 3))

    def place(self, parent_pc, child_pc_zalign):
        # a few candidate relative transforms; one near-upright (R[2,2]~1)
        out = [np.eye(4)]
        for ang in (0.3, 0.8, 1.4):
            R = _rotx(ang)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [0.0, 0.0, 0.02]
            out.append(T)
        return np.asarray(out)


def _T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
