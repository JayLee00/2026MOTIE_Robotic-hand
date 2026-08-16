"""Offline world<-camera extrinsic (no robot needed for perception dev).

On the live robot this comes from TF (tf.txt static + URDF). Here we reproduce it
from the same two measured/fixed transforms so captured RGB-D can be back-projected
into world without bringing up the robot:

    world ≡ base                                  (world_to_base TF = identity)
    base -> right_fr3_link0 : Trans(0.066,-0.122,0.359) · Rot_x(0.785)   (URDF, §2)
    right_fr3_link0 -> cam  : tf.txt  T(B<-C), B = right_fr3_link0        (§9-C)
    world <- cam = (base->link0) · tf.txt

⚠ tf.txt's C is assumed to be the COLOR OPTICAL frame (z fwd, x right, y down) to
match RealSense aligned-depth back-projection. Validate the sign/orientation on a
known point at bring-up (§9-E); if flipped, the camera frame convention differs.
"""
import numpy as np

# base -> right_fr3_link0 (URDF dual_fr3_kistar_v2: right_arm_xyz / right_arm_rpy)
ARM_XYZ = np.array([0.066, -0.122, 0.359])
ARM_ROLL = 0.785  # +45deg about x

# tf.txt  T(B <- C), B = right_fr3_link0, C = front_cam optical (measured; DO NOT EDIT)
T_LINK0_CAM = np.array([
    [-0.00767363, -0.780895, 0.624615, 0.0253511],
    [-0.699944, -0.441905, -0.561069, 0.389148],
    [0.714156, -0.441501, -0.543192, 0.170463],
    [0.0, 0.0, 0.0, 1.0],
])


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def world_from_link0():
    T = np.eye(4)
    T[:3, :3] = _rotx(ARM_ROLL)
    T[:3, 3] = ARM_XYZ
    return T


def world_from_cam():
    """4x4 world<-camera_optical (matches the live TF chain world<-front_cam_optical_calib)."""
    return world_from_link0() @ T_LINK0_CAM


if __name__ == "__main__":  # sanity-check against a real capture
    import os
    from .pointcloud import backproject
    f = os.path.join(os.path.dirname(__file__), "..", "fixtures", "scene.npz")
    d = np.load(f)
    Twc = world_from_cam()
    pc = backproject(d["depth"], d["K"], Twc)
    print("world<-cam =\n", np.round(Twc, 4))
    print("scene world bbox  min:", np.round(pc.min(0), 3), " max:", np.round(pc.max(0), 3))
    print("camera world position:", np.round(Twc[:3, 3], 3))
    print("N points:", len(pc))
