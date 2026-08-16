"""Headless check that the placement DESCENT is a STRAIGHT Cartesian line with the EE
orientation held FIXED (the arm-movement fix) — not an OMPL re-route that curves and
swings orientation. Run INSIDE the dex_ros container, AFTER bringing up the fake twin:

    ros2 launch franka_kistar_bringup dual_fr3_kistar_planning_pc_v2.launch.py \
        joint_state_mode:=fake use_rviz:=false &
    sleep 25
    python3 vision_pipeline/tools/verify_descent.py

It reads the CURRENT EE pose, builds a target 20 cm straight below it (same orientation),
asks the NEW `_cartesian_path` for the descent, FKs every trajectory point to an EE pose,
and asserts: x,y constant, z monotonically decreasing, orientation constant.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # host or /work
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/vision_pipeline/tools")
from vision_pipeline.backends.ros_backend import RosBackend, EE_LINK, ARM_JOINTS  # noqa: E402
import hand_fk  # noqa: E402


def ee_pose_of(positions, names):
    jv = {n: float(p) for n, p in zip(names, positions)}
    return hand_fk.fk(jv)[EE_LINK]


def main():
    b = RosBackend()
    T_now = b.tf(EE_LINK)                      # current EE pose (world)
    print("current EE pos:", np.round(T_now[:3, 3], 4).tolist())
    target = T_now.copy()
    target[2, 3] -= 0.20                       # 20 cm straight down, SAME orientation

    traj = b._cartesian_path(target)
    if traj is None or not traj.joint_trajectory.points:
        print("FAIL: cartesian path unavailable (fraction<0.9 or no service)")
        return
    names = list(traj.joint_trajectory.joint_names)
    pts = traj.joint_trajectory.points
    print(f"cartesian descent: {len(pts)} points, joints={names}")

    poses = [ee_pose_of(p.positions, names) for p in pts]
    xy = np.array([[T[0, 3], T[1, 3]] for T in poses])
    z = np.array([T[2, 3] for T in poses])
    R0 = poses[0][:3, :3]
    ang = [np.degrees(np.arccos(np.clip((np.trace(R0.T @ T[:3, :3]) - 1) / 2, -1, 1))) for T in poses]

    xy_drift = float(np.linalg.norm(xy - xy[0], axis=1).max()) * 1000    # mm
    z_mono = bool(np.all(np.diff(z) <= 1e-6))
    z_drop = float(z[0] - z[-1]) * 100                                   # cm
    ang_max = float(np.max(ang))                                        # deg
    print(f"  x,y drift max   = {xy_drift:.2f} mm     (should be ~0)")
    print(f"  z monotonic down= {z_mono}, total drop = {z_drop:.1f} cm  (~20)")
    print(f"  orientation max = {ang_max:.2f} deg     (should be ~0)")
    ok = xy_drift < 5 and z_mono and ang_max < 1.0
    print("RESULT:", "PASS — straight vertical descent, fixed orientation" if ok else "FAIL")


if __name__ == "__main__":
    main()
