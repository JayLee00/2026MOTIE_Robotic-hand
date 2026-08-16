"""Act-VH IGR completion service (run in conda env `anyplace_cu128`) — the live
pipeline's object shape-completion.

    POST /predict  npz{ partial:(N,3) meters,
                        hand_q?:(16,),                 # default = kistar_pose/tmp_pose.txt
                        arm_pose?:str }                # default = "child_pose"
                ->  npz{ dense:(P,3) meters, contacts:(4,3) meters }

The service computes the fingertip contact points itself (URDF FK + PaXini tip mesh,
host-side trimesh) so the ROS container needs neither meshes nor trimesh — it just
ships the partial cloud. Contacts = the 4 PaXini fingertip (지두) fingerprint-pad
CENTRES, from FK only (object-INDEPENDENT — the real pipeline never knows the object
surface). IGR then fits the partial + those 4 contacts.

Run:  conda activate anyplace_cu128 && python vision_pipeline/services/igr_service.py [--port 8816]
"""
import argparse
import os
import sys

import numpy as np

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place"
sys.path.insert(0, REPO)

from vision_pipeline.tools import hand_fk  # noqa: E402


def _joint_values(arm_pose, hand_q):
    """{joint: rad} for FK — arm from franka_pose.yaml[arm_pose], hand from hand_q
    if given else kistar_pose/tmp_pose.txt (the recorded grasp)."""
    if hand_q is None:
        return hand_fk.joint_values(arm_pose=arm_pose)
    import yaml
    arm = yaml.safe_load(open(os.path.join(REPO, "franka_pose.yaml")))["poses"][arm_pose]["joints"]
    jv = dict(arm)
    jv.update({n: float(hand_q[i]) for i, n in enumerate(hand_fk.HAND_JOINTS)})
    return jv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8816)
    a = ap.parse_args()

    import torch
    from vision_pipeline.core import igr_complete as IGR
    from vision_pipeline.services.rpc import serve

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[igr] device {device} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    print("[igr] building IGR network ...", flush=True)
    net = IGR.load_network(device)
    print("[igr] ready", flush=True)

    def predict(inp):
        arm_pose = str(inp["arm_pose"]) if "arm_pose" in inp else "child_pose"
        hand_q = inp["hand_q"] if "hand_q" in inp else None
        jv = _joint_values(arm_pose, hand_q)

        # hand_pc mode (step 3-14-A-1): PaXini-URDF FK hand surface cloud in world
        if "mode" in inp and str(inp["mode"]) == "hand_pc":
            n = int(inp["num_points"]) if "num_points" in inp else 2048
            hpc = hand_fk.hand_point_cloud(jv, n=n)
            print(f"[igr] hand_pc: {len(hpc)} pts (PaXini URDF FK)", flush=True)
            return {"hand_pc": np.asarray(hpc, np.float32)}

        # completion (step 3-12). The caller (orchestrator [3-11]) already DBSCAN-refined
        # the partial, so denoise_partial=False (no re-clean).
        partial = np.asarray(inp["partial"], np.float64)
        # contact_mode: "points" = 4 PaXini fingerprint CENTRES (default, expand into patches);
        # "cloud" = the 4 PaXini fingerprint pad SURFACES as a dense point cloud (used verbatim).
        mode = str(inp["contact_mode"]) if "contact_mode" in inp else "points"
        if mode == "cloud":
            contacts, per = hand_fk.fingertip_paxini_contact_cloud(jv)
            print(f"[igr] contacts: {len(contacts)} PaXini fingerprint-surface pts "
                  f"({', '.join(r[0] for r in per)})", flush=True)
            dense = IGR.complete(net, partial, contacts, device, seed=0, denoise_partial=False,
                                 expand_contact_patches=False)
        else:
            contacts, info = hand_fk.fingertip_paxini_contacts(jv)  # 4 fingerprint centres (FK only)
            print(f"[igr] contacts: {len(contacts)} PaXini fingertips "
                  f"({', '.join(r[0] for r in info)})", flush=True)
            dense = IGR.complete(net, partial, contacts, device, seed=0, denoise_partial=False)
        if dense is None:                              # no surface -> fall back to the partial
            dense = partial.astype(np.float32)
        # Act-VH latent optimisation transiently allocates ~1.7 GB whose CACHE the allocator
        # then HOLDS. On the shared 5090 (Molmo 17 + SAM 10 + IGR + AnyPlace ≈ 30 GB) that
        # leftover cache is what tips the LAST service (AnyPlace) into CUDA OOM. Release it so
        # co-resident services keep their headroom (cheap; this service reallocates next call).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"dense": np.asarray(dense, np.float32),
                "contacts": np.asarray(contacts, np.float32)}

    serve(predict, a.port, name="igr")


if __name__ == "__main__":
    main()
