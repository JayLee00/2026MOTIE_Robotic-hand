"""Run the child-object pipeline flow (steps 3-10..3-14) on a LOGGED partial + a hand-joint
config, and save each stage as a coloured layer for view_pipeline.py.

Flow (exactly the live pipeline):
  1. child_pc_i (logged)            --DBSCAN-->            child_pc_refined
  2. hand joints (child_pose grasp) --PaXini URDF FK-->    contacts (4 지두 지문 centres)
  3. child_pc_refined + contacts    --Act-VH IGR-->        child_pc_com
  4. hand joints                    --PaXini URDF FK-->    hand_pc
  5. child_pc_com + hand_pc         --concat (both world)->child_pc

    conda activate anyplace_cu128
    python vision_pipeline/tools/actvh_test/run_pipeline_viz.py
    python vision_pipeline/tools/actvh_test/view_pipeline.py       # then view
"""
import os
import sys

import numpy as np
import torch
import trimesh

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place"
sys.path.insert(0, REPO)
from vision_pipeline.core import igr_complete as IGR                # noqa: E402
from vision_pipeline.core.outlier_removal import clean              # noqa: E402
from vision_pipeline.tools import hand_fk                           # noqa: E402

LOG = f"{REPO}/vision_pipeline/test_logs/run_0705_224858/debug.npz"  # logged child_pc_i
OUT = f"{REPO}/vision_pipeline/fixtures/actvh_test/pipeline_viz"


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. child_pc_i (logged) -> DBSCAN -> child_pc_refined
    child_pc_i = np.load(LOG, allow_pickle=True)["child_pc_i"].astype(np.float64)
    child_pc_refined = clean(child_pc_i)

    # 2. arbitrary hand joints (child_pose arm + recorded grasp kistar_pose/tmp_pose.txt)
    #    -> the 4 PaXini fingertip (지두) fingerprint contact centres (URDF FK, object-independent)
    jv = hand_fk.joint_values()
    contacts, info = hand_fk.fingertip_paxini_contacts(jv)

    # 3. child_pc_refined + contacts -> Act-VH IGR -> child_pc_com
    net = IGR.load_network(device)
    child_pc_com = IGR.complete(net, child_pc_refined, contacts, device, seed=0,
                                denoise_partial=False)

    # 4. same hand joints -> PaXini-URDF-FK hand cloud
    hand_pc = hand_fk.hand_point_cloud(jv, n=3000)

    # 5. child_pc = concat(child_pc_com, hand_pc)  (both already in world -> 정합 = concat)
    child_pc = np.vstack([child_pc_com, hand_pc])

    for name, pc in [("child_pc_i", child_pc_i), ("child_pc_refined", child_pc_refined),
                     ("child_pc_com", child_pc_com), ("hand_pc", hand_pc), ("child_pc", child_pc)]:
        trimesh.PointCloud(np.asarray(pc, np.float64)).export(f"{OUT}/{name}.ply")
    np.save(f"{OUT}/contacts.npy", np.asarray(contacts, np.float32))

    print(f"saved -> {OUT}")
    print(f"1. child_pc_i {len(child_pc_i)} -> DBSCAN -> child_pc_refined {len(child_pc_refined)} "
          f"({len(child_pc_i) - len(child_pc_refined)} removed)")
    print(f"2. PaXini contacts {contacts.shape}: {', '.join(r[0] for r in info)}")
    print(f"3. child_pc_com {child_pc_com.shape}  "
          f"(extent {((child_pc_com.max(0)-child_pc_com.min(0)).max()*100):.1f}cm)")
    print(f"4. hand_pc {hand_pc.shape}")
    print(f"5. child_pc {child_pc.shape}")


if __name__ == "__main__":
    main()
