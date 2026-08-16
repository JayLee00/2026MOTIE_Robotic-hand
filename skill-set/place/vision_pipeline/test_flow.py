"""Offline flow test: run PlacePipeline with mock backend+models for both
scenarios, asserting the step order, frame math, and a valid T_act/ee_target."""
import sys

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")
from vision_pipeline.orchestrator import PlacePipeline
from vision_pipeline.backends.mock import MockBackend, MockModels


def check(scenario):
    b, m = MockBackend(), MockModels()
    pipe = PlacePipeline(b, m, log=lambda *a: None)
    R = pipe.run(scenario=scenario)

    # T_act / ee_target are valid homogeneous transforms
    for key in ("T_zalign", "T_act", "ee_target"):
        T = R[key]
        assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1]), f"{key} invalid"
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6, f"{key} rotation not SO(3)"

    # §9-A: chosen candidate is the most upright (max R[2,2]) -> the identity one here
    assert R["upright_score"] > 0.99, R["upright_score"]

    # step order: parent move -> async child move -> wait -> async left-arm clear ->
    # wait left-arm done -> T_preplace (move_joints) -> collision-free waypoint (ee) ->
    # monitored decel descent -> [release/retract] weak-open -> slow ascent -> linear lift
    # to waypoint -> parent_pose -> hand relax
    names = [c[0] for c in b.calls]
    assert names == ["move_joints", "start_move", "wait_move", "left_arm", "wait_move",
                     "move_joints", "move_ee", "descend_monitored",
                     "hand_release", "ascend", "move_ee_linear", "move_joints",
                     "hand_safe_shutdown"], names
    assert R["executed"] is True
    assert R["stop_reason"] == "reached" and R["release_reason"] == "noload" and R["retract_done"] is True

    # fruit yields >=1 region (multi holes); table exactly 1 candidate set
    assert R["candidates"].ndim == 3 and R["candidates"].shape[1:] == (4, 4)
    print(f"  {scenario}: OK  (local_crop={R['local_crop']:.3f}m, "
          f"{len(R['candidates'])} cand, upright={R['upright_score']:.3f}, "
          f"ee_target={np.round(R['ee_target'][:3,3],3).tolist()})")


if __name__ == "__main__":
    for s in ("table", "fruit"):
        check(s)
    print("FLOW TEST OK: orchestrator runs both scenarios end-to-end (mock backend)")
