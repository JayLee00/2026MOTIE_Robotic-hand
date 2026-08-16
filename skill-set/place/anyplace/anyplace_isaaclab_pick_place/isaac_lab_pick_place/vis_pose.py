import torch
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG


class PoseVisualization:
    def __init__(self, scene):
        self.env_origins = scene.env_origins

        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.pose1_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/pose1"))
        self.pose2_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/pose2"))
        self.pose3_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/pose2"))

    def vis(self, pose1: torch.Tensor, pose2: torch.Tensor, pose3: torch.Tensor):
        self.pose1_marker.set_visibility(True)
        self.pose2_marker.set_visibility(True)
        self.pose3_marker.set_visibility(True)
        self.pose1_marker.visualize(pose1[..., :3] + self.env_origins, pose1[..., 3:])
        self.pose2_marker.visualize(pose2[..., :3] + self.env_origins, pose2[..., 3:])
        self.pose3_marker.visualize(pose3[..., :3] + self.env_origins, pose3[..., 3:])

    def del_vis(self):
        self.pose1_marker.set_visibility(False)
        self.pose2_marker.set_visibility(False)
        self.pose3_marker.set_visibility(False)
