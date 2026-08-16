"""Reader for recorded Replicator sequences (format-identical to soldering_tool).

Pure numpy / cv2 / scipy / yaml. NO sim-side (recorder / Isaac) imports.

Directory layout per sequence (e.g. .../Replicator_02000):
    rgb/XXXXXX.png                                   BGR uint8 480x640 (cv2)
    distance_to_camera/XXXXXX.npy                    float32 (480,640) euclidean (m)
    semantic_segmentation/XXXXXX.png                 RGB-colorized seg
    semantic_segmentation/segmentation_labels_XXXXXX.json
                                                     {"<cls>": {"class":..,"color":"(r, g, b)"}}
    poses/XXXXXX.json                                camera/object/hand + joints (WXYZ)
    camera_pose.json                                 per-sequence extrinsic
    gt_shape.json                                    optional (fruit only); None if absent

The object mask is built by parsing the sidecar color STRING '(r, g, b)' for
the requested class, converting RGB->BGR, and matching against the cv2-loaded
(BGR) seg PNG with a per-channel tolerance.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .types import Frame, GtShape


def parse_color_string(color_str: str) -> np.ndarray:
    """Parse an RGB color STRING like '(128, 0, 0)' -> np.array([128,0,0]) (RGB)."""
    nums = re.findall(r"-?\d+", str(color_str))
    if len(nums) < 3:
        raise ValueError(f"could not parse color string {color_str!r}")
    return np.array([int(nums[0]), int(nums[1]), int(nums[2])], dtype=np.int32)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Reverse an (r, g, b) triple to (b, g, r) for cv2-loaded images."""
    rgb = np.asarray(rgb).reshape(3)
    return rgb[::-1].copy()


def mask_from_seg_bgr(
    seg_bgr: np.ndarray, target_bgr: np.ndarray, tol: int = 0
) -> np.ndarray:
    """Boolean mask where the BGR seg PNG matches target_bgr within ``tol``."""
    seg = seg_bgr.astype(np.int32)
    target = np.asarray(target_bgr, dtype=np.int32).reshape(3)
    diff = np.abs(seg - target[None, None, :])
    return np.all(diff <= int(tol), axis=2)


class ReplicatorReader:
    """Iterate frames of a recorded Replicator sequence.

    Args:
        seq_dir: path to a sequence directory (e.g. .../Replicator_02000).
        camera_yaml: path to camera.yaml (used only to carry intrinsics around;
            extrinsics come per-sequence from camera_pose.json).
        fruit_class_name: the semantic class name to segment (e.g. "apple",
            "soldering_tool"). Matched against the sidecar label JSON keys.
        color_tolerance: per-channel BGR tolerance for the GT color match.
    """

    def __init__(
        self,
        seq_dir: str,
        camera_yaml: Optional[str] = None,
        fruit_class_name: str = "apple",
        color_tolerance: int = 0,
    ) -> None:
        self.seq_dir = os.path.abspath(seq_dir)
        self.camera_yaml = camera_yaml
        self.fruit_class_name = fruit_class_name
        self.color_tolerance = int(color_tolerance)

        if not os.path.isdir(self.seq_dir):
            raise FileNotFoundError(f"sequence dir not found: {self.seq_dir}")

        self._rgb_dir = os.path.join(self.seq_dir, "rgb")
        self._dist_dir = os.path.join(self.seq_dir, "distance_to_camera")
        self._seg_dir = os.path.join(self.seq_dir, "semantic_segmentation")
        self._poses_dir = os.path.join(self.seq_dir, "poses")

        # Frame ids from the rgb dir (zero-padded stems).
        self._frame_ids: List[str] = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self._rgb_dir, "*.png"))
        )

        # Per-sequence camera extrinsic.
        self.camera_pose = self._load_camera_pose()

        # Optional GT shape (absent for soldering_tool -> None).
        self.gt_shape: Optional[GtShape] = self._load_gt_shape()

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._frame_ids)

    @property
    def frame_ids(self) -> List[str]:
        return list(self._frame_ids)

    # ------------------------------------------------------------------ #
    def _load_camera_pose(self) -> dict:
        path = os.path.join(self.seq_dir, "camera_pose.json")
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _load_gt_shape(self) -> Optional[GtShape]:
        path = os.path.join(self.seq_dir, "gt_shape.json")
        if not os.path.isfile(path):
            return None  # handle absence gracefully (soldering_tool)
        with open(path) as f:
            data = json.load(f)
        return GtShape(
            a=float(data["a"]),
            c=float(data["c"]),
            # scale is metadata only (scalar or [sx,sy,sz] list); stored raw,
            # never multiplied into a/c which are already absolute radii.
            scale=data.get("scale", 1.0),
            shape_class=str(data.get("shape_class", "spheroid")),
        )

    def _seg_label_path(self, fid: str) -> str:
        return os.path.join(self._seg_dir, f"segmentation_labels_{fid}.json")

    def _target_bgr(self, fid: str) -> Optional[np.ndarray]:
        """Resolve the BGR target color for the fruit class from the sidecar."""
        path = self._seg_label_path(fid)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            labels = json.load(f)
        entry = labels.get(self.fruit_class_name)
        if entry is None:
            # Fall back: first entry whose class == "object".
            for v in labels.values():
                if v.get("class") == "object":
                    entry = v
                    break
        if entry is None:
            return None
        rgb = parse_color_string(entry["color"])
        return rgb_to_bgr(rgb)

    # ------------------------------------------------------------------ #
    def read_frame(self, i: int) -> Frame:
        """Read frame at index ``i`` (0-based, in sorted frame-id order)."""
        fid = self._frame_ids[i]

        rgb = cv2.imread(os.path.join(self._rgb_dir, f"{fid}.png"))  # BGR uint8
        distance = np.load(
            os.path.join(self._dist_dir, f"{fid}.npy")
        ).astype(np.float32)
        seg_bgr = cv2.imread(os.path.join(self._seg_dir, f"{fid}.png"))  # BGR uint8

        target_bgr = self._target_bgr(fid)
        if target_bgr is None:
            object_mask = np.zeros(distance.shape, dtype=bool)
        else:
            object_mask = mask_from_seg_bgr(
                seg_bgr, target_bgr, tol=self.color_tolerance
            )

        with open(os.path.join(self._poses_dir, f"{fid}.json")) as f:
            pose = json.load(f)

        cam_pos = np.asarray(pose["camera"][0], dtype=np.float64)
        cam_quat = np.asarray(pose["camera"][1], dtype=np.float64)  # WXYZ
        obj_pos = np.asarray(pose["object"][0], dtype=np.float64)
        obj_quat = np.asarray(pose["object"][1], dtype=np.float64)  # WXYZ
        hand_pos = np.asarray(pose["hand"][0], dtype=np.float64)
        hand_quat = np.asarray(pose["hand"][1], dtype=np.float64)   # WXYZ
        hand_joints = np.asarray(pose.get("hand_joints", []), dtype=np.float64)
        hand_joint_names = list(pose.get("hand_joint_names", []))

        return Frame(
            idx=int(pose.get("seq_idx", i)),
            rgb=rgb,
            distance=distance,
            seg_color=seg_bgr,
            object_mask=object_mask,
            cam_pos=cam_pos,
            cam_quat_wxyz=cam_quat,
            gt_object_pos=obj_pos,
            gt_object_quat_wxyz=obj_quat,
            hand_pos=hand_pos,
            hand_quat_wxyz=hand_quat,
            hand_joints=hand_joints,
            hand_joint_names=hand_joint_names,
        )

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.read_frame(i)
