import copy
import numpy as np
from dataclasses import dataclass, fields
from scipy.spatial.transform import Rotation
from typing import Union, Optional


@dataclass
class Grasp:
    pos: Union[np.ndarray, list]
    orn: Union[np.ndarray, Rotation, list]
    width: float

    index: int = 0
    score: Optional[float] = None
    depth: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.pos, np.ndarray):
            self.pos = self.pos.tolist()
        if isinstance(self.orn, Rotation):
            self.orn = self.orn.as_quat()
        if isinstance(self.orn, np.ndarray):
            self.orn = self.orn.tolist()

    @property
    def pos_np(self):
        return np.array(self.pos)

    @property
    def orn_rot(self):
        return Rotation.from_quat(self.orn)

    def __iter__(self):
        for field in fields(self):
            val = getattr(self, field.name)
            if val is not None:
                yield (field.name, getattr(self, field.name))


def to_float_list(arr):
    return list(map(float, arr))


def tf_orn_mat_anygrasp(orn_mat):
    '''
    x, y, z axes usually correspond respectively to
    the axis perpendiculaire to the gripper surface,
    and those parallel to the gripper bottom and tail.
    While in the output result of anygrasp,
    they correspond to the gripper tail, bottom,
    and the one perpendiculaire to the gripper surface.
    So we need to transform the rotation matrix to the normal convention
    '''
    return orn_mat @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def tf_cam_to_base_frame(pos: list, orn: Rotation, tf_mat, is_anygrasp: bool = False, lift_dis: float = 0):
    tf_mat = np.array(tf_mat)
    pos_mat = np.linalg.inv(tf_mat) @ np.array([pos + [1]]).T
    orn_mat = orn.as_matrix()
    new_orn_mat = tf_mat[:3, :3].T @ orn_mat
    if is_anygrasp:
        new_orn_mat = tf_orn_mat_anygrasp(new_orn_mat)  # for anygrasp
    pos_vec = pos_mat[:3].flatten()
    pos_vec -= lift_dis * new_orn_mat[:, 2].T
    return pos_vec, Rotation.from_matrix(new_orn_mat)


def tf_grasp(grasp: Grasp, cam_ext_mat: np.ndarray, is_anygrasp: bool = False, lift_dis: float = 0):
    grasp = copy.deepcopy(grasp)
    new_pos, new_orn = tf_cam_to_base_frame(grasp.pos, grasp.orn_rot, cam_ext_mat, is_anygrasp, lift_dis)
    grasp.pos = new_pos.tolist()
    grasp.orn = new_orn.as_quat().tolist()
    return grasp
