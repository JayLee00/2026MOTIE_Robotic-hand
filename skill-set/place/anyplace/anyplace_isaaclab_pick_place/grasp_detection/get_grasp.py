import os
import sys
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from pxr import UsdGeom

from grasp_utils import Grasp, tf_grasp

sys.path.append(os.path.join(os.path.dirname(__file__), 'graspnet-baseline'))

from utils.data_utils import transform_point_cloud
from demo import get_net, get_grasps, collision_detection


class GetGraspByAnyGrasp:
    def __init__(self, args):
        self.args = args
        print('Using checkpoint:', args.checkpoint_path)

        self.net = get_net(args)

    def visualize(self):
        grippers = self.grasps.to_open3d_geometry_list()
        pts = np.asarray(self.cloud.points)
        mx_z = np.max(pts[:, 2])
        new_pts = pts[pts[:, 2] < mx_z - 1e-3].copy()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(new_pts)
        o3d.visualization.draw_geometries(
            [cloud, *grippers],
            front=[0, 0, -1],
            lookat=[0, 0, 0],
            up=[0, 1, 0],
            zoom=2,
        )

    def get_grasp(self, usd_mesh, trans_to_world_mat, scale=1):
        self.default_extr = None

        end_points, self.cloud = self.get_and_process_data(usd_mesh, trans_to_world_mat, scale)
        grasps = get_grasps(self.net, end_points)

        grasps.nms()
        grasps.sort_by_score()
        if self.args.collision_thresh > 0:
            grasps = collision_detection(self.args, grasps, np.array(self.cloud.points))

        if len(grasps) > self.args.max_grasps:
            grasps = grasps[:self.args.max_grasps]

        self.grasps = grasps

        # Transform these grasps
        tf_grasps = []
        for i, grasp in enumerate(grasps):
            pos = grasp.translation / scale
            if np.max(np.abs(pos[:2])) > 0.3: # Ignore the grasps on the border
                continue
            grasp = Grasp(
                index=i,
                score=grasp.score,
                width=grasp.width/scale,
                height=grasp.height/scale,
                depth=grasp.depth/scale,
                pos=pos,
                orn=Rotation.from_matrix(grasp.rotation_matrix)
            )
            print('pos:', pos)
            tf_grasps.append(tf_grasp(grasp, self.default_extr, is_anygrasp=True, lift_dis=-0.02))
            print(tf_grasp(grasp, self.default_extr, is_anygrasp=True, lift_dis=-0.02))

        return tf_grasps

    def get_and_process_data(self, usd_mesh: UsdGeom.Mesh, trans_to_world_mat, scale):
        plane_pts = np.zeros((4, 3), dtype=float)
        plane_pts[[0, 2], 0] = 1
        plane_pts[[1, 3], 0] = -1
        plane_pts[[0, 1], 1] = -1
        plane_pts[[2, 3], 1] = 1
        plane_pts *= 0.4

        plane_triangles = np.array([[0, 1, 2], [1, 2, 3]])

        obj_pts = np.array(usd_mesh.GetPointsAttr().Get())
        obj_triangles = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        obj_triangles = obj_triangles.reshape(-1, 3) + 4

        obj_pts = np.concatenate((obj_pts, np.ones((obj_pts.shape[0], 1))), axis=1)
        # obj_pts = (trans_to_world_mat @ obj_pts.T).T
        obj_pts = obj_pts @ trans_to_world_mat.T
        obj_pts = obj_pts[:, :3]

        plane_pts[:, 2] = np.min(obj_pts[:, 2]) - 1e-2

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.concatenate((plane_pts, obj_pts)))
        mesh.triangles = o3d.utility.Vector3iVector(np.concatenate((plane_triangles, obj_triangles)))

        pcd = mesh.sample_points_uniformly(number_of_points=int(2e5))
        cloud = np.asarray(pcd.points)

        color = np.ones((cloud.shape[0], 3), dtype=float) * .5
        color[cloud[:, 2] > plane_pts[0, 2] + 1e-3, 2] = 1

        trans_mat = np.diag([1., -1, -1, 1])
        trans_mat[2, 3] = 0.5
        self.default_extr = trans_mat

        cloud = transform_point_cloud(cloud, trans_mat)

        cloud *= scale

        return self.sample_points(cloud, color)

    def sample_points(self, cloud_arr, color_arr):
        # Sample points
        if len(cloud_arr) >= self.args.num_point:
            idxs = np.random.choice(
                len(cloud_arr), self.args.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_arr))
            idxs2 = np.random.choice(len(cloud_arr), self.args.num_point-len(cloud_arr), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_arr[idxs]
        color_sampled = color_arr[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_arr.astype(np.float32))
        # cloud.colors = o3d.utility.Vector3dVector(color_arr.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(
            cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        color_sampled = color_sampled[np.newaxis]
        cloud_sampled = cloud_sampled.to(device)
        print('cloud_sampled shape:', cloud_sampled.shape)
        print('color sampled shape:', color_sampled.shape)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud
