import open3d as o3d
import numpy as np

# Get data path
data_path = "PATH_TO_DATA_SUBFOLDER"
base_obj_path= f'{data_path}/base_obj_pointcloud.ply'
target_obj_path = f'{data_path}/target_obj_pointcloud.ply' 

# Load pointclouds and predicted placement poses
base_obj = o3d.io.read_point_cloud(base_obj_path)
target_obj = o3d.io.read_point_cloud(target_obj_path)
base_obj_pc = np.asarray(base_obj.points)
grasp_obj_pc = np.asarray(target_obj.points)
placement_pose = np.load(f"{data_path}/anyplace_diffusion_molmocrop_multitask_relative_pose_prediction.npy", allow_pickle=True).item()

# Visualize all predicted placement poses
relative_poses = placement_pose["relative"]
print(f"Number of relative poses: {len(relative_poses)}")
for relative_pose in relative_poses:
     grasp_pc = grasp_obj_pc.copy()
     grasp_pc = grasp_pc @ relative_pose[:3, :3].T + relative_pose[:3, 3]
     grasp_pcd_o3d = o3d.geometry.PointCloud()
     grasp_pcd_o3d.points = o3d.utility.Vector3dVector(grasp_pc)

     base_pcd_o3d = o3d.geometry.PointCloud()
     base_pcd_o3d.points = o3d.utility.Vector3dVector(base_obj_pc)
     print(relative_pose)
     o3d.visualization.draw_geometries([base_pcd_o3d, grasp_pcd_o3d])
