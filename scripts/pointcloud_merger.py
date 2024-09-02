import open3d as o3d
import numpy as np
import os

def preprocess_point_cloud(pcd, voxel_size):
    print(f"Downsampling with a voxel size of {voxel_size}.")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    print(f"Estimating normals with search radius of {radius_normal}.")
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def pairwise_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("Applying point-to-plane ICP")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result_icp.transformation

def crop_point_cloud(pcd, min_bound, max_bound):
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    cropped_pcd = pcd.crop(bounding_box)
    return cropped_pcd

def merge_point_clouds(file_path_pattern, num_files, voxel_size=0.05):
    pcds = []
    for i in range(num_files):
        file_path = file_path_pattern.format(i)
        print(f"Loading point cloud: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Crop the point cloud
        min_bound = np.array([-0.25, -0.3, -1])  # Adjust these bounds as needed
        max_bound = np.array([0.25, 0.5, 1])
        pcd = crop_point_cloud(pcd, min_bound, max_bound)

        pcds.append(preprocess_point_cloud(pcd, voxel_size))

    print("Combining point clouds")
    combined_pcd = pcds[0]
    for i in range(1, len(pcds)):
        transformation = pairwise_registration(pcds[i], combined_pcd, voxel_size)
        pcds[i].transform(transformation)
        combined_pcd += pcds[i]
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)

    return combined_pcd

if __name__ == "__main__":
    file_path_pattern = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/pointclouds/no_time_stamp/pointcloud_{}.ply"
    num_files = 5  # Number of point clouds to merge
    voxel_size = 0.001

    combined_pcd = merge_point_clouds(file_path_pattern, num_files, voxel_size)
    o3d.io.write_point_cloud("/home/kiyanoush/miscanthus_ws/src/viper_ros/data/pointclouds/no_time_stamp/merged_pointcloud1.ply", combined_pcd)
    o3d.visualization.draw_geometries([combined_pcd])
    print("Merged point cloud saved to /home/kiyanoush/miscanthus_ws/src/viper_ros/data/pointclouds/no_time_stamp/merged_pointcloud.ply")
