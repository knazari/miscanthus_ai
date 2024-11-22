import open3d as o3d
import numpy as np


# Apply a depth filter to keep only the points within the specified range
def crop_point_cloud_by_depth(pcd, min_z, max_z):
    points = np.asarray(pcd.points)
    mask = np.where((points[:, 2] >= min_z) & (points[:, 2] <= max_z))
    cropped_points = points[mask]
    
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    return cropped_pcd

def crop_point_cloud_by_width(pcd, min_x, max_x):
    points = np.asarray(pcd.points)
    mask = np.where((points[:, 0] >= min_x) & (points[:, 0] <= max_x))
    cropped_points = points[mask]
    
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    return cropped_pcd

# def visualize_point_cloud(file_path):
#     # Load the point cloud
#     point_cloud = o3d.io.read_point_cloud(file_path)

#     points = np.asarray(point_cloud.points)
#     min_bound = points.min(axis=0)
#     # max_bound = points.max(axis=0)
#     print(f"Min bound: {min_bound}")
#     # print(f"Max bound: {max_bound}")

#     # Check if the point cloud is loaded successfully
#     if not point_cloud:
#         print(f"Failed to load point cloud from {file_path}")
#         return

#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # Specify the path to the .ply file
    file_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/left_transformed.ply"

    # Visualize the point cloud
    # visualize_point_cloud(file_path)

    point_cloud = o3d.io.read_point_cloud(file_path)

    # points = np.asarray(point_cloud.points)
    # min_bound = points.min(axis=0)
    # # max_bound = points.max(axis=0)
    # print(f"Min bound: {min_bound}")
    # # print(f"Max bound: {max_bound}")

    # # Check if the point cloud is loaded successfully
    # if not point_cloud:
    #     print(f"Failed to load point cloud from {file_path}")
    #     return

    cropped_pcd = crop_point_cloud_by_depth(point_cloud, -16.81, -11.9)
    cropped_pcd1 = crop_point_cloud_by_width(cropped_pcd, -7.27, -6.0)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([cropped_pcd1])
    o3d.io.write_point_cloud("/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/left_transformed_cropped.ply", cropped_pcd1)
