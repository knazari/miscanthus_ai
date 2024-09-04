import open3d as o3d
import numpy as np


def visualize_point_cloud(file_path):
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(file_path)

    points = np.asarray(point_cloud.points)
    # min_bound = points.min(axis=0)
    # max_bound = points.max(axis=0)
    # print(f"Min bound: {min_bound}")
    # print(f"Max bound: {max_bound}")

    # Check if the point cloud is loaded successfully
    if not point_cloud:
        print(f"Failed to load point cloud from {file_path}")
        return

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # Specify the path to the .ply file
    file_path = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/Walled_Garden_24_07_29_arbitrary_DRONE.ply"

    # Visualize the point cloud
    visualize_point_cloud(file_path)
