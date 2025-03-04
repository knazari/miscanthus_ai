import open3d as o3d
import numpy as np

class DepthDiscontinuityRemover:
    def __init__(self, pointcloud_path):
        # Load the pointcloud
        self.pointcloud = o3d.io.read_point_cloud(pointcloud_path)
        self.kdtree = o3d.geometry.KDTreeFlann(self.pointcloud)  # Create the KDTree

    def remove_depth_discontinuities(self, depth_threshold=0.01, radius=0.02):
        """
        Remove points that have depth discontinuities based on a depth threshold.
        """
        points = np.asarray(self.pointcloud.points)
        filtered_points = []
        filtered_colors = []

        # For each point, compare its depth with its neighbors
        for i in range(len(points)):
            # Find neighbors within the specified radius using KDTree
            [k, idx, _] = self.kdtree.search_radius_vector_3d(self.pointcloud.points[i], radius)
            if k < 3:
                continue

            # Get the depth (z-component) of the current point and its neighbors
            current_depth = points[i][2]  # Z value of the current point
            neighbor_depths = points[idx[:k], 2]  # Z values of the neighbors

            # Calculate the maximum depth difference
            max_depth_diff = np.max(np.abs(neighbor_depths - current_depth))

            # If the maximum depth difference is less than the threshold, keep the point
            if max_depth_diff < depth_threshold:
                filtered_points.append(points[i])
                filtered_colors.append(np.asarray(self.pointcloud.colors)[i])

        # Create a new pointcloud with the filtered points and colors
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors))

        return filtered_pcd

    def visualize_pointcloud(self, pointcloud):
        """
        Visualize the pointcloud.
        """
        o3d.visualization.draw_geometries([pointcloud])

    def save_pointcloud(self, pointcloud, filename="filtered_pointcloud.pcd"):
        """
        Save the filtered pointcloud to a PCD file.
        """
        o3d.io.write_point_cloud(filename, pointcloud)
        print(f"Filtered pointcloud saved as {filename}")


if __name__ == "__main__":
    pointcloud_path = "/home/kiyanoush/Downloads/email/masked_pintcloud.ply"

    # Initialize the remover
    remover = DepthDiscontinuityRemover(pointcloud_path)

    # Remove depth discontinuities
    filtered_pointcloud = remover.remove_depth_discontinuities(depth_threshold=0.05, radius=0.02)

    # Visualize the filtered pointcloud
    remover.visualize_pointcloud(filtered_pointcloud)

    # Save the filtered pointcloud
    # remover.save_pointcloud(filtered_pointcloud, filename="filtered_pointcloud.pcd")
