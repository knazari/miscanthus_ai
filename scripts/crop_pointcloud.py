import open3d as o3d
import numpy as np

class PointCloudCropper:
    def __init__(self, pointcloud_path):
        # Load the pointcloud
        self.pointcloud = o3d.io.read_point_cloud(pointcloud_path)

    def crop_with_axis_aligned_bounding_box(self, min_bound, max_bound):
        """
        Crop the pointcloud with an axis-aligned bounding box.
        """
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        cropped_pcd = self.pointcloud.crop(aabb)
        return cropped_pcd

    def crop_with_oriented_bounding_box(self, center, extents, rotation_matrix):
        """
        Crop the pointcloud with an oriented bounding box.
        """
        obb = o3d.geometry.OrientedBoundingBox(center=center, extents=extents, R=rotation_matrix)
        cropped_pcd = self.pointcloud.crop(obb)
        return cropped_pcd

    def visualize(self, pointcloud):
        """
        Visualize the pointcloud.
        """
        o3d.visualization.draw_geometries([pointcloud])

    def save_pointcloud(self, pointcloud, filename="cropped_pointcloud.pcd"):
        """
        Save the cropped pointcloud to a PCD file.
        """
        o3d.io.write_point_cloud(filename, pointcloud)
        print(f"Cropped pointcloud saved as {filename}")


if __name__ == "__main__":
    pointcloud_path = "/home/kiyanoush/Downloads/miscanthus360_point_cloud.ply"

    # Initialize the cropper
    cropper = PointCloudCropper(pointcloud_path)

    # Define bounds for axis-aligned bounding box (adjust as needed)
    min_bound = np.array([-5, -15, -19])  # Minimum x, y, z coordinates
    max_bound = np.array([5, 6, 20])     # Maximum x, y, z coordinates
    cropped_pcd_aabb = cropper.crop_with_axis_aligned_bounding_box(min_bound, max_bound)

    # Visualize the axis-aligned cropped pointcloud
    print("Visualizing Axis-Aligned Bounding Box Crop...")
    cropper.visualize(cropped_pcd_aabb)

    # Save the axis-aligned cropped pointcloud
    save_file_name = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_4th_october/cropped_pointcloud_front.ply"
    # cropper.save_pointcloud(cropped_pcd_aabb, filename=save_file_name)

    # # Define parameters for oriented bounding box (adjust as needed)
    # center = np.array([0, 0, 0])                      # Center of the bounding box
    # extents = np.array([1.0, 0.5, 0.5])               # Size of the bounding box along x, y, z axes
    # rotation_matrix = np.eye(3)                       # Identity matrix for no rotation (or define custom rotation)
    # cropped_pcd_obb = cropper.crop_with_oriented_bounding_box(center, extents, rotation_matrix)

    # # Visualize the oriented bounding box cropped pointcloud
    # print("Visualizing Oriented Bounding Box Crop...")
    # cropper.visualize(cropped_pcd_obb)

    # # Save the oriented bounding box cropped pointcloud
    # cropper.save_pointcloud(cropped_pcd_obb, filename="cropped_pointcloud_obb.ply")
