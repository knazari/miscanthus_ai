import cv2
import numpy as np
import open3d as o3d

class PointCloudSegmenter:
    def __init__(self, pointcloud_path, segmented_image_path, intrinsic_matrix=None, extrinsic_matrix=None):
        # Load the pointcloud
        self.pointcloud = o3d.io.read_point_cloud(pointcloud_path)
        # Load and preprocess the segmented image
        self.segmented_image = cv2.imread(segmented_image_path)
        self.mask = self.create_mask(self.segmented_image)

        # Set intrinsic matrix (default is for RealSense D435)
        self.intrinsic_matrix = intrinsic_matrix if intrinsic_matrix is not None else np.array([
            [617.24, 0, 318.53],  # fx, 0, cx
            [0, 617.08, 238.35],  # 0, fy, cy
            [0, 0, 1]
        ])

        # Set extrinsic matrix (default is the identity matrix)
        self.extrinsic_matrix = extrinsic_matrix if extrinsic_matrix is not None else np.eye(4)

    def create_mask(self, segmented_image):
        """
        Convert the segmented image into a binary mask.
        """
        segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(segmented_gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def filter_pointcloud_by_mask(self):
        """
        Filter the pointcloud using the segmentation mask and intrinsic/extrinsic matrices.
        """
        points = np.asarray(self.pointcloud.points)
        colors = np.asarray(self.pointcloud.colors)

        # Convert points to homogeneous coordinates for projection
        points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        # Apply extrinsic matrix (transform points into camera coordinate system)
        points_camera_frame = (self.extrinsic_matrix @ points_homogeneous.T).T

        # Project the points onto the image plane using the intrinsic matrix
        projected_points = self.intrinsic_matrix @ points_camera_frame[:, :3].T
        projected_points /= projected_points[2, :]  # Normalize by z-coordinate to get 2D pixel coordinates

        u_coords = projected_points[0, :].astype(np.int32)
        v_coords = projected_points[1, :].astype(np.int32)

        # Filter points using the mask
        valid_points = []
        valid_colors = []
        h, w = self.mask.shape
        for i in range(len(u_coords)):
            if 0 <= u_coords[i] < w and 0 <= v_coords[i] < h and self.mask[v_coords[i], u_coords[i]] > 0:
                valid_points.append(points[i])
                valid_colors.append(colors[i])

        # Create a new pointcloud with the filtered points and colors
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(valid_colors))

        return filtered_pcd

    def save_pointcloud(self, filtered_pointcloud, filename):
        """
        Save the segmented pointcloud to a file.
        """
        o3d.io.write_point_cloud(filename, filtered_pointcloud)
        print(f"Segmented pointcloud saved as {filename}")


if __name__ == "__main__":
    pointcloud_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/front_transformed.ply"
    segmented_image_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/front_color_segmented.jpg"

    # Initialize the segmenter
    segmenter = PointCloudSegmenter(pointcloud_path, segmented_image_path)

    # Filter the pointcloud based on the mask
    filtered_pointcloud = segmenter.filter_pointcloud_by_mask()

    # Save the segmented pointcloud
    saved_file_name = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/front_pcd_segmented.ply"
    segmenter.save_pointcloud(filtered_pointcloud, filename=saved_file_name)

