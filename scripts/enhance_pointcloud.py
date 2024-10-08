import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata

class PointCloudInterpolator:
    def __init__(self, segmented_pointcloud_path, segmented_image_path, intrinsic_matrix=None, extrinsic_matrix=None):
        # Load the segmented pointcloud
        self.pointcloud = o3d.io.read_point_cloud(segmented_pointcloud_path)
        # Load the segmented image
        self.segmented_image = cv2.imread(segmented_image_path)

        # Set intrinsic matrix (default is for RealSense D435)
        self.intrinsic_matrix = intrinsic_matrix if intrinsic_matrix is not None else np.array([
            [617.24, 0, 318.53],  # fx, 0, cx
            [0, 617.08, 238.35],  # 0, fy, cy
            [0, 0, 1]
        ])

        # Set extrinsic matrix (default is the identity matrix)
        self.extrinsic_matrix = extrinsic_matrix if extrinsic_matrix is not None else np.eye(4)

    def project_pointcloud_to_image(self):
        """
        Project the pointcloud onto the 2D image plane and return the 2D coordinates (u, v).
        """
        points = np.asarray(self.pointcloud.points)

        # Convert points to homogeneous coordinates for projection
        points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        # Apply extrinsic matrix (transform points into camera coordinate system)
        points_camera_frame = (self.extrinsic_matrix @ points_homogeneous.T).T

        # Project the points onto the image plane using the intrinsic matrix
        projected_points = self.intrinsic_matrix @ points_camera_frame[:, :3].T
        projected_points /= projected_points[2, :]  # Normalize by z-coordinate to get 2D pixel coordinates

        u_coords = projected_points[0, :].astype(np.int32)
        v_coords = projected_points[1, :].astype(np.int32)

        return u_coords, v_coords

    def interpolate_2d_points(self, u_coords, v_coords, depth_values, colors, grid_resolution=640):
        """
        Interpolate 2D points using grid-based interpolation to increase density.
        """
        # Create a grid for interpolation
        grid_u, grid_v = np.mgrid[min(u_coords):max(u_coords):complex(grid_resolution),
                                  min(v_coords):max(v_coords):complex(grid_resolution)]

        # Interpolate depth values at the new grid locations
        grid_depth = griddata((u_coords, v_coords), depth_values, (grid_u, grid_v), method='linear')

        # Interpolate color values (R, G, B channels separately)
        grid_colors_r = griddata((u_coords, v_coords), colors[:, 0], (grid_u, grid_v), method='linear')
        grid_colors_g = griddata((u_coords, v_coords), colors[:, 1], (grid_u, grid_v), method='linear')
        grid_colors_b = griddata((u_coords, v_coords), colors[:, 2], (grid_u, grid_v), method='linear')

        # Flatten the grid to get the interpolated points
        interpolated_u = grid_u.flatten()
        interpolated_v = grid_v.flatten()
        interpolated_depth = grid_depth.flatten()
        interpolated_colors = np.vstack([grid_colors_r.flatten(), grid_colors_g.flatten(), grid_colors_b.flatten()]).T

        # Filter out invalid (NaN) points
        valid_mask = ~np.isnan(interpolated_depth)
        return interpolated_u[valid_mask], interpolated_v[valid_mask], interpolated_depth[valid_mask], interpolated_colors[valid_mask]

    def reproject_to_3d(self, u_coords, v_coords, depth_values):
        """
        Reconstruct the 3D points from the interpolated 2D points and depth values.
        """
        # Create homogeneous 2D points (u, v, 1)
        homogeneous_2d = np.vstack([u_coords, v_coords, np.ones_like(u_coords)])

        # Invert the intrinsic matrix
        intrinsic_inv = np.linalg.inv(self.intrinsic_matrix)

        # Reproject the 2D points back into 3D using the inverse of the intrinsic matrix
        points_camera_frame = intrinsic_inv @ (homogeneous_2d * depth_values)

        # Convert points from camera frame back to world frame using the inverse of the extrinsic matrix
        extrinsic_inv = np.linalg.inv(self.extrinsic_matrix)
        points_camera_frame_homogeneous = np.vstack([points_camera_frame, np.ones_like(u_coords)])
        points_world_frame = extrinsic_inv @ points_camera_frame_homogeneous

        return points_world_frame[:3].T  # Return only the x, y, z components

    def save_pointcloud(self, points, colors, filename="enhanced_pointcloud.pcd"):
        """
        Save the enhanced pointcloud to a PCD file.
        """
        enhanced_pcd = o3d.geometry.PointCloud()
        enhanced_pcd.points = o3d.utility.Vector3dVector(points)
        enhanced_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, enhanced_pcd)
        print(f"Saved enhanced pointcloud as {filename}")


if __name__ == "__main__":
    data_dir = "/home/kia/Kiyanoush/Github/miscanthus_ai/data"
    segmented_pointcloud_path = data_dir + "/masked_pintcloud.ply"
    segmented_image_path = data_dir + "/segmented_miscanthus_image.jpg"

    # Initialize the interpolator
    interpolator = PointCloudInterpolator(segmented_pointcloud_path, segmented_image_path)

    # Project the pointcloud to 2D image plane and get u, v coordinates
    u_coords, v_coords = interpolator.project_pointcloud_to_image()

    # Get depth values and colors for the points
    depth_values = np.asarray(interpolator.pointcloud.points)[:, 2]
    colors = np.asarray(interpolator.pointcloud.colors)

    # Interpolate points and colors
    interpolated_u, interpolated_v, interpolated_depth, interpolated_colors = interpolator.interpolate_2d_points(
        u_coords, v_coords, depth_values, colors
    )

    # Reproject interpolated points back to 3D
    new_points = interpolator.reproject_to_3d(interpolated_u, interpolated_v, interpolated_depth)

    # Save the new pointcloud with interpolated points
    interpolator.save_pointcloud(new_points, interpolated_colors, filename="/home/kia/Kiyanoush/Github/miscanthus_ai/data/enhanced_pointcloud.ply")
