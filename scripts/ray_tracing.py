import os
import cv2
import numpy as np
import open3d as o3d

def segment_plant_using_rgb(rgb_image):
    # Convert RGB image to HSV color space for better segmentation
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define the range of colors for the plant (e.g., green hues)
    lower_green = np.array([35, 25, 25])
    upper_green = np.array([111, 255, 255])

    # Threshold the image to get only the green parts (the plant)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply the mask to the original RGB image to visualize the segmentation
    segmented_plant = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

    # Show the original and segmented images side by side
    cv2.imshow('Original RGB Image', rgb_image)
    cv2.imshow('Segmented Plant', segmented_plant)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return mask

def project_pointcloud_to_image(pcd, intrinsic, extrinsic, image_shape):
    points = np.asarray(pcd.points)

    # Convert 3D points to 4D (homogeneous coordinates)
    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # Apply the extrinsic matrix (4x4) to transform the 3D points into camera coordinates
    transformed_points = (extrinsic @ points_homogeneous.T).T

    # Now apply the intrinsic matrix (3x3) to project the 3D points to 2D image plane
    projected_points = intrinsic @ transformed_points[:, :3].T

    # Normalize the projected points by the third (z) coordinate to get (u, v) pixel coordinates
    projected_points /= projected_points[2, :]

    # Clip to valid pixel indices
    u_coords = np.clip(projected_points[0, :], 0, image_shape[1] - 1).astype(np.int32)
    v_coords = np.clip(projected_points[1, :], 0, image_shape[0] - 1).astype(np.int32)

    return u_coords, v_coords

def apply_mask_to_pointcloud(pcd, mask, u_coords, v_coords):
    # Get points from the mask that are non-zero (i.e., part of the plant)
    valid_mask = mask[v_coords, u_coords] > 0

    # Filter points based on the mask
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    filtered_points = points[valid_mask, :]
    filtered_colors = colors[valid_mask, :]

    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd

def main(pcd_file, rgb_file):
    # Load pointcloud and RGB image
    pcd = o3d.io.read_point_cloud(pcd_file)
    rgb_image = cv2.imread(rgb_file)

    # Segment plant using RGB information
    mask = segment_plant_using_rgb(rgb_image)

    # Define the intrinsic matrix for RealSense D435
    intrinsic_matrix = np.array([
        [617.24, 0, 318.53],  # fx, 0, cx
        [0, 617.08, 238.35],  # 0, fy, cy
        [0, 0, 1]
    ])

    # Define the extrinsic matrix (identity in this case)
    extrinsic_matrix = np.eye(4)

    # Project point cloud to image coordinates
    u_coords, v_coords = project_pointcloud_to_image(pcd, intrinsic_matrix, extrinsic_matrix, rgb_image.shape)

    # Apply the mask to filter the point cloud
    filtered_pcd = apply_mask_to_pointcloud(pcd, mask, u_coords, v_coords)

    # Visualize the filtered pointcloud
    o3d.visualization.draw_geometries([filtered_pcd])

    # Save the filtered pointcloud
    o3d.io.write_point_cloud("filtered_pointcloud.pcd", filtered_pcd)

if __name__ == "__main__":
    # Get the path of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the relative path to the folder
    data_folder = os.path.join(current_dir, '..', 'data')
    pcd_file = data_folder + "/walled_garden_30th_july/front_cropped_no_rot.pcd"
    rgb_file = data_folder + "/walled_garden_30th_july/011/color_4.png"  # Optional
    depth_file_path = data_folder + "/walled_garden_30th_july/011/depth_4.png"  # Optional
    main(pcd_file, rgb_file)
