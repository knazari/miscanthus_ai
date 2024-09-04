import open3d as o3d
import numpy as np

root_path = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july"
# Load the point cloud from a .pcd file
pcd = o3d.io.read_point_cloud(root_path + "/011/pointcloud_4__20240730-125423.ply")

# Crop the point cloud based on the depth (z component)
# Adjust the min_z and max_z values according to your needs
min_z = 0.01  # minimum depth
max_z = 2.5  # maximum depth

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

cropped_pcd = crop_point_cloud_by_depth(pcd, min_z, max_z)

# Rotation around the x-axis and z-axis
# Define the rotation angles (in degrees) for each axis
rotation_angle_y_degrees = 180  # rotation angle around the x-axis
rotation_angle_z_degrees = 180  # rotation angle around the z-axis

# Convert the rotation angles to radians
rotation_angle_y_radians = np.deg2rad(rotation_angle_y_degrees)
rotation_angle_z_radians = np.deg2rad(rotation_angle_z_degrees)

# Create rotation matrices for x-axis and z-axis
R_y = cropped_pcd.get_rotation_matrix_from_axis_angle(rotation_angle_y_radians * np.array([0, 1, 0]))
R_z = cropped_pcd.get_rotation_matrix_from_axis_angle(rotation_angle_z_radians * np.array([0, 0, 1]))

# Apply the rotations to the point cloud
cropped_pcd.rotate(R_y, center=(0, 0, 0))
cropped_pcd.rotate(R_z, center=(0, 0, 0))

# Save the cropped point cloud to a file (e.g., as "cropped_pointcloud.pcd")
o3d.io.write_point_cloud(root_path + "/front_cropped.pcd", cropped_pcd)
print("Cropped point cloud saved as 'cropped_pointcloud.pcd'.")


# Visualize the processed point cloud
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Cropped and Rotated Point Cloud")

vis.add_geometry(cropped_pcd)

view_control = vis.get_view_control()

# Set the initial zoom level
view_control.set_zoom(0.25)  # Lower values mean zoomed out, higher values zoom in

vis.run()
vis.destroy_window()
