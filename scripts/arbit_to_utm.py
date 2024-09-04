import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_transformation(source_points, target_points):
    # Assuming source_points and target_points are numpy arrays of shape (N, 3)
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target
    H = centered_source.T @ centered_target
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    translation = centroid_target - rotation @ centroid_source
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

def apply_transformation(pcd, transformation_matrix):
    # Apply the transformation
    pcd.transform(transformation_matrix)

def save_point_cloud(pcd, file_path):
    # Save the transformed point cloud
    o3d.io.write_point_cloud(file_path, pcd)

def main():
    input_file = '/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/front_cropped.pcd'
    output_file = '/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/front_transformed.pcd'
    
    # Load the point cloud
    pcd = load_point_cloud(input_file)
    
    # Dummy corresponding points (replace with actual data)
    source_points = np.array([[0.030, -0.500, -1.350], [0.590, -0.460, -1.540], [-0.460, -0.470, -1.540]])
    target_points = np.array([[6.500, -2.500, -15.200], [7.200, -3.200, -15.300], [7.200, -1.600, -15.300]])
    
    # Compute the transformation matrix
    transformation_matrix = compute_transformation(source_points, target_points)
    
    # Apply the transformation
    apply_transformation(pcd, transformation_matrix)
    
    # Save the transformed point cloud
    save_point_cloud(pcd, output_file)

if __name__ == "__main__":
    main()




































































































