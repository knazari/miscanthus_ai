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
    input_file = '/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/left_cropped.ply'
    output_file = '/home/kia/Kiyanoush/Github/miscanthus_ai/data/walled_garden_4th_october/left_transformed.ply'
    
    # Load the point cloud
    pcd = load_point_cloud(input_file)
    
    # Front view Key points
    # source_points = np.array([[-0.190, -0.660, -2.160], [0.270, -0.540, -2.400], [0.850, -0.620, -2.180]])
    # target_points = np.array([[-7.100, -10.100, -16.900], [-6.800, -10.500, -16.900], [-6.900, -11.200, -16.900]])

    # Right view Key points
    # source_points = np.array([[-0.330, -0.470, -2.870], [-0.140, -0.560, -2.400], [-0.350, -0.720, -1.930]])
    # target_points = np.array([[-7.100, -10.100, -16.900], [-6.800, -10.500, -16.900], [-6.900, -11.200, -16.900]])
    # target_points = np.array([[664957.128282, 5904962.668081, 84.997], [664957.183929, 5904961.944679, 85.216], [664957.406514, 5904961.357302, 84.896]])

    # Left view Key points
    source_points = np.array([[0.660, -0.680, -1.490], [0.400, -0.530, -2.080], [0.700, -0.420, -2.590]])
    target_points = np.array([[-7.100, -10.100, -16.900], [-6.800, -10.500, -16.900], [-6.900, -11.200, -16.900]])
    
    # Compute the transformation matrix
    transformation_matrix = compute_transformation(source_points, target_points)
    
    # Apply the transformation
    apply_transformation(pcd, transformation_matrix)
    
    # Save the transformed point cloud
    save_point_cloud(pcd, output_file)

if __name__ == "__main__":
    main()




































































































