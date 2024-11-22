import numpy as np
import matplotlib.pyplot as plt

def load_points3D(file_path):
    """
    Loads 3D points from COLMAP's points3D.txt file.

    Args:
        file_path (str): Path to the points3D.txt file.

    Returns:
        np.ndarray: Array of 3D points (N, 3).
    """
    points = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip comments
            data = line.strip().split()
            if len(data) < 3:
                continue  # Skip invalid lines
            x, y, z = map(float, data[1:4])  # Extract X, Y, Z
            points.append([x, y, z])
    return np.array(points)

def parse_colmap_images(file_path):
    """
    Parse COLMAP images.txt file to extract camera poses.

    Args:
        file_path (str): Path to COLMAP's images.txt file.

    Returns:
        torch.Tensor: Poses tensor of shape (N, 4, 4), where N is the number of images.
    """
    poses = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Skip comment lines
            if line.startswith("#"):
                continue
            
            # Split the line into components
            data = line.strip().split()
            if len(data) > 11:
                continue
            
            # COLMAP format for each image line:
            # IMAGE_ID, qw, qx, qy, qz, tx, ty, tz, CAMERA_ID, NAME
            qw, qx, qy, qz = map(float, data[1:5])  # Quaternion components
            tx, ty, tz = map(float, data[5:8])      # Translation components
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz]).reshape((3, 1))  # Translation vector
            
            # Create a 4x4 transformation matrix
            pose = np.hstack((R, t))
            pose = np.vstack((pose, np.array([0, 0, 0, 1])))  # Convert to 4x4
            
            # Add pose to list
            poses.append(pose)
    
    # Convert list of poses to a PyTorch tensor
    poses = np.array(poses)
    
    return poses

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a 3x3 rotation matrix.

    Args:
        qw, qx, qy, qz (float): Quaternion components.

    Returns:
        np.array: 3x3 rotation matrix.
    """
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
   
    return R

def calculate_near_far(points3D, camera_positions):
    """
    Calculates near and far distances from camera positions to 3D points.

    Args:
        points3D (np.ndarray): 3D points (N, 3).
        camera_positions (np.ndarray): Camera positions (M, 3).

    Returns:
        float, float: Near and far distances.
    """
    distances = np.linalg.norm(points3D[:, None, :] - camera_positions[None, :, :], axis=-1)
    near = np.min(distances)
    far = np.max(distances)
    return near, far

# Load 3D points
points3D = load_points3D("/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/sparse_model/points3D.txt")

# Load camera poses
poses = parse_colmap_images("/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/sparse_model/images.txt")

# Load camera positions (example with your poses)
camera_positions = poses[:, :3, 3]  # Extract camera translations from poses

mean_position = np.mean(camera_positions, axis=0)
max_distance = np.max(np.linalg.norm(camera_positions - mean_position, axis=-1))
normalized_camera_positions = (camera_positions - mean_position) / max_distance

mean_3dposition = np.mean(points3D, axis=0)
max_3ddistance = np.max(np.linalg.norm(points3D - mean_3dposition, axis=-1))
normalized_3d_positions = (points3D - mean_3dposition) / max_3ddistance


# Calculate near and far
near, far = calculate_near_far(normalized_3d_positions, normalized_camera_positions)
print(f"Near: {near}, Far: {far}")
