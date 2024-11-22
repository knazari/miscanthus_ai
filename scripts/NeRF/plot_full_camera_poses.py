import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_colmap_images(file_path):
    """
    Parse COLMAP images.txt file to extract camera poses.

    Args:
        file_path (str): Path to COLMAP's images.txt file.

    Returns:
        np.ndarray: Array of camera positions (N, 3).
        np.ndarray: Array of camera orientations as rotation matrices (N, 3, 3).
    """
    positions = []
    rotations = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            data = line.strip().split()
            if len(data) > 11:  # Skip invalid lines
                continue
            # Parse pose information
            qw, qx, qy, qz = map(float, data[1:5])  # Quaternion components
            tx, ty, tz = map(float, data[5:8])      # Translation vector
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            positions.append(t)
            rotations.append(R)
    return np.array(positions), np.array(rotations)

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a 3x3 rotation matrix.

    Args:
        qw, qx, qy, qz (float): Quaternion components.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_camera_pyramids(positions, rotations, scale=0.3, fov=45):
    """
    Plots camera positions and orientations as pyramids with a red near plane in 3D.

    Args:
        positions (np.ndarray): Array of camera positions (N, 3).
        rotations (np.ndarray): Array of camera orientations as rotation matrices (N, 3, 3).
        scale (float): Scale for the camera pyramids.
        fov (float): Field of view (in degrees) for the camera pyramids.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each camera as a pyramid
    for i, (position, rotation) in enumerate(zip(positions, rotations)):
        if i % 1 == 0:
            # Camera position (apex of the pyramid)
            ax.scatter(position[0], position[1], position[2], color='red', s=1)

            # Add pose index text slightly above the apex
            ax.text(
            position[0], position[1], position[2] + scale * 0.5,  # Adjust Z-axis offset
            str(i), color='black', fontsize=8
            )

            # Define the near plane points
            aspect_ratio = 16 / 9  # Assuming a standard image aspect ratio
            half_fov_rad = np.radians(fov / 2)
            near = scale

            # Near plane corners in the camera's local frame
            near_height = 2 * np.tan(half_fov_rad) * near
            near_width = near_height * aspect_ratio
            near_plane = np.array([
                [-near_width / 2, -near_height / 2, -near],  # Bottom-left
                [near_width / 2, -near_height / 2, -near],   # Bottom-right
                [near_width / 2, near_height / 2, -near],    # Top-right
                [-near_width / 2, near_height / 2, -near]    # Top-left
            ])

            # Transform to world frame
            near_plane_world = (rotation @ near_plane.T).T + position

            # Plot the pyramid edges
            for corner in near_plane_world:
                ax.plot(
                    [position[0], corner[0]],
                    [position[1], corner[1]],
                    [position[2], corner[2]],
                    color='red',
                    linewidth=0.5
                )

            # Connect the near plane edges
            for j in range(4):
                ax.plot(
                    [near_plane_world[j, 0], near_plane_world[(j + 1) % 4, 0]],
                    [near_plane_world[j, 1], near_plane_world[(j + 1) % 4, 1]],
                    [near_plane_world[j, 2], near_plane_world[(j + 1) % 4, 2]],
                    color='red',
                    linewidth=0.5
                )

            # Add the near plane as a filled polygon with transparency
            near_plane_polygon = Poly3DCollection(
                [near_plane_world], color='red', alpha=0.3
            )
            ax.add_collection3d(near_plane_polygon)

    # Add labels and adjust view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.8, 1.5)
    ax.set_ylim(-3, 1.1)
    ax.set_zlim(0, 8)
    ax.set_title('Camera Poses with Triangular Pyramids')
    plt.show()

# File path to COLMAP's images.txt file
images_txt_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/sparse_model/images.txt"

# Parse the camera poses
positions, rotations = parse_colmap_images(images_txt_path)
# print(rotations.shape)
positions, rotations = positions[[0, 5, 20, 50, 100, 12, 18, 200, 240, 270]], rotations[[0, 5, 20, 50, 100, 12, 18, 200, 240, 270]]

# Visualize the camera poses as pyramids
plot_camera_pyramids(positions, rotations)
