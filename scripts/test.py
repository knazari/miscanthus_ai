import open3d as o3d
import numpy as np

# Load the point cloud from a .pcd file
pcd = o3d.io.read_point_cloud("/home/kiyanoush/Downloads/Walled Garden 24 07 29 arbitrary.ply")  # Change to your file path

# Create a small red sphere as the marker
def create_marker(radius=0.1, color=[1, 0, 0], position=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere

# Initial position of the marker
marker_position = [0, 0, 0]
marker = create_marker(position=marker_position)

# Function to update the marker's position
def update_marker_position(vis, new_position):
    global marker  # Declare marker as global before modifying it
    vis.remove_geometry(marker, reset_bounding_box=False)
    marker = create_marker(position=new_position)
    vis.add_geometry(marker, reset_bounding_box=False)

    # Print the new coordinates of the marker
    print(f"Marker position: x={new_position[0]:.3f}, y={new_position[1]:.3f}, z={new_position[2]:.3f}")

# Key callback functions to move the marker in different directions
def move_forward(vis):
    global marker_position
    marker_position[2] += 0.1  # Move along z-axis
    update_marker_position(vis, marker_position)

def move_backward(vis):
    global marker_position
    marker_position[2] -= 0.1  # Move along z-axis
    update_marker_position(vis, marker_position)

def move_left(vis):
    global marker_position
    marker_position[0] -= 0.1  # Move along x-axis
    update_marker_position(vis, marker_position)

def move_right(vis):
    global marker_position
    marker_position[0] += 0.1  # Move along x-axis
    update_marker_position(vis, marker_position)

def move_up(vis):
    global marker_position
    marker_position[1] += 0.1  # Move along y-axis
    update_marker_position(vis, marker_position)

def move_down(vis):
    global marker_position
    marker_position[1] -= 0.1  # Move along y-axis
    update_marker_position(vis, marker_position)

# Visualize the point cloud and the marker
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud Visualization with Marker")

# Add the point cloud and the initial marker
vis.add_geometry(pcd)
# vis.add_geometry(marker)


# Register key callbacks for arrow keys
vis.register_key_callback(265, move_up)     # Up Arrow (GLFW_KEY_UP = 265)
vis.register_key_callback(264, move_down)   # Down Arrow (GLFW_KEY_DOWN = 264)
vis.register_key_callback(263, move_left)   # Left Arrow (GLFW_KEY_LEFT = 263)
vis.register_key_callback(262, move_right)  # Right Arrow (GLFW_KEY_RIGHT = 262)

# Register callbacks for Q and E for forward and backward along z-axis
vis.register_key_callback(ord("Q"), move_forward)  # Move forward (up in z-axis)
vis.register_key_callback(ord("E"), move_backward) # Move backward (down in z-axis)


# Set the initial zoom level
view_control = vis.get_view_control()
view_control.set_zoom(0.5)

vis.run()
vis.destroy_window()