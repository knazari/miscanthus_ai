import open3d as o3d
import numpy as np

# Load the two point clouds from .pcd files
pcd1 = o3d.io.read_point_cloud("/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/right_transformed_2HR.pcd")  # Change to your first file path
pcd2 = o3d.io.read_point_cloud("/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/front_transformed_2HR.pcd")  # Change to your second file path
pcd3 = o3d.io.read_point_cloud("/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/left_transformed_2HR.pcd")

# Visualize the point clouds and the marker
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud Visualization with Marker")

# Add the two point clouds and the marker to the visualization
# vis.add_geometry(pcd1)
vis.add_geometry(pcd2)
vis.add_geometry(pcd3)

# Set the initial zoom level
view_control = vis.get_view_control()
view_control.set_zoom(0.5)

vis.run()
vis.destroy_window()
