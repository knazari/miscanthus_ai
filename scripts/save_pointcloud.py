import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def rs_to_o3d(points, color_frame):
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # XYZ coordinates
    tex = np.asanyarray(color_frame.get_data()).reshape(-1, 3) / 255.0  # RGB data
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(vtx)
    o3d_pcd.colors = o3d.utility.Vector3dVector(tex)
    return o3d_pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def pairwise_registration(source, target):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for source_id in range(len(pcds)):
        for target_id in range(source_id + 1, len(pcds)):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            if target_id == source_id + 1:  # Odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 information_icp,
                                                                                 uncertain=False))
            else:  # Loop closure case
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 information_icp,
                                                                                 uncertain=True))
    return pose_graph

def optimize_pose_graph(pose_graph, max_correspondence_distance_fine):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create a pointcloud object
pc = rs.pointcloud()
points = rs.points()

point_clouds = []
voxel_size = 0.05
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

try:
    while True:
        if input("Press Enter to capture a point cloud (or type 'exit' to finish): ").strip().lower() == 'exit':
            break

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Depth or Color frame not available. Skipping...")
            continue

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        o3d_pcd = rs_to_o3d(points, color_frame)
        point_clouds.append(preprocess_point_cloud(o3d_pcd, voxel_size))
        print(f"Captured point cloud {len(point_clouds)}")

        if len(point_clouds) >= 5:  # Example limit
            break

finally:
    pipeline.stop()

# Register all point clouds
pose_graph = full_registration(point_clouds, max_correspondence_distance_coarse, max_correspondence_distance_fine)
optimize_pose_graph(pose_graph, max_correspondence_distance_fine)

# Merge point clouds
unified_point_cloud = o3d.geometry.PointCloud()
for point_id in range(len(point_clouds)):
    pcd = point_clouds[point_id]
    pcd.transform(pose_graph.nodes[point_id].pose)
    unified_point_cloud += pcd

unified_point_cloud = unified_point_cloud.voxel_down_sample(voxel_size)

# Save the unified point cloud
o3d.io.write_point_cloud("unified_point_cloud.ply", unified_point_cloud)

# Visualize the unified point cloud
o3d.visualization.draw_geometries([unified_point_cloud])
