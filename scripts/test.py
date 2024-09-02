#!/usr/bin/env python

import rospy
import open3d as o3d
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def publish_point_cloud():
    rospy.init_node('rgb_depth_to_pointcloud')
    pub = rospy.Publisher('point_cloud_topic', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Load RGB and Depth images
    rgb_image_path = '/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/003/color_3.png'
    depth_image_path = '/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_30th_july/003/depth_3.png'

    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Get the intrinsic parameters of the RealSense camera
    width, height = rgb_image.shape[1], rgb_image.shape[0]
    fx, fy = 600, 600  # Adjust these values with your camera's intrinsic parameters
    cx, cy = width / 2, height / 2

    # Create the point cloud
    def generate_point_cloud(rgb, depth, intrinsic):
        height, width = depth.shape

        # Create a meshgrid for pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Get the depth scale
        depth_scale = 0.001  # RealSense depth scale (1 unit = 1 millimeter)

        # Get the z coordinates from the depth image
        z = depth * depth_scale

        # Calculate the x and y coordinates
        x = (u - intrinsic['cx']) * z / intrinsic['fx']
        y = (v - intrinsic['cy']) * z / intrinsic['fy']

        # Stack x, y, and z to create a 3D point cloud
        points = np.dstack((x, y, z)).reshape(-1, 3)

        # Get the colors from the RGB image
        colors = rgb.reshape(-1, 3)

        return points, colors

    # Define the intrinsic parameters
    intrinsic = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }

    # Generate the point cloud
    points, colors = generate_point_cloud(rgb_image, depth_image, intrinsic)

    # Pack RGB values into a single uint32 value
    def pack_rgb(colors):
        packed = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
        return packed

    packed_rgb = pack_rgb(colors)

    # Create the PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'  # You can set the frame ID to match your setup

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    # Combine points and packed RGB values
    points_with_colors = np.hstack((points, packed_rgb[:, np.newaxis])).tolist()
    point_cloud = pc2.create_cloud(header, fields, points_with_colors)

    while not rospy.is_shutdown():
        pub.publish(point_cloud)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_point_cloud()
    except rospy.ROSInterruptException:
        pass
