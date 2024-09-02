#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from viper_ros.srv import SavePointCloud, SavePointCloudResponse
import open3d as o3d
import numpy as np
import struct
import os
import time

current_pointcloud = None

def callback(pointcloud_msg):
    global current_pointcloud
    current_pointcloud = pointcloud_msg
    pub.publish(pointcloud_msg)

def handle_save_pointcloud(req):
    rospy.loginfo("Service handle_save_pointcloud called")
    
    if current_pointcloud is None:
        rospy.logwarn("No point cloud data available")
        return SavePointCloudResponse(success=False)
    
    rospy.loginfo("Processing point cloud data")
    
    # Get the field names to determine available data
    field_names = [field.name for field in current_pointcloud.fields]
    rospy.loginfo(f"Point cloud fields: {field_names}")

    # Adjust the reading based on available fields
    points_list = []
    for point in pc2.read_points(current_pointcloud, skip_nans=True):
        if 'rgb' in field_names:
            # Unpack RGB value from float
            rgb_packed = struct.unpack('I', struct.pack('f', point[3]))[0]
            r = (rgb_packed >> 16) & 0xFF
            g = (rgb_packed >> 8) & 0xFF
            b = rgb_packed & 0xFF
            points_list.append([point[0], point[1], point[2], r, g, b])
        else:
            points_list.append([point[0], point[1], point[2], 255, 255, 255])  # Default to white color if no RGB

    np_points = np.asarray(points_list)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np_points[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(np_points[:, 3:] / 255.0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{req.filename}_{timestamp}.ply"
    o3d.io.write_point_cloud(filename, o3d_pcd)
    rospy.loginfo(f"Saved point cloud to {filename}")
    return SavePointCloudResponse(success=True)

def main():
    rospy.init_node('realsense_publisher', anonymous=True)
    
    rospy.Subscriber('/camera/depth/color/points', PointCloud2, callback)
    global pub
    pub = rospy.Publisher('realsense/processed_points', PointCloud2, queue_size=10)

    save_service = rospy.Service('save_pointcloud', SavePointCloud, handle_save_pointcloud)
    rospy.loginfo("Service save_pointcloud is ready")

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
