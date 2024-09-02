#!/usr/bin/env python

import rospy
from viper_ros.srv import SavePointCloud
import time

def save_pointcloud_client(filename):
    rospy.wait_for_service('save_pointcloud')
    try:
        save_pointcloud = rospy.ServiceProxy('save_pointcloud', SavePointCloud)
        response = save_pointcloud(filename)
        return response.success
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return False

if __name__ == "__main__":
    rospy.init_node('capture_multiple_views', anonymous=True)

    base_filename = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/pointclouds/pointcloud"

    viewpoints = 5  # Number of different viewpoints
    for i in range(viewpoints):
        # Create a unique filename for each viewpoint
        unique_filename = f"{base_filename}_{i}"
        success = save_pointcloud_client(unique_filename)
        if success:
            print(f"Point cloud {i + 1} saved as {unique_filename}.ply")
        else:
            print(f"Failed to save point cloud {i + 1}")
        time.sleep(3)  # Sleep to give time to move the sensor

    print("All point clouds captured")
