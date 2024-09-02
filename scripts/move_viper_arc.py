#!/usr/bin/env python

import os
import tf
import sys
import cv2
import time
import copy
import math
import rospy
import random
import numpy as np
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
import pyrealsense2 as rs
from six.moves import input
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler, euler_from_quaternion


from viper_moveit_class import MoveGroupPythonIntefaceTutorial


class viper_robot:
    def __init__(self):
        self.viper_moveit = MoveGroupPythonIntefaceTutorial()
        self.save_dir = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/10"

    def go_sleep_pose(self):
        goal_pose = geometry_msgs.msg.Pose()
        goal_pose.position.x = 0.1342
        goal_pose.position.y = -0.0000
        goal_pose.position.z = 0.088
        goal_pose.orientation.x = 0.0007
        goal_pose.orientation.y = 0.2924
        goal_pose.orientation.z = -0.0002
        goal_pose.orientation.w = 0.9562
        self.viper_moveit.go_to_pose_goal(goal_pose)
    
    def go_home_pose(self):
        goal_pose = geometry_msgs.msg.Pose()
        goal_pose.position.x = 0.5418
        goal_pose.position.y = 0.0000
        goal_pose.position.z = 0.4022
        goal_pose.orientation.x = -0.0000
        goal_pose.orientation.y = 0.0000
        goal_pose.orientation.z = 0.0000
        goal_pose.orientation.w = 1.0000
        self.viper_moveit.go_to_pose_goal(goal_pose)
        self.home_pose = goal_pose
    
    
    def capture_camera_data(self, save_dir, pose_id):
        
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        config = rs.config()

        # Configure the pipeline to stream different resolutions of color and depth streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise Exception("Could not capture depth or color frame.")

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Save the images
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            depth_image_path = os.path.join(save_dir, f'depth_{pose_id}.png')
            color_image_path = os.path.join(save_dir, f'color_{pose_id}.png')

            cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(color_image_path, color_image)

            print(f'Saved depth image to {depth_image_path}')
            print(f'Saved color image to {color_image_path}')

        finally:
            # Stop streaming
            pipeline.stop()

def normalize_quaternion(quaternion):
    norm = math.sqrt(quaternion[0]**2 + quaternion[1]**2 + quaternion[2]**2 + quaternion[3]**2)
    return [quaternion[0] / norm, quaternion[1] / norm, quaternion[2] / norm, quaternion[3] / norm]

def compute_orientation(current_position, target_point):
    # Calculate the direction vector from current position to target point
    direction_vector = [
        target_point[0] - current_position.x,
        target_point[1] - current_position.y,
        target_point[2] - current_position.z
    ]
    norm = math.sqrt(sum([x**2 for x in direction_vector]))
    direction_vector = [x / norm for x in direction_vector]

    # Adjust the rotation matrix to align x-axis with the direction vector
    x_axis = direction_vector
    z_axis = [0, 0, 1]  # Assuming the z-axis remains unchanged (perpendicular to the plane)
    y_axis = [0, 0, 0]
    y_axis[0] = z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1]
    y_axis[1] = z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2]
    y_axis[2] = z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0]

    norm_y = math.sqrt(sum([x**2 for x in y_axis]))
    y_axis = [x / norm_y for x in y_axis]

    z_axis = [
        y_axis[1] * x_axis[2] - y_axis[2] * x_axis[1],
        y_axis[2] * x_axis[0] - y_axis[0] * x_axis[2],
        y_axis[0] * x_axis[1] - y_axis[1] * x_axis[0]
    ]

    rot_matrix = tf.transformations.identity_matrix()
    rot_matrix[:3, 0] = x_axis
    rot_matrix[:3, 1] = y_axis
    rot_matrix[:3, 2] = z_axis

    # Extract quaternion from the rotation matrix
    quaternion = tf.transformations.quaternion_from_matrix(rot_matrix)
    normalized_quaternion = normalize_quaternion(quaternion)

    return normalized_quaternion


def main():
    # Initialize the ViperMoveIt object
    robot = viper_robot()

    # Define the target point the end-effector's z-axis will point to
    target_point = [0.5, 0.0, 0.5]  # Example point in space

    # Define start pose
    start_pose = robot.viper_moveit.group.get_current_pose().pose
    start_pose.position.x = 0.2
    start_pose.position.y = 0.0
    start_pose.position.z = 0.3

    # Calculate the quaternion for the orientation
    quaternion = tf.transformations.quaternion_from_euler(0, 0, math.atan2(target_point[1] - start_pose.position.y, target_point[0] - start_pose.position.x))
    start_pose.orientation.x = quaternion[0]
    start_pose.orientation.y = quaternion[1]
    start_pose.orientation.z = quaternion[2]
    start_pose.orientation.w = quaternion[3]

    # Set the start state to the current state
    robot.viper_moveit.group.set_start_state_to_current_state()

    # Define waypoints for the arc trajectory
    waypoints = []

    # Start pose
    waypoints.append(start_pose)

    # Define the arc trajectory
    center_x = 0.5
    center_y = 0.0
    radius = 0.3
    num_points = 20

    for i in range(num_points + 1):
        angle = i * (math.pi / num_points)
        wpose = geometry_msgs.msg.Pose()
        wpose.position.x = center_x + radius * math.cos(angle)
        wpose.position.y = center_y + radius * math.sin(angle)
        wpose.position.z = start_pose.position.z

        # Calculate the quaternion for the orientation to keep the x-axis toward the target point
        quaternion = compute_orientation(wpose.position, target_point)
        wpose.orientation.x = quaternion[0]
        wpose.orientation.y = quaternion[1]
        wpose.orientation.z = quaternion[2]
        wpose.orientation.w = quaternion[3]

        waypoints.append(wpose)


    # Print waypoints for debugging
    for idx, waypoint in enumerate(waypoints):
        rospy.loginfo(f"Waypoint {idx}: Position - ({waypoint.position.x}, {waypoint.position.y}, {waypoint.position.z}), Orientation - ({waypoint.orientation.x}, {waypoint.orientation.y}, {waypoint.orientation.z}, {waypoint.orientation.w})")

    # Plan the trajectory
    (plan, fraction) = robot.viper_moveit.group.compute_cartesian_path(
        waypoints,   # waypoints to follow
        0.01,        # eef_step
        0.0)         # jump_threshold

    # Check the fraction of the trajectory that was planned
    rospy.loginfo(f"Planned fraction of the trajectory: {fraction}")

    if fraction > 0.95:
        # Execute the trajectory
        robot.viper_moveit.group.execute(plan, wait=True)
    else:
        rospy.logwarn("Only a small fraction of the trajectory was planned. Execution aborted.")


    # Execute the trajectory
    robot.viper_moveit.group.execute(plan, wait=True)

    time.sleep(5)

    robot.go_sleep_pose()

    # Shut down the MoveIt! commander
    moveit_commander.roscpp_shutdown()

    # Exit the script
    rospy.signal_shutdown("Trajectory complete")


if __name__ == "__main__":
    main()
