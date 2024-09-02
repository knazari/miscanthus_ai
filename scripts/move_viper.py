#!/usr/bin/env python

import sys
import copy
import rospy
import random
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from six.moves import input
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from moveit_commander.conversions import pose_to_list
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import os

from viper_moveit_class import MoveGroupPythonIntefaceTutorial


class viper_robot:
    def __init__(self):
        self.viper_moveit = MoveGroupPythonIntefaceTutorial()
        self.save_dir = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/11"

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
    
    def go_data_collection_poses(self):
        goal_joint_values = self.viper_moveit.group.get_current_joint_values()
        goal_joint_values[0] = -1.4741555452346802
        goal_joint_values[2] += 0.3236
        self.viper_moveit.group.go(goal_joint_values, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.viper_moveit.group.stop()

        time.sleep(1)
        
        # self.capture_camera_data(self.save_dir, "pose"+str(1))

        time.sleep(1)

        goal_joint_values[2] = goal_joint_values[2] - 1.5
        # goal_joint_values[4] = goal_joint_values[4] + 0.785398
        goal_joint_values[4] = goal_joint_values[4] + 1.6
        self.viper_moveit.group.go(goal_joint_values, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.viper_moveit.group.stop()

        time.sleep(1)
        
        # self.capture_camera_data(self.save_dir, "pose"+str(2))

        time.sleep(1)

        goal_joint_values[2] = goal_joint_values[2] + 2 * 0.785398 + 0.4
        goal_joint_values[4] = goal_joint_values[4] - 2 * 0.785398 - 0.5
        self.viper_moveit.group.go(goal_joint_values, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.viper_moveit.group.stop()

        time.sleep(1)
        
        # self.capture_camera_data(self.save_dir, "pose"+str(3))

        time.sleep(1)

        goal_joint_values[2] = goal_joint_values[2] - 0.785398 + 0.6
        goal_joint_values[4] = goal_joint_values[4] + 0.785398
        goal_joint_values[0] = goal_joint_values[0] + 0.785398
        goal_joint_values[3] = goal_joint_values[3] + 2 * 0.785398
        goal_joint_values[4] = goal_joint_values[4] - 1.5 * 0.785398
        goal_joint_values[5] = goal_joint_values[5] - 2.1 * 0.785398

        self.viper_moveit.group.go(goal_joint_values, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.viper_moveit.group.stop()

        time.sleep(1)
        
        # self.capture_camera_data(self.save_dir, "pose"+str(4))

        time.sleep(1)

        # goal_joint_values[2] = goal_joint_values[2] + 0.785398
        # goal_joint_values[4] = goal_joint_values[4] - 2 * 0.785398
        goal_joint_values[0] = goal_joint_values[0] - 2 * 0.785398
        goal_joint_values[3] = goal_joint_values[3] - 4 * 0.785398
        goal_joint_values[4] = goal_joint_values[4] + 0.1 * 0.785398
        goal_joint_values[5] = goal_joint_values[5] + 4.2 * 0.785398

        self.viper_moveit.group.go(goal_joint_values, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.viper_moveit.group.stop()

        time.sleep(1)
        
        # self.capture_camera_data(self.save_dir, "pose"+str(5))

        time.sleep(1)

    
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


def main():
    # Initialize the ViperMoveIt object
    robot = viper_robot()
    current_joints = robot.viper_moveit.group.get_current_joint_values()
    print("hiiiii")
    print(current_joints)
    print("======")
    print(type(current_joints))
    robot.go_home_pose()
    time.sleep(1)
    robot.go_data_collection_poses()
    time.sleep(2)
    robot.go_sleep_pose()

    # wpose = viper_moveit.group.get_current_pose().pose
    # print(wpose)
    # quat = quaternion_from_euler(goal_pose[3], goal_pose[4], goal_pose[5])
    # goal_pose = geometry_msgs.msg.Pose()
    # goal_pose.position.x = wpose.position.x
    # goal_pose.position.y = wpose.position.y
    # goal_pose.position.z = wpose.position.z + 0.2
    # goal_pose.orientation.x = wpose.orientation.x
    # goal_pose.orientation.y = wpose.orientation.y
    # goal_pose.orientation.z = wpose.orientation.z
    # goal_pose.orientation.w = wpose.orientation.w

    # print("============ Press `Enter` to execute a movement using a pose goal ...")
    # input()
    # viper_moveit.go_to_pose_goal(goal_pose)

    # initial_euler = euler_from_quaternion([goal_pose.orientation.x, goal_pose.orientation.y,\
    #                                        goal_pose.orientation.z, goal_pose.orientation.w])
    # print("initial: ", initial_euler)
    # initial_euler = list(initial_euler)
    # initial_euler[1] += 20
    # new_orientation_quat = quaternion_from_euler(initial_euler[0], initial_euler[1], initial_euler[2])
    # goal_pose.orientation.x = new_orientation_quat[0]
    # goal_pose.orientation.y = new_orientation_quat[1]
    # goal_pose.orientation.z = new_orientation_quat[2]
    # goal_pose.orientation.w = new_orientation_quat[3]
    # print("second: ", initial_euler)

    # print("============ Press `Enter` to execute a movement using a pose goal ...")
    # input()
    # viper_moveit.go_to_pose_goal(goal_pose)


    # Define a few target poses for the end-effector
    poses = [
        # Pose 1
        geometry_msgs.msg.Pose(
            position=geometry_msgs.msg.Point(0.2, 0.2, 0.2),
            orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)
        ),
        # Pose 2
        geometry_msgs.msg.Pose(
            position=geometry_msgs.msg.Point(0.3, -0.2, 0.4),
            orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)
        ),
        # Pose 3
        geometry_msgs.msg.Pose(
            position=geometry_msgs.msg.Point(0.1, 0.3, 0.3),
            orientation=geometry_msgs.msg.Quaternion(0.0, 0.0, 0.0, 1.0)
        )
    ]

    # Move to the defined poses
    # viper_moveit.move_to_poses(poses)

    # Shutdown the moveit_commander
    # viper_moveit.shutdown()

if __name__ == "__main__":
    main()
