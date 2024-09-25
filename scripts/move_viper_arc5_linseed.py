#!/usr/bin/env python

import os
import tf
import sys
import time
import copy
import math
import threading
import queue
import rospy
import cv2
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from viper_ros.srv import SavePointCloud
from viper_moveit_class import MoveGroupPythonIntefaceTutorial

class viper_robot:
    def __init__(self):
        self.viper_moveit = MoveGroupPythonIntefaceTutorial()
        self.save_dir = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_24th_Sep_linseed/010"
        self.viper_moveit.group.set_max_velocity_scaling_factor(0.6)
        self.collecting_data = False
        self.pose_id = 0
        self.bridge = CvBridge()

        self.color_image = None
        self.depth_image = None

        self.image_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.start()

        # Subscribers for color and depth images
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_image_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_image_callback)

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

    def start_data_collection(self):
        self.collecting_data = True
        self.pose_id = 0
        print("Started data collection")

    def stop_data_collection(self):
        self.collecting_data = False
        print("Stopped data collection")

    def color_image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_queue.put(('color', color_image))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.image_queue.put(('depth', depth_image))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def process_images(self):
        while not rospy.is_shutdown():
            try:
                image_type, image = self.image_queue.get(timeout=1)
                if image_type == 'color':
                    self.color_image = image
                elif image_type == 'depth':
                    self.depth_image = image
                self.image_queue.task_done()
            except queue.Empty:
                pass

    def capture_camera_data(self):
        if not self.collecting_data or self.color_image is None or self.depth_image is None:
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        depth_image_path = os.path.join(self.save_dir, f'depth_{self.pose_id}.png')
        color_image_path = os.path.join(self.save_dir, f'color_{self.pose_id}.png')

        cv2.imwrite(depth_image_path, self.depth_image)
        cv2.imwrite(color_image_path, self.color_image)

        print(f'Saved depth image to {depth_image_path}')
        print(f'Saved color image to {color_image_path}')

        # Save point cloud
        pointcloud_filename = os.path.join(self.save_dir, f'pointcloud_{self.pose_id}_')
        self.save_pointcloud(pointcloud_filename)

        self.pose_id += 1

    def save_pointcloud(self, filename):
        rospy.loginfo(f"Waiting for service '/save_pointcloud'")
        rospy.wait_for_service('/save_pointcloud', timeout=10)  # Increase the timeout
        try:
            save_pointcloud = rospy.ServiceProxy('/save_pointcloud', SavePointCloud)
            response = save_pointcloud(filename)
            if response.success:
                print(f"Point cloud saved as {filename}")
            else:
                print(f"Failed to save point cloud: {filename}")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        except rospy.ROSException as e:
            print(f"Service timeout: {e}")

    def normalize_quaternion(self, quaternion):
        norm = math.sqrt(quaternion[0]**2 + quaternion[1]**2 + quaternion[2]**2 + quaternion[3]**2)
        return [quaternion[0] / norm, quaternion[1] / norm, quaternion[2] / norm, quaternion[3] / norm]

    def set_start_pose(self):
        start_pose = self.viper_moveit.group.get_current_pose().pose
        start_pose.position.x = 0.3
        start_pose.position.y = 0.0
        start_pose.position.z = 0.4
        start_pose.orientation.x = -0.0000
        start_pose.orientation.y = 0.0000
        start_pose.orientation.z = 0.0000
        start_pose.orientation.w = 1.0000
        self.viper_moveit.go_to_pose_goal(start_pose)
        return start_pose

    def move_to_side_pose(self, start_pose):
        side_pose = copy.deepcopy(start_pose)
        side_pose.position.x = 0.0
        side_pose.position.y = -0.5
        side_pose.position.z = -0.05
        side_quat = np.asarray(tf.transformations.quaternion_from_euler(0.0, 0.0, -math.pi / 2))
        side_quat = self.normalize_quaternion(side_quat)
        side_pose.orientation.x = side_quat[0]
        side_pose.orientation.y = side_quat[1]
        side_pose.orientation.z = side_quat[2]
        side_pose.orientation.w = side_quat[3]

        self.viper_moveit.go_to_pose_goal(side_pose)
        return side_pose

    def move_to_side_down_pose(self, side_pose):
        side_down_pose = copy.deepcopy(side_pose)
        side_down_pose.position.z -= 0.05
        self.viper_moveit.go_to_pose_goal(side_down_pose)
        return side_down_pose

    def plan_and_execute_trajectory(self, waypoints, eef_step=0.01, jump_threshold=0.0, velocity_scaling=1.0):
        self.viper_moveit.group.set_max_velocity_scaling_factor(velocity_scaling)

        for i in range(len(waypoints) - 1):
            single_waypoint = [waypoints[i], waypoints[i + 1]]
            plan, fraction = self.viper_moveit.group.compute_cartesian_path(
                single_waypoint, eef_step, jump_threshold)

            display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                           moveit_msgs.msg.DisplayTrajectory,
                                                           queue_size=20)
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.viper_moveit.group.get_current_state()
            display_trajectory.trajectory.append(plan)
            display_trajectory_publisher.publish(display_trajectory)

            rospy.loginfo(f"Planned fraction of the trajectory: {fraction}")

            if fraction > 0.95:
                self.viper_moveit.group.execute(plan, wait=True)
                self.capture_camera_data()
                rospy.sleep(2)  # Pause for data capture
            else:
                rospy.logwarn("Only a small fraction of the trajectory was planned. Execution aborted.")

    def perform_motion_sequence(self):
        start_pose = self.set_start_pose()
        side_pose = self.move_to_side_pose(start_pose)
        # side_down_pose = self.move_to_side_down_pose(side_pose)

        self.start_data_collection()

        waypoints = [side_pose]
        radius = 0.8
        theta = np.linspace(0, 45, 10, True) * (math.pi / 180.0)

        for angle in theta:
            wpose = geometry_msgs.msg.Pose()
            wpose.position.x = side_pose.position.x
            wpose.position.y = side_pose.position.y - radius * (1 - math.cos(angle))
            wpose.position.z = side_pose.position.z + radius * math.sin(angle) * 1.2

            rot_x = 0.0
            rot_y = angle
            rot_z = -math.pi / 2

            new_quat = tf.transformations.quaternion_from_euler(rot_x, rot_y, rot_z)
            normalized_quaternion = self.normalize_quaternion(new_quat)
            wpose.orientation.x = normalized_quaternion[0]
            wpose.orientation.y = normalized_quaternion[1]
            wpose.orientation.z = normalized_quaternion[2]
            wpose.orientation.w = normalized_quaternion[3]

            waypoints.append(wpose)

        self.plan_and_execute_trajectory(waypoints, velocity_scaling=0.5)

        rospy.sleep(2)

        self.go_sleep_pose()

        self.stop_data_collection()

def main():
    robot = viper_robot()
    robot.perform_motion_sequence()

    moveit_commander.roscpp_shutdown()
    rospy.signal_shutdown("Trajectory complete")

if __name__ == "__main__":
    main()
