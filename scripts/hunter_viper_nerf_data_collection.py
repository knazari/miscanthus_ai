#!/usr/bin/env python3

import os
import tf
import cv2
import time
import math
import queue
import rospy
import threading
import numpy as np
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from viper_ros.srv import SavePointCloud
from cv_bridge import CvBridge, CvBridgeError
from viper_moveit_class import MoveGroupPythonIntefaceTutorial

class DataCollection:
    def __init__(self):
        # rospy.init_node('hunter_viper_nerf_data_collection', anonymous=True)

        # Initialize the Hunter SE velocity publisher
        self.hunter_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize the ViperX MoveIt interface
        self.viper_moveit = MoveGroupPythonIntefaceTutorial()
        self.viper_moveit.group.set_max_velocity_scaling_factor(0.6)

        # Image capture settings
        self.bridge = CvBridge()
        self.save_dir = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/indoor_miscanthus_dataset/025/"
        self.image_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.start()

        # Subscribers for RealSense camera
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_image_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_image_callback)

        # Initial pose ID for saved images
        self.pose_id = 0
        self.color_image = None
        self.depth_image = None
        self.collecting_data = False

    def move_hunter_in_circle(self, duration=30):
        """
        Moves the Hunter SE in a 2-meter radius circle for a set duration.
        """
        vel_msg = Twist()
        vel_msg.linear.x = 0.2  # Move forward
        motion_circle_radius = 2
        vel_msg.angular.z = vel_msg.linear.x / motion_circle_radius  # Rotate to maintain circular motion (0.5 / 2m radius)

        start_time = time.time()
        rospy.loginfo("Moving Hunter SE in a circle for {} seconds...".format(duration))

        while time.time() - start_time < duration and not rospy.is_shutdown():
            self.hunter_pub.publish(vel_msg)
            self.capture_camera_data()  # Capture camera images during motion
            time.sleep(0.1)  # Maintain loop rate

        # Stop the robot
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.hunter_pub.publish(vel_msg)
        rospy.loginfo("Hunter SE movement completed.")

    def move_viper_to_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Moves the ViperX arm to a specific pose.
        """
        goal_pose = geometry_msgs.msg.Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        goal_pose.position.z = z

        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        quaternion = self.normalize_quaternion(quaternion)
        goal_pose.orientation.x = quaternion[0]
        goal_pose.orientation.y = quaternion[1]
        goal_pose.orientation.z = quaternion[2]
        goal_pose.orientation.w = quaternion[3]

        self.viper_moveit.go_to_pose_goal(goal_pose)
        rospy.sleep(2)  # Allow time for movement
    
    def normalize_quaternion(self, quaternion):
        norm = math.sqrt(quaternion[0]**2 + quaternion[1]**2 + quaternion[2]**2 + quaternion[3]**2)
        return [quaternion[0] / norm, quaternion[1] / norm, quaternion[2] / norm, quaternion[3] / norm]

    def start_data_collection(self):
        """Starts image logging."""
        self.collecting_data = True
        rospy.loginfo("Started data collection.")

    def stop_data_collection(self):
        """Stops image logging."""
        self.collecting_data = False
        rospy.loginfo("Stopped data collection.")

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
        """Threaded function for processing camera images."""
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
        """Saves images from the wrist camera."""
        if not self.collecting_data or self.color_image is None or self.depth_image is None:
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        depth_path = os.path.join(self.save_dir, f'depth_{self.pose_id}.png')
        color_path = os.path.join(self.save_dir, f'color_{self.pose_id}.png')

        # cv2.imwrite(depth_path, self.depth_image)
        cv2.imwrite(color_path, self.color_image)

        rospy.loginfo(f'Saved images: {color_path}, {depth_path}')

        self.pose_id += 1

    def perform_data_collection(self):
        """Main sequence for capturing NeRF training data."""
        self.start_data_collection()

        # Define different arm heights for data collection
        arm_heights = [0.4, 0.5, 0.6]  # Adjust Z positions

        for height in arm_heights:
            rospy.loginfo(f"Moving arm to height {height}m")
            if height == 0.4:
                self.move_viper_to_pose(x=-0.1, y=-0.15, z=height, roll=0, pitch=0.2, yaw=1.57)
            else:
                self.move_viper_to_pose(x=-0.1, y=-0.15, z=height, roll=0, pitch=0.0, yaw=1.57)

            rospy.sleep(2)  # Wait for arm to stabilize
            self.move_hunter_in_circle(duration=55)  # Move Hunter SE in a circle while collecting images

        self.go_sleep_pose()
        self.stop_data_collection()
        rospy.loginfo("Data collection completed.")
    
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

def main():
    collector = DataCollection()
    collector.perform_data_collection()
    moveit_commander.roscpp_shutdown()
    rospy.signal_shutdown("Data collection completed.")

if __name__ == "__main__":
    main()
