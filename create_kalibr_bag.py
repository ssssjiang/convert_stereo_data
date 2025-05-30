#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def get_image_files(base_dir, cam_name):
    """Helper function to get sorted image files for a camera."""
    image_dir = os.path.join(base_dir, cam_name)
    files = glob.glob(os.path.join(image_dir, '*.png'))
    files.sort()  # Ensure chronological order based on filename
    return files

def create_rosbag(output_bag_name, base_data_dir):
    """
    Creates a ROS bag file from images in cam0 and cam1 directories.

    Args:
        output_bag_name (str): The name of the ROS bag file to create.
        base_data_dir (str): The base directory containing cam0 and cam1 folders.
    """
    bridge = CvBridge()
    cam0_files = get_image_files(base_data_dir, 'cam0')
    cam1_files = get_image_files(base_data_dir, 'cam1')

    if not cam0_files and not cam1_files:
        print("No image files found in cam0 or cam1 directories.")
        return

    print("Found {} images for cam0".format(len(cam0_files)))
    print("Found {} images for cam1".format(len(cam1_files)))

    with rosbag.Bag(output_bag_name, 'w') as bag:
        print("Creating ROS bag: {}".format(output_bag_name))

        # Process cam0 images
        for idx, fpath in enumerate(cam0_files):
            filename = os.path.basename(fpath)
            timestamp_ns_str = os.path.splitext(filename)[0]
            try:
                timestamp_ns = int(timestamp_ns_str)
            except ValueError:
                print("Warning: Could not parse timestamp from filename {}. Skipping.".format(filename))
                continue

            secs = timestamp_ns // 10**9
            nsecs = timestamp_ns % 10**9
            ros_time = rospy.Time(secs, nsecs)

            try:
                cv_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if cv_image is None:
                    print("Warning: Could not read image {}. Skipping.".format(fpath))
                    continue
                # Kalibr typically expects mono8 encoding for grayscale
                image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
                image_msg.header.stamp = ros_time
                image_msg.header.frame_id = "cam0" # Or your desired frame_id
                bag.write('/cam0/image_raw', image_msg, ros_time)
                if (idx + 1) % 10 == 0:
                    print("Processed {}/{} images for cam0...".format(idx + 1, len(cam0_files)))
            except Exception as e:
                print("Error processing image {} for cam0: {}".format(fpath, e))


        print("Finished processing cam0 images.")

        # Process cam1 images
        for idx, fpath in enumerate(cam1_files):
            filename = os.path.basename(fpath)
            timestamp_ns_str = os.path.splitext(filename)[0]
            try:
                timestamp_ns = int(timestamp_ns_str)
            except ValueError:
                print("Warning: Could not parse timestamp from filename {}. Skipping.".format(filename))
                continue

            secs = timestamp_ns // 10**9
            nsecs = timestamp_ns % 10**9
            ros_time = rospy.Time(secs, nsecs)

            try:
                cv_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if cv_image is None:
                    print("Warning: Could not read image {}. Skipping.".format(fpath))
                    continue
                image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
                image_msg.header.stamp = ros_time
                image_msg.header.frame_id = "cam1" # Or your desired frame_id
                bag.write('/cam1/image_raw', image_msg, ros_time)
                if (idx + 1) % 10 == 0:
                    print("Processed {}/{} images for cam1...".format(idx + 1, len(cam1_files)))
            except Exception as e:
                print("Error processing image {} for cam1: {}".format(fpath, e))

        print("Finished processing cam1 images.")
        print("ROS bag {} created successfully.".format(output_bag_name))

if __name__ == '__main__':
    # Assume the script is run from the directory containing cam0 and cam1,
    # or provide the correct path to your data.
    data_directory = '/home/roborock/下载/标定图像_20250527/25032711615'  # Current directory
    output_bagfile = '/home/roborock/下载/标定图像_20250527/25032711615/kalibr_data.bag'

    # Check if data directory exists
    if not os.path.isdir(os.path.join(data_directory, 'cam0')) or \
       not os.path.isdir(os.path.join(data_directory, 'cam1')):
        print("Error: 'cam0' or 'cam1' directory not found in {}".format(os.path.abspath(data_directory)))
        print("Please run this script from the directory containing 'cam0' and 'cam1' folders,")
        print("or modify the 'data_directory' variable in the script.")
    else:
        try:
            create_rosbag(output_bagfile, data_directory)
        except Exception as e:
            print("An error occurred: {}".format(e))
            print("Please ensure you have ROS environment sourced and necessary libraries (python-rosbag, opencv-python, cv_bridge) installed.") 