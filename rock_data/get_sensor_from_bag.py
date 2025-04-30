#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rosbag
import argparse
import os
import cv2
import numpy as np
import csv
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from cv_bridge import CvBridge

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract data from ROS bag and save in respective formats.")
    parser.add_argument('--bag_file', type=str, required=True, help="Path to the input ROS bag file.")
    parser.add_argument('--output_pose_file', type=str, help="Path to the output TUM pose file.")
    parser.add_argument('--output_imu_file', type=str, help="Path to the output IMU data file.")
    parser.add_argument('--output_dir', type=str, help="Directory to save camera images and data files.")
    parser.add_argument('--camera0_topic', type=str, default='/image', 
                        help="Topic name for camera0 images.")
    parser.add_argument('--camera1_topic', type=str, default='/image1', 
                        help="Topic name for camera1 images.")
    return parser.parse_args()

def save_pose_to_tum(bag_file, output_file, topic='/vio_pose'):
    """
    Extract /vio_pose data from a ROS bag and save it in TUM format.

    Parameters:
    - bag_file: Path to the input ROS bag file.
    - output_file: Path to the output TUM file.
    - topic: The topic to extract pose data from.
    """
    try:
        message_count = 0  # Counter for messages
        with rosbag.Bag(bag_file, 'r') as bag, open(output_file, 'w') as tum_file:
            print("Opening bag file: {}".format(bag_file))
            for topic_name, msg, t in bag.read_messages(topics=[topic]):
                print("Processing message on topic: {}".format(topic_name))  # Debug: Topic name
                # Extract timestamp
                timestamp = msg.header.stamp.to_sec()

                # Extract position (tx, ty, tz)
                tx = msg.pose.pose.position.x
                ty = msg.pose.pose.position.y
                tz = msg.pose.pose.position.z

                # Extract orientation (qx, qy, qz, qw)
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w

                # Write to TUM file
                tum_file.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    timestamp, tx, ty, tz, qx, qy, qz, qw))

                message_count += 1  # Increment counter
                print("Saved pose: timestamp {:.6f}, position ({:.6f}, {:.6f}, {:.6f})".format(
                    timestamp, tx, ty, tz))  # Debug: Pose details

            print("Processed {} messages from topic '{}' for pose data.".format(message_count, topic))
            if message_count == 0:
                print("No messages found on topic '{}' for pose data.".format(topic))

    except Exception as e:
        print("Error extracting /vio_pose: {}".format(e))

def save_imu_to_file(bag_file, output_file, topic='/imu0'):
    """
    Extract /imu0 data from a ROS bag and save it in a plain text format.

    Parameters:
    - bag_file: Path to the input ROS bag file.
    - output_file: Path to the output IMU data file.
    - topic: The topic to extract IMU data from.
    """
    try:
        message_count = 0  # Counter for messages
        with rosbag.Bag(bag_file, 'r') as bag, open(output_file, 'w') as imu_file:
            print("Opening bag file: {}".format(bag_file))
            for topic_name, msg, t in bag.read_messages(topics=[topic]):
                # Extract timestamp
                timestamp = msg.header.stamp.to_sec()

                # Extract angular velocity (ax, ay, az)
                ax = msg.angular_velocity.x
                ay = msg.angular_velocity.y
                az = msg.angular_velocity.z

                # Extract linear acceleration (lx, ly, lz)
                lx = msg.linear_acceleration.x
                ly = msg.linear_acceleration.y
                lz = msg.linear_acceleration.z

                # Write to IMU file
                imu_file.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    timestamp, ax, ay, az, lx, ly, lz))

                message_count += 1  # Increment counter

            print("Processed {} messages from topic '{}' for IMU data.".format(message_count, topic))
            if message_count == 0:
                print("No messages found on topic '{}' for IMU data.".format(topic))

    except Exception as e:
        print("Error extracting /imu0: {}".format(e))

def save_images_from_bag(bag_file, output_dir, camera0_topic, camera1_topic):
    """
    Extract camera images from a ROS bag and save them in specified directories.
    Also create a data.csv file in each camera directory listing timestamps and filenames.

    Parameters:
    - bag_file: Path to the input ROS bag file.
    - output_dir: Base directory to save images and data.
    - camera0_topic: Topic name for camera0 images.
    - camera1_topic: Topic name for camera1 images.
    """
    try:
        # Create directories
        camera0_dir = os.path.join(output_dir, 'camera', 'camera0')
        camera1_dir = os.path.join(output_dir, 'camera', 'camera1')
        
        for directory in [camera0_dir, camera1_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Open data.csv files
        camera0_csv = open(os.path.join(camera0_dir, 'data.csv'), 'w')
        camera1_csv = open(os.path.join(camera1_dir, 'data.csv'), 'w')
        
        # Write CSV headers
        camera0_csv.write("#timestamp [ns],filename\n")
        camera1_csv.write("#timestamp [ns],filename\n")
        
        bridge = CvBridge()
        
        # Process camera0 images
        message_count_cam0 = 0
        print("Processing images from topic: {}".format(camera0_topic))
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[camera0_topic]):
                # Get timestamp in nanoseconds
                timestamp_ns = msg.header.stamp.to_nsec()
                # Use milliseconds for filename
                timestamp_ms = int(timestamp_ns / 1000000)
                
                # Convert ROS image to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                
                # Save image
                filename = "{}.png".format(timestamp_ms)
                image_path = os.path.join(camera0_dir, filename)
                cv2.imwrite(image_path, cv_image)
                
                # Write to CSV
                camera0_csv.write("{},{}\n".format(timestamp_ns, filename))
                
                message_count_cam0 += 1
                if message_count_cam0 % 100 == 0:
                    print("Processed {} images from camera0".format(message_count_cam0))
        
        # Process camera1 images
        message_count_cam1 = 0
        print("Processing images from topic: {}".format(camera1_topic))
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[camera1_topic]):
                # Get timestamp in nanoseconds
                timestamp_ns = msg.header.stamp.to_nsec()
                # Use milliseconds for filename
                timestamp_ms = int(timestamp_ns / 1000000)
                
                # Convert ROS image to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                
                # Save image
                filename = "{}.png".format(timestamp_ms)
                image_path = os.path.join(camera1_dir, filename)
                cv2.imwrite(image_path, cv_image)
                
                # Write to CSV
                camera1_csv.write("{},{}\n".format(timestamp_ns, filename))
                
                message_count_cam1 += 1
                if message_count_cam1 % 100 == 0:
                    print("Processed {} images from camera1".format(message_count_cam1))
        
        # Close CSV files
        camera0_csv.close()
        camera1_csv.close()
        
        print("Extracted {} images from camera0 and {} images from camera1".format(
            message_count_cam0, message_count_cam1))
        
    except Exception as e:
        print("Error extracting camera images: {}".format(e))

def main():
    args = parse_arguments()
    
    if args.output_pose_file:
        save_pose_to_tum(args.bag_file, args.output_pose_file)
    
    if args.output_imu_file:
        save_imu_to_file(args.bag_file, args.output_imu_file)
    
    if args.output_dir:
        save_images_from_bag(args.bag_file, args.output_dir, args.camera0_topic, args.camera1_topic)

if __name__ == "__main__":
    main()
