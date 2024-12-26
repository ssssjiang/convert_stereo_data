#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rosbag
import argparse
import os
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract /vio_pose and /imu0 data and save in respective formats.")
    parser.add_argument('--bag_file', type=str, required=True, help="Path to the input ROS bag file.")
    parser.add_argument('--output_pose_file', type=str, required=True, help="Path to the output TUM pose file.")
    parser.add_argument('--output_imu_file', type=str, required=True, help="Path to the output IMU data file.")
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

def main():
    args = parse_arguments()
    save_pose_to_tum(args.bag_file, args.output_pose_file)
    save_imu_to_file(args.bag_file, args.output_imu_file)

if __name__ == "__main__":
    main()
