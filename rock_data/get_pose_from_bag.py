#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rosbag
import argparse
import os
from nav_msgs.msg import Odometry

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract /vio_pose data and save in TUM format.")
    parser.add_argument('--bag_file', type=str, required=True, help="Path to the input ROS bag file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output TUM file.")
    return parser.parse_args()

def save_to_tum(bag_file, output_file, topic='/vio_pose'):
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

            print("Processed {} messages from topic '{}'.".format(message_count, topic))
            if message_count == 0:
                print("No messages found on topic '{}'.".format(topic))

    except Exception as e:
        print("Error extracting /vio_pose: {}".format(e))

def main():
    args = parse_arguments()
    save_to_tum(args.bag_file, args.output_file)

if __name__ == "__main__":
    main()
