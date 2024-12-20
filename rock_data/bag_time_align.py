#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rosbag
import argparse
import os
from cv_bridge import CvBridge
import cv2
import rospy

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract and process data from a ROS bag file.")
    parser.add_argument('--bag_file', type=str, required=True, help="Path to the input ROS bag file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output ROS bag file.")
    parser.add_argument('--start_timestamp', type=int, default=0, help="Start timestamp for filtering messages.")
    parser.add_argument('--end_timestamp', type=int, default=999999999, help="End timestamp for filtering messages.")
    parser.add_argument('--camera_downsample', type=int, default=4, help="Downsampling factor for camera data.")
    return parser.parse_args()

def check_timestamps(bag_file):
    """Check if timestamps for each topic are strictly increasing."""
    last_timestamps = {}  # Dictionary to track the last timestamp for each topic
    is_strictly_increasing = True

    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                timestamp = msg.header.stamp.to_sec()

                # Check if the timestamp is strictly increasing for the topic
                if topic in last_timestamps and timestamp <= last_timestamps[topic]:
                    print "Timestamp issue on topic:", topic, "at timestamp:", timestamp
                    is_strictly_increasing = False

                # Update the last timestamp for the topic
                last_timestamps[topic] = timestamp

        if is_strictly_increasing:
            print "All topics have strictly increasing timestamps."
        else:
            print "Some topics have non-strictly increasing timestamps. Check logs for details."

    except Exception as e:
        print "Error checking timestamps in bag file:", e

def align_timestamps(bag_file):
    """Calculate timestamp offset and align all topics to /cam0/image_raw."""
    offsets = {}  # Store offsets for each topic
    camera0_first_stamp = None

    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                # Standardize topic names to ensure consistency
                if not topic.startswith('/'):
                    topic = '/' + topic

                if topic == '/cam0/image_raw' and camera0_first_stamp is None:
                    camera0_first_stamp = msg.header.stamp.to_sec()
                    print "Camera0 first timestamp:", camera0_first_stamp

                if camera0_first_stamp is not None and topic not in offsets:
                    offsets[topic] = msg.header.stamp.to_sec() - camera0_first_stamp

        if camera0_first_stamp is None:
            print "No messages found for /cam0/image_raw. Cannot align timestamps."
            return None

        for topic, offset in offsets.items():
            print "Offset for topic", topic, "is", offset, "seconds."

        return offsets

    except Exception as e:
        print "Error aligning timestamps in bag file:", e
        return None

def process_bag(args):
    bridge = CvBridge()
    cam0_count, cam1_count = 0, 0
    last_timestamps = {}  # Dictionary to track the last timestamp for each topic

    offsets = align_timestamps(args.bag_file)
    if offsets is None:
        print "Failed to calculate timestamp offsets. Exiting."
        return

    try:
        with rosbag.Bag(args.bag_file, 'r') as inbag, rosbag.Bag(args.output_file, 'w') as outbag:
            total_messages = inbag.get_message_count()
            print "Total messages in bag:", total_messages

            for idx, (topic, msg, t) in enumerate(inbag.read_messages(), start=1):
                # Standardize topic names
                if not topic.startswith('/'):
                    topic = '/' + topic

                timestamp = msg.header.stamp.to_sec()

                # Display progress
                if idx % 1000 == 0:
                    print "Processed", idx, "/", total_messages, "messages..."

                # Filter messages based on timestamp
                if timestamp < args.start_timestamp or timestamp > args.end_timestamp:
                    continue

                # Check if the timestamp is strictly increasing for the topic
                if topic in last_timestamps and timestamp <= last_timestamps[topic]:
                    print "Discarding out-of-order message on topic:", topic, "at timestamp:", timestamp
                    continue

                # Update the last timestamp for the topic
                last_timestamps[topic] = timestamp

                # Align the header timestamp to /cam0/image_raw
                if hasattr(msg, 'header') and topic in offsets:
                    aligned_time = msg.header.stamp.to_sec() - offsets[topic]
                    msg.header.stamp = rospy.Time.from_sec(aligned_time)

                if topic == '/cam0/image_raw':
                    cam0_count += 1
                    if cam0_count % args.camera_downsample != 0:
                        continue  # Skip frames not part of the downsample

                    # Resize image to 640x400
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    resized_image = cv2.resize(cv_image, (640, 400))
                    resized_msg = bridge.cv2_to_imgmsg(resized_image, "bgr8")
                    resized_msg.header = msg.header  # Retain the original header, including timestamp

                elif topic == '/cam1/image_raw':
                    cam1_count += 1
                    if cam1_count % args.camera_downsample != 0:
                        continue  # Skip frames not part of the downsample

                    # Resize image to 640x400
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    resized_image = cv2.resize(cv_image, (640, 400))
                    resized_msg = bridge.cv2_to_imgmsg(resized_image, "bgr8")
                    resized_msg.header = msg.header  # Retain the original header, including timestamp

                else:
                    # For other topics, do not apply downsampling or resizing
                    resized_msg = msg


                # Write processed message to output bag
                outbag.write(topic, resized_msg, t)

            print "Bag processing completed successfully. Output saved to:", args.output_file

    except Exception as e:
        print "Error processing bag file:", e

def main():
    args = parse_arguments()
    print "Checking timestamps in the input bag file..."
    check_timestamps(args.bag_file)
    print "Processing bag file..."
    process_bag(args)

if __name__ == "__main__":
    main()
