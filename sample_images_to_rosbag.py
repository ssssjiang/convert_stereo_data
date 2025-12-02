#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample images from folder by stride and create rosbag for Kalibr calibration.

Usage:
    python sample_images_to_rosbag.py -i /path/to/images -o output.bag -s 5
    python sample_images_to_rosbag.py -i /path/to/cam0 -o calib.bag -s 3 -t /cam0/image_raw -f cam0
"""

import os
import argparse
import rosbag
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy


def get_sorted_image_files(image_folder):
    """
    Get all image files from folder and sort by filename.
    
    Args:
        image_folder: Path to image folder
        
    Returns:
        List of sorted image filenames
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for file in os.listdir(image_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(file)
    
    # Sort by filename
    image_files.sort()
    
    return image_files


def sample_images_to_rosbag(image_folder, output_bag, stride=1, topic_name='/camera/image_raw', 
                            frame_id='camera', encoding='mono8', fps=20.0):
    """
    Sample images from folder by stride and create rosbag.
    
    Args:
        image_folder: Path to image folder
        output_bag: Path to output rosbag file
        stride: Sample stride (take 1 image every stride images)
        topic_name: ROS topic name
        frame_id: Camera frame_id
        encoding: Image encoding ('mono8' for grayscale, 'bgr8' for color)
        fps: Frames per second for timestamp generation
    """
    # Check if image folder exists
    if not os.path.exists(image_folder):
        print("Error: Image folder does not exist: {}".format(image_folder))
        return
    
    # Get all image files and sort
    image_files = get_sorted_image_files(image_folder)
    
    if not image_files:
        print("Error: No image files found in folder: {}".format(image_folder))
        return
    
    # Sample images by stride
    sampled_files = image_files[::stride]
    
    print("="*60)
    print("Image Sampling to RosBag")
    print("="*60)
    print("Image folder: {}".format(image_folder))
    print("Total images: {}".format(len(image_files)))
    print("Sample stride: {}".format(stride))
    print("Sampled images: {}".format(len(sampled_files)))
    print("Output bag: {}".format(output_bag))
    print("Topic name: {}".format(topic_name))
    print("Frame ID: {}".format(frame_id))
    print("Encoding: {}".format(encoding))
    print("FPS: {}".format(fps))
    print("="*60)
    
    # Create rosbag
    bridge = CvBridge()
    bag = rosbag.Bag(output_bag, 'w')
    
    time_interval = 1.0 / fps
    success_count = 0
    
    try:
        for idx, filename in enumerate(sampled_files):
            image_path = os.path.join(image_folder, filename)
            
            # Read image
            if encoding == 'mono8':
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Warning: Failed to read image: {}".format(image_path))
                continue
            
            # Create ROS message
            timestamp = rospy.Time.from_sec(idx * time_interval)
            
            try:
                img_msg = bridge.cv2_to_imgmsg(image, encoding=encoding)
                img_msg.header.stamp = timestamp
                img_msg.header.frame_id = frame_id
                
                # Write to bag
                bag.write(topic_name, img_msg, timestamp)
                success_count += 1
                
                if (idx + 1) % 10 == 0:
                    print("Processed: {}/{} images".format(idx + 1, len(sampled_files)))
                    
            except Exception as e:
                print("Warning: Failed to convert image {}: {}".format(filename, e))
                continue
        
        print("="*60)
        print("Success! Created rosbag with {} images".format(success_count))
        print("Output: {}".format(output_bag))
        print("="*60)
        
    except Exception as e:
        print("Error during bag creation: {}".format(e))
        
    finally:
        bag.close()


def sample_stereo_images_to_rosbag(cam0_folder, cam1_folder, output_bag, stride=1,
                                   cam0_topic='/cam0/image_raw', cam1_topic='/cam1/image_raw',
                                   cam0_frame='cam0', cam1_frame='cam1', 
                                   encoding='mono8', fps=20.0):
    """
    Sample stereo images from two folders and create rosbag.
    
    Args:
        cam0_folder: Path to camera 0 image folder
        cam1_folder: Path to camera 1 image folder
        output_bag: Path to output rosbag file
        stride: Sample stride
        cam0_topic: Camera 0 topic name
        cam1_topic: Camera 1 topic name
        cam0_frame: Camera 0 frame_id
        cam1_frame: Camera 1 frame_id
        encoding: Image encoding
        fps: Frames per second
    """
    # Check folders
    if not os.path.exists(cam0_folder):
        print("Error: Camera 0 folder does not exist: {}".format(cam0_folder))
        return
    if not os.path.exists(cam1_folder):
        print("Error: Camera 1 folder does not exist: {}".format(cam1_folder))
        return
    
    # Get image files
    cam0_files = get_sorted_image_files(cam0_folder)
    cam1_files = get_sorted_image_files(cam1_folder)
    
    if not cam0_files:
        print("Error: No images in camera 0 folder")
        return
    if not cam1_files:
        print("Error: No images in camera 1 folder")
        return
    
    # Check if same number of images
    if len(cam0_files) != len(cam1_files):
        print("Warning: Camera 0 has {} images, Camera 1 has {} images".format(
            len(cam0_files), len(cam1_files)))
        print("Will use minimum count")
    
    min_count = min(len(cam0_files), len(cam1_files))
    cam0_files = cam0_files[:min_count]
    cam1_files = cam1_files[:min_count]
    
    # Sample by stride
    cam0_sampled = cam0_files[::stride]
    cam1_sampled = cam1_files[::stride]
    
    print("="*60)
    print("Stereo Image Sampling to RosBag")
    print("="*60)
    print("Camera 0 folder: {}".format(cam0_folder))
    print("Camera 1 folder: {}".format(cam1_folder))
    print("Total stereo pairs: {}".format(min_count))
    print("Sample stride: {}".format(stride))
    print("Sampled pairs: {}".format(len(cam0_sampled)))
    print("Output bag: {}".format(output_bag))
    print("="*60)
    
    # Create rosbag
    bridge = CvBridge()
    bag = rosbag.Bag(output_bag, 'w')
    
    time_interval = 1.0 / fps
    success_count = 0
    
    try:
        for idx, (cam0_file, cam1_file) in enumerate(zip(cam0_sampled, cam1_sampled)):
            cam0_path = os.path.join(cam0_folder, cam0_file)
            cam1_path = os.path.join(cam1_folder, cam1_file)
            
            # Read images
            if encoding == 'mono8':
                cam0_img = cv2.imread(cam0_path, cv2.IMREAD_GRAYSCALE)
                cam1_img = cv2.imread(cam1_path, cv2.IMREAD_GRAYSCALE)
            else:
                cam0_img = cv2.imread(cam0_path, cv2.IMREAD_COLOR)
                cam1_img = cv2.imread(cam1_path, cv2.IMREAD_COLOR)
            
            if cam0_img is None or cam1_img is None:
                print("Warning: Failed to read images at index {}".format(idx))
                continue
            
            # Create timestamp
            timestamp = rospy.Time.from_sec(idx * time_interval)
            
            try:
                # Convert camera 0
                cam0_msg = bridge.cv2_to_imgmsg(cam0_img, encoding=encoding)
                cam0_msg.header.stamp = timestamp
                cam0_msg.header.frame_id = cam0_frame
                bag.write(cam0_topic, cam0_msg, timestamp)
                
                # Convert camera 1
                cam1_msg = bridge.cv2_to_imgmsg(cam1_img, encoding=encoding)
                cam1_msg.header.stamp = timestamp
                cam1_msg.header.frame_id = cam1_frame
                bag.write(cam1_topic, cam1_msg, timestamp)
                
                success_count += 1
                
                if (idx + 1) % 10 == 0:
                    print("Processed: {}/{} stereo pairs".format(idx + 1, len(cam0_sampled)))
                    
            except Exception as e:
                print("Warning: Failed to convert images at index {}: {}".format(idx, e))
                continue
        
        print("="*60)
        print("Success! Created rosbag with {} stereo pairs".format(success_count))
        print("Output: {}".format(output_bag))
        print("="*60)
        
    except Exception as e:
        print("Error during bag creation: {}".format(e))
        
    finally:
        bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample images from folder by stride and create rosbag for Kalibr calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single camera
  python sample_images_to_rosbag.py -i /path/to/images -o output.bag -s 5
  
  # Single camera with custom topic
  python sample_images_to_rosbag.py -i /path/to/cam0 -o calib.bag -s 3 -t /cam0/image_raw -f cam0
  
  # Stereo cameras
  python sample_images_to_rosbag.py --cam0 /path/to/cam0 --cam1 /path/to/cam1 -o stereo.bag -s 5
  
  # Color images
  python sample_images_to_rosbag.py -i /path/to/images -o output.bag -s 5 --encoding bgr8
        """
    )
    
    # Input options
    parser.add_argument('-i', '--image_folder', type=str,
                        help="Path to image folder (for single camera)")
    parser.add_argument('--cam0', type=str,
                        help="Path to camera 0 image folder (for stereo)")
    parser.add_argument('--cam1', type=str,
                        help="Path to camera 1 image folder (for stereo)")
    
    # Output options
    parser.add_argument('-o', '--output_bag', type=str, required=True,
                        help="Path to output rosbag file")
    
    # Sampling options
    parser.add_argument('-s', '--stride', type=int, default=1,
                        help="Sample stride (default: 1, no sampling)")
    
    # Topic options
    parser.add_argument('-t', '--topic', type=str, default='/camera/image_raw',
                        help="ROS topic name for single camera (default: /camera/image_raw)")
    parser.add_argument('--cam0_topic', type=str, default='/cam0/image_raw',
                        help="Camera 0 topic name for stereo (default: /cam0/image_raw)")
    parser.add_argument('--cam1_topic', type=str, default='/cam1/image_raw',
                        help="Camera 1 topic name for stereo (default: /cam1/image_raw)")
    
    # Frame ID options
    parser.add_argument('-f', '--frame_id', type=str, default='camera',
                        help="Camera frame_id for single camera (default: camera)")
    parser.add_argument('--cam0_frame', type=str, default='cam0',
                        help="Camera 0 frame_id for stereo (default: cam0)")
    parser.add_argument('--cam1_frame', type=str, default='cam1',
                        help="Camera 1 frame_id for stereo (default: cam1)")
    
    # Image options
    parser.add_argument('--encoding', type=str, default='mono8',
                        choices=['mono8', 'bgr8', 'rgb8'],
                        help="Image encoding (default: mono8)")
    parser.add_argument('--fps', type=float, default=20.0,
                        help="Frames per second for timestamp (default: 20.0)")
    
    args = parser.parse_args()
    
    # Check arguments
    if args.cam0 and args.cam1:
        # Stereo mode
        sample_stereo_images_to_rosbag(
            args.cam0, args.cam1, args.output_bag, args.stride,
            args.cam0_topic, args.cam1_topic,
            args.cam0_frame, args.cam1_frame,
            args.encoding, args.fps
        )
    elif args.image_folder:
        # Single camera mode
        sample_images_to_rosbag(
            args.image_folder, args.output_bag, args.stride,
            args.topic, args.frame_id, args.encoding, args.fps
        )
    else:
        parser.error("Must specify either -i/--image_folder for single camera, or --cam0 and --cam1 for stereo")

