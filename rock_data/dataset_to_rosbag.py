import os
import csv
import cv2
import rosbag
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import Header
import rospy

def read_csv_file(file_path, delimiter=','):
    """Reads a CSV file and returns the content as a list of rows."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader)  # Skip header
        return [row for row in reader]

def convert_image_to_ros_msg(image_path, timestamp, frame_id):
    """Converts an image to a ROS Image message with cropping."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        rospy.logerr("Failed to read image: {}".format(image_path))
        raise ValueError("Failed to read image: {}".format(image_path))

    # # Crop the image to keep the center and resize to 640x480
    # height, width = image.shape
    # crop_x = (width - 640) // 2
    # crop_y = (height - 480) // 2
    # cropped_image = image[crop_y:crop_y + 480, crop_x:crop_x + 640]

    cv2.imshow('image', image)
    cv2.waitKey(1)

    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(image, encoding='mono8')
    img_msg.header.stamp = rospy.Time.from_sec(timestamp)
    img_msg.header.frame_id = frame_id
    return img_msg

def convert_imu_to_ros_msg(data, timestamp):
    """Converts IMU data to a ROS IMU message."""
    try:
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.from_sec(timestamp)
        imu_msg.linear_acceleration.x = float(data[4])
        imu_msg.linear_acceleration.y = float(data[5])
        imu_msg.linear_acceleration.z = float(data[6])
        imu_msg.angular_velocity.x = float(data[1])
        imu_msg.angular_velocity.y = float(data[2])
        imu_msg.angular_velocity.z = float(data[3])
        return imu_msg
    except (ValueError, IndexError) as e:
        rospy.logwarn("Invalid IMU data at timestamp {}: {}, error: {}".format(timestamp, data, e))
        return None

def convert_pose_to_ros_msg(data, timestamp):
    """Converts pose data to a ROS Odometry message."""
    try:
        pose_msg = Odometry()
        pose_msg.header.stamp = rospy.Time.from_sec(timestamp)
        pose_msg.header.frame_id = "world"
        pose_msg.pose.pose.position.x = float(data[1])
        pose_msg.pose.pose.position.y = float(data[2])
        pose_msg.pose.pose.position.z = float(data[3])
        pose_msg.pose.pose.orientation.w = float(data[7])
        pose_msg.pose.pose.orientation.x = float(data[4])
        pose_msg.pose.pose.orientation.y = float(data[5])
        pose_msg.pose.pose.orientation.z = float(data[6])
        return pose_msg
    except (ValueError, IndexError) as e:
        rospy.logwarn("Invalid pose data at timestamp {}: {}, error: {}".format(timestamp, data, e))
        return None

def process_dataset(dataset_path, output_bag_file, start_time, end_time):
    """Converts a dataset to a ROS bag file."""
    bag = rosbag.Bag(output_bag_file, 'w')
    try:
        stats = {
            'cam0': 0,
            'cam1': 0,
            'imu': 0,
            'vio_pose': 0
        }

        # Process camera 0
        last_timestamp = -1
        cam0_csv = os.path.join(dataset_path, 'camera/camera0', 'data.csv')
        cam0_folder = os.path.join(dataset_path, 'camera/camera0')
        cam0_data = read_csv_file(cam0_csv)
        for row in cam0_data:
            try:
                timestamp = float(row[0]) / 1e3
                if timestamp <= last_timestamp:
                    rospy.logwarn("Skipping non-strictly-increasing timestamp in camera 0: {}".format(timestamp))
                    continue
                if start_time <= timestamp <= end_time:
                    image_path = os.path.join(cam0_folder, row[1])
                    img_msg = convert_image_to_ros_msg(image_path, timestamp, 'cam0')
                    bag.write('/cam0/image_raw', img_msg, rospy.Time.from_sec(timestamp))
                    stats['cam0'] += 1
                    last_timestamp = timestamp
            except (ValueError, IndexError) as e:
                rospy.logwarn("Invalid camera 0 data: {}, error: {}".format(row, e))

        # Print statistics
        print("Data processing statistics:")
        for key, count in stats.items():
            print("{}: {} messages".format(key, count))

        # Process camera 1
        last_timestamp = -1
        cam1_csv = os.path.join(dataset_path, 'camera/camera1', 'data.csv')
        cam1_folder = os.path.join(dataset_path, 'camera/camera1')
        cam1_data = read_csv_file(cam1_csv)
        for row in cam1_data:
            try:
                timestamp = float(row[0]) / 1e3
                if timestamp <= last_timestamp:
                    rospy.logwarn("Skipping non-strictly-increasing timestamp in camera 1: {}".format(timestamp))
                    continue
                if start_time <= timestamp <= end_time:
                    image_path = os.path.join(cam1_folder, row[1])
                    img_msg = convert_image_to_ros_msg(image_path, timestamp, 'cam1')
                    bag.write('/cam1/image_raw', img_msg, rospy.Time.from_sec(timestamp))
                    stats['cam1'] += 1
                    last_timestamp = timestamp
            except (ValueError, IndexError) as e:
                rospy.logwarn("Invalid camera 1 data: {}, error: {}".format(row, e))

        # Print statistics
        print("Data processing statistics:")
        for key, count in stats.items():
            print("{}: {} messages".format(key, count))

        # Process IMU
        last_timestamp = -1
        imu_csv = os.path.join(dataset_path, 'imu.csv')
        imu_data = read_csv_file(imu_csv)
        for row in imu_data:
            try:
                timestamp = float(row[0]) / 1e3
                if timestamp <= last_timestamp:
                    rospy.logwarn("Skipping non-strictly-increasing timestamp in IMU: {}".format(timestamp))
                    continue
                if start_time <= timestamp <= end_time:
                    imu_msg = convert_imu_to_ros_msg(row, timestamp)
                    if imu_msg:
                        bag.write('/imu0', imu_msg, rospy.Time.from_sec(timestamp))
                        stats['imu'] += 1
                        last_timestamp = timestamp
            except (ValueError, IndexError) as e:
                rospy.logwarn("Invalid IMU data: {}, error: {}".format(row, e))

        # Process VIO pose
        last_timestamp = -1
        vio_pose_csv = os.path.join(dataset_path, 'tof_pose.txt')
        if os.path.exists(vio_pose_csv):
            vio_pose_data = read_csv_file(vio_pose_csv, delimiter=' ')
            for row in vio_pose_data:
                try:
                    timestamp = float(row[0])
                    if timestamp <= last_timestamp:
                        rospy.logwarn("Skipping non-strictly-increasing timestamp in VIO pose: {}".format(timestamp))
                        continue
                    if start_time <= timestamp <= end_time:
                        pose_msg = convert_pose_to_ros_msg(row, timestamp)
                        if pose_msg:
                            bag.write('/vio_pose', pose_msg, rospy.Time.from_sec(timestamp))
                            stats['vio_pose'] += 1
                            last_timestamp = timestamp
                except (ValueError, IndexError) as e:
                    rospy.logwarn("Invalid VIO pose data: {}, error: {}".format(row, e))

        # Print statistics
        print("Data processing statistics:")
        for key, count in stats.items():
            print("{}: {} messages".format(key, count))

    finally:
        bag.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert dataset to ROS bag.")
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument('-o', '--output_bag_file', type=str, required=True, help="Path to the output ROS bag file.")
    parser.add_argument('-s', '--start_time', type=float, default=0, help="Start time in seconds.")
    parser.add_argument('-e', '--end_time', type=float, default=float('inf'), help="End time in seconds.")

    args = parser.parse_args()

    process_dataset(args.dataset_path, args.output_bag_file, args.start_time, args.end_time)


# check tools
# 1. rosbag info <bagfile>
# 2. rqt_bag <bagfile>
# 3. rosrun rosbag_editor rosbag_editor