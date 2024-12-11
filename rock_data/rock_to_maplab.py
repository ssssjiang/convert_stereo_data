import os
import csv
from glob import glob
from convert_stereo_data.convert_tof_traj import convert_vslam_to_tum, plot_tum_trajectory
from convert_stereo_data.imu.analyse_imu_data import process_imu_data


def create_maplab_structure(source_dir):
    # 创建 MapLab 格式的目录结构
    camera_dir = os.path.join(source_dir, "camera")
    imu_file_path = os.path.join(source_dir, "imu.csv")
    imu_log_file_path = os.path.join(source_dir, "RRLDR_fprintf.log")
    pose_log_pattern = os.path.join(source_dir, "*_SLAM_fprintf.log")
    pose_log_files = glob(pose_log_pattern)  # 匹配所有 *_SLAM_fprintf.log 文件
    tof_pose_path = os.path.join(source_dir, "tof_pose.txt")

    if not os.path.exists(camera_dir):
        raise FileNotFoundError(f"{camera_dir} not found in source directory.")

    # 分别生成 camera0 和 camera1 的 data.csv 文件，只保留时间戳重叠的项
    camera0_csv_path = os.path.join(camera_dir, "camera0", "data.csv")
    camera1_csv_path = os.path.join(camera_dir, "camera1", "data.csv")
    generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path)

    # 从 RRLDR_fprintf.log 提取 IMU 数据生成 imu.csv
    extract_imu_data(imu_log_file_path, imu_file_path)

    # 从 *_SLAM_fprintf.log 提取 SLAM pose 数据生成 tof_pose.txt
    if pose_log_files:
        convert_vslam_to_tum(pose_log_files[0], tof_pose_path)
        # 生成 tof_pose.png
        plot_tum_trajectory(tof_pose_path, tof_pose_path.replace(".txt", ".png"))
    else:
        raise FileNotFoundError(f"No *_SLAM_fprintf.log files found in {source_dir}")


def generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path):
    """
    分别为 camera0 和 camera1 生成 data.csv 文件，并只保留时间戳重叠的项。
    同时确保时间戳严格递增。
    """
    camera0_path = os.path.join(camera_dir, "camera0")
    camera1_path = os.path.join(camera_dir, "camera1")

    if not os.path.isdir(camera0_path) or not os.path.isdir(camera1_path):
        raise FileNotFoundError("camera0 or camera1 directory not found.")

    # 获取 camera0 和 camera1 的文件名及时间戳
    camera0_files = sorted(os.listdir(camera0_path))
    camera1_files = sorted(os.listdir(camera1_path))

    camera0_timestamps = {os.path.splitext(f)[0]: f for f in camera0_files}
    camera1_timestamps = {os.path.splitext(f)[0]: f for f in camera1_files}

    # 找到重叠的时间戳
    common_timestamps = sorted(set(camera0_timestamps.keys()) & set(camera1_timestamps.keys()))

    # 去除重复并确保时间戳递增
    last_timestamp = -1
    filtered_timestamps = []
    for timestamp in common_timestamps:
        current_timestamp = int(timestamp)
        if current_timestamp > last_timestamp:
            filtered_timestamps.append(timestamp)
            last_timestamp = current_timestamp

    # 分别生成 camera0 和 camera1 的 data.csv 文件
    with open(camera0_csv_path, "w", newline='') as camera0_csv, open(camera1_csv_path, "w", newline='') as camera1_csv:
        camera0_writer = csv.writer(camera0_csv)
        camera1_writer = csv.writer(camera1_csv)

        # 写入表头
        camera0_writer.writerow(["#timestamp [ns]", "filename"])
        camera1_writer.writerow(["#timestamp [ns]", "filename"])

        # 写入重叠时间戳的文件路径，按时间戳排序
        for timestamp in filtered_timestamps:
            camera0_writer.writerow([timestamp, camera0_timestamps[timestamp]])
            camera1_writer.writerow([timestamp, camera1_timestamps[timestamp]])


def extract_imu_data(log_file_path, imu_file_path):
    """
    从 RRLDR_fprintf.log 提取 IMU 数据，生成 imu.csv。
    数据格式：时间戳、陀螺仪和加速度数据，确保时间戳严格递增。
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"{log_file_path} not found.")

    imu_data = []

    with open(log_file_path, "r") as logfile:
        for line in logfile:
            # 假设 IMU 数据格式为：<timestamp> ... gyroOdo=<gx> <gy> <gz> accelOdo=<ax> <ay> <az>
            if "gyroOdo" in line:
                parts = line.split()
                timestamp = parts[0]
                gyro_data = parts[17:20]  # 假设第 18-20 列是陀螺仪数据
                accel_data = parts[11:14]  # 假设第 12-14 列是加速度数据
                imu_data.append([int(timestamp)] + gyro_data + accel_data)

    # 按时间戳排序并去重
    imu_data.sort(key=lambda x: x[0])
    unique_imu_data = []
    last_timestamp = -1
    for row in imu_data:
        if row[0] > last_timestamp:
            unique_imu_data.append(row)
            last_timestamp = row[0]

    with open(imu_file_path, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(["# timestamp", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
        # 写入排序后的 IMU 数据
        csvwriter.writerows(unique_imu_data)

    # plot imu data
    process_imu_data(imu_file_path, save_dir=os.path.dirname(imu_file_path))

# 使用示例
create_maplab_structure("/home/roborock/datasets/roborock/stereo/479-living-room")
