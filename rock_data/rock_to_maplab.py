import os
import csv
from glob import glob
from .convert_tof_traj import convert_vslam_to_tum, plot_tum_trajectory
from imu.analyse_imu_data import process_imu_data


def log(message, level="INFO"):
    """简单日志工具"""
    print(f"[{level}] {message}")


def create_maplab_structure(source_dir):
    """
    创建 MapLab 格式的目录结构并解析数据。
    """
    log(f"开始解析数据，目录：{source_dir}")

    camera_dir = os.path.join(source_dir, "camera")
    imu_file_path = os.path.join(source_dir, "imu.csv")
    imu_log_file_path = os.path.join(source_dir, "RRLDR_fprintf.log")
    pose_log_pattern = os.path.join(source_dir, "*SLAM_fprintf.log")
    pose_log_files = glob(pose_log_pattern)
    tof_pose_path = os.path.join(source_dir, "tof_pose.txt")

    if not os.path.exists(camera_dir):
        log(f"目录 {camera_dir} 不存在。", level="ERROR")
        raise FileNotFoundError(f"{camera_dir} not found in source directory.")

    # 生成 camera0 和 camera1 的 data.csv 文件
    try:
        log("生成 camera0 和 camera1 的 data.csv 文件")
        camera0_csv_path = os.path.join(camera_dir, "camera0", "data.csv")
        camera1_csv_path = os.path.join(camera_dir, "camera1", "data.csv")
        generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path)
    except Exception as e:
        log(f"生成 data.csv 失败: {e}", level="ERROR")
        raise

    # 生成 imu.csv
    try:
        log("从 RRLDR_fprintf.log 提取 IMU 数据")
        extract_imu_data(imu_log_file_path, imu_file_path)
    except FileNotFoundError:
        log(f"IMU 日志文件 {imu_log_file_path} 不存在，跳过 IMU 数据生成。", level="WARNING")

    # 生成 tof_pose.txt
    try:
        if pose_log_files:
            log(f"提取 SLAM pose 数据，使用文件 {pose_log_files[0]}")
            convert_vslam_to_tum(pose_log_files[0], tof_pose_path)
            log("生成 TOF 轨迹图")
            plot_tum_trajectory(tof_pose_path, tof_pose_path.replace(".txt", ".png"))
        else:
            log(f"未找到 *_SLAM_fprintf.log 文件，跳过轨迹数据生成。", level="WARNING")
    except Exception as e:
        log(f"生成 tof_pose 数据失败: {e}", level="ERROR")
        raise

    log("数据解析完成！")


def generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path):
    """
    分别为 camera0 和 camera1 生成 data.csv 文件，并只保留时间戳重叠的项。
    同时确保时间戳严格递增。
    """
    camera0_path = os.path.join(camera_dir, "camera0")
    camera1_path = os.path.join(camera_dir, "camera1")

    if not os.path.isdir(camera0_path) or not os.path.isdir(camera1_path):
        raise FileNotFoundError("camera0 或 camera1 目录不存在。")

    def extract_numeric_timestamps(file_list):
        """提取文件名中的有效数字时间戳"""
        timestamps = {}
        for file_name in file_list:
            base_name, _ = os.path.splitext(file_name)
            try:
                timestamps[int(base_name)] = file_name
            except ValueError:
                # 跳过无法转换为整数的文件名
                print(f"跳过无效文件名: {file_name}")
        return timestamps

    # 获取 camera0 和 camera1 的文件名及时间戳
    camera0_files = sorted(os.listdir(camera0_path))
    camera1_files = sorted(os.listdir(camera1_path))

    camera0_timestamps = extract_numeric_timestamps(camera0_files)
    camera1_timestamps = extract_numeric_timestamps(camera1_files)

    # 找到重叠的时间戳
    common_timestamps = sorted(set(camera0_timestamps.keys()) & set(camera1_timestamps.keys()))
    filtered_timestamps = []
    last_timestamp = -1

    for timestamp in common_timestamps:
        current_timestamp = int(timestamp)
        if current_timestamp > last_timestamp:
            filtered_timestamps.append(timestamp)
            last_timestamp = current_timestamp

    with open(camera0_csv_path, "w", newline='') as camera0_csv, open(camera1_csv_path, "w", newline='') as camera1_csv:
        camera0_writer = csv.writer(camera0_csv)
        camera1_writer = csv.writer(camera1_csv)
        camera0_writer.writerow(["#timestamp [ns]", "filename"])
        camera1_writer.writerow(["#timestamp [ns]", "filename"])
        for timestamp in filtered_timestamps:
            camera0_writer.writerow([timestamp, camera0_timestamps[timestamp]])
            camera1_writer.writerow([timestamp, camera1_timestamps[timestamp]])


def extract_imu_data(log_file_path, imu_file_path):
    """
    从 RRLDR_fprintf.log 提取 IMU 数据，生成 imu.csv。
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"{log_file_path} 不存在。")

    imu_data = []

    with open(log_file_path, "r") as logfile:
        for line in logfile:
            if "gyroOdo" in line:
                parts = line.split()
                timestamp = parts[0]
                gyro_data = parts[17:20]  # 假设第 18-20 列是陀螺仪数据
                accel_data = parts[11:14]  # 假设第 12-14 列是加速度数据
                imu_data.append([int(timestamp)] + gyro_data + accel_data)

    imu_data.sort(key=lambda x: x[0])
    unique_imu_data = []
    last_timestamp = -1
    for row in imu_data:
        if row[0] > last_timestamp:
            unique_imu_data.append(row)
            last_timestamp = row[0]

    with open(imu_file_path, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["# timestamp", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
        csvwriter.writerows(unique_imu_data)

    process_imu_data(imu_file_path, save_dir=os.path.dirname(imu_file_path))


if __name__ == "__main__":
    source_dir = input("请输入测试数据路径: ").strip()
    try:
        create_maplab_structure(source_dir)
    except Exception as e:
        log(f"程序执行失败: {e}", level="ERROR")
