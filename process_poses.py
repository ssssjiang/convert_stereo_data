import os
import re
import numpy as np
import glob
import transformations

def extract_vslam_pose(source_log_path, output_tum_path):
    """
    从 SLAM_normal.log 中提取 V-SLAM pose，并以 TUM 格式保存。
    此函数整合了 extract_log_lines.py 和 convert_vio_to_tum.py 的功能。

    Args:
        source_log_path (str): SLAM_normal.log 文件的路径。
        output_tum_path (str): 输出的 TUM 格式文件的路径。
    """
    print(f"正在从 {source_log_path} 提取 V-SLAM pose...")
    lines_processed = 0
    try:
        with open(source_log_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_tum_path, 'w', encoding='utf-8') as outfile:
            
            # TUM 格式的注释头 (可选)
            # outfile.write("#timestamp tx ty tz qx qy qz qw\n")

            for line in infile:
                if 'consume frontend vio state' in line:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        tum_data = parts[-8:]
                        
                        # 从TUM数据中提取时间戳、位置和旋转四元数
                        timestamp = tum_data[0]
                        tx, ty, tz = float(tum_data[1]), float(tum_data[2]), float(tum_data[3])
                        qx, qy, qz, qw = float(tum_data[4]), float(tum_data[5]), float(tum_data[6]), float(tum_data[7])
                        
                        # convert T_W_B to T_W_R
                        t_B_R = np.array([-0.036, 0, 0.113])  # 从机体坐标系到机器人坐标系的平移
                        
                        # 使用transformations库创建变换矩阵
                        # 1. 创建T_W_B变换矩阵 (世界到机体)
                        T_W_B = transformations.quaternion_matrix([qx, qy, qz, qw])
                        T_W_B[:3, 3] = [tx, ty, tz]
                        
                        # 2. 创建T_B_R变换矩阵 (机体到机器人)
                        T_B_R = np.eye(4)  # 单位旋转矩阵
                        T_B_R[:3, 3] = t_B_R  # 设置平移部分
                        
                        # 3. 计算T_W_R = T_W_B * T_B_R
                        T_W_R = np.dot(T_W_B, T_B_R)
                        
                        # 提取位置和四元数
                        t_W_R = T_W_R[:3, 3]
                        quat_W_R = transformations.quaternion_from_matrix(T_W_R)
                        
                        # 构造新的TUM格式数据
                        new_tum_data = [
                            timestamp,
                            f"{t_W_R[0]:.9f}", f"{t_W_R[1]:.9f}", f"{t_W_R[2]:.9f}",
                            f"{quat_W_R[0]:.9f}", f"{quat_W_R[1]:.9f}", 
                            f"{quat_W_R[2]:.9f}", f"{quat_W_R[3]:.9f}"
                        ]
                        
                        outfile.write(" ".join(new_tum_data) + "\n")
                        lines_processed += 1
        
        print(f"成功处理 {lines_processed} 行 V-SLAM pose，并保存至 {output_tum_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{source_log_path}' 未找到。")
    except Exception as e:
        print(f"提取 V-SLAM pose 时发生错误: {e}")

def extract_fusion_pose(slam_fprintf_log_path, output_tum_path):
    """
    从 SLAM_fprintf.log 中提取 6DoF pose (fusion 算法)，并以 TUM 格式保存。

    Args:
        slam_fprintf_log_path (str): SLAM_fprintf.log 文件的路径。
        output_tum_path (str): 输出的 TUM 格式文件的路径。
    """
    print(f"正在从 {slam_fprintf_log_path} 提取 Fusion pose...")
    lines_processed = 0
    try:
        with open(slam_fprintf_log_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_tum_path, 'w', encoding='utf-8') as outfile:

            # TUM 格式的注释头 (可选)
            # outfile.write("#timestamp tx ty tz qx qy qz qw\n")
            
            for line in infile:
                parts = line.strip().split()
                # 检查是否是 'estimate_3d' 行并且有足够的数据
                if len(parts) >= 9 and parts[1] == 'estimate_3d':
                    timestamp = parts[0]
                    # The first value is timestamp, followed by 7 pose values (tx, ty, tz, qx, qy, qz, qw)
                    pose_data = parts[2:9]
                    output_line = f"{timestamp} {' '.join(pose_data)}\n"
                    outfile.write(output_line)
                    lines_processed += 1
            
            print(f"成功处理 {lines_processed} 行 Fusion pose，并保存至 {output_tum_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{slam_fprintf_log_path}' 未找到。")
    except Exception as e:
        print(f"提取 Fusion pose 时发生错误: {e}")

def analyze_camera_frame_time_diff(source_log_path):
    """
    从 SLAM_normal.log 中提取相机帧数据，分析时间戳差值。
    
    Args:
        source_log_path (str): SLAM_normal.log 文件的路径。
    """
    print(f"正在从 {source_log_path} 分析相机帧时间差...")
    
    # 用于存储提取的时间戳
    timestamps = []
    pattern = r"receive camera data (\d+\.\d+)"
    
    try:
        with open(source_log_path, 'r', encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                if "receive camera data" in line:
                    match = re.search(pattern, line)
                    if match:
                        timestamp = float(match.group(1))
                        timestamps.append(timestamp)
        
        if len(timestamps) < 2:
            print("找到的相机帧数据少于2帧，无法计算时间差")
            return
        
        # 计算时间戳差值
        diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # 统计差值大于0.1的比例和数据
        large_diffs = []
        large_diff_timestamps = []
        
        for i in range(len(diffs)):
            if diffs[i] > 0.1:
                large_diffs.append(diffs[i])
                # 保存对应的时间戳对
                large_diff_timestamps.append((timestamps[i], timestamps[i+1], diffs[i]))
        
        total_diffs = len(diffs)
        large_diffs_count = len(large_diffs)
        
        if total_diffs == 0:
            print("没有计算出有效的时间差")
            return
        
        # 计算统计数据
        large_diff_ratio = large_diffs_count / total_diffs
        
        if large_diffs_count > 0:
            large_diff_mean = np.mean(large_diffs)
            large_diff_max = np.max(large_diffs)
        else:
            large_diff_mean = 0
            large_diff_max = 0
        
        # 打印结果
        print(f"\n相机帧时间差分析结果:")
        print(f"总共分析了 {total_diffs} 对相邻帧")
        print(f"时间差大于0.1秒的比例: {large_diff_ratio:.4f} ({large_diffs_count}/{total_diffs})")
        print(f"时间差大于0.1秒的均值: {large_diff_mean:.4f} 秒")
        print(f"时间差大于0.1秒的最大值: {large_diff_max:.4f} 秒")
        
        # 打印时间差大于0.1秒的两帧时间戳
        if large_diffs_count > 0:
            print("\n时间差大于0.1秒的帧对:")
            print("序号\t前一帧时间戳\t后一帧时间戳\t时间差(秒)")
            for i, (prev_ts, curr_ts, diff) in enumerate(large_diff_timestamps, 1):
                print(f"{i}\t{prev_ts:.6f}\t{curr_ts:.6f}\t{diff:.6f}")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{source_log_path}' 未找到。")
    except Exception as e:
        print(f"分析相机帧时间差时发生错误: {e}")

def find_log_file(base_path, base_name):
    """
    查找指定基础名称的日志文件，支持.log和.log.pl9后缀
    
    Args:
        base_path (str): 基础目录路径
        base_name (str): 日志文件基础名称(不含后缀)
    
    Returns:
        str: 找到的日志文件完整路径，如未找到则返回默认路径(.log后缀)
    """
    # 尝试查找可能的后缀
    possible_files = glob.glob(os.path.join(base_path, f"{base_name}*"))
    
    # 按特定顺序优先查找
    for ext in ['.log', '.log.pl9']:
        expected_file = os.path.join(base_path, f"{base_name}{ext}")
        if expected_file in possible_files:
            print(f"找到日志文件: {expected_file}")
            return expected_file
    
    # 如果没有找到，返回默认名称
    default_file = os.path.join(base_path, f"{base_name}.log")
    print(f"未找到匹配的日志文件，将使用默认路径: {default_file}")
    return default_file

if __name__ == '__main__':
    # --- 用户配置 ---
    # 请根据您的文件结构修改这些路径
    source_dir = input("请输入测试数据路径: ").strip()
    
    # 查找V-SLAM和Fusion日志文件，支持.log和.log.pl9后缀
    VSLAM_SOURCE_LOG = find_log_file(source_dir, 'SLAM_normal')
    FUSION_SOURCE_LOG = find_log_file(source_dir, 'SLAM_fprintf')

    # 输出文件
    VSLAM_OUTPUT_FILE = os.path.join(source_dir, 'vslam_pose_tum.txt')
    FUSION_OUTPUT_FILE = os.path.join(source_dir, 'fusion_pose_tum.txt')
    # --- 配置结束 ---

    # 提取 V-SLAM pose
    extract_vslam_pose(VSLAM_SOURCE_LOG, VSLAM_OUTPUT_FILE)

    # 提取 Fusion pose
    extract_fusion_pose(FUSION_SOURCE_LOG, FUSION_OUTPUT_FILE)
    
    # 分析相机帧时间差
    analyze_camera_frame_time_diff(VSLAM_SOURCE_LOG)

    print("\n所有处理完成。") 