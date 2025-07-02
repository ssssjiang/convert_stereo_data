import os

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
                        outfile.write(" ".join(tum_data) + "\n")
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

if __name__ == '__main__':
    # --- 用户配置 ---
    # 请根据您的文件结构修改这些路径
    # V-SLAM pose 来源日志
    source_dir = input("请输入测试数据路径: ").strip()
    VSLAM_SOURCE_LOG = os.path.join(source_dir, 'SLAM_normal.log')
    # Fusion pose 来源日志
    FUSION_SOURCE_LOG = os.path.join(source_dir, 'SLAM_fprintf.log')

    # 输出文件
    VSLAM_OUTPUT_FILE = os.path.join(source_dir, 'vslam_pose_tum.txt')
    FUSION_OUTPUT_FILE = os.path.join(source_dir, 'fusion_pose_tum.txt')
    # --- 配置结束 ---

    # 提取 V-SLAM pose
    extract_vslam_pose(VSLAM_SOURCE_LOG, VSLAM_OUTPUT_FILE)

    # 提取 Fusion pose
    extract_fusion_pose(FUSION_SOURCE_LOG, FUSION_OUTPUT_FILE)

    print("\n所有处理完成。") 