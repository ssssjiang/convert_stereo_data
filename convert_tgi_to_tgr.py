#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
坐标系转换工具：将T_G_I轨迹转换为T_G_R轨迹

输入：
1. TUM格式的T_G_I轨迹文件（IMU在全局坐标系下的位姿）
2. sensor.yaml文件（包含T_B_I和t_B_R参数）

输出：
- TUM格式的T_G_R轨迹文件（RTK在全局坐标系下的位姿）

坐标系变换关系：
T_G_R = T_G_I * T_I_B * T_B_R
其中：
- T_G_I: 从全局坐标系到IMU坐标系的变换（输入轨迹）
- T_I_B: 从IMU坐标系到机体坐标系的变换（T_B_I的逆）
- T_B_R: 从机体坐标系到RTK坐标系的变换
"""

import numpy as np
import yaml
import argparse
import os
import sys

# 导入transformations库
try:
    import transformations
except ImportError:
    print("错误：找不到transformations库，请确保transformations.py在当前目录")
    sys.exit(1)


def load_tum_trajectory(tum_file_path):
    """
    加载TUM格式的轨迹文件
    
    Args:
        tum_file_path: TUM轨迹文件路径
        
    Returns:
        list: 轨迹数据列表，每个元素为 [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    trajectory_data = []
    
    try:
        with open(tum_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) != 8:
                    print(f"警告：第{line_num}行数据格式不正确，跳过: {line}")
                    continue
                
                try:
                    timestamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    
                    trajectory_data.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
                except ValueError as e:
                    print(f"警告：第{line_num}行数据解析失败，跳过: {line} (错误: {e})")
                    continue
                    
    except FileNotFoundError:
        print(f"错误：找不到轨迹文件 {tum_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取轨迹文件失败: {e}")
        sys.exit(1)
    
    print(f"成功加载 {len(trajectory_data)} 个轨迹点")
    return trajectory_data


def load_sensor_config(sensor_yaml_path):
    """
    加载传感器配置文件
    
    Args:
        sensor_yaml_path: 传感器配置文件路径
        
    Returns:
        tuple: (T_B_I, t_B_R) T_B_I为4x4变换矩阵，t_B_R为3x1平移向量
    """
    try:
        with open(sensor_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到传感器配置文件 {sensor_yaml_path}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取传感器配置文件失败: {e}")
        sys.exit(1)
    
    # 提取T_B_I变换矩阵
    try:
        imu_config = config['sensor']['imu']
        T_B_I_data = imu_config['T_B_I']['data']
        T_B_I = np.array(T_B_I_data).reshape(4, 4)
        print("T_B_I变换矩阵:")
        print(T_B_I)
    except KeyError as e:
        print(f"错误：在传感器配置文件中找不到T_B_I参数: {e}")
        sys.exit(1)
    
    # 提取t_B_R平移向量
    try:
        rtk_config = config['sensor']['rtk']
        t_B_R = np.array(rtk_config['t_B_R'])
        print(f"t_B_R平移向量: {t_B_R}")
    except KeyError as e:
        print(f"错误：在传感器配置文件中找不到t_B_R参数: {e}")
        sys.exit(1)
    
    return T_B_I, t_B_R


def convert_tgi_to_tgr(trajectory_data, T_B_I, t_B_R):
    """
    将T_G_I轨迹转换为T_G_R轨迹
    
    Args:
        trajectory_data: TUM格式的轨迹数据
        T_B_I: IMU到机体的变换矩阵 4x4
        t_B_R: 机体到RTK的平移向量 3x1
        
    Returns:
        list: 转换后的T_G_R轨迹数据
    """
    print("开始坐标系转换...")
    
    # 计算T_I_B（T_B_I的逆）
    T_I_B = np.linalg.inv(T_B_I)
    print("T_I_B变换矩阵 (T_B_I的逆):")
    print(T_I_B)
    
    # 构建T_B_R变换矩阵（只有平移，没有旋转）
    T_B_R = np.eye(4)
    T_B_R[:3, 3] = t_B_R
    print("T_B_R变换矩阵:")
    print(T_B_R)
    
    # 预计算复合变换：T_I_R = T_I_B * T_B_R
    T_I_R = np.dot(T_I_B, T_B_R)
    print("T_I_R复合变换矩阵:")
    print(T_I_R)
    
    converted_data = []
    
    for i, (timestamp, tx, ty, tz, qx, qy, qz, qw) in enumerate(trajectory_data):
        try:
            # 构建T_G_I变换矩阵
            T_G_I = transformations.quaternion_matrix([qx, qy, qz, qw])
            T_G_I[:3, 3] = [tx, ty, tz]
            
            # 计算T_G_R = T_G_I * T_I_R
            T_G_R = np.dot(T_G_I, T_I_R)
            
            # 提取位置和四元数
            t_G_R = T_G_R[:3, 3]
            quat_G_R = transformations.quaternion_from_matrix(T_G_R)
            
            # 转换为TUM格式 [timestamp, tx, ty, tz, qx, qy, qz, qw]
            converted_data.append([
                timestamp,
                t_G_R[0], t_G_R[1], t_G_R[2],
                quat_G_R[0], quat_G_R[1], quat_G_R[2], quat_G_R[3]
            ])
            
        except Exception as e:
            print(f"警告：转换第{i+1}个轨迹点时出错，跳过: {e}")
            continue
    
    print(f"成功转换 {len(converted_data)} 个轨迹点")
    return converted_data


def save_tum_trajectory(trajectory_data, output_path):
    """
    保存TUM格式的轨迹文件
    
    Args:
        trajectory_data: 轨迹数据
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w') as f:
            # 写入TUM格式的注释头
            f.write("#timestamp tx ty tz qx qy qz qw\n")
            
            for data in trajectory_data:
                timestamp, tx, ty, tz, qx, qy, qz, qw = data
                f.write(f"{timestamp:.6f} {tx:.9f} {ty:.9f} {tz:.9f} "
                       f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        
        print(f"T_G_R轨迹已保存至: {output_path}")
        
    except Exception as e:
        print(f"错误：保存轨迹文件失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="将TUM格式的T_G_I轨迹转换为T_G_R轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python convert_tgi_to_tgr.py -i after_optimization_mapping_poses.txt -s sensor_MK1-5.yaml -o tgr_trajectory.txt
  
坐标系说明:
  T_G_I: 全局坐标系到IMU坐标系的变换
  T_G_R: 全局坐标系到RTK坐标系的变换
  T_B_I: 机体坐标系到IMU坐标系的变换
  t_B_R: 机体坐标系到RTK坐标系的平移
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入的TUM格式T_G_I轨迹文件')
    parser.add_argument('-s', '--sensor', required=True,
                       help='传感器配置YAML文件')
    parser.add_argument('-o', '--output', required=True,
                       help='输出的TUM格式T_G_R轨迹文件')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入轨迹文件不存在: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.sensor):
        print(f"错误：传感器配置文件不存在: {args.sensor}")
        sys.exit(1)
    
    print("=== T_G_I 到 T_G_R 轨迹转换工具 ===")
    print(f"输入轨迹文件: {args.input}")
    print(f"传感器配置文件: {args.sensor}")
    print(f"输出轨迹文件: {args.output}")
    print()
    
    # 步骤1：加载T_G_I轨迹
    print("步骤1：加载T_G_I轨迹文件...")
    trajectory_data = load_tum_trajectory(args.input)
    print()
    
    # 步骤2：加载传感器配置
    print("步骤2：加载传感器配置文件...")
    T_B_I, t_B_R = load_sensor_config(args.sensor)
    print()
    
    # 步骤3：执行坐标系转换
    print("步骤3：执行坐标系转换...")
    converted_data = convert_tgi_to_tgr(trajectory_data, T_B_I, t_B_R)
    print()
    
    # 步骤4：保存T_G_R轨迹
    print("步骤4：保存T_G_R轨迹文件...")
    save_tum_trajectory(converted_data, args.output)
    print()
    
    print("=== 转换完成 ===")


if __name__ == "__main__":
    main() 