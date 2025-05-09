#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import numpy as np

def transform_imu_data(input_file, output_file):
    """
    读取IMU数据并应用旋转矩阵变换
    
    参数:
        input_file: 输入的CSV文件路径
        output_file: 输出的CSV文件路径
    """
    # 定义旋转矩阵
    rotation_matrix = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    # 读取CSV文件并应用变换
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # 读取标题行
        header = next(reader)
        writer.writerow(header)  # 写入相同的标题
        
        # 处理每一行数据
        for row in reader:
            try:
                timestamp = row[0]
                gyro = np.array([float(row[1]), float(row[2]), float(row[3])])
                accel = np.array([float(row[4]), float(row[5]), float(row[6])])
                
                # 应用旋转矩阵
                gyro_transformed = rotation_matrix @ gyro
                accel_transformed = rotation_matrix @ accel
                
                # 构造新行并写入
                new_row = [
                    timestamp,
                    gyro_transformed[0], gyro_transformed[1], gyro_transformed[2],
                    accel_transformed[0], accel_transformed[1], accel_transformed[2]
                ]
                writer.writerow(new_row)
            except Exception as e:
                print(f"处理行时出错: {row}")
                print(f"错误信息: {e}")
                continue
    
    print(f"变换完成！结果保存到 {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python transform_imu_data.py <输入文件.csv> <输出文件.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    transform_imu_data(input_file, output_file) 