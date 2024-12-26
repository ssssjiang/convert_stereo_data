#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os

# 设置参数解析
parser = argparse.ArgumentParser(description="Process IMU and Odom data.")
parser.add_argument('--root_folder', type=str, required=True, help='Root folder containing the input files.')

args = parser.parse_args()

# 提取 root_folder
root_folder = args.root_folder

# 设置输入和输出文件路径
gyro_file_path = os.path.join(root_folder, 't265_gyroscope.txt')
odom_file_path = os.path.join(root_folder, 'odom.txt')
output_file = os.path.join(root_folder, 'RRLDR_fprintf.log')

# 初始化数据列表
odo_data = []
imu_data = []

# 读取 IMU 数据, 加速度计数据置零
with open(gyro_file_path, 'r') as f:
    for line in f:
        print(f"Raw line: {line.strip()}")  # 打印原始数据行
        if line.startswith('#'):  # 跳过注释行
            continue
        data = line.strip().split(' ')
        if len(data) < 4:  # 检查数据格式是否正确
            print(f"Invalid data line: {line.strip()}")
            continue
        timestamp = float(data[0])
        gyro_x = float(data[1])
        gyro_y = float(data[2])
        gyro_z = float(data[3])
        acc_x = 0
        acc_y = 0
        acc_z = 0
        imu_data.append([timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

# 检查解析后的 IMU 数据
if not imu_data:
    print("No valid IMU data found. Please check the input file and filtering conditions.")
    exit()

# 继续执行后续代码
imu_df = pd.DataFrame(imu_data, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
print(f"IMU DataFrame:\n{imu_df.head()}")

# 读取 Odom 数据
with open(odom_file_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        data = line.strip().split(' ')
        timestamp = float(data[0])
        pos_x = float(data[1])
        pos_y = float(data[2])
        odo_data.append([timestamp, pos_x, pos_y])

odo_df = pd.DataFrame(odo_data, columns=['timestamp', 'pos_x', 'pos_y'])

# 检查 Odom 数据
if odo_df.empty:
    print("No valid Odom data found. Please check the Odom file.")
    exit()

# 自定义函数来找到最近的 Odom 数据
def find_nearest_odo(row):
    lower_idx = odo_df[odo_df['timestamp'] <= row['timestamp']].index.max()
    upper_idx = odo_df[odo_df['timestamp'] > row['timestamp']].index.min()

    # 检查索引是否为空
    if pd.isna(lower_idx) or pd.isna(upper_idx):
        return pd.Series([0.0, 0.0])  # 返回默认值

    lower_data = odo_df.loc[lower_idx]
    upper_data = odo_df.loc[upper_idx]

    t1 = lower_data['timestamp']
    t2 = upper_data['timestamp']
    weight_upper = (row['timestamp'] - t1) / (t2 - t1)
    weight_lower = (t2 - row['timestamp']) / (t2 - t1)
    print(f"Interpolating between timestamps {t1} and {t2} with weights {weight_lower} and {weight_upper}")

    pos_x = weight_lower * lower_data['pos_x'] + weight_upper * upper_data['pos_x']
    pos_y = weight_lower * lower_data['pos_y'] + weight_upper * upper_data['pos_y']
    print(f"Interpolated pos_x: {pos_x}, pos_y: {pos_y}")

    return pd.Series([pos_x, pos_y])

# 应用插值计算
print(imu_df.info())

imu_df[['pos_x', 'pos_y']] = imu_df.apply(find_nearest_odo, axis=1, result_type='expand')
imu_df[['pos_x', 'pos_y']] = imu_df[['pos_x', 'pos_y']].diff().fillna(0) * 10000
odo_df = imu_df.copy(deep=True)

# 初始化外参矩阵
T_base_link_t265_imu = np.array([[-4.67072555e-04, -8.41766754e-03,  9.99964462e-01],
                                 [ 9.99777229e-01, -2.11047138e-02,  2.89326323e-04],
                                 [ 2.11015284e-02,  9.99741834e-01,  8.42564976e-03]])

# 转换陀螺仪和加速度计数据
for i in range(len(odo_df)):
    imu_vec = np.array([odo_df.loc[i, 'gyro_x'], odo_df.loc[i, 'gyro_y'], odo_df.loc[i, 'gyro_z']])
    acc_vec = np.array([odo_df.loc[i, 'acc_x'], odo_df.loc[i, 'acc_y'], odo_df.loc[i, 'acc_z']])

    gyro_transformed = np.dot(T_base_link_t265_imu, imu_vec)
    acc_transformed = np.dot(T_base_link_t265_imu, acc_vec)

    odo_df.loc[i, 'gyro_x'] = gyro_transformed[0]
    odo_df.loc[i, 'gyro_y'] = gyro_transformed[1]
    odo_df.loc[i, 'gyro_z'] = gyro_transformed[2]

    odo_df.loc[i, 'acc_x'] = acc_transformed[0]
    odo_df.loc[i, 'acc_y'] = acc_transformed[1]
    odo_df.loc[i, 'acc_z'] = acc_transformed[2]

# 计算累计里程数
odo_df['odo_left'] = odo_df[['pos_x', 'pos_y']].apply(lambda row: np.hypot(row['pos_x'], row['pos_y']), axis=1).cumsum().astype(int)
odo_df['odo_right'] = odo_df['odo_left']

# 定义列名
column_names = ['timestamp',
                'data_type',
                'pose_x',
                'pose_y',
                'pose_theta',
                'odo_gyro_pose_x',
                'odo_gyro_pose_y',
                'odo_gyro_pose_theta',
                'revised_pose_x',
                'revised_pose_y',
                'revised_pose_theta',
                'accel_x',
                'accel_y',
                'accel_z',
                'euler_roll',
                'euler_pitch',
                'euler_yaw',
                'gyro_odo_roll',
                'gyro_odo_pitch',
                'gyro_odo_yaw',
                'speed_v',
                'speed_w',
                'target_v',
                'target_w',
                'left_count',
                'right_count']

# 创建一个指定大小的 DataFrame，所有值初始化为 0
output_df = pd.DataFrame(0, index=range(odo_df.shape[0]), columns=column_names)

output_df['timestamp'] = (odo_df['timestamp'] * 1e3).astype(int)
output_df['data_type'] = 'gyroOdo'
output_df['gyro_odo_roll'] = odo_df['gyro_x']
output_df['gyro_odo_pitch'] = odo_df['gyro_y']
output_df['gyro_odo_yaw'] = odo_df['gyro_z']
output_df['left_count'] = odo_df['odo_left']
output_df['right_count'] = odo_df['odo_right']
output_df['accel_x'] = odo_df['acc_x']
output_df['accel_y'] = odo_df['acc_y']
output_df['accel_z'] = odo_df['acc_z']

# 查看最终输出的数据
print("\n最终输出的数据预览:")
print(output_df.tail())

# 将整合后的数据输出到 CSV 文件，以空格分隔, header以#开头
output_df.to_csv(output_file, index=False, sep=' ', header=True, float_format='%.6f')

print(f"\n数据已成功整合并输出到 {output_file}")
