import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import euler

import transformations as tf


# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/rr_stereo_grass_01/imu.txt'
imu_data = pd.read_csv(file_path, sep=' ', header=None)
# imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'euler_roll', 'euler_pitch', 'euler_yaw', 'gyro_x', 'gyro_y', 'gyro_z']
imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
gyro_data = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
acc_data = imu_data[['accel_x', 'accel_y', 'accel_z']].values
# euler_roll = imu_data['euler_roll'].values * 180 / np.pi
# euler_pitch = imu_data['euler_pitch'].values * 180 / np.pi
# euler_yaw = imu_data['euler_yaw'].values * 180 / np.pi

from ahrs.filters import Madgwick, EKF

sampling_rate = 140.0  # Hz
dt = 1.0 / sampling_rate  # 时间间隔

# 初始四元数
q = np.array([1.0, 0.0, 0.0, 0.0])

# 存储姿态（Roll, Pitch, Yaw）
euler_angles = []

# 初始化 Madgwick 滤波器
madgwick = Madgwick(frequency=sampling_rate, gain=0.00001)

# 滤波器迭代
for i in range(len(acc_data)):
    q = madgwick.updateIMU(q, gyr=gyro_data[i], acc=acc_data[i])
    q_h = [q[1], q[2], q[3], q[0]]  # 四元数转换为 [x, y, z, w] 格式
    euler = tf.euler_from_quaternion(q_h, axes='sxyz')  # 将四元数转换为欧拉角
    euler_angles.append(euler)

# 打印结果
euler_angles = np.degrees(euler_angles)  # 转换为角度制
for i, angles in enumerate(euler_angles):
    print(f"Sample {i+1}: Roll={angles[0]:.2f}, Pitch={angles[1]:.2f}, Yaw={angles[2]:.2f}")

# 绘制欧拉角
# x 轴为时间，y 轴为角度
# Plot the data with higher resolution
plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and DPI for higher resolution

# Plot Accelerometer Data
plt.subplot(211)
plt.plot(imu_data['timestamp'], euler_angles[:, 0], label='roll', alpha=0.7)
plt.plot(imu_data['timestamp'], euler_angles[:, 1], label='pitch', alpha=0.7)
plt.plot(imu_data['timestamp'], euler_angles[:, 2], label='yaw', alpha=0.7)
plt.title('Euler Angle')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.grid()
#
# plt.subplot(212)
# plt.plot(imu_data['timestamp'], euler_roll, label='roll', alpha=0.7)
# plt.plot(imu_data['timestamp'], euler_pitch, label='pitch', alpha=0.7)
# plt.plot(imu_data['timestamp'], euler_yaw, label='yaw', alpha=0.7)
# plt.title('Euler Angle')
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.grid()
#
# plt.legend()
# plt.show()
#
#
# # plot different euler angles
# plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and
# plt.plot(imu_data['timestamp'], euler_roll - euler_angles[:, 0], label='roll', alpha=0.7)
# plt.plot(imu_data['timestamp'], euler_pitch - euler_angles[:, 1], label='pitch', alpha=0.7)
# plt.plot(imu_data['timestamp'], euler_yaw - euler_angles[:, 2], label='yaw', alpha=0.7)
# plt.title('Euler Angle')
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.grid()

plt.legend()
plt.show()


# eular angle and 30cm / s speed, plot the trajectory
# 1. get the speed
# 2. get the pose
# 3. plot the trajectory


