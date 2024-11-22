# grep 'gyroOdo' RRLDR_fprintf.log | cut -d ' ' -f 1,12-14,18-20  > imu.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from convert_stereo_data.imu_acc_data import file_path

# # 设置最大行数和列数
# pd.set_option('display.max_rows', None)  # 显示所有行
# pd.set_option('display.max_columns', None)  # 显示所有列

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/mono/70_fov/sen1_close_clean/imu.txt'
# file_path = '/home/roborock/datasets/roborock/mono/vv-room8-open/imu.txt'
# file_path = '/home/roborock/datasets/roborock/mono/sen1_quick/imu.txt'
# 读取数据 timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
imu_data = pd.read_csv(file_path, sep=' ', header=None)
imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']


print(imu_data['accel_z'].describe())
print(imu_data['accel_z'].value_counts())

print(imu_data['gyro_x'].describe())
print(imu_data['gyro_x'].value_counts())

print(imu_data['gyro_y'].describe())
print(imu_data['gyro_y'].value_counts())

# print acc_z > 12.0
print(imu_data[imu_data['accel_z'] > 12.0])

# 定义低通滤波器函数
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 设置滤波器参数
cutoff_frequency = 10.0  # 截止频率（Hz）
sampling_rate = 100.0  # 假设采样率为 100 Hz
order = 2

# 对加速度数据进行滤波
imu_data['accel_z_filtered'] = butter_lowpass_filter(imu_data['accel_z'], cutoff_frequency, sampling_rate, order)

# 打印描述统计信息
print(imu_data['accel_z_filtered'].describe())

# plot the data
plt.figure()
plt.subplot(211)
plt.plot(imu_data['timestamp'], imu_data['accel_x'], label='accel_x')
plt.plot(imu_data['timestamp'], imu_data['accel_y'], label='accel_y')
# plt.plot(imu_data['timestamp'], imu_data['accel_z'], label='accel_z')
# plt.plot(imu_data['timestamp'], imu_data['accel_z_filtered'], label='accel_z_filtered')
plt.legend()
plt.title('Accelerometer Data')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration (m/s^2)')
plt.grid()

plt.subplot(212)
plt.plot(imu_data['timestamp'], imu_data['gyro_x'], label='gyro_x')
plt.plot(imu_data['timestamp'], imu_data['gyro_y'], label='gyro_y')
# plt.plot(imu_data['timestamp'], imu_data['gyro_z'], label='gyro_z')
plt.legend()

plt.title('Gyroscope Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()
plt.tight_layout()


plt.show()



