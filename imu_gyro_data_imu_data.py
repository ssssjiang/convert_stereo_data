import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/mono/vv-room8-open/imu.txt'
imu_data = pd.read_csv(file_path, sep=' ', header=None)
imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

print(imu_data['accel_z'].describe())
print(imu_data['accel_z'].value_counts())

print(imu_data['gyro_x'].describe())
print(imu_data['gyro_x'].value_counts())

print(imu_data['gyro_y'].describe())
print(imu_data['gyro_y'].value_counts())

print(imu_data['gyro_z'].describe())
print(imu_data['gyro_z'].value_counts())

# Print acc_z > 12.0
print(imu_data[imu_data['accel_z'] > 12.0])


# 频谱分析函数
def frequency_analysis(data, signal_column, time_column):
    # 提取信号和时间戳
    signal = data[signal_column].values
    timestamps = data[time_column].values

    # 计算采样间隔和采样率
    dt = np.mean(np.diff(timestamps)) / 1000  # 时间间隔
    fs = 1 / dt  # 采样率 (Hz)

    # FFT 计算
    n = len(signal)  # 数据点数量
    yf = fft(signal)  # 计算 FFT
    xf = fftfreq(n, dt)[:n // 2]  # 频率轴 (正频部分)

    # 计算功率谱
    ps = 2.0 / n * np.abs(yf[:n // 2])

    return xf, ps  # 返回频率和功率谱

# 执行频谱分析
frequencies_x, power_spectrum_x = frequency_analysis(imu_data, 'gyro_x', 'timestamp')
frequencies_y, power_spectrum_y = frequency_analysis(imu_data, 'gyro_y', 'timestamp')

# 绘制频谱
plt.figure(figsize=(10, 6))
# 透明度 0.7
plt.plot(frequencies_x, power_spectrum_x, label='gyro_x', alpha=0.7)
plt.plot(frequencies_y, power_spectrum_y, label='gyro_y', alpha=0.7)
plt.title('Frequency Spectrum Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.show()

# Define low-pass filter function
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Set filter parameters
cutoff_frequency = 20.0  # Cutoff frequency (Hz)
sampling_rate = 140.0  # Assume sampling rate is 100 Hz
order = 3

# Apply low-pass filter to accelerometer data
imu_data['gyro_x_filtered'] = butter_lowpass_filter(imu_data['gyro_x'], cutoff_frequency, sampling_rate, order)
imu_data['gyro_y_filtered'] = butter_lowpass_filter(imu_data['gyro_y'], cutoff_frequency, sampling_rate, order)
imu_data['gyro_z_filtered'] = butter_lowpass_filter(imu_data['gyro_z'], cutoff_frequency, sampling_rate, order)

# Print statistics of filtered data
print(imu_data['gyro_x_filtered'].describe())
print(imu_data['gyro_y_filtered'].describe())
print(imu_data['gyro_z_filtered'].describe())

# Plot the data with higher resolution
plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and DPI for higher resolution

# Plot Accelerometer Data
plt.subplot(211)
plt.plot(imu_data['timestamp'], imu_data['gyro_x'], label='gyro_x', linewidth=1.5)
plt.plot(imu_data['timestamp'], imu_data['gyro_y'], label='gyro_y', linewidth=1.5)
# plt.plot(imu_data['timestamp'], imu_data['gyro_z'], label='gyro_z', linewidth=1.5)

plt.legend()
plt.title('Gyroscope Raw Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()

plt.subplot(212)
plt.plot(imu_data['timestamp'], imu_data['gyro_x_filtered'], label='gyro_x_filtered', linewidth=1.5)
plt.plot(imu_data['timestamp'], imu_data['gyro_y_filtered'], label='gyro_y_filtered', linewidth=1.5)
# plt.plot(imu_data['timestamp'], imu_data['gyro_z_filtered'], label='gyro_z_filtered', linewidth=1.5)

plt.legend()
plt.title('Gyroscope filtered Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()
#
# # Show the plot
# plt.show()
