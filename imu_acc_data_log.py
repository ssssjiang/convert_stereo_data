import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/rr_stereo_grass_01/RRLDR_fprintf_10hz.log'
vslam_data = pd.read_csv(file_path, sep=' ', header=None)
# timestamp data_type pose_x pose_y pose_theta odo_gyro_pose_x odo_gyro_pose_y odo_gyro_pose_theta revised_pose_x revised_pose_y revised_pose_theta accel_x accel_y accel_z euler_roll euler_pitch euler_yaw gyro_x gyro_y gyro_z speed_v speed_w target_v target_w left_count right_count
vslam_data.columns = ['timestamp', 'data_type', 'pose_x', 'pose_y', 'pose_theta', 'odo_gyro_pose_x', 'odo_gyro_pose_y', 'odo_gyro_pose_theta', 'revised_pose_x', 'revised_pose_y', 'revised_pose_theta', 'accel_x', 'accel_y', 'accel_z', 'euler_roll', 'euler_pitch', 'euler_yaw', 'gyro_x', 'gyro_y', 'gyro_z', 'speed_v', 'speed_w', 'target_v', 'target_w', 'left_count', 'right_count']

print(vslam_data['gyro_x'].describe())
print(vslam_data['gyro_x'].value_counts())

print(vslam_data['gyro_y'].describe())
print(vslam_data['gyro_y'].value_counts())

print(vslam_data['gyro_z'].describe())
print(vslam_data['gyro_z'].value_counts())


# 频谱分析函数
def frequency_analysis(data, signal_column, time_column):
    # 提取信号和时间戳
    signal = data[signal_column].values
    timestamps = data[time_column].values

    # 计算采样间隔和采样率
    dt = np.mean(np.diff(timestamps)) / 1000  # 时间间隔
    fs = 1 / dt  # 采样率 (Hz)
    print(f"dt: {dt}, fs: {fs}")

    # FFT 计算
    n = len(signal)  # 数据点数量
    yf = fft(signal)  # 计算 FFT
    xf = fftfreq(n, dt)[:n // 2]  # 频率轴 (正频部分)

    # 计算功率谱
    ps = 2.0 / n * np.abs(yf[:n // 2])

    return xf, ps  # 返回频率和功率谱

# 执行频谱分析
frequencies_x, power_spectrum_x = frequency_analysis(vslam_data, 'gyro_x', 'timestamp')
frequencies_y, power_spectrum_y = frequency_analysis(vslam_data, 'gyro_y', 'timestamp')

# 绘制频谱
plt.figure(figsize=(10, 6))
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
cutoff_frequency = 10.0  # Cutoff frequency (Hz)
sampling_rate = 100.0  # Assume sampling rate is 100 Hz
order = 2

# Apply low-pass filter to accelerometer data
vslam_data['gyro_x_filtered'] = butter_lowpass_filter(vslam_data['gyro_x'], cutoff_frequency, sampling_rate, order)
vslam_data['gyro_y_filtered'] = butter_lowpass_filter(vslam_data['gyro_y'], cutoff_frequency, sampling_rate, order)
vslam_data['gyro_z_filtered'] = butter_lowpass_filter(vslam_data['gyro_z'], cutoff_frequency, sampling_rate, order)

# Print statistics of filtered data
print(vslam_data['gyro_x_filtered'].describe())
print(vslam_data['gyro_y_filtered'].describe())
print(vslam_data['gyro_z_filtered'].describe())

# Plot the data with higher resolution
plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and DPI for higher resolution

# Plot Accelerometer Data
plt.subplot(211)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_x'], label='gyro_x', linewidth=1.5)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_y'], label='gyro_y', linewidth=1.5)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_z'], label='gyro_z', linewidth=1.5)

plt.legend()
plt.title('Gyroscope Raw Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()

plt.subplot(212)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_x_filtered'], label='gyro_x_filtered', linewidth=1.5)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_y_filtered'], label='gyro_y_filtered', linewidth=1.5)
plt.plot(vslam_data['timestamp'], vslam_data['gyro_z_filtered'], label='gyro_z_filtered', linewidth=1.5)

plt.legend()
plt.title('Gyroscope filtered Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# # Show the plot
# plt.show()

# # replace gyro data with filtered data
# vslam_data['gyro_x'] = vslam_data['gyro_x_filtered']
# vslam_data['gyro_y'] = vslam_data['gyro_y_filtered']
# vslam_data['gyro_z'] = vslam_data['gyro_z_filtered']
#
# # drop filtered data
# vslam_data.drop(columns=['gyro_x_filtered', 'gyro_y_filtered', 'gyro_z_filtered'], inplace=True)
