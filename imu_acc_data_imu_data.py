import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/rr_stereo_grass_01/imu.txt'
imu_data = pd.read_csv(file_path, sep=' ', header=None)
imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

print(imu_data['accel_x'].describe())
print(imu_data['accel_y'].describe())
print(imu_data['accel_z'].describe())

# compute acc 模长
imu_data['accel_magnitude'] = np.sqrt(imu_data['accel_x']**2 + imu_data['accel_y']**2 + imu_data['accel_z']**2)

print(imu_data['accel_magnitude'].describe())

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
frequencies_x, power_spectrum_x = frequency_analysis(imu_data, 'accel_x', 'timestamp')
frequencies_y, power_spectrum_y = frequency_analysis(imu_data, 'accel_y', 'timestamp')
frequencies_z, power_spectrum_z = frequency_analysis(imu_data, 'accel_z', 'timestamp')
frequencies_m, power_spectrum_m = frequency_analysis(imu_data, 'accel_magnitude', 'timestamp')
# 绘制频谱
plt.figure(figsize=(10, 6))
# 透明度 0.7
plt.plot(frequencies_x, power_spectrum_x, label='accel_x', alpha=0.7)
plt.plot(frequencies_y, power_spectrum_y, label='accel_y', alpha=0.7)
plt.plot(frequencies_z, power_spectrum_z, label='accel_z', alpha=0.7)
plt.plot(frequencies_m, power_spectrum_m, label='accel_magnitude', alpha=0.7)
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
sampling_rate = 140.0  # Assume sampling rate is 100 Hz
order = 3

# # Apply low-pass filter to accelerometer data
# imu_data['accel_x_filtered'] = butter_lowpass_filter(imu_data['accel_x'], cutoff_frequency, sampling_rate, order)
# imu_data['accel_y_filtered'] = butter_lowpass_filter(imu_data['accel_y'], cutoff_frequency, sampling_rate, order)
# imu_data['accel_z_filtered'] = butter_lowpass_filter(imu_data['accel_z'], cutoff_frequency, sampling_rate, order)
# imu_data['accel_magnitude_filtered'] = butter_lowpass_filter(imu_data['accel_magnitude'], cutoff_frequency, sampling_rate, order)


# if imu date every 100ms, compute the mean of every 5 data
window_size = 14
imu_data['accel_x_filtered'] = imu_data['accel_x'].rolling(window=window_size).mean()
imu_data['accel_y_filtered'] = imu_data['accel_y'].rolling(window=window_size).mean()
imu_data['accel_z_filtered'] = imu_data['accel_z'].rolling(window=window_size).mean()
imu_data['accel_magnitude_filtered'] = np.sqrt(imu_data['accel_x_filtered']**2 + imu_data['accel_y_filtered']**2 + imu_data['accel_z_filtered']**2)

# Print statistics of filtered data
print(imu_data['accel_x_filtered'].describe())
print(imu_data['accel_y_filtered'].describe())
print(imu_data['accel_z_filtered'].describe())
print(imu_data['accel_magnitude_filtered'].describe())

# Plot the data with higher resolution
plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and DPI for higher resolution

# Plot Accelerometer Data
plt.subplot(211)
# y轴划分更细
plt.plot(imu_data['timestamp'], imu_data['accel_x'], label='accel_x', linewidth=1.5, alpha=0.7)
plt.plot(imu_data['timestamp'], imu_data['accel_y'], label='accel_y', linewidth=1.5, alpha=0.7)
# plt.plot(imu_data['timestamp'], imu_data['accel_z'], label='accel_z', linewidth=1.5, alpha=0.7)
plt.plot(imu_data['timestamp'], imu_data['accel_magnitude'], label='accel_magnitude', linewidth=1.5, alpha=0.7)

plt.legend()
plt.title('Accel Raw Data')
plt.xlabel('Timestamp')
plt.ylabel('acceleration (m/s^2)')
plt.grid()

plt.subplot(212)
plt.plot(imu_data['timestamp'], imu_data['accel_x_filtered'], label='accel_x_filtered', linewidth=1.5, alpha=0.7)
plt.plot(imu_data['timestamp'], imu_data['accel_y_filtered'], label='accel_y_filtered', linewidth=1.5, alpha=0.7)
plt.plot(imu_data['timestamp'], imu_data['accel_z_filtered'], label='accel_z_filtered', linewidth=1.5, alpha=0.7)
plt.plot(imu_data['timestamp'], imu_data['accel_magnitude_filtered'], label='accel_magnitude_filtered', linewidth=1.5, alpha=0.7)

plt.legend()
plt.title('Accel filtered Data')
plt.xlabel('Timestamp')
plt.ylabel('acceleration (m/s^2)')
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
