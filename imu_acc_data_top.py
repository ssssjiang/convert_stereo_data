import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/rr_stereo_grass_01/imu.txt'
imu_data = pd.read_csv(file_path, sep=' ', header=None)
imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

print(imu_data['accel_z'].describe())
print(imu_data['accel_z'].value_counts())

print(imu_data['gyro_x'].describe())
print(imu_data['gyro_x'].value_counts())

print(imu_data['gyro_y'].describe())
print(imu_data['gyro_y'].value_counts())

# Print acc_z > 12.0
print(imu_data[imu_data['accel_z'] > 12.0])

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
imu_data['accel_z_filtered'] = butter_lowpass_filter(imu_data['accel_z'], cutoff_frequency, sampling_rate, order)

# Print statistics of filtered data
print(imu_data['accel_z_filtered'].describe())

# Plot the data with higher resolution
plt.figure(figsize=(15, 10), dpi=150)  # Set figure size and DPI for higher resolution

# Plot Accelerometer Data
plt.subplot(211)
plt.plot(imu_data['timestamp'], imu_data['accel_x'], label='accel_x', linewidth=1.5)
plt.plot(imu_data['timestamp'], imu_data['accel_y'], label='accel_y', linewidth=1.5)
# Uncomment to plot accel_z and accel_z_filtered
# plt.plot(imu_data['timestamp'], imu_data['accel_z'], label='accel_z', linewidth=1.5)
# plt.plot(imu_data['timestamp'], imu_data['accel_z_filtered'], label='accel_z_filtered', linewidth=1.5)
plt.legend()
plt.title('Accelerometer Data')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration (m/s^2)')
plt.grid()

# Plot Gyroscope Data
plt.subplot(212)
plt.plot(imu_data['timestamp'], imu_data['gyro_x'], label='gyro_x', linewidth=1.5)
plt.plot(imu_data['timestamp'], imu_data['gyro_y'], label='gyro_y', linewidth=1.5)
# Uncomment to plot gyro_z
plt.plot(imu_data['timestamp'], imu_data['gyro_z'], label='gyro_z', linewidth=1.5)


plt.legend()
plt.title('Gyroscope Data')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
