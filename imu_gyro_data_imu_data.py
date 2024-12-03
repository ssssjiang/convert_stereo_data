import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import argparse

# /home/roborock/datasets/roborock/stereo/rr_stereo_grass_02/imu.txt
# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process IMU data and perform analysis.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the IMU data file.")
    parser.add_argument('--cutoff_frequency', type=float, default=10.0, help="Cutoff frequency for low-pass filter (Hz).")
    parser.add_argument('--sampling_rate', type=float, default=140.0, help="Sampling rate (Hz).")
    parser.add_argument('--order', type=int, default=2, help="Order of the low-pass filter.")
    return parser.parse_args()

# Main script
def main():
    args = parse_args()

    # Load the CSV file
    imu_data = pd.read_csv(args.file_path, sep=' ', header=None)
    imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Display statistics for gyroerometer data
    print(imu_data['gyro_x'].describe())
    print(imu_data['gyro_y'].describe())
    print(imu_data['gyro_z'].describe())

    # Frequency analysis function
    def frequency_analysis(data, signal_column, time_column):
        signal = data[signal_column].values
        timestamps = data[time_column].values

        # Calculate sampling interval and rate
        dt = np.mean(np.diff(timestamps)) / 1000
        fs = 1 / dt

        # Perform FFT
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, dt)[:n // 2]

        # Calculate power spectrum
        ps = 2.0 / n * np.abs(yf[:n // 2])
        return xf, ps

    # Perform frequency analysis
    frequencies_x, power_spectrum_x = frequency_analysis(imu_data, 'gyro_x', 'timestamp')
    frequencies_y, power_spectrum_y = frequency_analysis(imu_data, 'gyro_y', 'timestamp')
    frequencies_z, power_spectrum_z = frequency_analysis(imu_data, 'gyro_z', 'timestamp')

    # Plot frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_x, power_spectrum_x, label='gyro_x', alpha=0.7)
    plt.plot(frequencies_y, power_spectrum_y, label='gyro_y', alpha=0.7)
    plt.plot(frequencies_z, power_spectrum_z, label='gyro_z', alpha=0.7)
    plt.title('Frequency Spectrum Analysis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()

    # Plot the data
    plt.figure(figsize=(15, 10), dpi=300)

    plt.subplot(211)
    plt.plot(imu_data['timestamp'], imu_data['gyro_x'], label='gyro_x', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['gyro_y'], label='gyro_y', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['gyro_z'], label='gyro_z', linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.title('Gyroscope Raw Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()

    plt.subplot(212)
    plt.plot(imu_data['timestamp'], imu_data['gyro_x_filtered'], label='gyro_x_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['gyro_y_filtered'], label='gyro_y_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['gyro_z_filtered'], label='gyro_z_filtered', linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.title('Gyroscope filtered Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
