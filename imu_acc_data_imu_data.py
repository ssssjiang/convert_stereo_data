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
    parser.add_argument('--window_size', type=int, default=14, help="Window size for rolling mean filter.")
    return parser.parse_args()

# Main script
def main():
    args = parse_args()

    # Load the CSV file
    imu_data = pd.read_csv(args.file_path, sep=' ', header=None)
    imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Display statistics for accelerometer data
    print(imu_data['accel_x'].describe())
    print(imu_data['accel_y'].describe())
    print(imu_data['accel_z'].describe())

    # Compute accelerometer magnitude
    imu_data['accel_magnitude'] = np.sqrt(imu_data['accel_x']**2 + imu_data['accel_y']**2 + imu_data['accel_z']**2)
    print(imu_data['accel_magnitude'].describe())

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
    frequencies_x, power_spectrum_x = frequency_analysis(imu_data, 'accel_x', 'timestamp')
    frequencies_y, power_spectrum_y = frequency_analysis(imu_data, 'accel_y', 'timestamp')
    frequencies_z, power_spectrum_z = frequency_analysis(imu_data, 'accel_z', 'timestamp')
    frequencies_m, power_spectrum_m = frequency_analysis(imu_data, 'accel_magnitude', 'timestamp')

    # Plot frequency spectrum
    plt.figure(figsize=(10, 6))
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

    # Apply rolling mean filter
    imu_data['accel_x_filtered'] = imu_data['accel_x'].rolling(window=args.window_size).mean()
    imu_data['accel_y_filtered'] = imu_data['accel_y'].rolling(window=args.window_size).mean()
    imu_data['accel_z_filtered'] = imu_data['accel_z'].rolling(window=args.window_size).mean()
    imu_data['accel_magnitude_filtered'] = np.sqrt(
        imu_data['accel_x_filtered']**2 + imu_data['accel_y_filtered']**2 + imu_data['accel_z_filtered']**2
    )

    # Print statistics of filtered data
    print(imu_data['accel_x_filtered'].describe())
    print(imu_data['accel_y_filtered'].describe())
    print(imu_data['accel_z_filtered'].describe())
    print(imu_data['accel_magnitude_filtered'].describe())

    # Plot the data
    plt.figure(figsize=(15, 10), dpi=300)

    plt.subplot(211)
    plt.plot(imu_data['timestamp'], imu_data['accel_x'], label='accel_x', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['accel_y'], label='accel_y', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['accel_magnitude'], label='accel_magnitude', linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.title('Accel Raw Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid()

    plt.subplot(212)
    plt.plot(imu_data['timestamp'], imu_data['accel_x_filtered'], label='accel_x_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['accel_y_filtered'], label='accel_y_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['accel_z_filtered'], label='accel_z_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(imu_data['timestamp'], imu_data['accel_magnitude_filtered'], label='accel_magnitude_filtered', linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.title('Accel Filtered Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
