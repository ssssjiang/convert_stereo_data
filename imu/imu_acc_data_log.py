import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import argparse

# /home/roborock/datasets/roborock/stereo/rr_stereo_grass_02/RRLDR_fprintf.log
# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process VSLAM data and perform analysis.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the VSLAM data file.")
    parser.add_argument('--cutoff_frequency', type=float, default=10.0, help="Cutoff frequency for low-pass filter (Hz).")
    parser.add_argument('--sampling_rate', type=float, default=100.0, help="Sampling rate (Hz).")
    parser.add_argument('--order', type=int, default=2, help="Order of the low-pass filter.")
    return parser.parse_args()

# Main script
def main():
    args = parse_args()

    # Load the CSV file
    vslam_data = pd.read_csv(args.file_path, sep=' ', header=None)

    # Assign column names
    vslam_data.columns = ['timestamp', 'data_type', 'pose_x', 'pose_y', 'pose_theta', 'odo_gyro_pose_x', 'odo_gyro_pose_y', 'odo_gyro_pose_theta', 'revised_pose_x', 'revised_pose_y', 'revised_pose_theta', 'accel_x', 'accel_y', 'accel_z', 'euler_roll', 'euler_pitch', 'euler_yaw', 'gyro_x', 'gyro_y', 'gyro_z', 'speed_v', 'speed_w', 'target_v', 'target_w', 'left_count', 'right_count']

    # Display statistics for accelerometer data
    print(vslam_data['accel_x'].describe())
    print(vslam_data['accel_x'].value_counts())

    print(vslam_data['accel_y'].describe())
    print(vslam_data['accel_y'].value_counts())

    print(vslam_data['accel_z'].describe())
    print(vslam_data['accel_z'].value_counts())

    # Frequency analysis function
    def frequency_analysis(data, signal_column, time_column):
        # Extract signal and timestamps
        signal = data[signal_column].values
        timestamps = data[time_column].values

        # Calculate sampling interval and rate
        dt = np.mean(np.diff(timestamps)) / 1000  # Time interval in seconds
        fs = 1 / dt  # Sampling rate (Hz)
        print(f"dt: {dt}, fs: {fs}")

        # Perform FFT
        n = len(signal)  # Number of data points
        yf = fft(signal)  # FFT calculation
        xf = fftfreq(n, dt)[:n // 2]  # Frequency axis (positive frequencies only)

        # Calculate power spectrum
        ps = 2.0 / n * np.abs(yf[:n // 2])

        return xf, ps  # Return frequencies and power spectrum

    # Perform frequency analysis
    frequencies_x, power_spectrum_x = frequency_analysis(vslam_data, 'accel_x', 'timestamp')
    frequencies_y, power_spectrum_y = frequency_analysis(vslam_data, 'accel_y', 'timestamp')
    frequencies_z, power_spectrum_z = frequency_analysis(vslam_data, 'accel_z', 'timestamp')

    # Plot frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_x, power_spectrum_x, label='accel_x', alpha=0.7)
    plt.plot(frequencies_y, power_spectrum_y, label='accel_y', alpha=0.7)
    plt.plot(frequencies_z, power_spectrum_z, label='accel_z', alpha=0.7)
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

    # Apply low-pass filter to accelerometer data
    vslam_data['accel_x_filtered'] = butter_lowpass_filter(vslam_data['accel_x'], args.cutoff_frequency, args.sampling_rate, args.order)
    vslam_data['accel_y_filtered'] = butter_lowpass_filter(vslam_data['accel_y'], args.cutoff_frequency, args.sampling_rate, args.order)
    vslam_data['accel_z_filtered'] = butter_lowpass_filter(vslam_data['accel_z'], args.cutoff_frequency, args.sampling_rate, args.order)

    # Print statistics of filtered data
    print(vslam_data['accel_x_filtered'].describe())
    print(vslam_data['accel_y_filtered'].describe())
    print(vslam_data['accel_z_filtered'].describe())

    # Plot the data with higher resolution
    plt.figure(figsize=(15, 10), dpi=300)  # Set figure size and DPI for higher resolution

    # Plot Accelerometer Data
    plt.subplot(211)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_x'], label='accel_x', linewidth=1.5, alpha=0.7)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_y'], label='accel_y', linewidth=1.5, alpha=0.7)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_z'], label='accel_z', linewidth=1.5, alpha=0.7)

    plt.legend()
    plt.title('Accelerometer Raw Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid()

    plt.subplot(212)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_x_filtered'], label='accel_x_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_y_filtered'], label='accel_y_filtered', linewidth=1.5, alpha=0.7)
    plt.plot(vslam_data['timestamp'], vslam_data['accel_z_filtered'], label='accel_z_filtered', linewidth=1.5, alpha=0.7)

    plt.legend()
    plt.title('Accelerometer Filtered Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()

if __name__ == "__main__":
    main()