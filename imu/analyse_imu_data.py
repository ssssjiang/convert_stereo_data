import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

def process_imu_data(file_path, cutoff_frequency=10.0, sampling_rate=50.0, order=2, start_time=0.0, save_dir=".", analysis_target="all", save_filtered=False):
    """
    Processes IMU data and generates analysis plots for each axis and magnitude, saved to specified directory.

    Parameters:
        file_path (str): Path to the IMU data file.
        cutoff_frequency (float): Cutoff frequency for low-pass filter (Hz).
        sampling_rate (float): Sampling rate (Hz).
        order (int): Order of the low-pass filter.
        start_time (float): Start timestamp for analysis (in milliseconds).
        save_dir (str): Directory to save generated plots.
        analysis_target (str): Target data to analyze ("all", "accel", "gyro").

    Returns:
        None
    """
    # Load the CSV file
    imu_data = pd.read_csv(file_path, sep=',', header=None)
    imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']

    # Ensure all columns are numeric
    for col in ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']:
        imu_data[col] = pd.to_numeric(imu_data[col], errors='coerce')

    # Drop rows with invalid (NaN) values
    imu_data = imu_data.dropna()

    # Filter data to include only rows after start_time
    imu_data = imu_data[imu_data['timestamp'] >= start_time]
    if imu_data.empty:
        raise ValueError("No data points exist after the specified start_time.")

    print(imu_data.head())  # 查看前几行
    print(imu_data.dtypes)  # 检查每列的数据类型

    # Compute accelerometer magnitude if needed
    if analysis_target in ["all", "accel"]:
        imu_data['accel_magnitude'] = np.sqrt(imu_data['accel_x']**2 + imu_data['accel_y']**2 + imu_data['accel_z']**2)

    def frequency_analysis(data, signal_column, time_column):
        signal = data[signal_column].values
        timestamps = data[time_column].values

        # Calculate sampling interval and rate
        dt = np.mean(np.diff(timestamps)) / 1000  # Assuming timestamp in milliseconds
        fs = 1 / dt

        # Perform FFT
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, dt)[:n // 2]

        # Calculate power spectrum
        ps = 2.0 / n * np.abs(yf[:n // 2])
        return xf, ps

    # Select target axes for analysis
    target_axes = []
    if analysis_target in ["all", "accel"]:
        target_axes.extend(['accel_x', 'accel_y', 'accel_z', 'accel_magnitude'])
    if analysis_target in ["all", "gyro"]:
        target_axes.extend(['gyro_x', 'gyro_y', 'gyro_z'])

    # Perform frequency analysis for selected axes
    frequencies = {}
    power_spectra = {}
    for axis in target_axes:
        frequencies[axis], power_spectra[axis] = frequency_analysis(imu_data, axis, 'timestamp')

    # Define low-pass filter function
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Apply low-pass filter to selected axes
    for axis in target_axes:
        imu_data[f'{axis}_filtered'] = butter_lowpass_filter(imu_data[axis], cutoff_frequency, sampling_rate, order)

    # Save frequency spectrum plots for selected axes
    for axis in target_axes:
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[axis], power_spectra[axis], label=axis, alpha=0.7)
        plt.title(f'Frequency Spectrum Analysis - {axis}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.tight_layout()
        spectrum_plot_path = os.path.join(save_dir, f'frequency_spectrum_{axis}.png')
        plt.savefig(spectrum_plot_path)
        plt.close()

    # Save filtered and raw data plots for selected axes
    for axis in target_axes:
        plt.figure(figsize=(15, 8))

        plt.subplot(211)
        plt.plot(imu_data['timestamp'], imu_data[axis], label=f'{axis} (raw)', linewidth=1.5, alpha=0.7)
        plt.legend()
        plt.title(f'Raw {axis} Data')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Value')
        plt.grid()

        if save_filtered:
            plt.subplot(212)
            plt.plot(imu_data['timestamp'], imu_data[f'{axis}_filtered'], label=f'{axis} (filtered)', linewidth=1.5, alpha=0.7)
            plt.legend()
            plt.title(f'Filtered {axis} Data')
            plt.xlabel('Timestamp (ms)')
            plt.ylabel('Value')
            plt.grid()

        plt.tight_layout()
        filtered_plot_path = os.path.join(save_dir, f'filtered_data_{axis}.png')
        plt.savefig(filtered_plot_path)
        plt.close()

        print(f"Frequency spectrum plot for {axis} saved to: {os.path.join(save_dir, f'frequency_spectrum_{axis}.png')}")
        if save_filtered:
            print(f"Filtered data plot for {axis} saved to: {os.path.join(save_dir, f'filtered_data_{axis}.png')}")

        print(f"Frequency spectrum plot for {axis} saved to: {os.path.join(save_dir, f'frequency_spectrum_{axis}.png')}")
        print(f"Filtered data plot for {axis} saved to: {os.path.join(save_dir, f'filtered_data_{axis}.png')}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process IMU data and perform analysis.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the IMU data file.")
    parser.add_argument('--cutoff_frequency', type=float, default=10.0, help="Cutoff frequency for low-pass filter (Hz).")
    parser.add_argument('--sampling_rate', type=float, default=50.0, help="Sampling rate (Hz).")
    parser.add_argument('--order', type=int, default=2, help="Order of the low-pass filter.")
    parser.add_argument('--start_time', type=float, default=0.0, help="Start timestamp for analysis (in milliseconds).")
    parser.add_argument('--save_dir', type=str, default=".", help="Directory to save generated plots.")
    parser.add_argument('--analysis_target', type=str, choices=["all", "accel", "gyro"], default="all", help="Target data to analyze (all, accel, gyro).")

    args = parser.parse_args()

    process_imu_data(
        file_path=args.file_path,
        cutoff_frequency=args.cutoff_frequency,
        sampling_rate=args.sampling_rate,
        order=args.order,
        start_time=args.start_time,
        save_dir=args.save_dir,
        analysis_target=args.analysis_target
    )
