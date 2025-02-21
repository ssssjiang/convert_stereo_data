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
        save_filtered (bool): Whether to save filtered data plots.

    Returns:
        None
    """
    # Load the CSV file
    imu_data = pd.read_csv(file_path, sep=' ', header=None)
    imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']

    # Ensure all columns are numeric
    imu_data = imu_data.apply(pd.to_numeric, errors='coerce')
    imu_data = imu_data.dropna()

    # Filter data to include only rows after start_time
    imu_data = imu_data[imu_data['timestamp'] >= start_time]
    if imu_data.empty:
        raise ValueError("No data points exist after the specified start_time.")

    # Define low-pass filter function
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Select target axes for analysis
    target_axes = []
    if analysis_target in ["all", "accel"]:
        target_axes.extend(['accel_x', 'accel_y', 'accel_z'])
    if analysis_target in ["all", "gyro"]:
        target_axes.extend(['gyro_x', 'gyro_y', 'gyro_z'])

    # Apply low-pass filter to selected axes
    for axis in target_axes:
        imu_data[f'{axis}_filtered'] = butter_lowpass_filter(imu_data[axis], cutoff_frequency, sampling_rate, order)

    # Convert timestamp from milliseconds to seconds for plotting
    imu_data['time_seconds'] = imu_data['timestamp'] / 1000.0

    # Plot data based on analysis target
    if "accel" in analysis_target or analysis_target == "all":
        plt.figure(figsize=(20, 10))
        for i, axis in enumerate(['accel_x', 'accel_y', 'accel_z']):
            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(imu_data['time_seconds'], imu_data[axis], label=f'{axis} (raw)', linewidth=1.5)
            plt.title(f'{axis} Raw Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Value')
            plt.grid()

            if save_filtered:
                plt.subplot(3, 2, 2 * i + 2)
                plt.plot(imu_data['time_seconds'], imu_data[f'{axis}_filtered'], label=f'{axis} (filtered)', linewidth=1.5)
                plt.title(f'{axis} Filtered Data')
                plt.xlabel('Time (s)')
                plt.ylabel('Value')
                plt.grid()

        plt.tight_layout()
        accel_plot_path = os.path.join(save_dir, 'acceleration_data.png')
        plt.savefig(accel_plot_path)
        plt.close()

    if "gyro" in analysis_target or analysis_target == "all":
        plt.figure(figsize=(20, 10))
        for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(imu_data['time_seconds'], imu_data[axis], label=f'{axis} (raw)', linewidth=1.5)
            plt.title(f'{axis} Raw Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Value')
            plt.grid()

            if save_filtered:
                plt.subplot(3, 2, 2 * i + 2)
                plt.plot(imu_data['time_seconds'], imu_data[f'{axis}_filtered'], label=f'{axis} (filtered)', linewidth=1.5)
                plt.title(f'{axis} Filtered Data')
                plt.xlabel('Time (s)')
                plt.ylabel('Value')
                plt.grid()

        plt.tight_layout()
        gyro_plot_path = os.path.join(save_dir, 'gyroscope_data.png')
        plt.savefig(gyro_plot_path)
        plt.close()

    # Frequency analysis
    def frequency_analysis(signal, timestamps):
        dt = np.mean(np.diff(timestamps)) / 1000.0
        fs = 1 / dt
        n = len(signal)
        yf = fft(signal.to_numpy())
        xf = fftfreq(n, dt)[:n // 2]
        ps = 2.0 / n * np.abs(yf[:n // 2])
        return xf, ps

    if "accel" in analysis_target or analysis_target == "all":
        plt.figure(figsize=(20, 10))
        for i, axis in enumerate(['accel_x', 'accel_y', 'accel_z']):
            xf, ps = frequency_analysis(imu_data[axis], imu_data['timestamp'])
            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(xf, ps, label=f'{axis} Spectrum', linewidth=1.5)
            plt.title(f'{axis} Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid()

            if save_filtered:
                xf, ps = frequency_analysis(imu_data[f'{axis}_filtered'], imu_data['timestamp'])
                plt.subplot(3, 2, 2 * i + 2)
                plt.plot(xf, ps, label=f'{axis} Filtered Spectrum', linewidth=1.5)
                plt.title(f'{axis} Filtered Frequency Spectrum')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                plt.grid()

        plt.tight_layout()
        accel_spectrum_path = os.path.join(save_dir, 'acceleration_spectrum.png')
        plt.savefig(accel_spectrum_path)
        plt.close()

    if "gyro" in analysis_target or analysis_target == "all":
        plt.figure(figsize=(20, 10))
        for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            xf, ps = frequency_analysis(imu_data[axis], imu_data['timestamp'])
            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(xf, ps, label=f'{axis} Spectrum', linewidth=1.5)
            plt.title(f'{axis} Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid()

            if save_filtered:
                xf, ps = frequency_analysis(imu_data[f'{axis}_filtered'], imu_data['timestamp'])
                plt.subplot(3, 2, 2 * i + 2)
                plt.plot(xf, ps, label=f'{axis} Filtered Spectrum', linewidth=1.5)
                plt.title(f'{axis} Filtered Frequency Spectrum')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                plt.grid()

        plt.tight_layout()
        gyro_spectrum_path = os.path.join(save_dir, 'gyroscope_spectrum.png')
        plt.savefig(gyro_spectrum_path)
        plt.close()

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
    parser.add_argument('--save_filtered', action='store_true', help="Save filtered data plots.")

    args = parser.parse_args()

    process_imu_data(
        file_path=args.file_path,
        cutoff_frequency=args.cutoff_frequency,
        sampling_rate=args.sampling_rate,
        order=args.order,
        start_time=args.start_time,
        save_dir=args.save_dir,
        analysis_target=args.analysis_target,
        save_filtered=args.save_filtered
    )
