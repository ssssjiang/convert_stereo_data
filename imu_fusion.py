import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
import transformations as tf
import argparse

def load_imu_data(file_path, start_time):
    """
    Load IMU data from a CSV file and filter by start timestamp.
    :param file_path: Path to the IMU data file
    :param start_time: Start timestamp for filtering
    :return: Tuple (timestamps, gyro_data, acc_data)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IMU data file not found: {file_path}")

    imu_data = pd.read_csv(file_path, sep=' ', header=None)
    imu_data.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Filter data by start timestamp
    imu_data = imu_data[imu_data['timestamp'] >= start_time]
    if imu_data.empty:
        raise ValueError("No data points exist after the specified start_time.")

    timestamps = imu_data['timestamp'].values
    gyro_data = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
    acc_data = imu_data[['accel_x', 'accel_y', 'accel_z']].values

    return timestamps, gyro_data, acc_data

def compute_euler_angles(timestamps, gyro_data, acc_data, sampling_rate=140.0, gain=0.033):
    """
    Compute Euler angles using the Madgwick filter.
    :param timestamps: Array of timestamps
    :param gyro_data: Array of gyroscope data
    :param acc_data: Array of accelerometer data
    :param sampling_rate: Sampling rate of the IMU data
    :param gain: Gain for the Madgwick filter
    :return: Array of Euler angles in degrees
    """
    dt = 1.0 / sampling_rate
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
    euler_angles = []
    madgwick = Madgwick(frequency=sampling_rate, gain=gain)

    for i in range(len(acc_data)):
        try:
            q = madgwick.updateIMU(q, gyr=gyro_data[i], acc=acc_data[i])
            q_h = [q[1], q[2], q[3], q[0]]  # Convert quaternion format
            euler = tf.euler_from_quaternion(q_h, axes='sxyz')
            euler_angles.append(euler)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            euler_angles.append([np.nan, np.nan, np.nan])

    return np.degrees(euler_angles)

def plot_euler_angles(timestamps, euler_angles):
    """
    Plot Euler angles over time.
    :param timestamps: Array of timestamps
    :param euler_angles: Array of Euler angles (roll, pitch, yaw)
    """
    try:
        plt.figure(figsize=(15, 10), dpi=150)
        plt.plot(timestamps, euler_angles[:, 0], label='Roll', alpha=0.7)
        plt.plot(timestamps, euler_angles[:, 1], label='Pitch', alpha=0.7)
        plt.plot(timestamps, euler_angles[:, 2], label='Yaw', alpha=0.7)
        plt.title('Euler Angles over Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Error plotting Euler angles: {e}")

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="IMU Data Fusion and Euler Angles Visualization")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the IMU data file')
    parser.add_argument('--sampling_rate', type=float, default=50.0, help='Sampling rate of the IMU data (Hz)')
    parser.add_argument('--gain', type=float, default=0.03, help='Gain for the Madgwick filter')
    parser.add_argument('--start_time', type=float, required=True, help="Start timestamp for analysis (in milliseconds).")
    return parser.parse_args()

def main():
    # file_path = '/home/roborock/datasets/roborock/stereo/rr_stereo_grass_01/imu.txt'
    args = parse_args()
    file_path = args.file_path
    sampling_rate = args.sampling_rate
    gain = args.gain
    start_time = args.start_time

    try:
        timestamps, gyro_data, acc_data = load_imu_data(file_path, start_time)
        euler_angles = compute_euler_angles(timestamps, gyro_data, acc_data, sampling_rate=sampling_rate, gain=gain)
        plot_euler_angles(timestamps, euler_angles)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
