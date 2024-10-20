import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from convert_extrinsic import T_odo_imu

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/2020-08-14/imu0/data.csv'
imu_data = pd.read_csv(file_path)

# Extract individual columns
time = imu_data.iloc[:, 0].values
gyro_x = imu_data.iloc[:, 1].values
gyro_y = imu_data.iloc[:, 2].values
gyro_z = imu_data.iloc[:, 3].values
accel_x = imu_data.iloc[:, 4].values
accel_y = imu_data.iloc[:, 5].values
accel_z = imu_data.iloc[:, 6].values

# Convert timestamp to seconds (assuming nanoseconds)
time_seconds = (time - time[0]) / 1e9

# Kalman filter parameters
process_noise = 0.001
measurement_noise = 0.1
estimated_error = 1.0
dt = np.mean(np.diff(time_seconds))

# Reinitialize Kalman filter parameters for X, Y, and Z axes
P_x = estimated_error
P_y = estimated_error
P_z = estimated_error

# Initialize variables for Kalman filter with accelerometer correction for X, Y, and Z axes
kalman_angle_x_accel = np.zeros(len(time_seconds))
kalman_angle_y_accel = np.zeros(len(time_seconds))
kalman_angle_z_accel = np.zeros(len(time_seconds))

# Apply Kalman filter with accelerometer update for X, Y, and Z axes
for i in range(1, len(time_seconds)):
    # Predict step using gyroscope data
    angle_x_pred = kalman_angle_x_accel[i - 1] + gyro_x[i] * dt
    angle_y_pred = kalman_angle_y_accel[i - 1] + gyro_y[i] * dt
    angle_z_pred = kalman_angle_z_accel[i - 1] + gyro_z[i] * dt
    P_x_pred = P_x + process_noise
    P_y_pred = P_y + process_noise
    P_z_pred = P_z + process_noise

    # Measurement update using accelerometer data (calculate tilt angles)
    accel_angle_x = np.arctan2(accel_y[i], accel_z[i])
    accel_angle_y = np.arctan2(-accel_x[i], accel_z[i])
    accel_angle_z = 0  # Assuming no direct measurement available for Z angle (yaw)

    # Calculate Kalman gain for X, Y, and Z
    kalman_gain_x = P_x_pred / (P_x_pred + measurement_noise)
    kalman_gain_y = P_y_pred / (P_y_pred + measurement_noise)
    kalman_gain_z = P_z_pred / (P_z_pred + measurement_noise)

    # Update angles with accelerometer data for X and Y, keep gyroscope update for Z
    kalman_angle_x_accel[i] = angle_x_pred + kalman_gain_x * (accel_angle_x - angle_x_pred)
    kalman_angle_y_accel[i] = angle_y_pred + kalman_gain_y * (accel_angle_y - angle_y_pred)
    kalman_angle_z_accel[i] = angle_z_pred  # Z update relies on gyroscope only
    P_x = (1 - kalman_gain_x) * P_x_pred
    P_y = (1 - kalman_gain_y) * P_y_pred
    P_z = (1 - kalman_gain_z) * P_z_pred

# Convert angles from radians to degrees for visualization
kalman_angle_x_accel_deg = np.degrees(kalman_angle_x_accel)
kalman_angle_y_accel_deg = np.degrees(kalman_angle_y_accel)
kalman_angle_z_accel_deg = np.degrees(kalman_angle_z_accel)

# Plot the Kalman filtered angles with accelerometer correction for X, Y, and Z axes in degrees
plt.figure(figsize=(14, 10))

# Plot Kalman filtered angles with accelerometer correction for X, Y, and Z axes
plt.plot(time_seconds, kalman_angle_x_accel_deg, label='Kalman Filtered Angle X (With Accel)', color='b')
plt.plot(time_seconds, kalman_angle_y_accel_deg, label='Kalman Filtered Angle Y (With Accel)', color='r')
plt.plot(time_seconds, kalman_angle_z_accel_deg, label='Kalman Filtered Angle Z (With Accel)', color='g')

plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Kalman Filtered Angles with Accelerometer Correction for X, Y, and Z Axes')
plt.legend()
plt.grid(True)
plt.show()


# extinsic_matrix T_odo_imu:
# [[ 0.        1.        0.        0.104   ]
#  [-1.        0.        0.        0.      ]
#  [ 0.        0.        1.        0.003309]
#  [ 0.        0.        0.        1.      ]]

# T_odo_imu = np.array([
#     [0.0, 1.0, 0.0, 0.104],
#     [-1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.003309],
#     [0.0, 0.0, 0.0, 1.0]
# ])
#
# gyro_in_odo = T_odo_imu[:3, :3].dot(np.array([gyro_x, gyro_y, gyro_z]))
# print(gyro_in_odo)

