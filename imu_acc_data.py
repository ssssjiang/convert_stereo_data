import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/roborock/datasets/roborock/stereo/2020-08-14/imu0/data.csv'
imu_data = pd.read_csv(file_path)

# Extract individual columns
time = imu_data.iloc[:, 0].values
accel_x = imu_data.iloc[:, 4].values
accel_y = imu_data.iloc[:, 5].values
accel_z = imu_data.iloc[:, 6].values

# Convert timestamp to seconds (assuming nanoseconds)
time_seconds = (time - time[0]) / 1e9

dt = np.mean(np.diff(time_seconds))

# Subtract gravity from Z-axis acceleration (assuming gravity is approximately 9.81 m/s^2)
gravity = 9.81
accel_z = accel_z - gravity

# Kalman filter parameters
process_noise = 0.001
measurement_noise = 0.1
estimated_error = 1.0

# Initialize Kalman filter variables for velocity and displacement
velocity_x = np.zeros(len(time_seconds))
velocity_y = np.zeros(len(time_seconds))
velocity_z = np.zeros(len(time_seconds))
displacement_x = np.zeros(len(time_seconds))
displacement_y = np.zeros(len(time_seconds))
displacement_z = np.zeros(len(time_seconds))

P_x = estimated_error
P_y = estimated_error
P_z = estimated_error

# Apply Kalman filter to estimate velocity and displacement for X, Y, and Z axes
for i in range(1, len(time_seconds)):
    # Predict step for velocity
    vel_x_pred = velocity_x[i - 1] + accel_x[i] * dt
    vel_y_pred = velocity_y[i - 1] + accel_y[i] * dt
    vel_z_pred = velocity_z[i - 1] + accel_z[i] * dt
    P_x_pred = P_x + process_noise
    P_y_pred = P_y + process_noise
    P_z_pred = P_z + process_noise

    # Measurement update (assuming measurement is the predicted value itself for simplicity)
    kalman_gain_x = P_x_pred / (P_x_pred + measurement_noise)
    kalman_gain_y = P_y_pred / (P_y_pred + measurement_noise)
    kalman_gain_z = P_z_pred / (P_z_pred + measurement_noise)

    # Update velocity with Kalman gain
    velocity_x[i] = vel_x_pred + kalman_gain_x * (vel_x_pred - vel_x_pred)
    velocity_y[i] = vel_y_pred + kalman_gain_y * (vel_y_pred - vel_y_pred)
    velocity_z[i] = vel_z_pred + kalman_gain_z * (vel_z_pred - vel_z_pred)

    # Update error covariance
    P_x = (1 - kalman_gain_x) * P_x_pred
    P_y = (1 - kalman_gain_y) * P_y_pred
    P_z = (1 - kalman_gain_z) * P_z_pred

    # Integrate velocity to get displacement
    displacement_x[i] = displacement_x[i - 1] + velocity_x[i] * dt
    displacement_y[i] = displacement_y[i - 1] + velocity_y[i] * dt
    displacement_z[i] = displacement_z[i - 1] + velocity_z[i] * dt

# Plot the displacement for X, Y, and Z axes
plt.figure(figsize=(14, 10))

# Plot displacement for X, Y, and Z axes
plt.plot(time_seconds, displacement_x, label='Displacement X (Kalman Filter)', color='b')
plt.plot(time_seconds, displacement_y, label='Displacement Y (Kalman Filter)', color='r')
plt.plot(time_seconds, displacement_z, label='Displacement Z (Kalman Filter)', color='g')

plt.xlabel('Time (s)')
plt.ylabel('Displacement (meters)')
plt.title('Displacement Obtained by Integrating Accelerometer Data with Kalman Filter for X, Y, and Z Axes')
plt.legend()
plt.grid(True)
plt.show()
