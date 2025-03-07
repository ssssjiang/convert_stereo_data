import numpy as np
import matplotlib.pyplot as plt
import re

# Read log file
def read_imu_data(file_path):
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Only process lines containing "imu"
            if 'imu' in line:
                parts = line.strip().split()
                if len(parts) >= 11:  # Ensure enough data
                    timestamp = int(parts[0])
                    # Acceleration data (3 values) - multiply by -1
                    ax, ay, az = -1 * float(parts[2]), -1 * float(parts[3]), -1 * float(parts[4])
                    # Angular velocity data (3 values)
                    gx, gy, gz = float(parts[5]), float(parts[6]), float(parts[7])
                    
                    timestamps.append(timestamp)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)
    
    return {
        'timestamps': np.array(timestamps),
        'accel_x': np.array(accel_x),
        'accel_y': np.array(accel_y),
        'accel_z': np.array(accel_z),
        'gyro_x': np.array(gyro_x),
        'gyro_y': np.array(gyro_y),
        'gyro_z': np.array(gyro_z)
    }

# Plot data
def plot_imu_data(data):
    # Convert timestamps to relative time (seconds)
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0  # Assuming timestamp unit is milliseconds
    
    # Create charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot acceleration
    ax1.plot(rel_time, data['accel_x'], 'r-', label='X-axis')
    ax1.plot(rel_time, data['accel_y'], 'g-', label='Y-axis')
    ax1.plot(rel_time, data['accel_z'], 'b-', label='Z-axis')
    ax1.set_title('3-Axis Acceleration (Inverted)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration')
    ax1.legend()
    ax1.grid(True)
    
    # Plot angular velocity
    ax2.plot(rel_time, data['gyro_x'], 'r-', label='X-axis')
    ax2.plot(rel_time, data['gyro_y'], 'g-', label='Y-axis')
    ax2.plot(rel_time, data['gyro_z'], 'b-', label='Z-axis')
    ax2.set_title('3-Axis Angular Velocity')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('imu_data_plot.png')
    plt.show()

if __name__ == "__main__":
    # Specify log file path
    log_file = "Sensor_fprintf.log"
    
    # Read and parse data
    imu_data = read_imu_data(log_file)
    
    # Plot data
    plot_imu_data(imu_data)
    
    print(f"Parsed {len(imu_data['timestamps'])} IMU data points") 