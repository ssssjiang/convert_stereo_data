import numpy as np
import matplotlib.pyplot as plt
from parse_imu_data import read_imu_data

def analyze_imu_motion(data):
    """Analyze IMU motion patterns based on acceleration and angular velocity data."""
    
    # Convert timestamps to relative time (seconds)
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0
    
    # Calculate magnitude of acceleration
    accel_magnitude = np.sqrt(data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2)
    
    # Calculate magnitude of angular velocity
    gyro_magnitude = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
    
    # Create a more detailed visualization
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot acceleration components
    axs[0].plot(rel_time, data['accel_x'], 'r-', label='X-axis (forward)')
    axs[0].plot(rel_time, data['accel_y'], 'g-', label='Y-axis (left)')
    axs[0].plot(rel_time, data['accel_z'], 'b-', label='Z-axis (up)')
    axs[0].plot(rel_time, accel_magnitude, 'k--', label='Magnitude')
    axs[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axs[0].set_title('Acceleration Components')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot angular velocity components
    axs[1].plot(rel_time, data['gyro_x'], 'r-', label='X-axis (roll)')
    axs[1].plot(rel_time, data['gyro_y'], 'g-', label='Y-axis (pitch)')
    axs[1].plot(rel_time, data['gyro_z'], 'b-', label='Z-axis (yaw)')
    axs[1].plot(rel_time, gyro_magnitude, 'k--', label='Magnitude')
    axs[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axs[1].set_title('Angular Velocity Components')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot combined magnitudes for overall motion intensity
    axs[2].plot(rel_time, accel_magnitude, 'b-', label='Acceleration Magnitude')
    axs[2].plot(rel_time, gyro_magnitude, 'r-', label='Angular Velocity Magnitude')
    axs[2].set_title('Motion Intensity')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Magnitude')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_analysis.png')
    plt.show()
    
    # Analyze the motion patterns
    print("Motion Analysis:")
    
    # Check for gravity direction
    avg_accel_x = np.mean(data['accel_x'])
    avg_accel_y = np.mean(data['accel_y'])
    avg_accel_z = np.mean(data['accel_z'])
    
    print(f"Average acceleration: X={avg_accel_x:.2f}, Y={avg_accel_y:.2f}, Z={avg_accel_z:.2f}")
    
    # Determine dominant gravity component
    gravity_components = [abs(avg_accel_x), abs(avg_accel_y), abs(avg_accel_z)]
    max_gravity_idx = np.argmax(gravity_components)
    gravity_direction = ["forward/backward", "left/right", "up/down"][max_gravity_idx]
    
    print(f"Dominant gravity direction: {gravity_direction}")
    
    # Check for rotational motion
    avg_gyro_x = np.mean(abs(data['gyro_x']))
    avg_gyro_y = np.mean(abs(data['gyro_y']))
    avg_gyro_z = np.mean(abs(data['gyro_z']))
    
    print(f"Average angular velocity: X={avg_gyro_x:.4f}, Y={avg_gyro_y:.4f}, Z={avg_gyro_z:.4f}")
    
    # Determine dominant rotation axis
    rotation_components = [avg_gyro_x, avg_gyro_y, avg_gyro_z]
    max_rotation_idx = np.argmax(rotation_components)
    rotation_axis = ["roll (around X)", "pitch (around Y)", "yaw (around Z)"][max_rotation_idx]
    
    print(f"Dominant rotation: {rotation_axis}")
    
    # Analyze acceleration patterns for linear motion
    accel_x_std = np.std(data['accel_x'])
    accel_y_std = np.std(data['accel_y'])
    accel_z_std = np.std(data['accel_z'])
    
    print(f"Acceleration variation: X={accel_x_std:.2f}, Y={accel_y_std:.2f}, Z={accel_z_std:.2f}")
    
    # Provide a summary of the motion
    print("\nMotion Summary:")
    
    # Check if the device is mostly stationary
    if np.max(accel_magnitude) < 10.5 and np.max(gyro_magnitude) < 0.5:
        print("- The device appears to be mostly stationary")
    else:
        print("- The device is in motion")
    
    # Check for significant linear acceleration
    if accel_x_std > 0.5:
        print("- Significant forward/backward motion detected")
    if accel_y_std > 0.5:
        print("- Significant left/right motion detected")
    if accel_z_std > 0.5:
        print("- Significant up/down motion detected")
    
    # Check for significant rotation
    if avg_gyro_x > 0.1:
        print("- Significant roll rotation detected")
    if avg_gyro_y > 0.1:
        print("- Significant pitch rotation detected")
    if avg_gyro_z > 0.1:
        print("- Significant yaw rotation detected")

if __name__ == "__main__":
    # Load the IMU data
    log_file = "Sensor_fprintf.log"
    imu_data = read_imu_data(log_file)
    
    # Analyze the motion
    analyze_imu_motion(imu_data) 