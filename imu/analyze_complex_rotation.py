import numpy as np
import matplotlib.pyplot as plt
from imu.parse_imu_data import read_imu_data
from scipy.signal import find_peaks
import math

def detect_rotation_sequences(data):
    """
    Detect complex rotation sequences in IMU data.
    Looking for patterns of 90-degree rotations around different axes.
    """
    # Convert timestamps to relative time (seconds)
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0
    
    # Calculate magnitudes
    accel_magnitude = np.sqrt(data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2)
    gyro_magnitude = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
    
    # Create a figure with subplots for detailed analysis
    fig, axs = plt.subplots(4, 1, figsize=(14, 16))
    
    # Plot acceleration components with annotations for gravity direction changes
    axs[0].plot(rel_time, data['accel_x'], 'r-', label='X-axis (forward)')
    axs[0].plot(rel_time, data['accel_y'], 'g-', label='Y-axis (left)')
    axs[0].plot(rel_time, data['accel_z'], 'b-', label='Z-axis (up)')
    axs[0].set_title('Acceleration Components - Gravity Direction Changes')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot angular velocity for rotation detection
    axs[1].plot(rel_time, data['gyro_x'], 'r-', label='X-axis (roll)')
    axs[1].plot(rel_time, data['gyro_y'], 'g-', label='Y-axis (pitch)')
    axs[1].plot(rel_time, data['gyro_z'], 'b-', label='Z-axis (yaw)')
    axs[1].set_title('Angular Velocity - Rotation Detection')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot gyro magnitude to identify significant rotations
    axs[2].plot(rel_time, gyro_magnitude, 'k-', label='Angular Velocity Magnitude')
    axs[2].set_title('Angular Velocity Magnitude - Rotation Intensity')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Magnitude (rad/s)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Detect peaks in gyro magnitude to find significant rotations
    # Adjust min_height based on the data
    min_height = 0.1  # Minimum peak height to consider
    peaks, _ = find_peaks(gyro_magnitude, height=min_height, distance=10)
    
    # Mark the peaks on the plot
    axs[2].plot(rel_time[peaks], gyro_magnitude[peaks], 'ro', label='Detected Rotations')
    axs[2].legend()
    
    # Plot dominant axis for each rotation
    colors = ['r', 'g', 'b']
    labels = ['X-dominant', 'Y-dominant', 'Z-dominant']
    dominant_axis = np.zeros_like(rel_time, dtype=int)
    
    for i in range(len(rel_time)):
        gyro_abs = [abs(data['gyro_x'][i]), abs(data['gyro_y'][i]), abs(data['gyro_z'][i])]
        dominant_axis[i] = np.argmax(gyro_abs) if gyro_magnitude[i] > min_height else -1
    
    for axis in range(3):
        mask = dominant_axis == axis
        if np.any(mask):
            axs[3].scatter(rel_time[mask], np.ones_like(rel_time[mask])*axis, 
                          color=colors[axis], label=labels[axis], s=10)
    
    axs[3].set_title('Dominant Rotation Axis Over Time')
    axs[3].set_xlabel('Time (seconds)')
    axs[3].set_yticks([0, 1, 2])
    axs[3].set_yticklabels(['X-axis', 'Y-axis', 'Z-axis'])
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('complex_rotation_analysis.png')
    plt.show()
    
    # Analyze the sequence of rotations
    print("Complex Rotation Sequence Analysis:")
    
    # Identify segments where different axes dominate
    segments = []
    current_axis = -1
    segment_start = 0
    
    for i in range(len(rel_time)):
        if dominant_axis[i] != current_axis and dominant_axis[i] != -1:
            if current_axis != -1:
                segments.append({
                    'axis': current_axis,
                    'start_idx': segment_start,
                    'end_idx': i-1,
                    'start_time': rel_time[segment_start],
                    'end_time': rel_time[i-1],
                    'duration': rel_time[i-1] - rel_time[segment_start]
                })
            current_axis = dominant_axis[i]
            segment_start = i
    
    # Add the last segment
    if current_axis != -1:
        segments.append({
            'axis': current_axis,
            'start_idx': segment_start,
            'end_idx': len(rel_time)-1,
            'start_time': rel_time[segment_start],
            'end_time': rel_time[-1],
            'duration': rel_time[-1] - rel_time[segment_start]
        })
    
    # Filter out very short segments (noise)
    min_duration = 0.05  # seconds
    segments = [s for s in segments if s['duration'] > min_duration]
    
    # Print the rotation sequence
    axis_names = ['X', 'Y', 'Z']
    print(f"Detected {len(segments)} significant rotation segments:")
    
    for i, segment in enumerate(segments):
        axis = axis_names[segment['axis']]
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = segment['duration']
        
        # Determine rotation direction
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        mid_idx = (start_idx + end_idx) // 2
        
        gyro_value = 0
        if segment['axis'] == 0:
            gyro_value = data['gyro_x'][mid_idx]
        elif segment['axis'] == 1:
            gyro_value = data['gyro_y'][mid_idx]
        else:
            gyro_value = data['gyro_z'][mid_idx]
        
        direction = "clockwise" if gyro_value < 0 else "counter-clockwise"
        
        print(f"Segment {i+1}: Rotation around {axis}-axis ({direction})")
        print(f"  Time: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        
        # Estimate rotation angle (very approximate)
        # Integrate angular velocity over time
        if segment['axis'] == 0:
            gyro_data = data['gyro_x'][start_idx:end_idx+1]
        elif segment['axis'] == 1:
            gyro_data = data['gyro_y'][start_idx:end_idx+1]
        else:
            gyro_data = data['gyro_z'][start_idx:end_idx+1]
        
        time_data = rel_time[start_idx:end_idx+1]
        dt = np.diff(time_data)
        dt = np.append(dt, dt[-1])  # Repeat last dt for the last point
        
        angle_rad = np.sum(gyro_data * dt)
        angle_deg = math.degrees(abs(angle_rad))
        
        print(f"  Estimated rotation angle: {angle_deg:.1f} degrees")
        print()
    
    # Check for gravity direction changes
    print("\nGravity Direction Analysis:")
    
    # Split the data into segments and analyze gravity direction in each
    num_segments = 10  # Divide the data into segments
    segment_length = len(rel_time) // num_segments
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(rel_time)
        
        avg_accel_x = np.mean(data['accel_x'][start_idx:end_idx])
        avg_accel_y = np.mean(data['accel_y'][start_idx:end_idx])
        avg_accel_z = np.mean(data['accel_z'][start_idx:end_idx])
        
        # Determine dominant gravity component
        gravity_components = [abs(avg_accel_x), abs(avg_accel_y), abs(avg_accel_z)]
        max_gravity_idx = np.argmax(gravity_components)
        gravity_direction = ["X (forward/backward)", "Y (left/right)", "Z (up/down)"][max_gravity_idx]
        gravity_value = [avg_accel_x, avg_accel_y, avg_accel_z][max_gravity_idx]
        gravity_sign = "positive" if gravity_value > 0 else "negative"
        
        segment_start_time = rel_time[start_idx]
        segment_end_time = rel_time[end_idx-1]
        
        print(f"Time {segment_start_time:.2f}s - {segment_end_time:.2f}s:")
        print(f"  Dominant gravity: {gravity_direction} axis ({gravity_sign})")
        print(f"  Acceleration: X={avg_accel_x:.2f}, Y={avg_accel_y:.2f}, Z={avg_accel_z:.2f}")
        print()
    
    # Compare with expected rotation sequence
    print("\nComparison with Expected Rotation Sequence:")
    print("Expected sequence:")
    print("1. Initial position: X forward, Y left, Z up")
    print("2. Rotate to Y up (90° around X)")
    print("3. Rotate 90° clockwise around Y axis")
    print("4. Rotate 90° counter-clockwise around Y axis")
    print("5. Rotate to Z up (90° around X)")
    print("6. Rotate 90° counter-clockwise around Z axis")
    print("7. Rotate 90° clockwise around Z axis")
    print("8. Rotate to X up (90° around Y)")
    print("9. Rotate 90° counter-clockwise around X axis")
    print("10. Rotate 90° clockwise around X axis")
    
    # Final summary
    print("\nSummary of Detected Motion:")
    print("The IMU data shows a complex sequence of rotations around different axes.")
    print("The gravity vector shifts between different axes, indicating the device")
    print("was rotated to different orientations during the recording period.")

if __name__ == "__main__":
    # Load the IMU data
    log_file = "Sensor_fprintf.log"
    imu_data = read_imu_data(log_file)
    
    # Analyze the complex rotation sequence
    detect_rotation_sequences(imu_data) 