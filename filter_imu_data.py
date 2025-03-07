import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from imu.parse_imu_data import read_imu_data
from analyze_complex_rotation import detect_rotation_sequences

def apply_filters(data):
    """Apply various filters to IMU data to reduce noise."""
    
    # Create a copy of the original data
    filtered_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': data['accel_x'].copy(),
        'accel_y': data['accel_y'].copy(),
        'accel_z': data['accel_z'].copy(),
        'gyro_x': data['gyro_x'].copy(),
        'gyro_y': data['gyro_y'].copy(),
        'gyro_z': data['gyro_z'].copy()
    }
    
    # 1. Moving Average Filter (Simple but effective for IMU data)
    window_size = 5  # Adjust based on your sampling rate
    
    # Apply moving average to acceleration data
    for axis in ['accel_x', 'accel_y', 'accel_z']:
        filtered_data[axis] = np.convolve(data[axis], np.ones(window_size)/window_size, mode='same')
    
    # Apply moving average to gyroscope data
    for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
        filtered_data[axis] = np.convolve(data[axis], np.ones(window_size)/window_size, mode='same')
    
    # 2. Low-Pass Butterworth Filter (Better for frequency-based filtering)
    # Define filter parameters
    order = 4  # Filter order
    fs = 100.0  # Sampling frequency (Hz) - estimate based on timestamps
    cutoff = 5.0  # Cutoff frequency (Hz)
    
    # Normalize cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Design the filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Create a second filtered dataset using Butterworth filter
    butterworth_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': signal.filtfilt(b, a, data['accel_x']),
        'accel_y': signal.filtfilt(b, a, data['accel_y']),
        'accel_z': signal.filtfilt(b, a, data['accel_z']),
        'gyro_x': signal.filtfilt(b, a, data['gyro_x']),
        'gyro_y': signal.filtfilt(b, a, data['gyro_y']),
        'gyro_z': signal.filtfilt(b, a, data['gyro_z'])
    }
    
    # 3. Median Filter (Good for removing spikes)
    median_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': signal.medfilt(data['accel_x'], kernel_size=5),
        'accel_y': signal.medfilt(data['accel_y'], kernel_size=5),
        'accel_z': signal.medfilt(data['accel_z'], kernel_size=5),
        'gyro_x': signal.medfilt(data['gyro_x'], kernel_size=5),
        'gyro_y': signal.medfilt(data['gyro_y'], kernel_size=5),
        'gyro_z': signal.medfilt(data['gyro_z'], kernel_size=5)
    }
    
    # 4. Savitzky-Golay Filter (Preserves features better than moving average)
    savgol_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': signal.savgol_filter(data['accel_x'], window_length=15, polyorder=3),
        'accel_y': signal.savgol_filter(data['accel_y'], window_length=15, polyorder=3),
        'accel_z': signal.savgol_filter(data['accel_z'], window_length=15, polyorder=3),
        'gyro_x': signal.savgol_filter(data['gyro_x'], window_length=15, polyorder=3),
        'gyro_y': signal.savgol_filter(data['gyro_y'], window_length=15, polyorder=3),
        'gyro_z': signal.savgol_filter(data['gyro_z'], window_length=15, polyorder=3)
    }
    
    # Visualize the filtering results
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0
    
    # Plot comparison of original vs filtered data
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot acceleration data
    for i, axis in enumerate(['accel_x', 'accel_y', 'accel_z']):
        axs[i, 0].plot(rel_time, data[axis], 'k-', alpha=0.5, label='Original')
        axs[i, 0].plot(rel_time, filtered_data[axis], 'r-', label='Moving Avg')
        axs[i, 0].plot(rel_time, butterworth_data[axis], 'g-', label='Butterworth')
        axs[i, 0].plot(rel_time, median_data[axis], 'b-', label='Median')
        axs[i, 0].plot(rel_time, savgol_data[axis], 'm-', label='Savitzky-Golay')
        axs[i, 0].set_title(f'{axis} Filtering Comparison')
        axs[i, 0].set_xlabel('Time (seconds)')
        axs[i, 0].set_ylabel('Acceleration (m/s²)')
        axs[i, 0].legend()
        axs[i, 0].grid(True)
    
    # Plot gyroscope data
    for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
        axs[i, 1].plot(rel_time, data[axis], 'k-', alpha=0.5, label='Original')
        axs[i, 1].plot(rel_time, filtered_data[axis], 'r-', label='Moving Avg')
        axs[i, 1].plot(rel_time, butterworth_data[axis], 'g-', label='Butterworth')
        axs[i, 1].plot(rel_time, median_data[axis], 'b-', label='Median')
        axs[i, 1].plot(rel_time, savgol_data[axis], 'm-', label='Savitzky-Golay')
        axs[i, 1].set_title(f'{axis} Filtering Comparison')
        axs[i, 1].set_xlabel('Time (seconds)')
        axs[i, 1].set_ylabel('Angular Velocity (rad/s)')
        axs[i, 1].legend()
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png')
    plt.show()
    
    # Return the Butterworth filtered data as default (usually works well for IMU)
    # You can change this to return any of the other filtered datasets
    return butterworth_data

def threshold_gyro_data(data, threshold=0.05):
    """Apply threshold to gyro data to eliminate small movements."""
    
    thresholded_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': data['accel_x'].copy(),
        'accel_y': data['accel_y'].copy(),
        'accel_z': data['accel_z'].copy(),
        'gyro_x': data['gyro_x'].copy(),
        'gyro_y': data['gyro_y'].copy(),
        'gyro_z': data['gyro_z'].copy()
    }
    
    # Apply threshold to gyro data
    for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
        mask = np.abs(data[axis]) < threshold
        thresholded_data[axis][mask] = 0.0
    
    return thresholded_data

def segment_rotations(data, min_angle=10.0):
    """
    Segment the data into distinct rotation events.
    Only consider rotations with angles greater than min_angle degrees.
    """
    # Convert timestamps to relative time (seconds)
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0
    
    # Calculate gyro magnitude
    gyro_magnitude = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
    
    # Find segments where gyro magnitude exceeds threshold
    threshold = 0.1  # rad/s
    is_rotating = gyro_magnitude > threshold
    
    # Find the start and end of each rotation segment
    rotation_segments = []
    in_segment = False
    start_idx = 0
    
    for i in range(len(is_rotating)):
        if is_rotating[i] and not in_segment:
            # Start of a new segment
            in_segment = True
            start_idx = i
        elif not is_rotating[i] and in_segment:
            # End of a segment
            in_segment = False
            end_idx = i - 1
            
            # Calculate dominant axis and direction
            segment_gyro_x = data['gyro_x'][start_idx:end_idx+1]
            segment_gyro_y = data['gyro_y'][start_idx:end_idx+1]
            segment_gyro_z = data['gyro_z'][start_idx:end_idx+1]
            
            # Determine dominant axis
            avg_abs_gyro = [
                np.mean(np.abs(segment_gyro_x)),
                np.mean(np.abs(segment_gyro_y)),
                np.mean(np.abs(segment_gyro_z))
            ]
            dominant_axis = np.argmax(avg_abs_gyro)
            
            # Determine direction
            if dominant_axis == 0:
                direction = np.mean(segment_gyro_x) < 0
                gyro_data = segment_gyro_x
            elif dominant_axis == 1:
                direction = np.mean(segment_gyro_y) < 0
                gyro_data = segment_gyro_y
            else:
                direction = np.mean(segment_gyro_z) < 0
                gyro_data = segment_gyro_z
            
            # Calculate rotation angle
            segment_time = rel_time[start_idx:end_idx+1]
            dt = np.diff(segment_time)
            dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.01)
            angle_rad = np.sum(np.abs(gyro_data) * dt)
            angle_deg = np.degrees(angle_rad)
            
            # Only include significant rotations
            if angle_deg >= min_angle:
                rotation_segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': rel_time[start_idx],
                    'end_time': rel_time[end_idx],
                    'duration': rel_time[end_idx] - rel_time[start_idx],
                    'axis': dominant_axis,
                    'direction': 'clockwise' if direction else 'counter-clockwise',
                    'angle_deg': angle_deg
                })
    
    # Handle the case where the last segment extends to the end of the data
    if in_segment:
        end_idx = len(is_rotating) - 1
        
        # Calculate dominant axis and direction
        segment_gyro_x = data['gyro_x'][start_idx:end_idx+1]
        segment_gyro_y = data['gyro_y'][start_idx:end_idx+1]
        segment_gyro_z = data['gyro_z'][start_idx:end_idx+1]
        
        # Determine dominant axis
        avg_abs_gyro = [
            np.mean(np.abs(segment_gyro_x)),
            np.mean(np.abs(segment_gyro_y)),
            np.mean(np.abs(segment_gyro_z))
        ]
        dominant_axis = np.argmax(avg_abs_gyro)
        
        # Determine direction
        if dominant_axis == 0:
            direction = np.mean(segment_gyro_x) < 0
            gyro_data = segment_gyro_x
        elif dominant_axis == 1:
            direction = np.mean(segment_gyro_y) < 0
            gyro_data = segment_gyro_y
        else:
            direction = np.mean(segment_gyro_z) < 0
            gyro_data = segment_gyro_z
        
        # Calculate rotation angle
        segment_time = rel_time[start_idx:end_idx+1]
        dt = np.diff(segment_time)
        dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.01)
        angle_rad = np.sum(np.abs(gyro_data) * dt)
        angle_deg = np.degrees(angle_rad)
        
        # Only include significant rotations
        if angle_deg >= min_angle:
            rotation_segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': rel_time[start_idx],
                'end_time': rel_time[end_idx],
                'duration': rel_time[end_idx] - rel_time[start_idx],
                'axis': dominant_axis,
                'direction': 'clockwise' if direction else 'counter-clockwise',
                'angle_deg': angle_deg
            })
    
    return rotation_segments, rel_time

def analyze_filtered_data(data):
    """Analyze the filtered IMU data to detect rotation sequences."""
    
    # Segment the rotations
    rotation_segments, rel_time = segment_rotations(data, min_angle=30.0)
    
    # Plot the segmented rotations
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot gyro data with segments highlighted
    axs[0].plot(rel_time, data['gyro_x'], 'r-', label='X-axis (roll)')
    axs[0].plot(rel_time, data['gyro_y'], 'g-', label='Y-axis (pitch)')
    axs[0].plot(rel_time, data['gyro_z'], 'b-', label='Z-axis (yaw)')
    axs[0].set_title('Filtered Angular Velocity with Rotation Segments')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Angular Velocity (rad/s)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Highlight the rotation segments
    for segment in rotation_segments:
        start_time = segment['start_time']
        end_time = segment['end_time']
        axis = segment['axis']
        color = ['r', 'g', 'b'][axis]
        axs[0].axvspan(start_time, end_time, alpha=0.2, color=color)
    
    # Plot acceleration data
    axs[1].plot(rel_time, data['accel_x'], 'r-', label='X-axis')
    axs[1].plot(rel_time, data['accel_y'], 'g-', label='Y-axis')
    axs[1].plot(rel_time, data['accel_z'], 'b-', label='Z-axis')
    axs[1].set_title('Filtered Acceleration')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot rotation angles over time
    segment_times = [(s['start_time'] + s['end_time'])/2 for s in rotation_segments]
    segment_angles = [s['angle_deg'] for s in rotation_segments]
    segment_axes = [s['axis'] for s in rotation_segments]
    
    colors = ['r', 'g', 'b']
    for axis in range(3):
        mask = [a == axis for a in segment_axes]
        if any(mask):
            times = [segment_times[i] for i in range(len(mask)) if mask[i]]
            angles = [segment_angles[i] for i in range(len(mask)) if mask[i]]
            axs[2].scatter(times, angles, color=colors[axis], 
                          label=f"{'X' if axis==0 else 'Y' if axis==1 else 'Z'}-axis rotations")
    
    axs[2].set_title('Significant Rotation Angles')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Rotation Angle (degrees)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('filtered_rotation_analysis.png')
    plt.show()
    
    # Print the rotation segments
    axis_names = ['X', 'Y', 'Z']
    print(f"Detected {len(rotation_segments)} significant rotation segments (>30°):")
    
    for i, segment in enumerate(rotation_segments):
        axis = axis_names[segment['axis']]
        direction = segment['direction']
        angle = segment['angle_deg']
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = segment['duration']
        
        print(f"Segment {i+1}: Rotation around {axis}-axis ({direction})")
        print(f"  Time: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        print(f"  Rotation angle: {angle:.1f} degrees")
        print()
    
    # Analyze gravity direction changes
    analyze_gravity_changes(data, rotation_segments)
    
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

def analyze_gravity_changes(data, rotation_segments):
    """Analyze gravity direction changes around significant rotations."""
    
    print("\nGravity Direction Analysis Around Significant Rotations:")
    
    # Convert timestamps to relative time (seconds)
    t0 = data['timestamps'][0]
    rel_time = (data['timestamps'] - t0) / 1000.0
    
    for i, segment in enumerate(rotation_segments):
        # Get data before rotation
        if segment['start_idx'] > 20:
            before_start = segment['start_idx'] - 20
            before_end = segment['start_idx'] - 1
            
            avg_accel_x_before = np.mean(data['accel_x'][before_start:before_end])
            avg_accel_y_before = np.mean(data['accel_y'][before_start:before_end])
            avg_accel_z_before = np.mean(data['accel_z'][before_start:before_end])
            
            gravity_components_before = [abs(avg_accel_x_before), abs(avg_accel_y_before), abs(avg_accel_z_before)]
            max_gravity_idx_before = np.argmax(gravity_components_before)
            gravity_direction_before = ["X", "Y", "Z"][max_gravity_idx_before]
            gravity_value_before = [avg_accel_x_before, avg_accel_y_before, avg_accel_z_before][max_gravity_idx_before]
            gravity_sign_before = "+" if gravity_value_before > 0 else "-"
        else:
            gravity_direction_before = "unknown"
            gravity_sign_before = ""
        
        # Get data after rotation
        if segment['end_idx'] < len(rel_time) - 20:
            after_start = segment['end_idx'] + 1
            after_end = segment['end_idx'] + 20
            
            avg_accel_x_after = np.mean(data['accel_x'][after_start:after_end])
            avg_accel_y_after = np.mean(data['accel_y'][after_start:after_end])
            avg_accel_z_after = np.mean(data['accel_z'][after_start:after_end])
            
            gravity_components_after = [abs(avg_accel_x_after), abs(avg_accel_y_after), abs(avg_accel_z_after)]
            max_gravity_idx_after = np.argmax(gravity_components_after)
            gravity_direction_after = ["X", "Y", "Z"][max_gravity_idx_after]
            gravity_value_after = [avg_accel_x_after, avg_accel_y_after, avg_accel_z_after][max_gravity_idx_after]
            gravity_sign_after = "+" if gravity_value_after > 0 else "-"
        else:
            gravity_direction_after = "unknown"
            gravity_sign_after = ""
        
        axis = ["X", "Y", "Z"][segment['axis']]
        direction = segment['direction']
        angle = segment['angle_deg']
        
        print(f"Segment {i+1}: {angle:.1f}° {direction} rotation around {axis}-axis")
        print(f"  Before: Gravity along {gravity_sign_before}{gravity_direction_before}")
        print(f"  After: Gravity along {gravity_sign_after}{gravity_direction_after}")
        
        if gravity_direction_before != gravity_direction_after:
            print(f"  => Gravity direction changed from {gravity_direction_before} to {gravity_direction_after}")
        print()

if __name__ == "__main__":
    # Load the IMU data
    log_file = "Sensor_fprintf.log"
    imu_data = read_imu_data(log_file)
    
    # Apply filters to reduce noise
    filtered_data = apply_filters(imu_data)
    
    # Apply threshold to eliminate small movements
    thresholded_data = threshold_gyro_data(filtered_data, threshold=0.03)
    
    # Analyze the filtered data
    analyze_filtered_data(thresholded_data) 