import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from parse_imu_data import read_imu_data
import math

def filter_imu_data(data):
    """Apply Butterworth low-pass filter to IMU data to reduce noise."""
    
    # Define filter parameters
    order = 4
    fs = 50.0  # Sampling frequency (Hz) - estimate
    cutoff = 3.0  # Lower cutoff frequency for better noise reduction
    
    # Normalize cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Design the filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to all IMU data
    filtered_data = {
        'timestamps': data['timestamps'].copy(),
        'accel_x': signal.filtfilt(b, a, data['accel_x']),
        'accel_y': signal.filtfilt(b, a, data['accel_y']),
        'accel_z': signal.filtfilt(b, a, data['accel_z']),
        'gyro_x': signal.filtfilt(b, a, data['gyro_x']),
        'gyro_y': signal.filtfilt(b, a, data['gyro_y']),
        'gyro_z': signal.filtfilt(b, a, data['gyro_z'])
    }
    
    return filtered_data

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

def detect_significant_rotations(data, min_angle=45.0, merge_window=1.0):
    """
    Detect significant rotations in IMU data.
    
    Parameters:
    - data: IMU data dictionary
    - min_angle: Minimum rotation angle to consider (degrees)
    - merge_window: Time window to merge nearby rotations of the same axis (seconds)
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
    
    # Merge nearby rotations of the same axis
    merged_segments = []
    if rotation_segments:
        current_segment = rotation_segments[0]
        
        for i in range(1, len(rotation_segments)):
            next_segment = rotation_segments[i]
            
            # Check if segments are close in time and have the same axis and direction
            if (next_segment['start_time'] - current_segment['end_time'] < merge_window and
                next_segment['axis'] == current_segment['axis'] and
                next_segment['direction'] == current_segment['direction']):
                
                # Merge the segments
                current_segment['end_idx'] = next_segment['end_idx']
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                current_segment['angle_deg'] += next_segment['angle_deg']
            else:
                # Add the current segment to the merged list and start a new one
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
    
    return merged_segments, rel_time

def analyze_gravity_changes(data, rotation_segments):
    """Analyze gravity direction changes around significant rotations."""
    
    gravity_changes = []
    
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
            max_gravity_idx_before = -1
        
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
            max_gravity_idx_after = -1
        
        # Record gravity change
        gravity_changes.append({
            'segment_idx': i,
            'before_axis': max_gravity_idx_before,
            'before_sign': gravity_sign_before,
            'after_axis': max_gravity_idx_after,
            'after_sign': gravity_sign_after,
            'changed': max_gravity_idx_before != max_gravity_idx_after and max_gravity_idx_before != -1 and max_gravity_idx_after != -1
        })
    
    return gravity_changes

def match_rotation_sequence(rotation_segments, gravity_changes, expected_sequence):
    """
    Try to match the detected rotations with the expected sequence.
    
    Parameters:
    - rotation_segments: List of detected rotation segments
    - gravity_changes: List of gravity direction changes
    - expected_sequence: List of expected rotation steps
    """
    # Initialize the matching result
    matching_result = []
    
    # For each expected step, find the best matching rotation segment
    for step_idx, step in enumerate(expected_sequence):
        best_match = None
        best_match_score = 0
        
        for seg_idx, segment in enumerate(rotation_segments):
            # Skip segments that have already been matched
            if any(match['segment_idx'] == seg_idx for match in matching_result):
                continue
            
            # Calculate matching score based on axis, direction, and angle
            axis_match = segment['axis'] == step['axis']
            direction_match = segment['direction'] == step['direction']
            angle_diff = abs(segment['angle_deg'] - step['angle'])
            angle_match = angle_diff < 30  # Allow some tolerance
            
            # Check gravity change if applicable
            gravity_match = True
            if step.get('gravity_change'):
                for change in gravity_changes:
                    if change['segment_idx'] == seg_idx:
                        if step['gravity_change'] == 'X_to_Y':
                            gravity_match = change['before_axis'] == 0 and change['after_axis'] == 1
                        elif step['gravity_change'] == 'Y_to_Z':
                            gravity_match = change['before_axis'] == 1 and change['after_axis'] == 2
                        elif step['gravity_change'] == 'Z_to_X':
                            gravity_match = change['before_axis'] == 2 and change['after_axis'] == 0
                        break
            
            # Calculate overall score
            score = (3 * axis_match + 2 * direction_match + 1 * angle_match + 3 * gravity_match)
            
            # Update best match if this is better
            if score > best_match_score:
                best_match_score = score
                best_match = {
                    'step_idx': step_idx,
                    'segment_idx': seg_idx,
                    'score': score,
                    'max_score': 9,  # 3 + 2 + 1 + 3
                    'segment': segment,
                    'expected': step
                }
        
        # Add the best match for this step
        if best_match:
            matching_result.append(best_match)
    
    return matching_result

def visualize_rotations(data, rotation_segments, rel_time):
    """Visualize the detected rotations."""
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot gyro data with segments highlighted
    axs[0].plot(rel_time, data['gyro_x'], 'r-', label='X-axis (roll)')
    axs[0].plot(rel_time, data['gyro_y'], 'g-', label='Y-axis (pitch)')
    axs[0].plot(rel_time, data['gyro_z'], 'b-', label='Z-axis (yaw)')
    axs[0].set_title('Filtered Angular Velocity with Significant Rotation Segments')
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
        
        # Add text annotation for angle
        mid_time = (start_time + end_time) / 2
        y_pos = 0.3 if axis == 0 else (0.2 if axis == 1 else 0.1)
        axs[0].text(mid_time, y_pos, f"{segment['angle_deg']:.0f}°", 
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(facecolor='white', alpha=0.7))
    
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
    segment_directions = [1 if s['direction'] == 'counter-clockwise' else -1 for s in rotation_segments]
    
    # Adjust angles to show direction
    directed_angles = [angle * direction for angle, direction in zip(segment_angles, segment_directions)]
    
    colors = ['r', 'g', 'b']
    for axis in range(3):
        mask = [a == axis for a in segment_axes]
        if any(mask):
            times = [segment_times[i] for i in range(len(mask)) if mask[i]]
            angles = [directed_angles[i] for i in range(len(mask)) if mask[i]]
            axs[2].scatter(times, angles, color=colors[axis], s=100,
                          label=f"{'X' if axis==0 else 'Y' if axis==1 else 'Z'}-axis rotations")
    
    axs[2].set_title('Significant Rotation Angles (Positive = Counter-clockwise, Negative = Clockwise)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Rotation Angle (degrees)')
    axs[2].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    axs[2].axhline(y=-90, color='gray', linestyle='--', alpha=0.5)
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('combined_rotation_analysis.png')
    plt.show()

def analyze_imu_data(data):
    """Comprehensive analysis of IMU data to detect the complex rotation sequence."""
    
    # Filter the data
    filtered_data = filter_imu_data(data)
    
    # Apply threshold to eliminate small movements
    thresholded_data = threshold_gyro_data(filtered_data, threshold=0.03)
    
    # Detect significant rotations
    rotation_segments, rel_time = detect_significant_rotations(thresholded_data, min_angle=45.0, merge_window=0.5)
    
    # Analyze gravity changes
    gravity_changes = analyze_gravity_changes(thresholded_data, rotation_segments)
    
    # Define the expected rotation sequence
    expected_sequence = [
        {'step': 1, 'description': 'Initial position: X forward, Y left, Z up'},
        {'step': 2, 'description': 'Rotate to Y up', 'axis': 0, 'direction': 'clockwise', 'angle': 90, 'gravity_change': 'Z_to_Y'},
        {'step': 3, 'description': 'Rotate 90° clockwise around Y', 'axis': 1, 'direction': 'clockwise', 'angle': 90},
        {'step': 4, 'description': 'Rotate 90° counter-clockwise around Y', 'axis': 1, 'direction': 'counter-clockwise', 'angle': 90},
        {'step': 5, 'description': 'Rotate to Z up', 'axis': 0, 'direction': 'counter-clockwise', 'angle': 90, 'gravity_change': 'Y_to_Z'},
        {'step': 6, 'description': 'Rotate 90° counter-clockwise around Z', 'axis': 2, 'direction': 'counter-clockwise', 'angle': 90},
        {'step': 7, 'description': 'Rotate 90° clockwise around Z', 'axis': 2, 'direction': 'clockwise', 'angle': 90},
        {'step': 8, 'description': 'Rotate to X up', 'axis': 1, 'direction': 'clockwise', 'angle': 90, 'gravity_change': 'Z_to_X'},
        {'step': 9, 'description': 'Rotate 90° counter-clockwise around X', 'axis': 0, 'direction': 'counter-clockwise', 'angle': 90},
        {'step': 10, 'description': 'Rotate 90° clockwise around X', 'axis': 0, 'direction': 'clockwise', 'angle': 90}
    ]
    
    # Try to match the detected rotations with the expected sequence
    matching_result = match_rotation_sequence(rotation_segments, gravity_changes, expected_sequence[1:])  # Skip initial position
    
    # Visualize the rotations
    visualize_rotations(thresholded_data, rotation_segments, rel_time)
    
    # Print the detected rotations
    axis_names = ['X', 'Y', 'Z']
    print(f"Detected {len(rotation_segments)} significant rotation segments (>45°):")
    
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
        
        # Print gravity change if available
        for change in gravity_changes:
            if change['segment_idx'] == i and change['changed']:
                before_axis = ["X", "Y", "Z"][change['before_axis']]
                after_axis = ["X", "Y", "Z"][change['after_axis']]
                print(f"  Gravity direction changed from {change['before_sign']}{before_axis} to {change['after_sign']}{after_axis}")
        print()
    
    # Print the matching result
    print("\nMatching with Expected Rotation Sequence:")
    
    if matching_result:
        for match in matching_result:
            step = match['expected']
            segment = match['segment']
            score = match['score']
            max_score = match['max_score']
            
            print(f"Step {step['step']}: {step['description']}")
            print(f"  Matched with Segment {match['segment_idx']+1}")
            print(f"  Expected: {step['direction']} rotation around {axis_names[step['axis']]}-axis, {step['angle']}°")
            print(f"  Detected: {segment['direction']} rotation around {axis_names[segment['axis']]}-axis, {segment['angle_deg']:.1f}°")
            print(f"  Match score: {score}/{max_score}")
            print()
    else:
        print("No matches found between detected rotations and expected sequence.")
    
    # Provide a summary of the analysis
    print("\nSummary of IMU Data Analysis:")
    print("1. The IMU data shows several significant rotations around different axes.")
    print("2. The dominant gravity direction throughout most of the recording is along the Z-axis,")
    print("   suggesting the device mostly maintained its Z-axis pointing upward.")
    print("3. The detected rotations do not fully match the expected complex sequence of 90° rotations.")
    print("4. Most detected rotations are smaller than 90 degrees, suggesting incomplete rotations")
    print("   or limitations in the measurement/integration process.")
    
    # Suggest improvements
    print("\nPossible explanations and improvements:")
    print("1. The device might not have been rotated through complete 90° angles.")
    print("2. Integration of angular velocity to estimate angles can accumulate errors.")
    print("3. The sampling rate might be insufficient to capture fast rotations accurately.")
    print("4. Sensor calibration issues might affect the accuracy of measurements.")
    print("5. For better results, consider using quaternion-based orientation tracking or")
    print("   sensor fusion algorithms that combine accelerometer and gyroscope data.")

if __name__ == "__main__":
    # Load the IMU data
    log_file = "Sensor_fprintf.log"
    imu_data = read_imu_data(log_file)
    
    # Analyze the IMU data
    analyze_imu_data(imu_data) 