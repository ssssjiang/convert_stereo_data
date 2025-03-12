#!/usr/bin/env python3
import os
import csv
import sys
from glob import glob
import numpy as np  # Local import to reduce dependencies when not needed

# Add parent directory to path to allow relative imports to work when executed directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rock_data.convert_tof_traj import convert_vslam_to_tum, plot_tum_trajectory
from imu.analyse_imu_data import process_imu_data


def log(message, level="INFO"):
    """简单日志工具"""
    print(f"[{level}] {message}")


def create_maplab_structure(source_dir):
    """
    Create a MapLab format directory structure and parse data.
    
    Args:
        source_dir: Source directory containing camera data and log files
        
    Raises:
        FileNotFoundError: If required directories or files are missing
        Exception: If any processing step fails
    """
    log(f"Starting data parsing, directory: {source_dir}")

    # Define all the required paths
    camera_dir = os.path.join(source_dir, "camera")
    imu_file_path = os.path.join(source_dir, "imu.csv")
    imu_log_file_path = os.path.join(source_dir, "RRLDR_fprintf.log")
    wheel_log_file_path = os.path.join(source_dir, "RRLDR_fprintf.log")
    wheel_file_path = os.path.join(source_dir, "wheel_encoder.csv")
    pose_log_pattern = os.path.join(source_dir, "*SLAM_fprintf*.log")
    pose_log_files = glob(pose_log_pattern)
    tof_pose_path = os.path.join(source_dir, "tof_pose.txt")
    rtk_pose_path = os.path.join(source_dir, "rtk_pose.txt")

    # Validate source directory
    if not os.path.exists(source_dir):
        log(f"Source directory {source_dir} does not exist.", level="ERROR")
        raise FileNotFoundError(f"Source directory {source_dir} not found.")
    
    if not os.path.exists(camera_dir):
        log(f"Camera directory {camera_dir} does not exist.", level="ERROR")
        raise FileNotFoundError(f"Camera directory not found in source directory.")

    # Process camera data
    try:
        log("Generating data.csv files for camera0 and camera1")
        camera0_csv_path = os.path.join(camera_dir, "camera0", "data.csv")
        camera1_csv_path = os.path.join(camera_dir, "camera1", "data.csv")
        generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path)
        log("Camera data CSV files generated successfully")
    except Exception as e:
        log(f"Failed to generate camera data CSV files: {e}", level="ERROR")
        raise

    # Process IMU data
    try:
        log("Extracting IMU data from RRLDR_fprintf.log")
        if os.path.exists(imu_log_file_path):
            extract_imu_data(imu_log_file_path, imu_file_path)
            log("IMU data extracted successfully")
        else:
            log(f"IMU log file {imu_log_file_path} does not exist, skipping IMU data generation.", level="WARNING")
    except Exception as e:
        log(f"Failed to extract IMU data: {e}", level="ERROR")
        # Don't raise here to continue processing other data

    # Process RTK data
    try:
        log("Extracting RTK data from RRLDR_fprintf.log")
        if os.path.exists(imu_log_file_path):  # RTK data is in the same log file as IMU
            extract_rtk_data(imu_log_file_path, rtk_pose_path)
            log("RTK data extracted successfully")
            
            # Optionally plot the RTK trajectory
            try:
                rtk_plot_path = rtk_pose_path.replace(".txt", ".png")
                plot_tum_trajectory(rtk_pose_path, rtk_plot_path)
                log(f"RTK trajectory plot saved to {rtk_plot_path}")
            except Exception as e:
                log(f"Failed to generate RTK trajectory plot: {e}", level="WARNING")
        else:
            log(f"Log file {imu_log_file_path} does not exist, skipping RTK data generation.", level="WARNING")
    except Exception as e:
        log(f"Failed to extract RTK data: {e}", level="ERROR")
        # Don't raise here to continue processing other data

    # Process wheel data
    try:
        log("Extracting wheel data from RRLDR_fprintf.log")
        if os.path.exists(wheel_log_file_path):
            extract_wheel_data(wheel_log_file_path, wheel_file_path)
            log("Wheel data extracted successfully")
        else:
            log(f"Wheel log file {wheel_log_file_path} does not exist, skipping wheel data generation.", level="WARNING")
    except Exception as e:
        log(f"Failed to extract wheel data: {e}", level="ERROR")
        # Don't raise here to continue processing other data

    # Process pose data
    try:
        if pose_log_files:
            log(f"Extracting SLAM pose data using file {pose_log_files[0]}")
            convert_vslam_to_tum(pose_log_files[0], tof_pose_path)
            log("Generating TOF trajectory plot")
            
            plot_file_path = tof_pose_path.replace(".txt", ".png")
            plot_tum_trajectory(tof_pose_path, plot_file_path)
            log(f"TOF trajectory plot saved to {plot_file_path}")
        else:
            log("No *_SLAM_fprintf.log files found, skipping trajectory data generation.", level="WARNING")
    except Exception as e:
        log(f"Failed to generate TOF pose data: {e}", level="ERROR")
        # Don't raise here either to still report overall completion

    log("Data parsing complete!")
    return {
        "camera_csv_files": [camera0_csv_path, camera1_csv_path] if os.path.exists(camera_dir) else [],
        "imu_csv": imu_file_path if os.path.exists(imu_file_path) else None,
        "wheel_csv": wheel_file_path if os.path.exists(wheel_file_path) else None,
        "tof_pose": tof_pose_path if os.path.exists(tof_pose_path) else None,
        "rtk_pose": rtk_pose_path if os.path.exists(rtk_pose_path) else None
    }


def generate_data_csv(camera_dir, camera0_csv_path, camera1_csv_path):
    """
    Generate data.csv files for camera0 and camera1, keeping only overlapping timestamps.
    Ensures timestamps are strictly increasing.
    
    Args:
        camera_dir: Directory containing camera0 and camera1 subdirectories
        camera0_csv_path: Path to save camera0 data.csv
        camera1_csv_path: Path to save camera1 data.csv
        
    Raises:
        FileNotFoundError: If camera directories don't exist
        ValueError: If there are issues with file processing
    """
    camera0_path = os.path.join(camera_dir, "camera0")
    camera1_path = os.path.join(camera_dir, "camera1")

    if not os.path.isdir(camera0_path) or not os.path.isdir(camera1_path):
        raise FileNotFoundError("camera0 or camera1 directory does not exist.")

    def extract_numeric_timestamps(file_list):
        """Extract valid numeric timestamps from filenames"""
        timestamps = {}
        for file_name in file_list:
            base_name, _ = os.path.splitext(file_name)
            try:
                timestamps[int(base_name)] = file_name
            except ValueError:
                log(f"Skipping invalid filename: {file_name}", level="WARNING")
        return timestamps

    try:
        # Get filenames and timestamps for camera0 and camera1
        log(f"Reading files from {camera0_path} and {camera1_path}")
        camera0_files = [f for f in os.listdir(camera0_path) if os.path.isfile(os.path.join(camera0_path, f))]
        camera1_files = [f for f in os.listdir(camera1_path) if os.path.isfile(os.path.join(camera1_path, f))]

        log(f"Found {len(camera0_files)} files in camera0 and {len(camera1_files)} files in camera1")
        
        camera0_timestamps = extract_numeric_timestamps(camera0_files)
        camera1_timestamps = extract_numeric_timestamps(camera1_files)
        
        log(f"Extracted {len(camera0_timestamps)} valid timestamps from camera0 and {len(camera1_timestamps)} from camera1")

        # Find overlapping timestamps and ensure they're strictly increasing
        common_timestamps = set(camera0_timestamps.keys()) & set(camera1_timestamps.keys())
        log(f"Found {len(common_timestamps)} overlapping timestamps between cameras")
        
        # Create a sorted list of filtered timestamps (strictly increasing)
        filtered_timestamps = []
        last_timestamp = -1
        
        for timestamp in sorted(common_timestamps):
            if timestamp > last_timestamp:
                filtered_timestamps.append(timestamp)
                last_timestamp = timestamp
        
        log(f"After filtering, {len(filtered_timestamps)} strictly increasing timestamps remain")
        
        # Make sure target directories exist
        os.makedirs(os.path.dirname(camera0_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(camera1_csv_path), exist_ok=True)
        
        # Write CSV files
        with open(camera0_csv_path, "w", newline='') as camera0_csv, open(camera1_csv_path, "w", newline='') as camera1_csv:
            camera0_writer = csv.writer(camera0_csv)
            camera1_writer = csv.writer(camera1_csv)
            camera0_writer.writerow(["#timestamp [ns]", "filename"])
            camera1_writer.writerow(["#timestamp [ns]", "filename"])
            
            for timestamp in filtered_timestamps:
                camera0_writer.writerow([timestamp, camera0_timestamps[timestamp]])
                camera1_writer.writerow([timestamp, camera1_timestamps[timestamp]])
        
        log(f"Successfully wrote {len(filtered_timestamps)} entries to {camera0_csv_path} and {camera1_csv_path}")
        
    except Exception as e:
        log(f"Error generating camera data CSV files: {e}", level="ERROR")
        raise


def extract_imu_data(log_file_path, imu_file_path):
    """
    Extract IMU data from RRLDR_fprintf.log and generate imu.csv.
    
    Args:
        log_file_path: Path to the log file containing IMU data
        imu_file_path: Path to save the output IMU CSV file
        
    Raises:
        FileNotFoundError: If the input log file doesn't exist
        ValueError: If there are issues parsing the log file
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"{log_file_path} does not exist.")

    log(f"Extracting IMU data from {log_file_path}")
    
    # Predefine the rotation matrix for coordinate transformation
    R_O_I = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    
    # Use dictionary to track unique timestamps (more efficient than sorting later)
    unique_imu_records = {}
    
    try:
        line_count = 0
        imu_line_count = 0
        
        with open(log_file_path, "r") as logfile:
            for line_num, line in enumerate(logfile, 1):
                line_count += 1
                if "imu" in line:
                    imu_line_count += 1
                    try:
                        parts = line.split()
                        if len(parts) < 8:  # Ensure we have enough parts
                            continue
                            
                        timestamp = int(parts[0])
                        # Only process this entry if we haven't seen this timestamp before
                        if timestamp not in unique_imu_records:
                            # Convert string data to float arrays for NumPy operations
                            try:
                                gyro_data = np.array([float(x) for x in parts[5:8]])
                                accel_data = np.array([float(x) for x in parts[2:5]])
                                
                                # Apply coordinate transformation
                                gyro_transformed = R_O_I.dot(gyro_data)
                                accel_transformed = R_O_I.dot(accel_data)
                                
                                # Store the transformed data as a list for CSV writing
                                unique_imu_records[timestamp] = gyro_transformed.tolist() + accel_transformed.tolist()
                            except (ValueError, IndexError) as e:
                                log(f"Error converting data in line {line_num}: {e}", level="WARNING")
                                continue
                    except (IndexError, ValueError) as e:
                        log(f"Error parsing line {line_num}: {e}", level="WARNING")
                        continue
        
        log(f"Processed {line_count} lines, found {imu_line_count} IMU lines, with {len(unique_imu_records)} unique timestamps")
    except Exception as e:
        log(f"Error reading log file: {e}", level="ERROR")
        raise
    
    if not unique_imu_records:
        log("No valid IMU records found in the log file", level="WARNING")
        return
    
    # Prepare sorted data for writing to CSV
    sorted_imu_data = [
        [timestamp] + data 
        for timestamp, data in sorted(unique_imu_records.items())
    ]
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(imu_file_path)), exist_ok=True)
        
        with open(imu_file_path, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["# timestamp", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
            csvwriter.writerows(sorted_imu_data)
            log(f"IMU data saved to {imu_file_path} ({len(sorted_imu_data)} records)")
    except Exception as e:
        log(f"Error writing IMU CSV file: {e}", level="ERROR")
        raise

    # process_imu_data(imu_file_path, save_dir=os.path.dirname(imu_file_path))

def extract_wheel_data(log_file_path, wheel_file_path):
    """
    Extract wheel data from RRLDR_fprintf.log and generate wheel.csv.
    
    Args:
        log_file_path: Path to the log file containing wheel data
        wheel_file_path: Path to save the output wheel CSV file
    """

    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"{log_file_path} does not exist.")

    log(f"Extracting wheel data from {log_file_path}")

    # Use dictionary to track unique timestamps (more efficient than sorting later)
    unique_wheel_records = {}   

    try:
        with open(log_file_path, "r") as logfile:
            for line_num, line in enumerate(logfile, 1):
                if "rawgyroodo" in line:
                    try:
                        parts = line.split()
                        if len(parts) < 8:  # Ensure we have enough parts
                            continue

                        timestamp = int(parts[0])
                        # Only process this entry if we haven't seen this timestamp before
                        if timestamp not in unique_wheel_records:
                            wheel_data = parts[-4:-2] # 倒数第四和第三列
                            unique_wheel_records[timestamp] = wheel_data
                    except (IndexError, ValueError) as e:
                        log(f"Error parsing line {line_num}: {e}", level="WARNING")
                        continue
    except Exception as e:
        log(f"Error reading log file: {e}", level="ERROR")
        raise
    
    log(f"Found {len(unique_wheel_records)} unique wheel records")

    # Prepare sorted data for writing to CSV
    sorted_wheel_data = [
        [timestamp] + list(data) 
        for timestamp, data in sorted(unique_wheel_records.items())
    ]

    try:
        with open(wheel_file_path, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["# timestamp", "left_ticks", "right_ticks"])
            csvwriter.writerows(sorted_wheel_data)
            log(f"Wheel data saved to {wheel_file_path}")
    except Exception as e:
        log(f"Error writing wheel CSV file: {e}", level="ERROR")
        raise


def extract_rtk_data(log_file_path, rtk_file_path, apply_transform=True):
    """
    Extract RTK data from RRLDR_fprintf.log and generate a TUM format trajectory file.
    
    Args:
        log_file_path: Path to the log file containing RTK data
        rtk_file_path: Path to save the output RTK trajectory file in TUM format
        apply_transform: Whether to apply coordinate system transformation (default: True)
        
    Raises:
        FileNotFoundError: If the input log file doesn't exist
        ValueError: If there are issues parsing the log file
        
    Returns:
        Path to the created RTK file or None if no data was extracted
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"{log_file_path} does not exist.")

    log(f"Extracting RTK data from {log_file_path}")
    
    # List to store the RTK data
    rtk_data = []
    
    try:
        line_count = 0
        rtk_line_count = 0
        valid_rtk_count = 0
        
        with open(log_file_path, "r") as logfile:
            for line_num, line in enumerate(logfile, 1):
                line_count += 1
                if "rtk" in line:
                    rtk_line_count += 1
                    try:
                        values = line.strip().split()
                        
                        # Ensure we have enough fields
                        if len(values) < 19:  # We need at least the position and lat/long fields
                            continue
                        
                        # Extract the timestamp (convert to seconds)
                        timestamp = float(values[0]) / 1000.0
                        
                        # Extract number of satellites and status information
                        num_satellites = int(values[2]) if len(values) > 2 else 0
                        solution_status = int(values[3]) if len(values) > 3 else 0
                        velocity_status = int(values[4]) if len(values) > 4 else 0
                        
                        # Extract position data
                        pos_x = float(values[5]) if len(values) > 5 else 0.0
                        pos_y = float(values[6]) if len(values) > 6 else 0.0
                        pos_z = float(values[7]) if len(values) > 7 else 0.0
                        
                        # Extract position standard deviation (if available)
                        pos_std_x = float(values[8]) if len(values) > 8 else 0.0
                        pos_std_y = float(values[9]) if len(values) > 9 else 0.0
                        pos_std_z = float(values[10]) if len(values) > 10 else 0.0
                        
                        # Extract velocity data (if available)
                        velocity_x = float(values[11]) if len(values) > 11 else 0.0
                        velocity_y = float(values[12]) if len(values) > 12 else 0.0
                        velocity_z = float(values[13]) if len(values) > 13 else 0.0
                        
                        # Extract velocity standard deviation (if available)
                        vel_std_x = float(values[14]) if len(values) > 14 else 0.0
                        vel_std_y = float(values[15]) if len(values) > 15 else 0.0
                        vel_std_z = float(values[16]) if len(values) > 16 else 0.0
                        
                        # Extract geodetic coordinates for reference
                        lat = float(values[17]) if len(values) > 17 else 0.0
                        lon = float(values[18]) if len(values) > 18 else 0.0
                        alt = float(values[19]) if len(values) > 19 else 0.0
                        
                        # Extract reference position (if available)
                        ref_pos_x = float(values[20]) if len(values) > 20 else 0.0
                        ref_pos_y = float(values[21]) if len(values) > 21 else 0.0
                        ref_pos_z = float(values[22]) if len(values) > 22 else 0.0
                        
                        # For TUM format, we need orientation as quaternion
                        # If not available in RTK data, use identity quaternion (no rotation)
                        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                        
                        # Add to our data list (only if valid solution status)
                        if solution_status >= 1:  # Only use RTK data with valid solution
                            rtk_entry = [
                                timestamp,  # Timestamp in seconds
                                pos_x, pos_y, pos_z,  # Position
                                qx, qy, qz, qw,  # Orientation (quaternion)
                                num_satellites, solution_status, velocity_status,  # Status information
                                lat, lon, alt,  # Geodetic coordinates
                                pos_std_x, pos_std_y, pos_std_z,  # Position standard deviation
                                velocity_x, velocity_y, velocity_z,  # Velocity
                                vel_std_x, vel_std_y, vel_std_z,  # Velocity standard deviation
                                ref_pos_x, ref_pos_y, ref_pos_z  # Reference position
                            ]
                            rtk_data.append(rtk_entry)
                            valid_rtk_count += 1
                        
                    except (IndexError, ValueError) as e:
                        log(f"Error parsing RTK line {line_num}: {e}", level="WARNING")
                        continue
        
        log(f"Processed {line_count} lines, found {rtk_line_count} RTK lines, with {valid_rtk_count} valid positions")
    except Exception as e:
        log(f"Error reading log file: {e}", level="ERROR")
        raise
    
    if not rtk_data:
        log("No valid RTK records found in the log file", level="WARNING")
        return None
    
    # Sort by timestamp
    rtk_data.sort(key=lambda x: x[0])
    
    # Apply coordinate transform if needed (RTK to local ENU frame)
    if apply_transform and len(rtk_data) > 0:
        try:
            # Define coordinate transformation if needed
            # This is often needed to convert from global to local coordinates
            # The transformation depends on your specific RTK setup
            
            # Example transformation (identity by default)
            R_transform = np.eye(3)  # Identity rotation
            t_transform = np.zeros(3)  # No translation
            
            # Apply transformation to each position
            for i in range(len(rtk_data)):
                # Extract position
                pos = np.array([rtk_data[i][1], rtk_data[i][2], rtk_data[i][3]])
                
                # Apply transformation: R * pos + t
                transformed_pos = R_transform.dot(pos) + t_transform
                
                # Update data with transformed position
                rtk_data[i][1] = transformed_pos[0]
                rtk_data[i][2] = transformed_pos[1]
                rtk_data[i][3] = transformed_pos[2]
                
                # Transform velocity if present (rotation only, no translation)
                vel = np.array([rtk_data[i][17], rtk_data[i][18], rtk_data[i][19]])
                transformed_vel = R_transform.dot(vel)
                rtk_data[i][17] = transformed_vel[0]
                rtk_data[i][18] = transformed_vel[1]
                rtk_data[i][19] = transformed_vel[2]
                
            log("Applied coordinate transformation to RTK positions and velocities")
        except Exception as e:
            log(f"Failed to apply coordinate transformation: {e}", level="WARNING")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(rtk_file_path)), exist_ok=True)
    
    # Write data in TUM format (basic pose only)
    try:
        with open(rtk_file_path, "w") as outfile:
            # Write header
            outfile.write("# timestamp tx ty tz qx qy qz qw\n")
            
            # Write the TUM format data (timestamp, pos_x, pos_y, pos_z, qx, qy, qz, qw)
            for data in rtk_data:
                outfile.write(f"{data[0]:.6f} {data[1]:.6f} {data[2]:.6f} {data[3]:.6f} {data[4]:.6f} {data[5]:.6f} {data[6]:.6f} {data[7]:.6f}\n")
        
        log(f"Saved {len(rtk_data)} RTK records to TUM format file: {rtk_file_path}")
        
        # Save the full RTK data with all available information
        full_rtk_path = rtk_file_path.replace('.txt', '_full.txt')
        with open(full_rtk_path, "w") as outfile:
            # Write comprehensive header matching the C++ struct format
            outfile.write("# timestamp tx ty tz qx qy qz qw num_sats solution_status velocity_status lat lon alt " +
                         "pos_std_x pos_std_y pos_std_z velocity_x velocity_y velocity_z vel_std_x vel_std_y vel_std_z " +
                         "ref_pos_x ref_pos_y ref_pos_z\n")
            
            # Write all available data fields
            for data in rtk_data:
                line_parts = [f"{data[0]:.6f}"]  # timestamp
                
                # Position and orientation
                line_parts.extend([f"{data[i]:.6f}" for i in range(1, 8)])
                
                # Status information (integers)
                line_parts.extend([f"{int(data[i])}" for i in range(8, 11)])
                
                # Geodetic coordinates
                line_parts.extend([f"{data[11]:.8f}", f"{data[12]:.8f}", f"{data[13]:.3f}"])
                
                # Position standard deviation
                line_parts.extend([f"{data[i]:.6f}" for i in range(14, 17)])
                
                # Velocity
                line_parts.extend([f"{data[i]:.6f}" for i in range(17, 20)])
                
                # Velocity standard deviation
                line_parts.extend([f"{data[i]:.6f}" for i in range(20, 23)])
                
                # Reference position
                line_parts.extend([f"{data[i]:.6f}" for i in range(23, 26)])
                
                # Write the complete line
                outfile.write(" ".join(line_parts) + "\n")
        
        log(f"Saved complete RTK data with velocity and standard deviations to: {full_rtk_path}")
        
        # Create Maplab compatible format
        maplab_rtk_path = rtk_file_path.replace('.txt', '_maplab.csv')
        with open(maplab_rtk_path, "w", newline='') as outfile:
            csvwriter = csv.writer(outfile)
            # Maplab format header
            csvwriter.writerow(["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]", 
                               "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"])
            
            for data in rtk_data:
                # Convert time to nanoseconds
                time_ns = int(data[0] * 1e9)
                # Maplab uses w,x,y,z quaternion order (different from TUM)
                csvwriter.writerow([time_ns, data[1], data[2], data[3], data[7], data[4], data[5], data[6]])
                
        log(f"Saved Maplab-compatible RTK data to: {maplab_rtk_path}")
        
    except Exception as e:
        log(f"Error writing RTK trajectory file: {e}", level="ERROR")
        raise
    
    return rtk_file_path


if __name__ == "__main__":
    source_dir = input("请输入测试数据路径: ").strip()
    try:
        create_maplab_structure(source_dir)
    except Exception as e:
        log(f"程序执行失败: {e}", level="ERROR")
