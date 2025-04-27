#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import re

def load_opencv_calibration(yaml_path):
    """Load calibration parameters from a YAML file using OpenCV."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"YAML file at path '{yaml_path}' could not be loaded.")

    # Read camera matrices and distortion coefficients
    M1 = fs.getNode("M1").mat()
    D1 = fs.getNode("D1").mat().flatten()
    M2 = fs.getNode("M2").mat()
    D2 = fs.getNode("D2").mat().flatten()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat().flatten()

    fs.release()
    
    return M1, D1, M2, D2, R, T

def read_yaml_safely(file_path):
    """Read YAML file safely, handling syntax issues."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix problematic data arrays with leading commas
    content = re.sub(r'data: \[\s*,', 'data: [', content)
    
    # Parse the fixed content
    return yaml.safe_load(content)

def load_sensor_yaml(yaml_path):
    """Load camera parameters from sensor_sy YAML file."""
    sensor_data = read_yaml_safely(yaml_path)
    
    camera0 = sensor_data['sensor']['cameras'][0]
    camera1 = sensor_data['sensor']['cameras'][1]
    
    # Extract intrinsics
    intrinsics0 = camera0['camera']['intrinsics']['data']
    intrinsics1 = camera1['camera']['intrinsics']['data']
    
    # Extract distortion coefficients
    distortion0 = camera0['camera']['distortion']['data']
    distortion1 = camera1['camera']['distortion']['data']
    
    # Extract transformation matrices
    T_B_C0 = np.array(camera0['T_B_C']['data']).reshape(4, 4)
    T_B_C1 = np.array(camera1['T_B_C']['data']).reshape(4, 4)
    
    return intrinsics0, distortion0, intrinsics1, distortion1, T_B_C0, T_B_C1

def create_T_B_C(R, T):
    """Create a 4x4 transformation matrix from R and T."""
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = R
    T_B_C[:3, 3] = T.reshape(3)
    return T_B_C

def main():
    input_path = '/home/roborock/下载/37_stereo.yml'
    output_path = '/home/roborock/下载/sensor_sy_19.yaml'
    
    # Make a backup of the original sensor YAML
    backup_path = output_path + '.bak'
    if not os.path.exists(backup_path):
        with open(output_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Created backup of {output_path} to {backup_path}")
    
    # Load original stereo calibration data
    print(f"Loading original calibration from {input_path}")
    M1, D1, M2, D2, R, T = load_opencv_calibration(input_path)
    
    # Print original values
    print("\nOriginal Camera Parameters:")
    print("M1 (Camera Matrix 1):")
    print(M1)
    print("\nD1 (Distortion Coefficients 1):")
    print(D1[:8])  # Show only first 8 coefficients
    
    print("\nM2 (Camera Matrix 2):")
    print(M2)
    print("\nD2 (Distortion Coefficients 2):")
    print(D2[:8])  # Show only first 8 coefficients
    
    print("\nR (Rotation Matrix):")
    print(R)
    print("\nT (Translation Vector):")
    print(T)
    
    # Create transformation matrix for camera0
    T_B_C_cam0 = create_T_B_C(R, T)
    print("\nT_B_C for camera0 (from R and T):")
    print(T_B_C_cam0)
    
    # Run the conversion
    import subprocess
    print("\nRunning conversion script...")
    result = subprocess.run(
        ['/home/roborock/tools/convert_stereo_data/convert_stereo_yaml.py', 
         '--input', input_path, 
         '--output', output_path],
        capture_output=True, 
        text=True
    )
    print(result.stdout)
    
    # Load the updated sensor YAML
    print(f"\nLoading updated parameters from {output_path}")
    intrinsics0, distortion0, intrinsics1, distortion1, T_B_C0, T_B_C1 = load_sensor_yaml(output_path)
    
    # Print updated values
    print("\nUpdated Camera Parameters:")
    print("Intrinsics 0 (from M1 divided by 2):")
    print(f"fx: {intrinsics0[0]}")
    print(f"fy: {intrinsics0[1]}")
    print(f"cx: {intrinsics0[2]}")
    print(f"cy: {intrinsics0[3]}")
    
    print("\nDistortion 0 (from D1):")
    print(distortion0)
    
    print("\nIntrinsics 1 (from M2 divided by 2):")
    print(f"fx: {intrinsics1[0]}")
    print(f"fy: {intrinsics1[1]}")
    print(f"cx: {intrinsics1[2]}")
    print(f"cy: {intrinsics1[3]}")
    
    print("\nDistortion 1 (from D2):")
    print(distortion1)
    
    print("\nT_B_C for camera0 (from R and T):")
    print(T_B_C0)
    
    print("\nT_B_C for camera1 (identity matrix):")
    print(T_B_C1)
    
    # Verify that the expected changes were made
    expected_fx0 = M1[0, 0] / 2
    actual_fx0 = intrinsics0[0]
    print(f"\nVerification - Camera 0 fx: Expected {expected_fx0}, Got {actual_fx0}")
    
    expected_fx1 = M2[0, 0] / 2
    actual_fx1 = intrinsics1[0]
    print(f"Verification - Camera 1 fx: Expected {expected_fx1}, Got {actual_fx1}")
    
    # Verify that T_B_C0 is derived from R and T
    print(f"\nVerification - T_B_C0 matches T_B_C derived from R and T: {np.allclose(T_B_C0, T_B_C_cam0)}")
    
    # Verify that T_B_C1 is identity
    print(f"Verification - T_B_C1 is identity: {np.allclose(T_B_C1, np.eye(4))}")
    
    print("\nConversion verification complete.")

if __name__ == "__main__":
    main() 