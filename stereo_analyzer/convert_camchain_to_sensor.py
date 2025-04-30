#!/usr/bin/env python3
import numpy as np
import yaml
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Convert calibration parameters from camchain-imucam and rbc0 to sensor.yaml format")
    parser.add_argument('--camchain', type=str, default='/home/roborock/下载/sensor_yaml/MK1-5/MK1-5_equi_vio-camchain-imucam.yaml', 
                        help="Path to the input camchain-imucam YAML file")
    parser.add_argument('--rbc0', type=str, default='/home/roborock/下载/sensor_yaml/MK1-5/MK1-5_equi_rbc0.txt', 
                        help="Path to the input rbc0 txt file")
    parser.add_argument('--output', type=str, default='sensor_output.yaml',
                        help="Path to the output sensor YAML file")
    parser.add_argument('--template', type=str, default='/home/roborock/下载/sensor_656.yaml', 
                        help="Path to the template sensor YAML file")
    parser.add_argument('--divide_intrinsics', action='store_true', 
                        help="Divide intrinsics by 2 (for half-resolution images)")
    return parser.parse_args()

def read_yaml_safely(file_path):
    """Read YAML file safely, handling syntax issues."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix problematic data arrays with leading commas
    content = re.sub(r'data: \[\s*,', 'data: [', content)
    
    # Parse the fixed content
    return yaml.safe_load(content)

def read_rbc0_txt(file_path):
    """Read the rbc0 txt file and extract the transformation matrix."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find matrix data (typically in lines 3-5)
    matrix_lines = []
    for line in lines:
        if line.strip() and all(c.isdigit() or c in '.-e' or c.isspace() for c in line.strip()):
            matrix_lines.append(line.strip())
    
    if len(matrix_lines) < 3:
        raise ValueError(f"Could not find transformation matrix in {file_path}")
    
    # Parse the 3x4 matrix
    matrix = np.zeros((4, 4))
    for i, line in enumerate(matrix_lines[:3]):
        values = [float(val) for val in line.split()]
        if len(values) >= 4:
            matrix[i, :4] = values[:4]
    
    # Add the last row [0, 0, 0, 1]
    matrix[3, :] = [0, 0, 0, 1]
    
    return matrix

def map_distortion_model(distortion_model):
    """Map distortion model names from camchain format to sensor.yaml format."""
    mapping = {
        'radtan': 'radial-tangential',
        'radtan8': 'radial-tangential8',
        'equidistant': 'kannala-brandt',
        'none': 'double-sphere',
    }
    return mapping.get(distortion_model, distortion_model)

def convert_to_sensor_yaml(camchain_path, rbc0_path, output_path, template_path, divide_intrinsics=False):
    """Convert calibration data to sensor.yaml format."""
    # Load camchain YAML file
    with open(camchain_path, 'r') as file:
        camchain_data = yaml.safe_load(file)
    
    # Check if the required data exists in the input file
    if 'cam0' not in camchain_data or 'cam1' not in camchain_data:
        raise ValueError("Input YAML file does not contain expected cam0 and cam1 entries.")
    
    # Load T_B_C0 from rbc0 txt file
    T_B_C0 = read_rbc0_txt(rbc0_path)
    print("T_B_C0 from rbc0.txt:")
    print(T_B_C0)
    
    # Extract data from camchain YAML
    cam0 = camchain_data['cam0']
    cam1 = camchain_data['cam1']
    
    # Get T_cam_imu transformations
    T_cam0_imu = np.array(cam0['T_cam_imu'])
    T_cam1_imu = np.array(cam1['T_cam_imu'])
    
    # Get relative transformation between cameras
    T_c1_c0 = np.array(cam1['T_cn_cnm1'])
    
    T_B_I = T_B_C0 @ T_cam0_imu
    T_B_C1 = T_B_C0 @ np.linalg.inv(T_c1_c0)
    
    print("Calculated T_B_I:")
    print(T_B_I)
    print("Calculated T_B_C1:")
    print(T_B_C1)
    
    # Get intrinsics
    cam0_intrinsics = np.array(cam0['intrinsics'])
    cam1_intrinsics = np.array(cam1['intrinsics'])
    
    # Process intrinsics based on divide_intrinsics flag
    if divide_intrinsics:
        cam0_intrinsics_processed = cam0_intrinsics / 2
        cam1_intrinsics_processed = cam1_intrinsics / 2
    else:
        cam0_intrinsics_processed = cam0_intrinsics.copy()
        cam1_intrinsics_processed = cam1_intrinsics.copy()
    
    # Get distortion model and coefficients
    cam0_distortion_model = cam0.get('distortion_model', 'none')
    cam1_distortion_model = cam1.get('distortion_model', 'none')
    
    # Map distortion models to sensor.yaml format
    cam0_distortion_type = map_distortion_model(cam0_distortion_model)
    cam1_distortion_type = map_distortion_model(cam1_distortion_model)
    
    # Get distortion coefficients
    cam0_distortion = np.array(cam0.get('distortion_coeffs', []))
    cam1_distortion = np.array(cam1.get('distortion_coeffs', []))
    
    # Load the template YAML file
    sensor_data = read_yaml_safely(template_path)
    
    # Update camera parameters in sensor.yaml
    if 'cameras' in sensor_data['sensor'] and len(sensor_data['sensor']['cameras']) > 1:
        # Get camera objects
        camera0 = sensor_data['sensor']['cameras'][0]
        camera1 = sensor_data['sensor']['cameras'][1]
        
        # Update camera0 parameters
        camera0['camera']['distortion']['data'] = cam0_distortion.tolist()
        camera0['camera']['distortion']['cols'] = 1
        camera0['camera']['distortion']['rows'] = len(cam0_distortion)
        camera0['camera']['distortion_type'] = cam0_distortion_type
        
        camera0['camera']['intrinsics']['data'] = [
            float(cam0_intrinsics_processed[0]),  # fx
            float(cam0_intrinsics_processed[1]),  # fy
            float(cam0_intrinsics_processed[2]),  # cx
            float(cam0_intrinsics_processed[3])   # cy
        ]
        
        # Set resolution if available
        if 'resolution' in cam0:
            camera0['camera']['image_width'] = cam0['resolution'][0]
            camera0['camera']['image_height'] = cam0['resolution'][1]
        
        # Update camera1 parameters
        camera1['camera']['distortion']['data'] = cam1_distortion.tolist()
        camera1['camera']['distortion']['cols'] = 1
        camera1['camera']['distortion']['rows'] = len(cam1_distortion)
        camera1['camera']['distortion_type'] = cam1_distortion_type
        
        camera1['camera']['intrinsics']['data'] = [
            float(cam1_intrinsics_processed[0]),  # fx
            float(cam1_intrinsics_processed[1]),  # fy
            float(cam1_intrinsics_processed[2]),  # cx
            float(cam1_intrinsics_processed[3])   # cy
        ]
        
        # Set resolution if available
        if 'resolution' in cam1:
            camera1['camera']['image_width'] = cam1['resolution'][0]
            camera1['camera']['image_height'] = cam1['resolution'][1]
        
        # Convert T_B_C0 matrix to flattened list for the YAML format
        T_B_C0_list = [
            float(T_B_C0[0, 0]), float(T_B_C0[0, 1]), float(T_B_C0[0, 2]), float(T_B_C0[0, 3]),
            float(T_B_C0[1, 0]), float(T_B_C0[1, 1]), float(T_B_C0[1, 2]), float(T_B_C0[1, 3]),
            float(T_B_C0[2, 0]), float(T_B_C0[2, 1]), float(T_B_C0[2, 2]), float(T_B_C0[2, 3]),
            float(T_B_C0[3, 0]), float(T_B_C0[3, 1]), float(T_B_C0[3, 2]), float(T_B_C0[3, 3])
        ]
        camera0['T_B_C']['data'] = T_B_C0_list
        
        # Convert T_B_C1 matrix to flattened list for the YAML format
        T_B_C1_list = [
            float(T_B_C1[0, 0]), float(T_B_C1[0, 1]), float(T_B_C1[0, 2]), float(T_B_C1[0, 3]),
            float(T_B_C1[1, 0]), float(T_B_C1[1, 1]), float(T_B_C1[1, 2]), float(T_B_C1[1, 3]),
            float(T_B_C1[2, 0]), float(T_B_C1[2, 1]), float(T_B_C1[2, 2]), float(T_B_C1[2, 3]),
            float(T_B_C1[3, 0]), float(T_B_C1[3, 1]), float(T_B_C1[3, 2]), float(T_B_C1[3, 3])
        ]
        camera1['T_B_C']['data'] = T_B_C1_list
    
    # Update IMU parameters
    if 'imu' in sensor_data['sensor']:
        T_B_I_list = [
            float(T_B_I[0, 0]), float(T_B_I[0, 1]), float(T_B_I[0, 2]), float(T_B_I[0, 3]),
            float(T_B_I[1, 0]), float(T_B_I[1, 1]), float(T_B_I[1, 2]), float(T_B_I[1, 3]),
            float(T_B_I[2, 0]), float(T_B_I[2, 1]), float(T_B_I[2, 2]), float(T_B_I[2, 3]),
            float(T_B_I[3, 0]), float(T_B_I[3, 1]), float(T_B_I[3, 2]), float(T_B_I[3, 3])
        ]
        sensor_data['sensor']['imu']['T_B_I']['data'] = T_B_I_list
    
    # Write the updated YAML to the output file
    with open(output_path, 'w') as file:
        yaml.dump(sensor_data, file, default_flow_style=False)
    
    print(f"Updated {output_path} with calibration parameters from {camchain_path} and {rbc0_path}")
    print(f"Distortion model mapping: cam0: {cam0_distortion_model} -> {cam0_distortion_type}, cam1: {cam1_distortion_model} -> {cam1_distortion_type}")
    if divide_intrinsics:
        print("Intrinsics were divided by 2 for half-resolution images")

def main():
    args = parse_args()
    convert_to_sensor_yaml(args.camchain, args.rbc0, args.output, args.template, args.divide_intrinsics)

if __name__ == "__main__":
    main() 