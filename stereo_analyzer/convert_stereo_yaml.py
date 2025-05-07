#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Convert stereo calibration parameters from OpenCV YAML format to sensor.yaml format")
    parser.add_argument('--input', type=str, default='/home/roborock/下载/37_stereo.yml', help="Path to the input stereo calibration YAML file")
    parser.add_argument('--output', type=str, help="Path to the output sensor YAML file")
    parser.add_argument('--template', type=str, default='/home/roborock/下载/sy_sensor_yaml/sensor_mk1_4_657.yaml', help="Path to the template sensor YAML file. If not provided, output file will be used as template.")
    parser.add_argument('--camchain', action='store_true', help="Use camchain-imucam.yaml format instead of stereo YAML format")
    parser.add_argument('--swap', action='store_true', help="Swap left and right cameras (camera0 and camera1)")
    parser.add_argument('--no_divide_intrinsics', action='store_true', help="Do not divide intrinsics by 2")
    return parser.parse_args()

def load_calibration(yaml_path):
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

    if M1 is None or D1 is None or M2 is None or D2 is None or R is None or T is None:
        raise ValueError("Failed to load calibration parameters from the YAML file.")

    return M1, D1, M2, D2, R, T

def create_T_B_C(R, T):
    """Create a 4x4 transformation matrix from R and T."""
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = R
    T_B_C[:3, 3] = T.reshape(3)
    return T_B_C

def read_yaml_safely(file_path):
    """Read YAML file safely, handling syntax issues."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix problematic data arrays with leading commas
    content = re.sub(r'data: \[\s*,', 'data: [', content)
    
    # Parse the fixed content
    return yaml.safe_load(content)

def map_distortion_model(distortion_model):
    """Map distortion model names from camchain format to sensor.yaml format."""
    mapping = {
        'radtan8': 'radial-tangential8',
        'equidistant': 'kannala-brandt',
        'none': 'double-sphere',
    }
    return mapping.get(distortion_model, distortion_model)

def limit_distortion_params(distortion_params, max_dim=8):
    """Limit distortion parameters to a maximum dimension.
    
    Args:
        distortion_params: List or numpy array of distortion parameters
        max_dim: Maximum number of dimensions allowed (default: 8)
        
    Returns:
        List of distortion parameters limited to max_dim
    """
    # Convert to list if it's not already
    if isinstance(distortion_params, np.ndarray):
        distortion_list = distortion_params.flatten().tolist()
    else:
        distortion_list = list(distortion_params)
    
    # Limit to max_dim parameters
    if len(distortion_list) > max_dim:
        limited_params = distortion_list[:max_dim]
        print(f"Warning: Truncated distortion parameters from {len(distortion_list)} to {max_dim} dimensions")
        return limited_params
    
    return distortion_list

def update_sensor_yaml(input_path, output_path, swap_cameras=False, template_path=None, divide_intrinsics=True):
    """Update sensor.yaml with calibration parameters from OpenCV YAML format."""
    # Load calibration parameters from stereo YAML
    M1, D1, M2, D2, R, T = load_calibration(input_path)
    
    # Process camera matrix parameters based on divide_intrinsics flag
    M1_processed = M1.copy()
    M2_processed = M2.copy()
    
    if divide_intrinsics:
        # Divide camera matrix parameters by 2 if requested
        M1_processed[0, 0] /= 2  # fx
        M1_processed[1, 1] /= 2  # fy
        M1_processed[0, 2] /= 2  # cx
        M1_processed[1, 2] /= 2  # cy
        
        M2_processed[0, 0] /= 2  # fx
        M2_processed[1, 1] /= 2  # fy
        M2_processed[0, 2] /= 2  # cx
        M2_processed[1, 2] /= 2  # cy
    
    # Create transformation matrices
    T_B_C_from_stereo = create_T_B_C(R, T)
    identity_matrix = np.eye(4)
    
    # Extract distortion coefficients and limit to 8 dimensions
    D1_list = limit_distortion_params(D1.flatten().tolist())
    D2_list = limit_distortion_params(D2.flatten().tolist())
    
    # Load the template YAML file safely
    template_file = template_path if template_path else output_path
    sensor_data = read_yaml_safely(template_file)
    
    # Update camera parameters in sensor.yaml
    if 'cameras' in sensor_data['sensor'] and len(sensor_data['sensor']['cameras']) > 0:
        # Get camera objects
        camera0 = sensor_data['sensor']['cameras'][0]
        camera1 = sensor_data['sensor']['cameras'][1] if len(sensor_data['sensor']['cameras']) > 1 else None
        
        if swap_cameras and camera1 is not None:
            # When swapping, camera0 gets cam2 data, camera1 gets cam1 data
            camera0['camera']['distortion']['data'] = D2_list
            camera0['camera']['distortion']['cols'] = 1
            camera0['camera']['distortion']['rows'] = len(D2_list)
            
            camera0['camera']['intrinsics']['data'] = [
                float(M2_processed[0, 0]),  # fx
                float(M2_processed[1, 1]),  # fy
                float(M2_processed[0, 2]),  # cx
                float(M2_processed[1, 2])   # cy
            ]
            
            # Create identity matrix for camera0 when swapping
            T_B_C_cam0 = identity_matrix
            
            # Use T_B_C from stereo for camera1 when swapping
            T_B_C_cam1 = T_B_C_from_stereo
            
            # Update camera1 with cam1 data
            camera1['camera']['distortion']['data'] = D1_list
            camera1['camera']['distortion']['cols'] = 1
            camera1['camera']['distortion']['rows'] = len(D1_list)
            
            camera1['camera']['intrinsics']['data'] = [
                float(M1_processed[0, 0]),  # fx
                float(M1_processed[1, 1]),  # fy
                float(M1_processed[0, 2]),  # cx
                float(M1_processed[1, 2])   # cy
            ]
        else:
            # Default/no-swap case: camera0 gets cam1 data
            camera0['camera']['distortion']['data'] = D1_list
            camera0['camera']['distortion']['cols'] = 1
            camera0['camera']['distortion']['rows'] = len(D1_list)
            
            camera0['camera']['intrinsics']['data'] = [
                float(M1_processed[0, 0]),  # fx
                float(M1_processed[1, 1]),  # fy
                float(M1_processed[0, 2]),  # cx
                float(M1_processed[1, 2])   # cy
            ]
            
            # Use T_B_C for camera0 in non-swap mode
            T_B_C_cam0 = T_B_C_from_stereo
            
            # Update camera1 if it exists (no swap case)
            if camera1 is not None:
                camera1['camera']['distortion']['data'] = D2_list
                camera1['camera']['distortion']['cols'] = 1
                camera1['camera']['distortion']['rows'] = len(D2_list)
                
                camera1['camera']['intrinsics']['data'] = [
                    float(M2_processed[0, 0]),  # fx
                    float(M2_processed[1, 1]),  # fy
                    float(M2_processed[0, 2]),  # cx
                    float(M2_processed[1, 2])   # cy
                ]
                
                # Use identity matrix for camera1 in non-swap mode
                T_B_C_cam1 = identity_matrix
        
        # Convert T_B_C matrices to flattened lists for the YAML format
        T_B_C_cam0_list = [
            float(T_B_C_cam0[0, 0]), float(T_B_C_cam0[0, 1]), float(T_B_C_cam0[0, 2]), float(T_B_C_cam0[0, 3]),
            float(T_B_C_cam0[1, 0]), float(T_B_C_cam0[1, 1]), float(T_B_C_cam0[1, 2]), float(T_B_C_cam0[1, 3]),
            float(T_B_C_cam0[2, 0]), float(T_B_C_cam0[2, 1]), float(T_B_C_cam0[2, 2]), float(T_B_C_cam0[2, 3]),
            float(T_B_C_cam0[3, 0]), float(T_B_C_cam0[3, 1]), float(T_B_C_cam0[3, 2]), float(T_B_C_cam0[3, 3])
        ]
        camera0['T_B_C']['data'] = T_B_C_cam0_list
        
        # Update camera1 T_B_C if it exists
        if camera1 is not None:
            T_B_C_cam1_list = [
                float(T_B_C_cam1[0, 0]), float(T_B_C_cam1[0, 1]), float(T_B_C_cam1[0, 2]), float(T_B_C_cam1[0, 3]),
                float(T_B_C_cam1[1, 0]), float(T_B_C_cam1[1, 1]), float(T_B_C_cam1[1, 2]), float(T_B_C_cam1[1, 3]),
                float(T_B_C_cam1[2, 0]), float(T_B_C_cam1[2, 1]), float(T_B_C_cam1[2, 2]), float(T_B_C_cam1[2, 3]),
                float(T_B_C_cam1[3, 0]), float(T_B_C_cam1[3, 1]), float(T_B_C_cam1[3, 2]), float(T_B_C_cam1[3, 3])
            ]
            camera1['T_B_C']['data'] = T_B_C_cam1_list
    
    # Write the updated YAML to the output file
    with open(output_path, 'w') as file:
        yaml.dump(sensor_data, file, default_flow_style=False)
    
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    template_msg = f"using template from {template_file}" if template_path else "using output as template"
    print(f"Updated {output_path} with calibration parameters from {input_path} ({swap_msg}, {divide_msg}, {template_msg})")
    
    # Print T_B_C assignments for verification
    if swap_cameras:
        print("Camera0 assigned identity matrix for T_B_C")
        print("Camera1 assigned transformation matrix from stereo calibration for T_B_C")
    else:
        print("Camera0 assigned transformation matrix from stereo calibration for T_B_C")
        print("Camera1 assigned identity matrix for T_B_C")

def update_sensor_yaml_from_camchain(input_path, output_path, swap_cameras=False, template_path=None, divide_intrinsics=True):
    """Update sensor.yaml with calibration parameters from camchain-imucam.yaml format."""
    # Load camchain YAML file
    with open(input_path, 'r') as file:
        camchain_data = yaml.safe_load(file)
    
    # Check if the required data exists in the input file
    if 'cam0' not in camchain_data or 'cam1' not in camchain_data:
        raise ValueError("Input YAML file does not contain expected cam0 and cam1 entries.")
    
    # Extract data from camchain YAML
    cam0 = camchain_data['cam0']
    cam1 = camchain_data['cam1']
    
    # Get intrinsics
    cam0_intrinsics = np.array(cam0['intrinsics'])
    cam1_intrinsics = np.array(cam1['intrinsics'])
    
    # Process intrinsics based on divide_intrinsics flag
    if divide_intrinsics:
        # Divide intrinsics by 2 if requested
        cam0_intrinsics_processed = cam0_intrinsics / 2
        cam1_intrinsics_processed = cam1_intrinsics / 2
    else:
        # Use original intrinsics
        cam0_intrinsics_processed = cam0_intrinsics.copy()
        cam1_intrinsics_processed = cam1_intrinsics.copy()
    
    # Get distortion model and coefficients
    cam0_distortion_model = cam0.get('distortion_model', 'none')
    cam1_distortion_model = cam1.get('distortion_model', 'none')
    
    # Map distortion models to sensor.yaml format
    cam0_distortion_type = map_distortion_model(cam0_distortion_model)
    cam1_distortion_type = map_distortion_model(cam1_distortion_model)
    
    # Get distortion coefficients and limit to 8 dimensions
    cam0_distortion = limit_distortion_params(np.array(cam0.get('distortion_coeffs', [])))
    cam1_distortion = limit_distortion_params(np.array(cam1.get('distortion_coeffs', [])))
    
    # Get T_cn_cnm1 transformation matrix
    if 'T_cn_cnm1' not in cam1:
        raise ValueError("T_cn_cnm1 not found in cam1 data")
    
    T_cn_cnm1 = np.array(cam1['T_cn_cnm1'])
    identity_matrix = np.eye(4)
    
    # Load the template YAML file safely
    template_file = template_path if template_path else output_path
    sensor_data = read_yaml_safely(template_file)
    
    # Update camera parameters in sensor.yaml based on swap setting
    if 'cameras' in sensor_data['sensor'] and len(sensor_data['sensor']['cameras']) > 1:
        # Get camera objects
        camera0 = sensor_data['sensor']['cameras'][0]
        camera1 = sensor_data['sensor']['cameras'][1]
        
        if swap_cameras:
            # When swapping, camera0 gets cam1 data, camera1 gets cam0 data
            camera0['camera']['distortion']['data'] = cam1_distortion
            camera0['camera']['distortion']['cols'] = 1
            camera0['camera']['distortion']['rows'] = len(cam1_distortion)
            camera0['camera']['distortion_type'] = cam1_distortion_type
            
            camera0['camera']['intrinsics']['data'] = [
                float(cam1_intrinsics_processed[0]),  # fx
                float(cam1_intrinsics_processed[1]),  # fy
                float(cam1_intrinsics_processed[2]),  # cx
                float(cam1_intrinsics_processed[3])   # cy
            ]
            
            camera1['camera']['distortion']['data'] = cam0_distortion
            camera1['camera']['distortion']['cols'] = 1
            camera1['camera']['distortion']['rows'] = len(cam0_distortion)
            camera1['camera']['distortion_type'] = cam0_distortion_type
            
            camera1['camera']['intrinsics']['data'] = [
                float(cam0_intrinsics_processed[0]),  # fx
                float(cam0_intrinsics_processed[1]),  # fy
                float(cam0_intrinsics_processed[2]),  # cx
                float(cam0_intrinsics_processed[3])   # cy
            ]
            
            # When swapping, according to requirement:
            # camera0 gets identity matrix
            # camera1 gets T_cn_cnm1
            T_B_C_cam0 = identity_matrix
            T_B_C_cam1 = T_cn_cnm1
        else:
            # When not swapping, camera0 gets cam0 data, camera1 gets cam1 data
            camera0['camera']['distortion']['data'] = cam0_distortion
            camera0['camera']['distortion']['cols'] = 1
            camera0['camera']['distortion']['rows'] = len(cam0_distortion)
            camera0['camera']['distortion_type'] = cam0_distortion_type
            
            camera0['camera']['intrinsics']['data'] = [
                float(cam0_intrinsics_processed[0]),  # fx
                float(cam0_intrinsics_processed[1]),  # fy
                float(cam0_intrinsics_processed[2]),  # cx
                float(cam0_intrinsics_processed[3])   # cy
            ]
            
            camera1['camera']['distortion']['data'] = cam1_distortion
            camera1['camera']['distortion']['cols'] = 1
            camera1['camera']['distortion']['rows'] = len(cam1_distortion)
            camera1['camera']['distortion_type'] = cam1_distortion_type
            
            camera1['camera']['intrinsics']['data'] = [
                float(cam1_intrinsics_processed[0]),  # fx
                float(cam1_intrinsics_processed[1]),  # fy
                float(cam1_intrinsics_processed[2]),  # cx
                float(cam1_intrinsics_processed[3])   # cy
            ]
            
            # When not swapping, reverse the transformation logic:
            # camera0 gets T_cn_cnm1 inverse
            # camera1 gets identity matrix
            T_B_C_cam0 = T_cn_cnm1
            T_B_C_cam1 = identity_matrix
        
        # Convert T_B_C matrices to flattened lists for the YAML format
        T_B_C_cam0_list = [
            float(T_B_C_cam0[0, 0]), float(T_B_C_cam0[0, 1]), float(T_B_C_cam0[0, 2]), float(T_B_C_cam0[0, 3]),
            float(T_B_C_cam0[1, 0]), float(T_B_C_cam0[1, 1]), float(T_B_C_cam0[1, 2]), float(T_B_C_cam0[1, 3]),
            float(T_B_C_cam0[2, 0]), float(T_B_C_cam0[2, 1]), float(T_B_C_cam0[2, 2]), float(T_B_C_cam0[2, 3]),
            float(T_B_C_cam0[3, 0]), float(T_B_C_cam0[3, 1]), float(T_B_C_cam0[3, 2]), float(T_B_C_cam0[3, 3])
        ]
        camera0['T_B_C']['data'] = T_B_C_cam0_list
        
        T_B_C_cam1_list = [
            float(T_B_C_cam1[0, 0]), float(T_B_C_cam1[0, 1]), float(T_B_C_cam1[0, 2]), float(T_B_C_cam1[0, 3]),
            float(T_B_C_cam1[1, 0]), float(T_B_C_cam1[1, 1]), float(T_B_C_cam1[1, 2]), float(T_B_C_cam1[1, 3]),
            float(T_B_C_cam1[2, 0]), float(T_B_C_cam1[2, 1]), float(T_B_C_cam1[2, 2]), float(T_B_C_cam1[2, 3]),
            float(T_B_C_cam1[3, 0]), float(T_B_C_cam1[3, 1]), float(T_B_C_cam1[3, 2]), float(T_B_C_cam1[3, 3])
        ]
        camera1['T_B_C']['data'] = T_B_C_cam1_list
        
        # Print distortion model mapping information
        print(f"Mapped distortion models:")
        print(f"  cam0: {cam0_distortion_model} -> {cam0_distortion_type}")
        print(f"  cam1: {cam1_distortion_model} -> {cam1_distortion_type}")
    
    # Write the updated YAML to the output file
    with open(output_path, 'w') as file:
        yaml.dump(sensor_data, file, default_flow_style=False)
    
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    template_msg = f"using template from {template_file}" if template_path else "using output as template"
    print(f"Updated {output_path} with calibration parameters from {input_path} (camchain format, {swap_msg}, {divide_msg}, {template_msg})")
    
    # Print T_B_C assignments for verification
    if swap_cameras:
        print("Camera0 assigned identity matrix for T_B_C")
        print("Camera1 assigned T_cn_cnm1 for T_B_C")
    else:
        print("Camera0 assigned inverse of T_cn_cnm1 for T_B_C")
        print("Camera1 assigned identity matrix for T_B_C")

def main():
    args = parse_args()
    template_path = args.template if args.template else None
    divide_intrinsics = not args.no_divide_intrinsics
    
    if args.camchain:
        update_sensor_yaml_from_camchain(args.input, args.output, args.swap, template_path, divide_intrinsics)
    else:
        update_sensor_yaml(args.input, args.output, args.swap, template_path, divide_intrinsics)

if __name__ == "__main__":
    main() 