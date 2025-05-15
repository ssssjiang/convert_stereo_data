#!/usr/bin/env python3
import numpy as np
import yaml
import argparse
import os
import re
import sys
from stereo_utils import (
    read_yaml_safely, 
    map_distortion_model, 
    limit_distortion_params, 
    matrix_to_yaml_list,
    scale_template_resolution,
    process_camera_resolution,
    load_camchain_data,
    extract_camera_params,
    update_camera_params,
    process_sensor_cameras,
    process_okvis_resolution
)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert calibration parameters from camchain-imucam and Tbc0/Tbc1 to sensor.yaml format")
    parser.add_argument('--camchain', type=str, default='/home/roborock/下载/sensor_yaml/MK1-5/MK1-5_equi_vio-camchain-imucam.yaml', 
                        help="Path to the input camchain-imucam YAML file")
    parser.add_argument('--Tbc', type=str, default='/home/roborock/下载/sensor_yaml/MK1-5/MK1-5_equi_Tbc0.txt',
                        help="Path to the input Tbc0 or Tbc1 txt file")
    parser.add_argument('--use_Tbc1', action='store_true',
                        help="If set, the input Tbc file is treated as Tbc1 instead of Tbc0")
    parser.add_argument('--output', type=str, default='sensor_output.yaml',
                        help="Path to the output file prefix, suffixes will be added based on format")
    parser.add_argument('--sensor_template', type=str, default='/home/roborock/下载/sensor_656.yaml', 
                        help="Path to the template sensor YAML file")
    parser.add_argument('--divide_intrinsics', action='store_true', 
                        help="Divide intrinsics by 2 (for half-resolution images)")
    parser.add_argument('--swap', action='store_true', 
                        help="Swap left and right cameras (camera0 and camera1)")
    parser.add_argument('--format', type=str, choices=['sensor', 'okvis', 'all'], default='sensor',
                        help="Output format: 'sensor' for sensor.yaml, 'okvis' for mower_stereo_light.yaml, 'all' for both")
    parser.add_argument('--okvis_template', type=str, default='/home/roborock/repos/okvis2/config/mower_stereo_light.yaml',
                        help="Path to the template OKVIS YAML file")
    return parser.parse_args()

def read_Tbc_txt(file_path):
    """Read the Tbc txt file and extract the transformation matrix."""
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

def convert_to_sensor_yaml(camchain_data, output_path, swap_cameras=False, 
                       sensor_template=None, divide_intrinsics=True, print_debug=False,
                       Tbc_path=None, use_Tbc1=False):
    """Convert camchain data to sensor.yaml format.
    
    Args:
        camchain_data: Dictionary containing camchain data
        output_path: Path to the output sensor YAML file
        swap_cameras: Whether to swap cameras
        sensor_template: Path to the template sensor YAML file
        divide_intrinsics: Whether to divide intrinsics by 2
        print_debug: Whether to print debug information
        Tbc_path: Path to the Tbc0 or Tbc1 txt file (optional)
        use_Tbc1: If True, the input Tbc is treated as Tbc1 instead of Tbc0
    """
    # Extract data for the two cameras
    cam0 = camchain_data['cam0']
    cam1 = camchain_data['cam1']
    
    # Set identity matrix
    identity_matrix = np.eye(4)
    
    # Get transformation from cam0 to cam1 (T_cn_cnm1 in cam1)
    if 'T_cn_cnm1' not in cam1:
        # If not present, we need to calculate it from T_cn_i
        if 'T_cam_imu' in cam0 and 'T_cam_imu' in cam1:
            print("T_cn_cnm1 not found, calculating from T_cam_imu...")
            T_c0_i = np.array(cam0['T_cam_imu'])  # T_c0_i
            T_c1_i = np.array(cam1['T_cam_imu'])  # T_c1_i
            
            # T_c0_c1 = T_c0_i * inv(T_c1_i)
            T_c0_c1 = T_c0_i @ np.linalg.inv(T_c1_i)
            
            # We need T_c1_c0 which is inv(T_c0_c1)
            T_cn_cnm1 = np.linalg.inv(T_c0_c1)
        else:
            raise ValueError("Neither T_cn_cnm1 nor T_cam_imu is provided in the camchain data")
    else:
        T_cn_cnm1 = np.array(cam1['T_cn_cnm1'])
    
    # 如果提供了 Tbc 文件，则处理外部变换矩阵
    if Tbc_path:
        # 读取 Tbc 文件
        T_B_C_from_file = read_Tbc_txt(Tbc_path)
        
        # 获取 T_cam_imu 变换
        T_cam0_imu = np.array(cam0['T_cam_imu'])
        T_cam1_imu = np.array(cam1['T_cam_imu'])
        
        # 根据 use_Tbc1 标志决定如何处理变换
        if use_Tbc1:
            # 如果使用 Tbc1，那么输入文件包含 T_B_C1
            T_B_C1 = T_B_C_from_file
            print(f"Using provided file as T_B_C1:")
            print(T_B_C1)
            
            # 使用 T_B_C1 和 T_c1_c0 计算 T_B_C0
            T_B_C0 = T_B_C1 @ T_cn_cnm1
            print("Calculated T_B_C0 from T_B_C1 and T_c1_c0:")
            print(T_B_C0)
            
            # 使用 T_B_C1 和 T_cam1_imu 计算 T_B_I
            T_B_I = T_B_C1 @ T_cam1_imu
            print("Calculated T_B_I from T_B_C1 and T_cam1_imu:")
            print(T_B_I)
        else:
            # 如果使用 Tbc0（默认情况），那么输入文件包含 T_B_C0
            T_B_C0 = T_B_C_from_file
            print(f"Using provided file as T_B_C0:")
            print(T_B_C0)
            
            # 使用 T_B_C0 和 T_c1_c0 计算 T_B_C1
            T_B_C1 = T_B_C0 @ np.linalg.inv(T_cn_cnm1)
            print("Calculated T_B_C1 from T_B_C0 and inv(T_c1_c0):")
            print(T_B_C1)
            
            # 使用 T_B_C0 和 T_cam0_imu 计算 T_B_I
            T_B_I = T_B_C0 @ T_cam0_imu
            print("Calculated T_B_I from T_B_C0 and T_cam0_imu:")
            print(T_B_I)
    else:
        # 如果没有提供 Tbc 文件，则使用原始逻辑设置转换矩阵
        if swap_cameras:
            T_B_C0 = identity_matrix
            T_B_C1 = T_cn_cnm1
        else:
            T_B_C0 = T_cn_cnm1  # T_c1_c0
            T_B_C1 = identity_matrix
        
        # 没有提供 Tbc 文件时，不设置 T_B_I
        T_B_I = None
    
    # Load template sensor.yaml
    if sensor_template:
        sensor_data = read_yaml_safely(sensor_template)
    else:
        # Create a minimal sensor.yaml structure if no template is provided
        sensor_data = {'sensor': {'cameras': [], 'imu': {}}}
    
    # 对于camchain格式，支持分辨率更新
    update_resolution = True
    
    # Process cameras and update sensor data
    sensor_data, model_info = process_sensor_cameras(
        sensor_data, cam0, cam1, swap_cameras, divide_intrinsics,
        T_B_C0, T_B_C1, identity_matrix, update_resolution
    )
    
    # 如果有 T_B_I，更新传感器数据中的 IMU 部分
    if T_B_I is not None and 'imu' in sensor_data['sensor']:
        # 更新 T_B_I
        T_B_I_dict = {
            'cols': 4,
            'rows': 4,
            'data': matrix_to_yaml_list(T_B_I)
        }
        sensor_data['sensor']['imu']['T_B_I'] = T_B_I_dict
    
    # 使用自定义格式写入YAML文件而不是使用yaml.dump
    with open(output_path, 'w') as file:
        # 写入sensor节点
        file.write("sensor:\n")
        
        # 写入cameras部分
        if 'cameras' in sensor_data['sensor']:
            file.write("  cameras:\n")
            for i, camera in enumerate(sensor_data['sensor']['cameras']):
                file.write(f"    - camera:\n")
                
                # 写入相机内参
                if 'intrinsics' in camera['camera']:
                    intrinsics = camera['camera']['intrinsics']
                    file.write(f"        intrinsics:\n")
                    file.write(f"          cols: {intrinsics.get('cols', 1)}\n")
                    file.write(f"          rows: {intrinsics.get('rows', 4)}\n")
                    file.write(f"          data: [")
                    data_str = ", ".join([f"{float(x)}" for x in intrinsics['data']])
                    file.write(f"{data_str}]\n")
                
                # 写入畸变参数
                if 'distortion' in camera['camera']:
                    distortion = camera['camera']['distortion']
                    file.write(f"        distortion:\n")
                    file.write(f"          cols: {distortion.get('cols', 1)}\n")
                    file.write(f"          rows: {distortion.get('rows', len(distortion['data']))}\n")
                    file.write(f"          data: [")
                    data_str = ", ".join([f"{float(x)}" for x in distortion['data']])
                    file.write(f"{data_str}]\n")
                
                # 写入畸变类型
                if 'distortion_type' in camera['camera']:
                    file.write(f"        distortion_type: {camera['camera']['distortion_type']}\n")
                
                # 写入图像尺寸
                if 'image_width' in camera['camera']:
                    file.write(f"        image_width: {camera['camera']['image_width']}\n")
                if 'image_height' in camera['camera']:
                    file.write(f"        image_height: {camera['camera']['image_height']}\n")
                
                # 写入T_B_C
                if 'T_B_C' in camera:
                    t_b_c = camera['T_B_C']
                    file.write(f"      T_B_C:\n")
                    file.write(f"        cols: {t_b_c.get('cols', 4)}\n")
                    file.write(f"        rows: {t_b_c.get('rows', 4)}\n")
                    file.write(f"        data: [")
                    # 为了美观，将矩阵数据分行显示
                    data = t_b_c['data']
                    lines = [
                        ", ".join([f"{float(data[i*4+j])}" for j in range(4)]) 
                        for i in range(4)
                    ]
                    file.write(f"{lines[0]},\n")
                    file.write(f"                {lines[1]},\n")
                    file.write(f"                {lines[2]},\n")
                    file.write(f"                {lines[3]}]\n")
                
                # 写入其他相机参数
                for key, value in camera['camera'].items():
                    if key not in ['intrinsics', 'distortion', 'distortion_type', 'image_width', 'image_height']:
                        if isinstance(value, dict):
                            file.write(f"        {key}:\n")
                            for sub_key, sub_value in value.items():
                                file.write(f"          {sub_key}: {sub_value}\n")
                        else:
                            file.write(f"        {key}: {value}\n")
        
        # 写入IMU部分
        if 'imu' in sensor_data['sensor']:
            file.write("  imu:\n")
            for key, value in sensor_data['sensor']['imu'].items():
                if key == 'T_B_I':
                    file.write(f"    T_B_I:\n")
                    file.write(f"      cols: {value.get('cols', 4)}\n")
                    file.write(f"      rows: {value.get('rows', 4)}\n")
                    file.write(f"      data: [")
                    # 为了美观，将矩阵数据分行显示
                    data = value['data']
                    lines = [
                        ", ".join([f"{float(data[i*4+j])}" for j in range(4)]) 
                        for i in range(4)
                    ]
                    file.write(f"{lines[0]},\n")
                    file.write(f"              {lines[1]},\n")
                    file.write(f"              {lines[2]},\n")
                    file.write(f"              {lines[3]}]\n")
                else:
                    if isinstance(value, dict):
                        file.write(f"    {key}:\n")
                        for sub_key, sub_value in value.items():
                            file.write(f"      {sub_key}: {sub_value}\n")
                    else:
                        file.write(f"    {key}: {value}\n")
        
        # 写入其他部分
        for key, value in sensor_data['sensor'].items():
            if key not in ['cameras', 'imu']:
                if isinstance(value, dict):
                    file.write(f"  {key}:\n")
                    for sub_key, sub_value in value.items():
                        file.write(f"    {sub_key}: {sub_value}\n")
                else:
                    file.write(f"  {key}: {value}\n")
    
    # Print informative message
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    template_msg = f"using template from {sensor_template}" if sensor_template else "using minimal template"
    tbc_msg = ""
    if Tbc_path:
        tbc_type = "Tbc1" if use_Tbc1 else "Tbc0"
        tbc_msg = f", using {tbc_type} from {Tbc_path}"
    print(f"Converted {output_path} from camchain format ({swap_msg}, {divide_msg}, {template_msg}{tbc_msg})")
    
    # Print distortion model mapping information
    if print_debug:
        print(f"Mapped distortion models:")
        print(f"  cam0: {model_info['cam0_model']} -> {model_info['cam0_type']}")
        print(f"  cam1: {model_info['cam1_model']} -> {model_info['cam1_type']}")
    
    return output_path

def map_okvis_distortion_model(kalibr_model):
    """Map Kalibr distortion models to OKVIS distortion types."""
    mapping = {
        'radtan8': 'radialtangential8',
        'equidistant': 'equidistant',  # 直接支持equidistant格式
        'radtan': 'radialtangential',   # 直接支持radtan格式
    }
    
    # 输出调试信息，帮助理解映射过程
    if kalibr_model not in mapping:
        print(f"Warning: Unknown distortion model '{kalibr_model}', mapping to 'unknown'")
    else:
        print(f"Mapping distortion model from '{kalibr_model}' to '{mapping[kalibr_model]}'")
        
    return mapping.get(kalibr_model, 'unknown')

def limit_okvis_distortion_params(distortion_coeffs, distortion_type):
    """Limit distortion parameters based on the OKVIS distortion type."""
    if distortion_type == 'equidistant':
        # Equidistant model typically uses 4 parameters
        return distortion_coeffs[:4]
    elif distortion_type == 'radialtangential':
        # Radial-tangential model typically uses 4 parameters
        return distortion_coeffs[:4]
    elif distortion_type == 'radialtangential8':
        # 8-parameter radial-tangential model, may need padding to 14 parameters
        coeffs = distortion_coeffs[:8] if len(distortion_coeffs) >= 8 else distortion_coeffs
        # Pad with zeros to reach 14 parameters as required by OKVIS
        while len(coeffs) < 14:
            coeffs.append(0.0)
        return coeffs
    
    # Default case - return original coefficients
    return distortion_coeffs

def process_okvis_cameras(okvis_data, cam0, cam1, swap_cameras, divide_intrinsics, T_B_C0, T_B_C1, T_B_I):
    """Process camera parameters and update them to OKVIS format.
    
    Args:
        okvis_data: The template OKVIS YAML data
        cam0, cam1: Camera parameters from camchain
        swap_cameras: Whether to swap cameras
        divide_intrinsics: Whether to divide intrinsics by 2
        T_B_C0, T_B_C1: Body to camera transformations
        T_B_I: Body to IMU transformation
    
    Returns:
        Updated OKVIS data and model mapping information
    """
    # If swapping cameras
    if swap_cameras:
        cam0, cam1 = cam1, cam0
        T_B_C0, T_B_C1 = T_B_C1, T_B_C0
    
    # 输出调试信息，显示从camchain读取到的原始信息
    print(f"Original camchain distortion models: cam0: {cam0.get('distortion_model', 'undefined')}, cam1: {cam1.get('distortion_model', 'undefined')}")
    
    # Extract camera intrinsics
    # extract_camera_params返回(intrinsics_processed, distortion_type, distortion_coeffs, distortion_model)
    cam0_intrinsics_arr, _, cam0_distortion_coeffs, cam0_distortion_model = extract_camera_params(cam0, divide_intrinsics)
    cam1_intrinsics_arr, _, cam1_distortion_coeffs, cam1_distortion_model = extract_camera_params(cam1, divide_intrinsics)
    
    # 创建内参字典以便后续处理
    cam0_intrinsics = {
        'focal_length_x': float(cam0_intrinsics_arr[0]),
        'focal_length_y': float(cam0_intrinsics_arr[1]),
        'principal_point_x': float(cam0_intrinsics_arr[2]),
        'principal_point_y': float(cam0_intrinsics_arr[3])
    }
    
    cam1_intrinsics = {
        'focal_length_x': float(cam1_intrinsics_arr[0]),
        'focal_length_y': float(cam1_intrinsics_arr[1]),
        'principal_point_x': float(cam1_intrinsics_arr[2]),
        'principal_point_y': float(cam1_intrinsics_arr[3])
    }
    
    # Calculate T_SC (inverse of T_cam_imu)
    T_SC0 = np.linalg.inv(np.array(cam0['T_cam_imu']))
    T_SC1 = np.linalg.inv(np.array(cam1['T_cam_imu']))
    
    # Create model info dictionary
    model_info = {
        'cam0_model': cam0_distortion_model,
        'cam1_model': cam1_distortion_model
    }
    
    # Map distortion models to OKVIS types
    cam0_distortion_type = map_okvis_distortion_model(cam0_distortion_model)
    cam1_distortion_type = map_okvis_distortion_model(cam1_distortion_model)
    
    model_info.update({
        'cam0_type': cam0_distortion_type,
        'cam1_type': cam1_distortion_type
    })
    
    # Process distortion coefficients
    cam0_distortion = limit_okvis_distortion_params(cam0_distortion_coeffs, cam0_distortion_type)
    cam1_distortion = limit_okvis_distortion_params(cam1_distortion_coeffs, cam1_distortion_type)
    
    # Update first camera
    okvis_data['cameras'][0]['T_SC'] = matrix_to_yaml_list(T_SC0)
    okvis_data['cameras'][0]['distortion_coefficients'] = cam0_distortion
    okvis_data['cameras'][0]['distortion_type'] = cam0_distortion_type
    okvis_data['cameras'][0]['focal_length'] = [cam0_intrinsics['focal_length_x'], cam0_intrinsics['focal_length_y']]
    okvis_data['cameras'][0]['principal_point'] = [cam0_intrinsics['principal_point_x'], cam0_intrinsics['principal_point_y']]
    
    # Update image resolution (检查是否存在分辨率信息)
    if 'resolution' in cam0:
        okvis_data['cameras'][0]['image_dimension'] = process_okvis_resolution(cam0['resolution'], divide_intrinsics)
    else:
        print("Warning: No resolution information found for camera0, keeping template resolution")
    
    # Update second camera
    okvis_data['cameras'][1]['T_SC'] = matrix_to_yaml_list(T_SC1)
    okvis_data['cameras'][1]['distortion_coefficients'] = cam1_distortion
    okvis_data['cameras'][1]['distortion_type'] = cam1_distortion_type
    okvis_data['cameras'][1]['focal_length'] = [cam1_intrinsics['focal_length_x'], cam1_intrinsics['focal_length_y']]
    okvis_data['cameras'][1]['principal_point'] = [cam1_intrinsics['principal_point_x'], cam1_intrinsics['principal_point_y']]
    
    # Update image resolution (检查是否存在分辨率信息)
    if 'resolution' in cam1:
        okvis_data['cameras'][1]['image_dimension'] = process_okvis_resolution(cam1['resolution'], divide_intrinsics)
    else:
        print("Warning: No resolution information found for camera1, keeping template resolution")
    
    # Update wheel encoder parameters T_BS if present
    if 'wheel_encoder_parameters' in okvis_data:
        # T_BS is wheel to IMU transformation, which is T_B_I
        # okvis 中 IMU 是 body，所以需要取逆
        T_I_B = np.linalg.inv(T_B_I)
        okvis_data['wheel_encoder_parameters']['T_BS'] = matrix_to_yaml_list(T_I_B)
    
    return okvis_data, model_info

def convert_to_okvis_yaml(camchain_path, Tbc_path, output_path, template_path, 
                          divide_intrinsics=False, swap_cameras=False, use_Tbc1=False):
    """Convert calibration data to OKVIS YAML format.
    
    Args:
        camchain_path: Path to the camchain YAML file
        Tbc_path: Path to the Tbc0 or Tbc1 txt file
        output_path: Path to the output OKVIS YAML file
        template_path: Path to the template OKVIS YAML file
        divide_intrinsics: Whether to divide intrinsics by 2
        swap_cameras: Whether to swap cameras
        use_Tbc1: If True, the input Tbc is treated as Tbc1 instead of Tbc0
    """
    # Load camchain YAML file
    camchain_data = load_camchain_data(camchain_path)
    
    # Extract camera data
    cam0 = camchain_data['cam0']
    cam1 = camchain_data['cam1']
    
    # Load T_B_C from Tbc txt file
    T_B_C_from_file = read_Tbc_txt(Tbc_path)
    
    # Get T_cam_imu transformations
    T_cam0_imu = np.array(cam0['T_cam_imu'])
    T_cam1_imu = np.array(cam1['T_cam_imu'])
    
    # Get relative transformation between cameras
    T_c1_c0 = np.array(cam1['T_cn_cnm1'])
    
    # Calculate transformations based on whether we're using Tbc0 or Tbc1
    if use_Tbc1:
        # If using Tbc1, the input file contains T_B_C1
        T_B_C1 = T_B_C_from_file
        print(f"Using provided file as T_B_C1:")
        print(T_B_C1)
        
        # Calculate T_B_C0 using T_B_C1 and T_c1_c0
        T_B_C0 = T_B_C1 @ T_c1_c0
        print("Calculated T_B_C0 from T_B_C1 and T_c1_c0:")
        print(T_B_C0)
        
        # Calculate T_B_I using T_B_C1 and T_cam1_imu
        T_B_I = T_B_C1 @ T_cam1_imu
        print("Calculated T_B_I from T_B_C1 and T_cam1_imu:")
        print(T_B_I)
    else:
        # If using Tbc0 (default case), the input file contains T_B_C0
        T_B_C0 = T_B_C_from_file
        print(f"Using provided file as T_B_C0:")
        print(T_B_C0)
        
        # Calculate T_B_C1 using T_B_C0 and T_c1_c0
        T_B_C1 = T_B_C0 @ np.linalg.inv(T_c1_c0)
        print("Calculated T_B_C1 from T_B_C0 and inv(T_c1_c0):")
        print(T_B_C1)
        
        # Calculate T_B_I using T_B_C0 and T_cam0_imu
        T_B_I = T_B_C0 @ T_cam0_imu
        print("Calculated T_B_I from T_B_C0 and T_cam0_imu:")
        print(T_B_I)
    
    # Load the template OKVIS YAML file
    okvis_data = read_yaml_safely(template_path)
    
    # Process camera parameters
    okvis_data, model_info = process_okvis_cameras(
        okvis_data, cam0, cam1, swap_cameras, divide_intrinsics,
        T_B_C0, T_B_C1, T_B_I
    )
    
    # 使用更美观的格式写入YAML文件
    with open(output_path, 'w') as file:
        # 首先写入YAML版本标记
        file.write("%YAML:1.0\n")
        
        # 写入cameras部分
        file.write("cameras:\n")
        for i, camera in enumerate(okvis_data['cameras']):
            file.write(f"     - {{T_SC:\n")
            # 写入T_SC矩阵，4行4列，每行4个元素
            T_SC = camera['T_SC']
            file.write(f"        [{T_SC[0]:.8f}, {T_SC[1]:.8f}, {T_SC[2]:.8f}, {T_SC[3]:.8f},\n")
            file.write(f"        {T_SC[4]:.8f}, {T_SC[5]:.8f}, {T_SC[6]:.8f}, {T_SC[7]:.8f},\n")
            file.write(f"        {T_SC[8]:.8f}, {T_SC[9]:.8f}, {T_SC[10]:.8f}, {T_SC[11]:.8f},\n")
            file.write(f"        {T_SC[12]:.8f}, {T_SC[13]:.8f}, {T_SC[14]:.8f}, {T_SC[15]:.8f}],\n")
            
            # 写入image_dimension
            file.write(f"        image_dimension: [{camera['image_dimension'][0]}, {camera['image_dimension'][1]}],\n")
            
            # 写入distortion_coefficients
            file.write("        distortion_coefficients: [")
            coeffs = camera['distortion_coefficients']
            if len(coeffs) <= 4:
                # 少于4个系数的情况
                coeffs_str = ", ".join([f"{c:.16f}" for c in coeffs])
                file.write(f"{coeffs_str}],\n")
            else:
                # 处理长系数列表，每行4个系数
                for j in range(0, len(coeffs), 4):
                    if j > 0:
                        file.write("            ")
                    end_idx = min(j+4, len(coeffs))
                    coeffs_batch = coeffs[j:end_idx]
                    coeffs_str = ", ".join([f"{c:.16f}" for c in coeffs_batch])
                    if end_idx < len(coeffs):
                        file.write(f"{coeffs_str},\n")
                    else:
                        file.write(f"{coeffs_str}],\n")
            
            # 写入其他参数
            file.write(f"        distortion_type: {camera['distortion_type']},\n")
            file.write(f"        focal_length: [{camera['focal_length'][0]:.14f}, {camera['focal_length'][1]:.14f}],\n")
            file.write(f"        principal_point: [{camera['principal_point'][0]:.14f}, {camera['principal_point'][1]:.14f}],\n")
            file.write(f"        camera_type: {camera['camera_type']}, #gray, rgb, gray+depth, rgb+depth\n")
            file.write(f"        slam_use: {camera['slam_use']}}}" + (" #none, okvis, okvis-depth, okvis-virtual\n" if i == 0 else "\n\n"))
        
        # 写入其他参数部分
        for section in ["camera_parameters", "imu_parameters", "wheel_encoder_parameters",
                       "frontend_parameters", "estimator_parameters", "output_parameters"]:
            if section in okvis_data:
                file.write(f"\n# {section.replace('_', ' ')}\n")
                file.write(f"{section}:\n")
                
                # 特殊处理wheel_encoder_parameters中的T_BS
                if section == "wheel_encoder_parameters" and "T_BS" in okvis_data[section]:
                    T_BS = okvis_data[section]["T_BS"]
                    # 先写入其他参数
                    for key, value in okvis_data[section].items():
                        if key != "T_BS":
                            file.write(f"    {key}: {value}\n")
                    # 单独写入T_BS
                    file.write(f"    # transform Body-Sensor (WheelEncoder)\n")
                    file.write(f"    T_BS:\n")
                    file.write(f"        [{T_BS[0]:.8f}, {T_BS[1]:.8f}, {T_BS[2]:.8f}, {T_BS[3]:.8f},\n")
                    file.write(f"         {T_BS[4]:.8f}, {T_BS[5]:.8f}, {T_BS[6]:.8f}, {T_BS[7]:.8f},\n")
                    file.write(f"         {T_BS[8]:.8f}, {T_BS[9]:.8f}, {T_BS[10]:.8f}, {T_BS[11]:.8f},\n")
                    file.write(f"         {T_BS[12]:.8f}, {T_BS[13]:.8f}, {T_BS[14]:.8f}, {T_BS[15]:.8f}]\n")
                # 特殊处理imu_parameters中的T_BS
                elif section == "imu_parameters" and "T_BS" in okvis_data[section]:
                    T_BS = okvis_data[section]["T_BS"]
                    # 先写入其他参数
                    for key, value in okvis_data[section].items():
                        if key != "T_BS":
                            if isinstance(value, list):
                                file.write(f"    {key}: [ {', '.join([str(x) for x in value])} ]\n")
                            else:
                                file.write(f"    {key}: {value}\n")
                    # 单独写入T_BS
                    file.write(f"    # transform Body-Sensor (IMU)\n")
                    file.write(f"    T_BS:\n")
                    file.write(f"        [{T_BS[0]:.4f}, {T_BS[1]:.4f}, {T_BS[2]:.4f}, {T_BS[3]:.4f},\n")
                    file.write(f"         {T_BS[4]:.4f}, {T_BS[5]:.4f}, {T_BS[6]:.4f}, {T_BS[7]:.4f},\n")
                    file.write(f"         {T_BS[8]:.4f}, {T_BS[9]:.4f}, {T_BS[10]:.4f}, {T_BS[11]:.4f},\n")
                    file.write(f"         {T_BS[12]:.4f}, {T_BS[13]:.4f}, {T_BS[14]:.4f}, {T_BS[15]:.4f}]\n")
                else:
                    # 写入其他普通参数
                    for key, value in okvis_data[section].items():
                        if isinstance(value, dict):
                            file.write(f"    {key}:\n")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, list):
                                    file.write(f"        {subkey}: [ {', '.join([str(x) for x in subvalue])} ]\n")
                                else:
                                    file.write(f"        {subkey}: {subvalue}\n")
                        elif isinstance(value, list):
                            file.write(f"    {key}: [ {', '.join([str(x) for x in value])} ]\n")
                        else:
                            file.write(f"    {key}: {value}\n")
    
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    tbc_type = "Tbc1" if use_Tbc1 else "Tbc0"
    print(f"Updated {output_path} with calibration parameters from {camchain_path} and {Tbc_path} (using {tbc_type}, {swap_msg}, {divide_msg})")
    print(f"Distortion model mapping: cam0: {model_info['cam0_model']} -> {model_info['cam0_type']}, cam1: {model_info['cam1_model']} -> {model_info['cam1_type']}")
    if divide_intrinsics:
        print("Intrinsics were divided by 2 for half-resolution images")
        print("Image resolution was also divided by 2")

def main():
    args = parse_args()
    
    # 生成输出文件路径
    sensor_output = f"{args.output}_sensor.yaml"
    okvis_output = f"{args.output}_okvis.yaml"
    
    # 根据指定格式进行转换
    if args.format == 'sensor' or args.format == 'all':
        # 加载camchain数据，然后传递给convert_to_sensor_yaml
        camchain_data = load_camchain_data(args.camchain)
        convert_to_sensor_yaml(camchain_data, sensor_output, args.swap, args.sensor_template, 
                              args.divide_intrinsics, Tbc_path=args.Tbc, use_Tbc1=args.use_Tbc1)
        
    if args.format == 'okvis' or args.format == 'all':
        if not args.okvis_template and args.format == 'okvis':
            print("错误: OKVIS格式需要提供模板文件，请使用--okvis_template指定")
            sys.exit(1)
        elif args.okvis_template:
            convert_to_okvis_yaml(args.camchain, args.Tbc, okvis_output, args.okvis_template, 
                                 args.divide_intrinsics, args.swap, args.use_Tbc1)
        else:
            print("警告: 未提供OKVIS模板文件，跳过OKVIS格式转换")

if __name__ == "__main__":
    main() 