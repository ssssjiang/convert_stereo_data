#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import argparse
import os
import re
from stereo_utils import (
    read_yaml_safely, 
    map_distortion_model, 
    limit_distortion_params, 
    create_T_B_C, 
    matrix_to_yaml_list,
    scale_template_resolution,
    process_camera_resolution,
    load_camchain_data, 
    extract_camera_params, 
    update_camera_params, 
    process_sensor_cameras
)

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

def write_sensor_yaml_formatted(sensor_data, output_path):
    """使用美观易读的格式写入sensor.yaml文件"""
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
    
    # Create camchain-style camera data for process_sensor_cameras function
    cam0_data = {
        'intrinsics': [
            float(M1_processed[0, 0]),  # fx
            float(M1_processed[1, 1]),  # fy
            float(M1_processed[0, 2]),  # cx
            float(M1_processed[1, 2])   # cy
        ],
        'distortion_coeffs': D1_list,
        'distortion_model': 'radtan8'  # Default for OpenCV stereo
    }
    
    cam1_data = {
        'intrinsics': [
            float(M2_processed[0, 0]),  # fx
            float(M2_processed[1, 1]),  # fy
            float(M2_processed[0, 2]),  # cx
            float(M2_processed[1, 2])   # cy
        ],
        'distortion_coeffs': D2_list,
        'distortion_model': 'radtan8'  # Default for OpenCV stereo
    }
    
    # Set appropriate transformation matrices based on swap setting
    if swap_cameras:
        T_B_C0 = identity_matrix
        T_B_C1 = T_B_C_from_stereo
    else:
        T_B_C0 = T_B_C_from_stereo
        T_B_C1 = identity_matrix
    
    # 对于非 camchain 模式：
    # 1. 我们不修改分辨率，保持与模板一致
    # 2. 已经处理过内参，不需要再次除以2
    update_resolution = False
    
    # Process camera parameters 
    sensor_data, model_info = process_sensor_cameras(
        sensor_data, cam0_data, cam1_data, swap_cameras, False,
        T_B_C0, T_B_C1, identity_matrix, update_resolution
    )
    
    # 使用美观易读的格式写入YAML文件
    write_sensor_yaml_formatted(sensor_data, output_path)
    
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    template_msg = f"using template from {template_file}" if template_path else "using output as template"
    print(f"Updated {output_path} with calibration parameters from {input_path} ({swap_msg}, {divide_msg}, {template_msg})")

def update_sensor_yaml_from_camchain(input_path, output_path, swap_cameras=False, template_path=None, divide_intrinsics=True):
    """Update sensor.yaml with calibration parameters from camchain-imucam.yaml format."""
    # Load camchain YAML file
    camchain_data = load_camchain_data(input_path)
    
    # Extract camera data
    cam0 = camchain_data['cam0']
    cam1 = camchain_data['cam1']
    
    # Get T_cn_cnm1 transformation matrix
    if 'T_cn_cnm1' not in cam1:
        raise ValueError("T_cn_cnm1 not found in cam1 data")
    
    T_cn_cnm1 = np.array(cam1['T_cn_cnm1'])
    identity_matrix = np.eye(4)
    
    # Load the template YAML file safely
    template_file = template_path if template_path else output_path
    sensor_data = read_yaml_safely(template_file)
    
    # Set appropriate transformation matrices based on swap setting
    if swap_cameras:
        T_B_C0 = identity_matrix
        T_B_C1 = T_cn_cnm1
    else:
        T_B_C0 = T_cn_cnm1
        T_B_C1 = identity_matrix
    
    # 对于 camchain 模式，我们允许更新分辨率（如果有）
    update_resolution = True
    
    # Process camera parameters
    sensor_data, model_info = process_sensor_cameras(
        sensor_data, cam0, cam1, swap_cameras, divide_intrinsics,
        T_B_C0, T_B_C1, identity_matrix, update_resolution
    )
    
    # 使用美观易读的格式写入YAML文件
    write_sensor_yaml_formatted(sensor_data, output_path)
    
    swap_msg = "with camera swap" if swap_cameras else "without camera swap"
    divide_msg = "with divided intrinsics" if divide_intrinsics else "with original intrinsics"
    template_msg = f"using template from {template_file}" if template_path else "using output as template"
    print(f"Updated {output_path} with calibration parameters from {input_path} ({swap_msg}, {divide_msg}, {template_msg})")
    
    # Print distortion model mapping information
    print(f"Mapped distortion models:")
    print(f"  cam0: {model_info['cam0_model']} -> {model_info['cam0_type']}")
    print(f"  cam1: {model_info['cam1_model']} -> {model_info['cam1_type']}")

def main():
    args = parse_args()
    # Set divide_intrinsics to the opposite of no_divide_intrinsics flag
    divide_intrinsics = not args.no_divide_intrinsics
    
    # Choose the appropriate function based on input file format
    if args.camchain:
        update_sensor_yaml_from_camchain(args.input, args.output, args.swap, args.template, divide_intrinsics)
    else:
        update_sensor_yaml(args.input, args.output, args.swap, args.template, divide_intrinsics)

if __name__ == "__main__":
    main() 