#!/usr/bin/env python3
import numpy as np
import json
import yaml
import argparse
import os
import sys
import re
from stereo_utils import read_yaml_safely, matrix_to_yaml_list

class CaseInsensitiveDict(dict):
    """大小写不敏感的字典类，用于处理键名大小写变化"""
    
    def __init__(self, data=None):
        super(CaseInsensitiveDict, self).__init__()
        if data:
            self.update(data)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # 首先尝试直接获取
            try:
                return super(CaseInsensitiveDict, self).__getitem__(key)
            except KeyError:
                # 尝试查找所有可能的大小写变体
                for k in self.keys():
                    if isinstance(k, str) and k.lower() == key.lower():
                        return super(CaseInsensitiveDict, self).__getitem__(k)
        # 如果上述方法都失败，抛出KeyError异常
        raise KeyError(key)
    
    def __contains__(self, key):
        if isinstance(key, str):
            # 直接检查是否存在
            if super(CaseInsensitiveDict, self).__contains__(key):
                return True
            # 检查小写形式是否匹配
            for k in self.keys():
                if isinstance(k, str) and k.lower() == key.lower():
                    return True
        return False
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def update(self, other=None, **kwargs):
        if other is not None:
            for key, value in other.items():
                # 递归处理嵌套字典
                if isinstance(value, dict):
                    value = CaseInsensitiveDict(value)
                elif isinstance(value, list):
                    # 递归处理列表中的字典
                    value = [CaseInsensitiveDict(item) if isinstance(item, dict) else item for item in value]
                self[key] = value
        if kwargs:
            self.update(kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="将dualcamera_calibration.json转换为sensor和okvis格式的YAML文件")
    parser.add_argument('--input', type=str, required=True, 
                        help="输入的dualcamera_calibration.json文件路径")
    parser.add_argument('--output', type=str, required=True,
                        help="输出的YAML文件路径前缀，会根据模式添加后缀")
    parser.add_argument('--format', type=str, choices=['sensor', 'okvis', 'all'], default='sensor',
                        help="输出格式: 'sensor'为sensor.yaml, 'okvis'为okvis YAML格式, 'all'为同时输出两种格式")
    parser.add_argument('--sensor_template', type=str, default=None, 
                        help="sensor格式的模板YAML文件路径")
    parser.add_argument('--okvis_template', type=str, default=None, 
                        help="okvis格式的模板YAML文件路径")
    return parser.parse_args()

def read_json_file(file_path):
    """读取JSON文件并返回数据"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            # 将JSON数据转换成大小写不敏感的字典
            return CaseInsensitiveDict(data)
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        sys.exit(1)

def extrinsic_to_transform_matrix(extrinsic):
    """
    将外参数据转换为4x4变换矩阵
    
    外参数据格式:
    {
        "tx": x, "ty": y, "tz": z,
        "r00": xx, "r01": xy, "r02": xz,
        "r10": yx, "r11": yy, "r12": yz,
        "r20": zx, "r21": zy, "r22": zz
    }
    """
    matrix = np.eye(4)
    
    # 旋转部分，使用大小写不敏感字典
    for r_idx in [('r00', 0, 0), ('r01', 0, 1), ('r02', 0, 2),
                 ('r10', 1, 0), ('r11', 1, 1), ('r12', 1, 2),
                 ('r20', 2, 0), ('r21', 2, 1), ('r22', 2, 2)]:
        key, i, j = r_idx
        if key in extrinsic:
            matrix[i, j] = extrinsic[key]
    
    # 平移部分，使用大小写不敏感字典
    for t_idx in [('tx', 0, 3), ('ty', 1, 3), ('tz', 2, 3)]:
        key, i, j = t_idx
        if key in extrinsic:
            matrix[i, j] = extrinsic[key]
    
    return matrix

def toi_to_transform_matrix(toi):
    """从toi数据创建4x4变换矩阵，支持12元素或16元素的T数组"""
    if 't' in toi:  # 大小写不敏感字典会自动处理't'和'T'
        t_data = toi['t']
        # 检查T数组的长度
        if len(t_data) == 16:
            # 已经是4x4矩阵，直接reshape
            matrix = np.array(t_data).reshape(4, 4)
        elif len(t_data) == 12:
            # 只有3x4矩阵，需要补充最后一行[0,0,0,1]
            matrix = np.eye(4)
            # 填充前3行
            matrix[:3, :] = np.array(t_data).reshape(3, 4)
            print(f"注意: TOI数据中的T矩阵只有12个元素，已自动补充为4x4矩阵")
        else:
            print(f"警告: TOI数据中的T矩阵有{len(t_data)}个元素，既不是12个也不是16个，使用单位矩阵")
            return np.eye(4)
        return matrix
    else:
        print("警告: TOI数据中未找到T矩阵，使用单位矩阵")
        return np.eye(4)

def process_distortion_params(distortion):
    """处理畸变参数，返回radtan8格式的系数列表"""
    # 对于radtan8模型，参数顺序为: [k1, k2, p1, p2, k3, k4, k5, k6]
    # JSON中的顺序为: [k1, k2, k3, k4, k5, k6, p1, p2]
    # 需要重新排序
    keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    result = []
    
    for key in keys:
        if key in distortion:
            result.append(distortion[key])
        else:
            print(f"警告: 在畸变参数中未找到{key}，使用0.0作为默认值")
            result.append(0.0)
    
    return result

def convert_to_sensor_yaml(json_data, output_path, template_path=None):
    """将JSON数据转换为sensor.yaml格式"""
    # 固定的相机参数
    width = 640
    height = 544
    distortion_type = "radtan8"
    
    # 加载模板（如果提供）
    if template_path and os.path.exists(template_path):
        sensor_data = read_yaml_safely(template_path)
        if not sensor_data:
            print(f"无法加载模板文件，将创建默认结构")
            sensor_data = {'sensor': {'cameras': [], 'imu': {}}}
    else:
        # 创建默认结构
        sensor_data = {'sensor': {'cameras': [], 'imu': {}}}
    
    # 保存模板中除了相机和IMU T_B_I之外的其他字段
    template_extras = {}
    if 'sensor' in sensor_data:
        for key, value in sensor_data['sensor'].items():
            if key != 'cameras' and key != 'imu':
                template_extras[key] = value
        
        # 保存IMU中除了T_B_I之外的字段
        if 'imu' in sensor_data['sensor']:
            imu_extras = {}
            for key, value in sensor_data['sensor']['imu'].items():
                if key != 'T_B_I':
                    imu_extras[key] = value
    
    # 确保cameras列表存在
    if 'cameras' not in sensor_data['sensor']:
        sensor_data['sensor']['cameras'] = []
    
    # 清空现有的相机列表（如果有）
    sensor_data['sensor']['cameras'] = []
    
    # 处理左相机
    left_camera = json_data['left']
    left_intrinsic = left_camera['intrinsic']
    left_distortion = left_camera['distortion']
    left_extrinsic = left_camera['extrinsic']
    
    # 计算变换矩阵（相机到机身）
    T_B_C0 = extrinsic_to_transform_matrix(left_extrinsic)
    
    # 创建左相机数据
    left_cam_data = {
        'camera': {
            'distortion_type': distortion_type,
            'distortion': {
                'cols': 1,
                'rows': 8,
                'data': process_distortion_params(left_distortion)
            },
            'intrinsics': {
                'cols': 1,
                'rows': 4,
                'data': [
                    left_intrinsic['fx'] / 2,  # 内参除以2
                    left_intrinsic['fy'] / 2,
                    left_intrinsic['cx'] / 2,
                    left_intrinsic['cy'] / 2
                ]
            },
            'image_width': width, 
            'image_height': height
        },
        'T_B_C': {
            'cols': 4,
            'rows': 4,
            'data': matrix_to_yaml_list(T_B_C0)
        }
    }
    
    # 处理右相机
    right_camera = json_data['right']
    right_intrinsic = right_camera['intrinsic']
    right_distortion = right_camera['distortion']
    right_extrinsic = right_camera['extrinsic']
    
    # 计算变换矩阵（相机到机身）
    T_B_C1 = extrinsic_to_transform_matrix(right_extrinsic)
    
    # 创建右相机数据
    right_cam_data = {
        'camera': {
            'distortion_type': distortion_type,
            'distortion': {
                'cols': 1,
                'rows': 8,
                'data': process_distortion_params(right_distortion)
            },
            'intrinsics': {
                'cols': 1,
                'rows': 4,
                'data': [
                    right_intrinsic['fx'] / 2,  # 内参除以2
                    right_intrinsic['fy'] / 2,
                    right_intrinsic['cx'] / 2,
                    right_intrinsic['cy'] / 2
                ]
            },
            'image_width': width,
            'image_height': height
        },
        'T_B_C': {
            'cols': 4,
            'rows': 4,
            'data': matrix_to_yaml_list(T_B_C1)
        }
    }
    
    # 添加相机数据到sensor_data
    sensor_data['sensor']['cameras'].append(left_cam_data)
    sensor_data['sensor']['cameras'].append(right_cam_data)
    
    # 处理IMU数据，如果存在，用新的T_B_I更新
    if 'toi' in json_data:
        T_B_I = toi_to_transform_matrix(json_data['toi'])
        
        # 确保imu字典存在
        if 'imu' not in sensor_data['sensor']:
            sensor_data['sensor']['imu'] = {}
        
        # 更新T_B_I
        sensor_data['sensor']['imu']['T_B_I'] = {
            'cols': 4,
            'rows': 4,
            'data': matrix_to_yaml_list(T_B_I)
        }
        
        # 如果有保存的IMU额外字段，恢复它们
        if 'imu_extras' in locals():
            for key, value in imu_extras.items():
                sensor_data['sensor']['imu'][key] = value
    
    # 恢复模板中的其他字段
    if template_extras:
        for key, value in template_extras.items():
            sensor_data['sensor'][key] = value
    
    # 自定义格式写入YAML文件
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
        
        # 写入其他部分（来自模板的额外字段）
        for key, value in sensor_data['sensor'].items():
            if key not in ['cameras', 'imu']:
                if isinstance(value, dict):
                    file.write(f"  {key}:\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            file.write(f"    {sub_key}: [{', '.join([str(x) for x in sub_value])}]\n")
                        elif isinstance(sub_value, dict):
                            file.write(f"    {sub_key}:\n")
                            for subsub_key, subsub_value in sub_value.items():
                                file.write(f"      {subsub_key}: {subsub_value}\n")
                        else:
                            file.write(f"    {sub_key}: {sub_value}\n")
                elif isinstance(value, list):
                    file.write(f"  {key}: [{', '.join([str(x) for x in value])}]\n")
                else:
                    file.write(f"  {key}: {value}\n")
    
    print(f"已将JSON数据转换为sensor格式并保存到 {output_path}")
    return True

def convert_to_okvis_yaml(json_data, output_path, template_path=None):
    """将JSON数据转换为OKVIS YAML格式"""
    # 固定的相机参数
    width = 640
    height = 544
    distortion_type = "radialtangential8"  # OKVIS中radtan8的对应类型
    
    # 加载模板文件
    if template_path and os.path.exists(template_path):
        okvis_data = read_yaml_safely(template_path)
        if not okvis_data:
            print(f"无法加载OKVIS模板文件")
            return False
    else:
        print(f"需要OKVIS模板文件")
        return False
    
    # 处理左相机
    left_camera = json_data['left']
    left_intrinsic = left_camera['intrinsic']
    left_distortion = left_camera['distortion']
    left_extrinsic = left_camera['extrinsic']
    
    # 处理右相机
    right_camera = json_data['right']
    right_intrinsic = right_camera['intrinsic']
    right_distortion = right_camera['distortion']
    right_extrinsic = right_camera['extrinsic']
    
    # 计算变换矩阵（相机到机身）
    T_B_C0 = extrinsic_to_transform_matrix(left_extrinsic)
    T_B_C1 = extrinsic_to_transform_matrix(right_extrinsic)
    
    # 从TOI获取IMU到机身的变换
    if 'toi' in json_data:
        T_B_I = toi_to_transform_matrix(json_data['toi'])
    else:
        T_B_I = np.eye(4)
        print("警告: 未找到TOI数据，使用单位矩阵作为T_B_I")
    
    # 计算T_SC (T_S_C = T_I_B @ T_B_C)
    T_SC0 = np.linalg.inv(T_B_I) @ T_B_C0
    T_SC1 = np.linalg.inv(T_B_I) @ T_B_C1
    
    # 确保cameras列表存在并足够长
    if 'cameras' not in okvis_data or len(okvis_data['cameras']) < 2:
        okvis_data['cameras'] = [{}, {}]
    
    # 更新第一个相机参数
    okvis_data['cameras'][0]['T_SC'] = matrix_to_yaml_list(T_SC0)
    okvis_data['cameras'][0]['distortion_coefficients'] = process_distortion_params(left_distortion)
    okvis_data['cameras'][0]['distortion_type'] = distortion_type
    okvis_data['cameras'][0]['focal_length'] = [
        left_intrinsic['fx'] / 2,  # 内参除以2
        left_intrinsic['fy'] / 2
    ]
    okvis_data['cameras'][0]['principal_point'] = [
        left_intrinsic['cx'] / 2,
        left_intrinsic['cy'] / 2
    ]
    okvis_data['cameras'][0]['image_dimension'] = [width, height]
    
    # 更新第二个相机参数
    okvis_data['cameras'][1]['T_SC'] = matrix_to_yaml_list(T_SC1)
    okvis_data['cameras'][1]['distortion_coefficients'] = process_distortion_params(right_distortion)
    okvis_data['cameras'][1]['distortion_type'] = distortion_type
    okvis_data['cameras'][1]['focal_length'] = [
        right_intrinsic['fx'] / 2,  # 内参除以2
        right_intrinsic['fy'] / 2
    ]
    okvis_data['cameras'][1]['principal_point'] = [
        right_intrinsic['cx'] / 2,
        right_intrinsic['cy'] / 2
    ]
    okvis_data['cameras'][1]['image_dimension'] = [width, height]
    
    # 更新wheel encoder参数（如果存在）
    if 'wheel_encoder_parameters' in okvis_data:
        # 对于轮式编码器，T_BS也是T_B_I的逆
        okvis_data['wheel_encoder_parameters']['T_BS'] = matrix_to_yaml_list(np.linalg.inv(T_B_I))
    
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
            
            # 写入camera_type和slam_use（如果存在）
            if 'camera_type' in camera:
                file.write(f"        camera_type: {camera['camera_type']}, #gray, rgb, gray+depth, rgb+depth\n")
            if 'slam_use' in camera:
                file.write(f"        slam_use: {camera['slam_use']}}}" + (" #none, okvis, okvis-depth, okvis-virtual\n" if i == 0 else "\n\n"))
            else:
                file.write(f"        }}" + ("\n" if i == 0 else "\n\n"))
        
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
    
    print(f"已将JSON数据转换为OKVIS格式并保存到 {output_path}")
    return True

def main():
    args = parse_args()
    
    # 读取JSON数据
    json_data = read_json_file(args.input)
    
    # 生成输出文件路径
    sensor_output = f"{args.output}_sensor.yaml"
    okvis_output = f"{args.output}_okvis.yaml"
    
    # 根据指定格式进行转换
    if args.format == 'sensor' or args.format == 'all':
        convert_to_sensor_yaml(json_data, sensor_output, args.sensor_template)
        
    if args.format == 'okvis' or args.format == 'all':
        if not args.okvis_template and args.format == 'okvis':
            print("错误: OKVIS格式需要提供模板文件，请使用--okvis_template指定")
            sys.exit(1)
        elif args.okvis_template:
            convert_to_okvis_yaml(json_data, okvis_output, args.okvis_template)
        else:
            print("警告: 未提供OKVIS模板文件，跳过OKVIS格式转换")

if __name__ == "__main__":
    main() 