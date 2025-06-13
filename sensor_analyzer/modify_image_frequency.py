#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量修改YAML文件中的参数
- 支持从模板文件补充缺失参数
- 可修改image_frequency、use_only_main_camera、image_delay、wheel_delay和sigma_omega参数
- 保持与模板文件相同的格式
"""

import os
import re
import argparse
import sys

try:
    import yaml
except ImportError:
    print("错误: 本脚本需要 PyYAML 模块来处理YAML文件。")
    print("请运行 'pip install PyYAML' 来安装它。")
    sys.exit(1)

def find_yaml_files(root_dir):
    """查找所有以_okvis.yaml结尾的文件"""
    yaml_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_okvis.yaml'):
                yaml_files.append(os.path.join(dirpath, filename))
    return yaml_files

def merge_dicts(target, source, file_path="未知文件"):
    """将source字典中的键值对合并到target字典中，递归处理嵌套的字典，仅添加缺失的参数"""
    if not isinstance(source, dict) or not isinstance(target, dict):
        return  # 确保两者都是字典

    for key, value in source.items():
        # 如果target中没有这个键，直接添加
        if key not in target:
            target[key] = value
            print(f"在 {file_path} 中添加模板参数: {key}")
        elif isinstance(value, dict) and isinstance(target[key], dict):
            # 如果值是字典且target中对应的值也是字典，递归合并
            merge_dicts(target[key], value, file_path)
        # 如果键已存在且不是字典，保留原值，不覆盖

def modify_specific_parameters_in_dict(yaml_data, file_path, new_frequency=None, use_only_main_camera=None, 
                                     new_image_delay=None, new_wheel_delay=None, new_sigma_omega=None,
                                     new_sigma_v=None, new_unobs_info=None,
                                     detection_threshold=None, matching_threshold=None, max_num_keypoints=None,
                                     enable_debug_recording=None, use_async_processing=None, online_mode=None,
                                     max_batch_size=None):
    """修改YAML数据字典中的特定参数"""
    modified = False

    if not yaml_data:
        return modified

    # 修改image_frequency参数
    if new_frequency is not None and 'image_frequency' in yaml_data:
        old_frequency = yaml_data['image_frequency']
        yaml_data['image_frequency'] = new_frequency
        print(f"修改文件: {file_path}")
        print(f"  image_frequency: {old_frequency} -> {new_frequency}")
        modified = True

    # 修改frontend_parameters部分的参数
    if 'frontend_parameters' in yaml_data:
        # 修改use_only_main_camera参数
        if use_only_main_camera is not None:
            if 'use_only_main_camera' in yaml_data['frontend_parameters']:
                old_value = yaml_data['frontend_parameters']['use_only_main_camera']
                yaml_data['frontend_parameters']['use_only_main_camera'] = use_only_main_camera == 'True'
                print(f"修改文件: {file_path}")
                print(f"  use_only_main_camera: {old_value} -> {use_only_main_camera}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['frontend_parameters']['use_only_main_camera'] = use_only_main_camera == 'True'
                print(f"修改文件: {file_path}")
                print(f"  添加use_only_main_camera: {use_only_main_camera}")
                modified = True
                
        # 修改detection_threshold参数
        if detection_threshold is not None:
            if 'detection_threshold' in yaml_data['frontend_parameters']:
                old_value = yaml_data['frontend_parameters']['detection_threshold']
                yaml_data['frontend_parameters']['detection_threshold'] = float(detection_threshold)
                print(f"修改文件: {file_path}")
                print(f"  detection_threshold: {old_value} -> {detection_threshold}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['frontend_parameters']['detection_threshold'] = float(detection_threshold)
                print(f"修改文件: {file_path}")
                print(f"  添加detection_threshold: {detection_threshold}")
                modified = True
                
        # 修改matching_threshold参数
        if matching_threshold is not None:
            if 'matching_threshold' in yaml_data['frontend_parameters']:
                old_value = yaml_data['frontend_parameters']['matching_threshold']
                yaml_data['frontend_parameters']['matching_threshold'] = float(matching_threshold)
                print(f"修改文件: {file_path}")
                print(f"  matching_threshold: {old_value} -> {matching_threshold}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['frontend_parameters']['matching_threshold'] = float(matching_threshold)
                print(f"修改文件: {file_path}")
                print(f"  添加matching_threshold: {matching_threshold}")
                modified = True
                
        # 修改max_num_keypoints参数
        if max_num_keypoints is not None:
            if 'max_num_keypoints' in yaml_data['frontend_parameters']:
                old_value = yaml_data['frontend_parameters']['max_num_keypoints']
                yaml_data['frontend_parameters']['max_num_keypoints'] = int(max_num_keypoints)
                print(f"修改文件: {file_path}")
                print(f"  max_num_keypoints: {old_value} -> {max_num_keypoints}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['frontend_parameters']['max_num_keypoints'] = int(max_num_keypoints)
                print(f"修改文件: {file_path}")
                print(f"  添加max_num_keypoints: {max_num_keypoints}")
                modified = True

    # 修改image_delay参数
    if new_image_delay is not None and 'camera_parameters' in yaml_data:
        if 'image_delay' in yaml_data['camera_parameters']:
            old_delay = yaml_data['camera_parameters']['image_delay']
            # 格式化新的delay值
            if float(new_image_delay) == 0.0:
                formatted_delay = "0.0"
            else:
                temp = f"{float(new_image_delay):.17f}"
                formatted_delay = temp.rstrip('0').rstrip('.')
                if not formatted_delay or formatted_delay == "-":
                    formatted_delay = temp
            
            yaml_data['camera_parameters']['image_delay'] = float(formatted_delay)
            print(f"修改文件: {file_path}")
            print(f"  image_delay: {old_delay} -> {formatted_delay}")
            modified = True
        else:
            # 如果不存在，添加参数
            if float(new_image_delay) == 0.0:
                formatted_delay = "0.0"
            else:
                temp = f"{float(new_image_delay):.17f}"
                formatted_delay = temp.rstrip('0').rstrip('.')
                if not formatted_delay or formatted_delay == "-":
                    formatted_delay = temp
                    
            yaml_data['camera_parameters']['image_delay'] = float(formatted_delay)
            print(f"修改文件: {file_path}")
            print(f"  添加image_delay: {formatted_delay}")
            modified = True

    # 修改wheel_delay参数
    if new_wheel_delay is not None and 'wheel_encoder_parameters' in yaml_data:
        if 'wheel_delay' in yaml_data['wheel_encoder_parameters']:
            old_delay = yaml_data['wheel_encoder_parameters']['wheel_delay']
            yaml_data['wheel_encoder_parameters']['wheel_delay'] = float(new_wheel_delay)
            print(f"修改文件: {file_path}")
            print(f"  wheel_delay: {old_delay} -> {new_wheel_delay}")
            modified = True
        else:
            # 如果不存在，添加参数
            yaml_data['wheel_encoder_parameters']['wheel_delay'] = float(new_wheel_delay)
            print(f"修改文件: {file_path}")
            print(f"  添加wheel_delay: {new_wheel_delay}")
            modified = True

    # 修改sigma_omega参数
    if new_sigma_omega is not None and 'wheel_encoder_parameters' in yaml_data:
        if 'sigma_omega' in yaml_data['wheel_encoder_parameters']:
            old_value = yaml_data['wheel_encoder_parameters']['sigma_omega']
            yaml_data['wheel_encoder_parameters']['sigma_omega'] = float(new_sigma_omega)
            print(f"修改文件: {file_path}")
            print(f"  sigma_omega: {old_value} -> {new_sigma_omega}")
            modified = True
        else:
            # 如果不存在，添加参数
            yaml_data['wheel_encoder_parameters']['sigma_omega'] = float(new_sigma_omega)
            print(f"修改文件: {file_path}")
            print(f"  添加sigma_omega: {new_sigma_omega}")
            modified = True
    
    # 修改sigma_v参数
    if new_sigma_v is not None and 'wheel_encoder_parameters' in yaml_data:
        if 'sigma_v' in yaml_data['wheel_encoder_parameters']:
            old_value = yaml_data['wheel_encoder_parameters']['sigma_v']
            yaml_data['wheel_encoder_parameters']['sigma_v'] = float(new_sigma_v)
            print(f"修改文件: {file_path}")
            print(f"  sigma_v: {old_value} -> {new_sigma_v}")
            modified = True
        else:
            # 如果不存在，添加参数
            yaml_data['wheel_encoder_parameters']['sigma_v'] = float(new_sigma_v)
            print(f"修改文件: {file_path}")
            print(f"  添加sigma_v: {new_sigma_v}")
            modified = True
            
    # 修改unobs_info参数
    if new_unobs_info is not None and 'wheel_encoder_parameters' in yaml_data:
        if 'unobs_info' in yaml_data['wheel_encoder_parameters']:
            old_value = yaml_data['wheel_encoder_parameters']['unobs_info']
            yaml_data['wheel_encoder_parameters']['unobs_info'] = float(new_unobs_info)
            print(f"修改文件: {file_path}")
            print(f"  unobs_info: {old_value} -> {new_unobs_info}")
            modified = True
        else:
            # 如果不存在，添加参数
            yaml_data['wheel_encoder_parameters']['unobs_info'] = float(new_unobs_info)
            print(f"修改文件: {file_path}")
            print(f"  添加unobs_info: {new_unobs_info}")
            modified = True
    
    # 修改enable_debug_recording参数
    if enable_debug_recording is not None and 'output_parameters' in yaml_data:
        if 'enable_debug_recording' in yaml_data['output_parameters']:
            old_value = yaml_data['output_parameters']['enable_debug_recording']
            yaml_data['output_parameters']['enable_debug_recording'] = enable_debug_recording == 'True'
            print(f"修改文件: {file_path}")
            print(f"  enable_debug_recording: {old_value} -> {enable_debug_recording}")
            modified = True
        else:
            # 如果不存在，添加参数
            yaml_data['output_parameters']['enable_debug_recording'] = enable_debug_recording == 'True'
            print(f"修改文件: {file_path}")
            print(f"  添加enable_debug_recording: {enable_debug_recording}")
            modified = True
            
    # 修改estimator_parameters部分的参数
    if 'estimator_parameters' in yaml_data:
        # 修改use_async_processing参数
        if use_async_processing is not None:
            if 'use_async_processing' in yaml_data['estimator_parameters']:
                old_value = yaml_data['estimator_parameters']['use_async_processing']
                yaml_data['estimator_parameters']['use_async_processing'] = use_async_processing == 'True'
                print(f"修改文件: {file_path}")
                print(f"  use_async_processing: {old_value} -> {use_async_processing}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['estimator_parameters']['use_async_processing'] = use_async_processing == 'True'
                print(f"修改文件: {file_path}")
                print(f"  添加use_async_processing: {use_async_processing}")
                modified = True
                
        # 修改online_mode参数
        if online_mode is not None:
            if 'online_mode' in yaml_data['estimator_parameters']:
                old_value = yaml_data['estimator_parameters']['online_mode']
                yaml_data['estimator_parameters']['online_mode'] = online_mode == 'True'
                print(f"修改文件: {file_path}")
                print(f"  online_mode: {old_value} -> {online_mode}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['estimator_parameters']['online_mode'] = online_mode == 'True'
                print(f"修改文件: {file_path}")
                print(f"  添加online_mode: {online_mode}")
                modified = True
                
        # 修改max_batch_size参数
        if max_batch_size is not None:
            if 'max_batch_size' in yaml_data['estimator_parameters']:
                old_value = yaml_data['estimator_parameters']['max_batch_size']
                yaml_data['estimator_parameters']['max_batch_size'] = int(max_batch_size)
                print(f"修改文件: {file_path}")
                print(f"  max_batch_size: {old_value} -> {max_batch_size}")
                modified = True
            else:
                # 如果不存在，添加参数
                yaml_data['estimator_parameters']['max_batch_size'] = int(max_batch_size)
                print(f"修改文件: {file_path}")
                print(f"  添加max_batch_size: {max_batch_size}")
                modified = True
            
    return modified

def modify_yaml_file(file_path, template_data=None, new_frequency=None, use_only_main_camera=None, 
                        new_image_delay=None, new_wheel_delay=None, new_sigma_omega=None, 
                        new_sigma_v=None, new_unobs_info=None,
                        detection_threshold=None, matching_threshold=None, max_num_keypoints=None, 
                        enable_debug_recording=None, use_async_processing=None, online_mode=None,
                        max_batch_size=None, debug=False):
    """修改YAML文件，可以从模板添加缺失参数，并更新指定的参数值"""
    raw_content = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_content = file.read()
    except IOError as e:
        print(f"错误: 读取文件 {file_path} 失败: {e}。跳过此文件。")
        return False

    corrected_yaml_directive = None
    current_yaml_data = {}

    if raw_content:
        lines = raw_content.splitlines(True)
        original_first_line = ""
        processed_first_line = ""

        if lines:
            original_first_line = lines[0].strip()
            # 移除 BOM (如果存在)
            if lines[0].startswith('\ufeff'):
                if debug:
                    print(f"文件 {file_path}: 检测到并移除了BOM字符。")
                processed_first_line = lines[0].lstrip('\ufeff')
            else:
                processed_first_line = lines[0]
            
            if debug:
                print(f"DEBUG: 文件 {file_path}, 第一行 (BOM移除后): '{processed_first_line.strip()}'")

            # 处理YAML指令
            match_directive = re.match(r"^%YAML:(\d+\.\d+)", processed_first_line)
            if match_directive:
                yaml_version = match_directive.group(1)
                line_ending = '\r\n' if processed_first_line.endswith('\r\n') else '\n'
                
                # 使用OpenCV兼容的YAML指令格式 %YAML:1.0
                corrected_first_line_content = f"%YAML:{yaml_version}"
                corrected_full_first_line = f"{corrected_first_line_content}{line_ending}"
                
                if debug:
                    print(f"文件 {file_path}: 保留OpenCV兼容的YAML指令 '{corrected_first_line_content}'")
                
                # 更新第一行
                lines[0] = corrected_full_first_line 
                
                # 添加文档开始标记 '---'
                if len(lines) > 1:
                    indent_match = re.match(r'^(\s*)', lines[1])
                    indent = indent_match.group(1) if indent_match else ""
                    # 检查是否已存在文档分隔符
                    if not lines[1].strip() == '---':
                        lines.insert(1, f"{indent}---{line_ending}")
                        if debug:
                            print(f"文件 {file_path}: 添加文档开始标记 '---'")
                
                raw_content = "".join(lines)
                corrected_yaml_directive = corrected_first_line_content
        
        # 解析YAML内容 - 处理带有文档分隔符的YAML
        yaml_content = raw_content
        if raw_content and raw_content.startswith('%YAML:'):
            # 临时替换YAML指令为PyYAML兼容格式，仅用于解析
            yaml_content = re.sub(r'^%YAML:(\d+\.\d+)', r'%YAML \1', raw_content, count=1)
        
        try:
            # 使用safe_load_all来处理带有文档分隔符的YAML
            docs = list(yaml.safe_load_all(yaml_content))
            # 取第一个文档（如果有多个）
            current_yaml_data = docs[0] if docs else {}
            
            if debug and len(docs) > 1:
                print(f"文件 {file_path}: 包含 {len(docs)} 个YAML文档，使用第一个文档")
                
        except yaml.YAMLError as e:
            print(f"警告: 解析YAML文件 {file_path} 失败: {e}。跳过此文件。")
            return False
    
    # 从模板合并参数
    data_changed = False
    if template_data:
        if debug:
            print(f"正在为 {file_path} 合并模板参数...")
        
        # 合并前复制一份数据用于比较
        import copy
        original_data = copy.deepcopy(current_yaml_data)
        
        # 合并模板数据
        merge_dicts(current_yaml_data, template_data, file_path)
        
        # 检查是否有变化
        data_changed = current_yaml_data != original_data
        
        if debug:
            print(f"模板参数合并完成，{'有' if data_changed else '无'}变化")
            
    # 修改指定的特定参数
    params_modified = modify_specific_parameters_in_dict(
        current_yaml_data, 
        file_path, 
        new_frequency, 
        use_only_main_camera, 
        new_image_delay, 
        new_wheel_delay, 
        new_sigma_omega,
        new_sigma_v,
        new_unobs_info,
        detection_threshold,
        matching_threshold,
        max_num_keypoints,
        enable_debug_recording,
        use_async_processing,
        online_mode,
        max_batch_size
    )

    # 确定是否需要写回文件
    should_write = bool(corrected_yaml_directive) or data_changed or params_modified

    if not current_yaml_data and not corrected_yaml_directive:
        if debug:
            print(f"文件 {file_path} 内容为空且无需修正，不执行写入。")
        return True

    # 写入修改后的文件
    if should_write:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                # 写入OpenCV兼容的YAML指令和文档开始标记（如果需要）
                if corrected_yaml_directive:
                    file.write(f"%YAML:1.0\n")  # 固定使用OpenCV兼容的格式
                    file.write("---\n")
                else:
                    # 如果没有检测到YAML指令，添加OpenCV兼容的指令
                    file.write("%YAML:1.0\n")
                    file.write("---\n")

                # 写入YAML数据
                if current_yaml_data:
                    write_yaml_with_template_format(file, current_yaml_data)
            
            if debug:
                reasons = []
                if corrected_yaml_directive: reasons.append("YAML指令已修正为OpenCV兼容格式")
                if data_changed: reasons.append("模板参数已合并")
                if params_modified: reasons.append("特定参数已修改")
                print(f"文件 {file_path} 已更新: {'; '.join(reasons)}")
        except Exception as e:
            print(f"错误: 写入文件 {file_path} 失败: {e}")
            return False
    elif debug:
        print(f"文件 {file_path} 无需修改。")
        
    return True

def write_yaml_with_template_format(file, data):
    """使用直接写入的方式将数据写入YAML文件，保持与模板相同的格式
    
    Args:
        file: 要写入的文件对象
        data: YAML数据字典
    """
    # 写入cameras部分
    if 'cameras' in data:
        file.write("cameras:\n")
        for i, camera in enumerate(data['cameras']):
            # 写入相机定义开始
            file.write(f"     - {{T_SC:\n")
            
            # 写入T_SC矩阵，格式化为4行4列
            T_SC = camera['T_SC']
            file.write(f"        [{T_SC[0]:.8f}, {T_SC[1]:.8f}, {T_SC[2]:.8f}, {T_SC[3]:.8f},\n")
            file.write(f"        {T_SC[4]:.8f}, {T_SC[5]:.8f}, {T_SC[6]:.8f}, {T_SC[7]:.8f},\n")
            file.write(f"        {T_SC[8]:.8f}, {T_SC[9]:.8f}, {T_SC[10]:.8f}, {T_SC[11]:.8f},\n")
            file.write(f"        {T_SC[12]:.8f}, {T_SC[13]:.8f}, {T_SC[14]:.8f}, {T_SC[15]:.8f}],\n")
            
            # 写入image_dimension
            file.write(f"        image_dimension: [{camera['image_dimension'][0]}, {camera['image_dimension'][1]}],\n")
            
            # 写入distortion_coefficients
            _write_distortion_coefficients(file, camera['distortion_coefficients'])
            
            # 写入其他相机参数
            file.write(f"        distortion_type: {camera['distortion_type']},\n")
            file.write(f"        focal_length: [{camera['focal_length'][0]:.14f}, {camera['focal_length'][1]:.14f}],\n")
            file.write(f"        principal_point: [{camera['principal_point'][0]:.14f}, {camera['principal_point'][1]:.14f}],\n")
            
            # 写入camera_type
            camera_type = camera.get('camera_type', 'gray')
            file.write(f"        camera_type: {camera_type}, #gray, rgb, gray+depth, rgb+depth\n")
            
            # 写入slam_use，并添加注释
            slam_use = camera.get('slam_use', 'okvis')
            is_first_camera = (i == 0)
            comment = " #none, okvis, okvis-depth, okvis-virtual" if is_first_camera else ""
            file.write(f"        slam_use: {slam_use}}}{comment}\n")
            
            # 在第一个相机后不添加空行，在第二个相机后添加空行
            if not is_first_camera:
                file.write("\n")
    
    # 写入其他参数部分
    _write_parameter_sections(file, data)

def _write_distortion_coefficients(file, dist_coef):
    """写入畸变系数，格式化输出"""
    if len(dist_coef) <= 4:
        # 对于少于等于4个系数的情况，单行输出
        dist_str = ", ".join([f"{c:.16f}" for c in dist_coef])
        file.write(f"        distortion_coefficients: [{dist_str}],\n")
    else:
        # 对于多于4个系数的情况，每4个一行
        file.write("        distortion_coefficients: [")
        for j in range(len(dist_coef)):
            if j > 0:
                if j % 4 == 0:
                    file.write(",\n            ")
                else:
                    file.write(", ")
            file.write(f"{dist_coef[j]:.16f}")
        file.write("],\n")

def _write_parameter_sections(file, data):
    """写入其他参数部分（camera_parameters, imu_parameters等）"""
    for section in ["camera_parameters", "imu_parameters", "wheel_encoder_parameters",
                   "frontend_parameters", "estimator_parameters", "output_parameters"]:
        if section in data:
            file.write(f"\n# {section.replace('_', ' ')}\n")
            file.write(f"{section}:\n")
            
            # 特殊处理wheel_encoder_parameters中的T_BS
            if section == "wheel_encoder_parameters" and "T_BS" in data[section]:
                _write_wheel_encoder_section(file, data[section])
            # 特殊处理imu_parameters中的T_BS
            elif section == "imu_parameters" and "T_BS" in data[section]:
                _write_imu_section(file, data[section])
            else:
                # 写入普通参数
                _write_regular_parameters(file, data[section])

def _write_wheel_encoder_section(file, wheel_data):
    """写入wheel_encoder_parameters部分，特殊处理T_BS矩阵"""
    # 先写入除T_BS外的其他参数
    for key, value in wheel_data.items():
        if key != "T_BS":
            file.write(f"    {key}: {value}\n")
            
    # 单独写入T_BS矩阵
    T_BS = wheel_data["T_BS"]
    file.write(f"    # transform Body-Sensor (WheelEncoder)\n")
    file.write(f"    T_BS:\n")
    file.write(f"        [{T_BS[0]:.8f}, {T_BS[1]:.8f}, {T_BS[2]:.8f}, {T_BS[3]:.8f},\n")
    file.write(f"         {T_BS[4]:.8f}, {T_BS[5]:.8f}, {T_BS[6]:.8f}, {T_BS[7]:.8f},\n")
    file.write(f"         {T_BS[8]:.8f}, {T_BS[9]:.8f}, {T_BS[10]:.8f}, {T_BS[11]:.8f},\n")
    file.write(f"         {T_BS[12]:.8f}, {T_BS[13]:.8f}, {T_BS[14]:.8f}, {T_BS[15]:.8f}]\n")

def _write_imu_section(file, imu_data):
    """写入imu_parameters部分，特殊处理T_BS矩阵"""
    # 先写入除T_BS外的其他参数
    for key, value in imu_data.items():
        if key != "T_BS":
            if isinstance(value, list):
                file.write(f"    {key}: [ {', '.join([str(x) for x in value])} ]\n")
            else:
                file.write(f"    {key}: {value}\n")
                
    # 单独写入T_BS矩阵
    T_BS = imu_data["T_BS"]
    file.write(f"    # transform Body-Sensor (IMU)\n")
    file.write(f"    T_BS:\n")
    file.write(f"        [{T_BS[0]:.4f}, {T_BS[1]:.4f}, {T_BS[2]:.4f}, {T_BS[3]:.4f},\n")
    file.write(f"         {T_BS[4]:.4f}, {T_BS[5]:.4f}, {T_BS[6]:.4f}, {T_BS[7]:.4f},\n")
    file.write(f"         {T_BS[8]:.4f}, {T_BS[9]:.4f}, {T_BS[10]:.4f}, {T_BS[11]:.4f},\n")
    file.write(f"         {T_BS[12]:.4f}, {T_BS[13]:.4f}, {T_BS[14]:.4f}, {T_BS[15]:.4f}]\n")

def _write_regular_parameters(file, section_data):
    """写入普通参数，处理不同类型和添加注释"""
    for key, value in section_data.items():
        if isinstance(value, dict):
            # 写入嵌套字典
            file.write(f"    {key}:\n")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    file.write(f"        {subkey}: [ {', '.join([str(x) for x in subvalue])} ]\n")
                else:
                    file.write(f"        {subkey}: {subvalue}\n")
        elif isinstance(value, list):
            # 写入列表
            file.write(f"    {key}: [ {', '.join([str(x) for x in value])} ]\n")
        else:
            # 处理特殊参数，添加注释
            if key == "debug_output_dir":
                file.write(f'    {key}: "{value}" # directory for debug output files\n')
            elif key == "wheel_delay":
                file.write(f"    {key}: {value} # [s] timestamp_wheel_correct = timestamp_wheel - wheel_delay\n")
            elif key == "sigma_omega":
                file.write(f"    {key}: {value} # angular velocity noise [rad/s]\n")
            elif key == "sigma_v":
                file.write(f"    {key}: {value} # linear velocity noise [m/s]\n")
            elif key == "unobs_info":
                file.write(f"    {key}: {value} # unobservable state information matrix value\n")
            elif key == "use_only_main_camera":
                file.write(f"    {key}: {str(value).lower()} # if true, only camera0 is used for matchToMap and other operations (except matchStereo)\n")
            elif key == "image_delay":
                file.write(f"    {key}: {value} # [s] timestamp_camera_correct = timestamp_camera - image_delay\n")
            elif key == "detection_threshold":
                file.write(f"    {key}: {value} # detection threshold. By default the uniformity radius in pixels\n")
            elif key == "matching_threshold":
                file.write(f"    {key}: {value} # BRISK descriptor matching threshold\n")
            elif key == "max_num_keypoints":
                file.write(f"    {key}: {value} # restrict to a maximum of this many keypoints per image (strongest ones)\n")
            elif key == "enable_debug_recording":
                file.write(f"    {key}: {str(value).lower()} # enable debug recording of frontend/backend packages\n")
            elif key == "use_async_processing":
                file.write(f"    {key}: {str(value).lower()} # enable asynchronous frontend/backend processing\n")
            elif key == "online_mode":
                file.write(f"    {key}: {str(value).lower()} # whether to run in online mode\n")
            elif key == "max_batch_size":
                file.write(f"    {key}: {value} # maximum number of frontend packages to batch for backend processing\n")
            else:
                # 普通参数
                file.write(f"    {key}: {value}\n")

def main():
    """主函数：解析命令行参数并处理YAML文件"""
    parser = argparse.ArgumentParser(description='批量修改YAML文件中的参数，从模板补充缺失参数。')
    parser.add_argument('root_dir', help='包含待处理YAML文件的根目录路径')
    parser.add_argument('--template-file', type=str, help='模板YAML文件路径，用于补充目标文件中缺失的参数')
    parser.add_argument('--frequency', type=float, help='设置image_frequency参数值')
    parser.add_argument('--use-only-main-camera', choices=['True', 'False'], help='设置use_only_main_camera参数值')
    parser.add_argument('--image-delay', type=float, help='设置image_delay参数值')
    parser.add_argument('--wheel-delay', type=float, help='设置wheel_delay参数值')
    parser.add_argument('--sigma-omega', type=float, help='设置sigma_omega参数值')
    parser.add_argument('--sigma-v', type=float, help='设置sigma_v参数值')
    parser.add_argument('--unobs-info', type=float, help='设置unobs_info参数值')
    parser.add_argument('--detection-threshold', type=float, help='设置detection_threshold参数值')
    parser.add_argument('--matching-threshold', type=float, help='设置matching_threshold参数值')
    parser.add_argument('--max-num-keypoints', type=int, help='设置max_num_keypoints参数值')
    parser.add_argument('--enable-debug-recording', choices=['True', 'False'], help='设置enable_debug_recording参数值')
    parser.add_argument('--use-async-processing', choices=['True', 'False'], help='设置use_async_processing参数值')
    parser.add_argument('--online-mode', choices=['True', 'False'], help='设置online_mode参数值')
    parser.add_argument('--max-batch-size', type=int, help='设置max_batch_size参数值')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，打印详细日志')
    parser.add_argument('--format-only', action='store_true', help='仅格式化文件，不修改参数')
    args = parser.parse_args()

    # 打印修改原则说明
    print("YAML文件修改原则:")
    print("1. 模板文件只会添加目标文件中缺失的参数，不会覆盖已有参数")
    print("2. 命令行参数(如--sigma-omega)将会覆盖目标文件中对应的参数值")
    print("3. 输出YAML文件将保持与模板相同的美观格式")
    print("4. 将使用OpenCV兼容的YAML指令格式(%YAML:1.0)\n")
    
    # 加载模板文件
    template_data = None
    if args.template_file:
        if not os.path.exists(args.template_file):
            print(f"错误: 模板文件 {args.template_file} 不存在，将不使用模板。")
        else:
            try:
                # 读取模板文件内容
                with open(args.template_file, 'r', encoding='utf-8') as tf:
                    template_content = tf.read()
                
                # 处理模板文件YAML指令
                template_yaml_content = template_content
                if template_content.startswith('%YAML:'):
                    # 临时将YAML指令修改为PyYAML兼容格式，仅用于解析
                    template_yaml_content = re.sub(r'^%YAML:(\d+\.\d+)', r'%YAML \1', template_content, count=1)
                    if args.debug:
                        print(f"模板文件: 临时将YAML指令转换为PyYAML兼容格式进行解析")
                
                # 解析模板YAML内容，使用safe_load_all处理可能的多文档YAML
                try:
                    docs = list(yaml.safe_load_all(template_yaml_content))
                    template_data = docs[0] if docs else {}
                    
                    if args.debug and len(docs) > 1:
                        print(f"模板文件: 包含 {len(docs)} 个YAML文档，使用第一个文档")
                    
                    if template_data is None:
                        template_data = {}
                        print(f"警告: 模板文件 {args.template_file} 内容为空或无效。")
                    elif args.debug:
                        print(f"成功加载模板文件 {args.template_file}")
                except yaml.YAMLError as e:
                    print(f"错误: 解析模板文件 {args.template_file} 失败: {e}。将不使用模板。")
                    template_data = None
            except Exception as e:
                print(f"错误: 处理模板文件 {args.template_file} 失败: {e}。将不使用模板。")
                template_data = None

    # 检查是否有任何修改参数
    if (args.frequency is None and args.use_only_main_camera is None and 
        args.image_delay is None and args.wheel_delay is None and 
        args.sigma_omega is None and args.sigma_v is None and
        args.unobs_info is None and args.detection_threshold is None and
        args.matching_threshold is None and args.max_num_keypoints is None and
        args.enable_debug_recording is None and args.use_async_processing is None and
        args.online_mode is None and args.max_batch_size is None and
        template_data is None and not args.format_only):
        parser.error("请至少指定一个要修改的参数或提供模板文件。")
    
    # 查找需要处理的YAML文件
    yaml_files = find_yaml_files(args.root_dir)
    
    if not yaml_files:
        print(f"未在 {args.root_dir} 及其子文件夹中找到任何*_okvis.yaml文件")
        return
    
    print(f"找到 {len(yaml_files)} 个*_okvis.yaml文件")
    
    # 处理每个YAML文件
    success_count = 0
    for file_path in yaml_files:
        if args.debug:
            print(f"\n处理文件: {file_path}")
            
        # 修改YAML文件
        if modify_yaml_file(
            file_path, 
            template_data, 
            args.frequency, 
            args.use_only_main_camera, 
            args.image_delay, 
            args.wheel_delay, 
            args.sigma_omega,
            args.sigma_v,
            args.unobs_info,
            args.detection_threshold,
            args.matching_threshold,
            args.max_num_keypoints,
            args.enable_debug_recording,
            args.use_async_processing,
            args.online_mode,
            args.max_batch_size,
            args.debug
        ):
            success_count += 1
        elif args.debug:
            print(f"文件 {file_path} 处理失败。")
            
    print(f"\n成功处理文件数: {success_count}/{len(yaml_files)}")

if __name__ == "__main__":
    main()