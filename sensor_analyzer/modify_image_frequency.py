#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量修改子文件夹中的xxx_okvis.yaml文件中的image_frequency参数、use_only_main_camera参数、
image_delay参数、wheel_delay参数和sigma_omega参数
"""

import os
import re
import argparse

def find_yaml_files(root_dir):
    """查找所有以_okvis.yaml结尾的文件"""
    yaml_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_okvis.yaml'):
                yaml_files.append(os.path.join(dirpath, filename))
    return yaml_files

def modify_use_only_main_camera(file_path, new_value):
    """修改YAML文件中的use_only_main_camera参数"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 查找frontend_parameters部分
    frontend_params_match = re.search(r'frontend_parameters\s*:\s*\n', content)
    if not frontend_params_match:
        print(f"警告: 无法在文件 {file_path} 中找到 frontend_parameters 部分")
        return False
    
    # 查找use_only_main_camera参数
    pattern = r'(use_only_main_camera\s*:\s*)(True|False)'
    match = re.search(pattern, content, re.IGNORECASE)
    
    if match:
        # 找到use_only_main_camera参数，直接替换
        old_value = match.group(2)
        new_content = re.sub(pattern, f'\\1{new_value}', content, flags=re.IGNORECASE)
        
        # 写回文件
        with open(file_path, 'w') as file:
            file.write(new_content)
            
        print(f"修改文件: {file_path}")
        print(f"  use_only_main_camera: {old_value} -> {new_value}")
        return True
    else:
        # 没有找到use_only_main_camera参数，尝试添加
        lines = content.split('\n')
        frontend_params_start = 0
        for i, line in enumerate(lines):
            if re.search(r'frontend_parameters\s*:', line):
                frontend_params_start = i
                break
        
        # 查找frontend_parameters部分的最后一个参数
        last_param_line = 0
        for i in range(frontend_params_start + 1, len(lines)):
            if i + 1 >= len(lines) or not lines[i + 1].strip().startswith(('detection', 'absolute', 'matching', 'octaves', 'max_num', 'keyframe', 'use_cnn', 'parallelise', 'num_matching')):
                last_param_line = i
                break
        
        if last_param_line > 0:
            # 获取缩进
            indent = re.match(r'(\s*)', lines[last_param_line]).group(1)
            # 在最后一个参数后添加use_only_main_camera
            lines.insert(last_param_line + 1, f"{indent}use_only_main_camera: {new_value}")
            
            # 写回文件
            with open(file_path, 'w') as file:
                file.write('\n'.join(lines))
                
            print(f"修改文件: {file_path}")
            print(f"  添加use_only_main_camera: {new_value}")
            return True
        else:
            print(f"警告: 无法在文件 {file_path} 中找到适合添加 use_only_main_camera 的位置")
            return False

def modify_image_delay(file_path, new_delay):
    """修改YAML文件中的image_delay参数"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 查找camera_parameters部分
    camera_params_match = re.search(r'camera_parameters\s*:\s*\n', content)
    if not camera_params_match:
        print(f"警告: 无法在文件 {file_path} 中找到 camera_parameters 部分")
        return False
    
    # 尝试转换为浮点数
    try:
        float_delay = float(new_delay)
        
        # 决定最终写入的字符串格式
        if float_delay == 0.0:
            final_value_str = "0.0"  # 直接为0.0指定输出
        else:
            # 对于非0.0的值，使用之前的精度和去除逻辑
            temp_formatted = f"{float_delay:.17f}"
            stripped_val = temp_formatted.rstrip('0').rstrip('.')
            # 处理 stripping 可能产生空字符串或无效格式的情况 (尽管对于非零值不太可能)
            if stripped_val == "" or stripped_val == "-" or stripped_val == "." or stripped_val == "-.":
                final_value_str = temp_formatted # 异常情况回退到完整精度格式
            else:
                final_value_str = stripped_val
        
        # 查找image_delay参数
        # 使用更健壮的正则匹配浮点数，包括科学计数法
        pattern = r'(image_delay\s*:\s*)(-?\d+\.?\d*(?:[eE][-+]?\d+)?)'
        match = re.search(pattern, content)
        
        if match:
            # 找到image_delay参数，直接替换
            old_delay_str = match.group(2)
            new_content = re.sub(pattern, rf'\g<1>{final_value_str}', content)
            
            # 写回文件
            if content != new_content:
                with open(file_path, 'w') as file:
                    file.write(new_content)
            else:
                # 如果内容没有变化，也打印信息，表示值可能已经是目标值
                pass # 之前这里没有打印，可以考虑添加，但为了最小化更改，暂时保留
                
            print(f"修改文件: {file_path}")
            print(f"  image_delay: {old_delay_str} -> {final_value_str}")
            return True
        else:
            # 没有找到image_delay参数，尝试添加
            lines = content.split('\n')
            camera_params_start = 0
            for i, line in enumerate(lines):
                if re.search(r'camera_parameters\s*:', line):
                    camera_params_start = i
                    break
            
            image_delay_added = False
            for i in range(camera_params_start + 1, len(lines)):
                if 'sync_cameras:' in lines[i]:
                    indent = re.match(r'(\s*)', lines[i]).group(1)
                    lines.insert(i + 1, f"{indent}image_delay: {final_value_str}")
                    image_delay_added = True
                    break
            
            if image_delay_added:
                with open(file_path, 'w') as file:
                    file.write('\n'.join(lines))
                print(f"修改文件: {file_path}")
                print(f"  添加image_delay: {final_value_str}")
                return True
    except ValueError:
        print(f"错误: 提供的delay值 '{new_delay}' 不是有效的数值")
        return False

def modify_wheel_delay(file_path, new_delay):
    """修改YAML文件中的wheel_delay参数"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 查找wheel_encoder_parameters部分
    wheel_params_match = re.search(r'wheel_encoder_parameters\s*:\s*\n', content)
    if not wheel_params_match:
        print(f"警告: 无法在文件 {file_path} 中找到 wheel_encoder_parameters 部分")
        return False
    
    # 尝试转换为浮点数
    try:
        float_delay = float(new_delay)
        formatted_delay = f"{float_delay}"
        
        # 查找wheel_delay参数
        pattern = r'(wheel_delay\s*:\s*)(-?[0-9.]+)'
        match = re.search(pattern, content)
        
        if match:
            # 找到wheel_delay参数，直接替换
            old_delay = float(match.group(2))
            new_content = re.sub(pattern, rf'\g<1>{formatted_delay}', content)
            
            # 写回文件
            with open(file_path, 'w') as file:
                file.write(new_content)
                
            print(f"修改文件: {file_path}")
            print(f"  wheel_delay: {old_delay} -> {formatted_delay}")
            return True
        else:
            # 没有找到wheel_delay参数，尝试添加
            lines = content.split('\n')
            wheel_params_start = 0
            for i, line in enumerate(lines):
                if re.search(r'wheel_encoder_parameters\s*:', line):
                    wheel_params_start = i
                    break
            
            # 查找合适的位置添加wheel_delay
            wheel_delay_added = False
            for i in range(wheel_params_start + 1, len(lines)):
                if 'use:' in lines[i]:
                    # 获取缩进
                    indent = re.match(r'(\s*)', lines[i]).group(1)
                    # 在use:后面添加wheel_delay
                    lines.insert(i + 1, f"{indent}wheel_delay: {formatted_delay} # [s] timestamp_wheel_correct = timestamp_wheel - wheel_delay")
                    wheel_delay_added = True
                    break
            
            if wheel_delay_added:
                # 写回文件
                with open(file_path, 'w') as file:
                    file.write('\n'.join(lines))
                    
                print(f"修改文件: {file_path}")
                print(f"  添加wheel_delay: {formatted_delay}")
                return True
            else:
                print(f"警告: 无法在文件 {file_path} 中找到适合添加 wheel_delay 的位置")
                return False
    except ValueError:
        print(f"错误: 提供的delay值 '{new_delay}' 不是有效的数值")
        return False

def modify_sigma_omega(file_path, new_value=100.0):
    """修改YAML文件中的sigma_omega参数"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 查找wheel_encoder_parameters部分
    wheel_params_match = re.search(r'wheel_encoder_parameters\s*:\s*\n', content)
    if not wheel_params_match:
        print(f"警告: 无法在文件 {file_path} 中找到 wheel_encoder_parameters 部分")
        return False
    
    # 尝试转换为浮点数
    try:
        float_value = float(new_value)
        formatted_value = f"{float_value}"
        
        # 查找sigma_omega参数（正常格式）
        pattern = r'(sigma_omega\s*:\s*)([0-9.]+)'
        match = re.search(pattern, content)
        
        if match:
            # 找到sigma_omega参数，直接替换
            old_value = match.group(2)
            new_content = re.sub(pattern, rf'\g<1>{formatted_value}', content)
            
            # 写回文件
            with open(file_path, 'w') as file:
                file.write(new_content)
                
            print(f"修改文件: {file_path}")
            print(f"  sigma_omega: {old_value} -> {formatted_value}")
            return True
        else:
            # 没有找到正常格式的sigma_omega参数，检查异常格式或添加新参数
            lines = content.split('\n')
            wheel_params_start = 0
            for i, line in enumerate(lines):
                if re.search(r'wheel_encoder_parameters\s*:', line):
                    wheel_params_start = i
                    break
            
            # 查找异常格式的sigma_omega值（如H0.0）
            abnormal_value_found = False
            for i in range(wheel_params_start + 1, len(lines)):
                # 检查是否有异常格式（如H0.0）的行
                if re.match(r'\s+[A-Za-z]+[0-9.]+\s*$', lines[i]):
                    # 如果这行在sigma_v之后
                    if i > 0 and 'sigma_v:' in lines[i-1]:
                        # 替换异常行为正确的sigma_omega
                        indent = re.match(r'(\s*)', lines[i]).group(1)
                        lines[i] = f"{indent}sigma_omega: {formatted_value}"
                        abnormal_value_found = True
                        
                        # 写回文件
                        with open(file_path, 'w') as file:
                            file.write('\n'.join(lines))
                            
                        print(f"修改文件: {file_path}")
                        print(f"  修复并设置sigma_omega: 异常值 -> {formatted_value}")
                        return True
            
            # 如果没找到异常值，查找sigma_v参数的位置，在其后添加sigma_omega
            if not abnormal_value_found:
                sigma_omega_added = False
                for i in range(wheel_params_start + 1, len(lines)):
                    if 'sigma_v:' in lines[i]:
                        # 获取缩进
                        indent = re.match(r'(\s*)', lines[i]).group(1)
                        # 在sigma_v后面添加sigma_omega
                        lines.insert(i + 1, f"{indent}sigma_omega: {formatted_value}")
                        sigma_omega_added = True
                        break
                
                if sigma_omega_added:
                    # 写回文件
                    with open(file_path, 'w') as file:
                        file.write('\n'.join(lines))
                        
                    print(f"修改文件: {file_path}")
                    print(f"  添加sigma_omega: {formatted_value}")
                    return True
                else:
                    print(f"警告: 无法在文件 {file_path} 中找到适合添加 sigma_omega 的位置")
                    return False
    except ValueError:
        print(f"错误: 提供的值 '{new_value}' 不是有效的数值")
        return False

def modify_yaml_file(file_path, new_frequency=None, use_only_main_camera=None, new_image_delay=None, new_wheel_delay=None, new_sigma_omega=None):
    """修改YAML文件中的参数"""
    result = True
    
    # 修改image_frequency参数（如果提供）
    if new_frequency is not None:
        # 读取文件内容
        with open(file_path, 'r') as file:
            content = file.read()
        
        # 查找camera_parameters部分
        camera_params_match = re.search(r'camera_parameters\s*:\s*\n', content)
        if not camera_params_match:
            print(f"警告: 无法在文件 {file_path} 中找到 camera_parameters 部分")
            result = False
        else:
            # 确保new_frequency是有效的数值格式
            try:
                # 将new_frequency转换为浮点数，确保格式正确
                float_frequency = float(new_frequency)
                # 使用字符串格式化确保YAML格式正确
                formatted_frequency = f"{float_frequency}"
                
                # 正常情况下查找image_frequency参数
                pattern = r'(image_frequency\s*:\s*)([0-9.]+)'
                match = re.search(pattern, content)
                
                if match:
                    # 找到正常的image_frequency参数，直接替换
                    old_frequency = float(match.group(2))
                    new_content = re.sub(pattern, rf'\g<1>{formatted_frequency}', content)
                    
                    # 写回文件
                    with open(file_path, 'w') as file:
                        file.write(new_content)
                        
                    print(f"修改文件: {file_path}")
                    print(f"  image_frequency: {old_frequency} -> {formatted_frequency}")
                else:
                    # 没有找到正常的image_frequency参数，检查是否有异常格式
                    lines = content.split('\n')
                    camera_params_start = 0
                    for i, line in enumerate(lines):
                        if re.search(r'camera_parameters\s*:', line):
                            camera_params_start = i
                            break
                    
                    # 查找online_calibration部分的结束位置
                    online_calib_end = 0
                    for i in range(camera_params_start, len(lines)):
                        if 'sigma_alpha:' in lines[i]:
                            online_calib_end = i
                            break
                    
                    # 检查是否有异常格式的行（如J.0）
                    abnormal_found = False
                    for i in range(online_calib_end + 1, min(online_calib_end + 10, len(lines))):
                        # 检查是否有独立的异常值行（如J.0）
                        if re.match(r'\s+[A-Za-z]+\.[0-9]+\s*$', lines[i]):
                            # 替换异常行
                            indent = re.match(r'(\s*)', lines[i]).group(1)
                            lines[i] = f"{indent}image_frequency: {formatted_frequency}"
                            abnormal_found = True
                            print(f"修改文件: {file_path}")
                            print(f"  修复并设置image_frequency: 异常值 -> {formatted_frequency}")
                            break
                    
                    # 如果没找到异常值，在start_time前添加image_frequency
                    if not abnormal_found:
                        for i in range(online_calib_end + 1, min(online_calib_end + 10, len(lines))):
                            if 'start_time:' in lines[i]:
                                # 获取缩进
                                indent = re.match(r'(\s*)', lines[i]).group(1)
                                # 在start_time前插入image_frequency
                                lines.insert(i, f"{indent}image_frequency: {formatted_frequency}")
                                abnormal_found = True
                                print(f"修改文件: {file_path}")
                                print(f"  添加image_frequency: {formatted_frequency}")
                                break
                    
                    if abnormal_found:
                        # 写回文件
                        with open(file_path, 'w') as file:
                            file.write('\n'.join(lines))
                    else:
                        print(f"警告: 无法在文件 {file_path} 中找到适合添加 image_frequency 的位置")
                        result = False
            except ValueError:
                print(f"错误: 提供的frequency值 '{new_frequency}' 不是有效的数值")
                result = False
    
    # 修改use_only_main_camera参数（如果提供）
    if use_only_main_camera is not None:
        if not modify_use_only_main_camera(file_path, use_only_main_camera):
            result = False
    
    # 修改image_delay参数（如果提供）
    if new_image_delay is not None:
        if not modify_image_delay(file_path, new_image_delay):
            result = False
    
    # 修改wheel_delay参数（如果提供）
    if new_wheel_delay is not None:
        if not modify_wheel_delay(file_path, new_wheel_delay):
            result = False
    
    # 修改sigma_omega参数（如果提供）
    if new_sigma_omega is not None:
        if not modify_sigma_omega(file_path, new_sigma_omega):
            result = False
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量修改xxx_okvis.yaml文件中的参数')
    parser.add_argument('root_dir', help='包含子文件夹的根目录路径')
    parser.add_argument('--frequency', type=float, help='新的image_frequency值')
    parser.add_argument('--use-only-main-camera', choices=['True', 'False'], help='设置use_only_main_camera参数值')
    parser.add_argument('--image-delay', type=float, help='新的image_delay值')
    parser.add_argument('--wheel-delay', type=float, help='新的wheel_delay值')
    parser.add_argument('--sigma-omega', type=float, help='新的sigma_omega值，默认为100.0')
    parser.add_argument('--debug', action='store_true', help='打印更多调试信息')
    args = parser.parse_args()
    
    # 检查是否至少提供了一个要修改的参数
    if (args.frequency is None and args.use_only_main_camera is None and 
        args.image_delay is None and args.wheel_delay is None and args.sigma_omega is None):
        parser.error("请至少指定一个要修改的参数: --frequency, --use-only-main-camera, --image-delay, --wheel-delay 或 --sigma-omega")
    
    # 查找所有yaml文件
    yaml_files = find_yaml_files(args.root_dir)
    
    if not yaml_files:
        print(f"未在 {args.root_dir} 及其子文件夹中找到任何*_okvis.yaml文件")
        return
    
    print(f"找到 {len(yaml_files)} 个*_okvis.yaml文件")
    
    # 修改文件
    success_count = 0
    for file_path in yaml_files:
        if modify_yaml_file(file_path, args.frequency, args.use_only_main_camera, 
                           args.image_delay, args.wheel_delay, args.sigma_omega):
            success_count += 1
    
    print(f"成功修改 {success_count}/{len(yaml_files)} 个文件")

if __name__ == "__main__":
    main()