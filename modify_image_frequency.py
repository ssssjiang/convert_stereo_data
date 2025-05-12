#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量修改子文件夹中的xxx_okvis.yaml文件中的image_frequency参数
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

def modify_yaml_file(file_path, new_frequency):
    """修改YAML文件中的image_frequency参数"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 查找camera_parameters部分
    camera_params_match = re.search(r'camera_parameters\s*:\s*\n', content)
    if not camera_params_match:
        print(f"警告: 无法在文件 {file_path} 中找到 camera_parameters 部分")
        return False
    
    # 正常情况下查找image_frequency参数
    pattern = r'(image_frequency\s*:\s*)([0-9.]+)'
    match = re.search(pattern, content)
    
    if match:
        # 找到正常的image_frequency参数，直接替换
        old_frequency = float(match.group(2))
        new_content = re.sub(pattern, f'\\1{new_frequency}', content)
        
        # 写回文件
        with open(file_path, 'w') as file:
            file.write(new_content)
            
        print(f"修改文件: {file_path}")
        print(f"  image_frequency: {old_frequency} -> {new_frequency}")
        return True
    else:
        # 没有找到正常的image_frequency参数，检查是否有异常格式
        # 查找在camera_parameters部分中，在online_calibration后面的行
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
        
        # 检查接下来的几行是否有异常的image_frequency或者缺少image_frequency
        inserted = False
        for i in range(online_calib_end + 1, min(online_calib_end + 5, len(lines))):
            # 如果是start_time行前面没有image_frequency，则插入
            if 'start_time:' in lines[i] and 'image_frequency:' not in lines[i-1]:
                # 检查前一行是否有异常值（如J.0）
                if re.match(r'\s+[A-Za-z]+\.[0-9]+\s*$', lines[i-1]):
                    # 替换异常行
                    lines[i-1] = f"    image_frequency: {new_frequency}"
                    inserted = True
                    print(f"修改文件: {file_path}")
                    print(f"  修复并设置image_frequency: 异常值 -> {new_frequency}")
                    break
                else:
                    # 在start_time前插入image_frequency
                    indent = re.match(r'(\s*)', lines[i]).group(1)
                    lines.insert(i, f"{indent}image_frequency: {new_frequency}")
                    inserted = True
                    print(f"修改文件: {file_path}")
                    print(f"  添加image_frequency: {new_frequency}")
                    break
        
        if inserted:
            # 写回文件
            with open(file_path, 'w') as file:
                file.write('\n'.join(lines))
            return True
        else:
            print(f"警告: 无法在文件 {file_path} 中找到适合添加 image_frequency 的位置")
            return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量修改xxx_okvis.yaml文件中的image_frequency参数')
    parser.add_argument('root_dir', help='包含子文件夹的根目录路径')
    parser.add_argument('--frequency', type=float, required=True, help='新的image_frequency值')
    parser.add_argument('--debug', action='store_true', help='打印更多调试信息')
    args = parser.parse_args()
    
    # 查找所有yaml文件
    yaml_files = find_yaml_files(args.root_dir)
    
    if not yaml_files:
        print(f"未在 {args.root_dir} 及其子文件夹中找到任何*_okvis.yaml文件")
        return
    
    print(f"找到 {len(yaml_files)} 个*_okvis.yaml文件")
    
    # 修改文件
    success_count = 0
    for file_path in yaml_files:
        if modify_yaml_file(file_path, args.frequency):
            success_count += 1
    
    print(f"成功修改 {success_count}/{len(yaml_files)} 个文件")

if __name__ == "__main__":
    main()