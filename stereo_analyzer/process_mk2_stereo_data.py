#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import shutil
import argparse
import subprocess
import re
import json
from pathlib import Path
from datetime import datetime

# 添加大小写不敏感的字典类，用于处理JSON文件中大写键名
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
    parser = argparse.ArgumentParser(description="处理MK2样机的双目相机数据")
    parser.add_argument('--csv', type=str, required=True,
                        help="MK2样机清单CSV文件路径")
    parser.add_argument('--root', type=str, required=True,
                        help="包含双目编号子文件夹的根目录路径")
    parser.add_argument('--format', type=str, choices=['sensor', 'okvis', 'all'], default='sensor',
                        help="输出格式: 'sensor', 'okvis', 或 'all'同时输出两种格式")
    parser.add_argument('--sensor_template', type=str, default=None,
                        help="用于生成sensor.yaml的模板文件路径")
    parser.add_argument('--okvis_template', type=str, default=None,
                        help="用于生成okvis.yaml的模板文件路径")
    parser.add_argument('--keep-original', action='store_true',
                        help="保留原始文件夹（默认会重命名）")
    parser.add_argument('--report', type=str, default=None,
                        help="生成处理报告的文件路径")
    return parser.parse_args()

# 添加用于读取和处理JSON的辅助函数
def read_json_file(file_path):
    """读取JSON文件并返回大小写不敏感的字典"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return CaseInsensitiveDict(data)
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None

def read_correspondence_from_csv(csv_path):
    """从CSV文件中读取整机编号和双目编号的对应关系"""
    correspondence = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # 跳过标题行
            next(reader)
            
            for row in reader:
                if len(row) >= 5:
                    machine_id = row[0].strip()  # 整机编号
                    stereo_id = row[4].strip()   # 双目编号
                    
                    # 确保两个ID都不为空
                    if machine_id and stereo_id:
                        # 处理machine_id，保证命名一致性
                        # 如果machine_id不以"MK2-"开头，且不是纯数字，则规范化为"MK2-xxx"格式
                        if not machine_id.startswith("MK2-") and not machine_id.isdigit():
                            if machine_id.startswith("MK2"):
                                # 如果是"MK2xxx"格式，修改为"MK2-xxx"
                                machine_id = "MK2-" + machine_id[3:]
                        
                        print(f"读取对应关系: {stereo_id} -> {machine_id}")
                        
                        # 存储原始ID
                        correspondence[stereo_id] = machine_id
                        
                        # 如果是数字格式，处理各种变体
                        if stereo_id.isdigit():
                            # 1. 去除前导零的版本
                            stripped_id = stereo_id.lstrip('0')
                            if stripped_id == '':  # 如果全是0，至少保留一个0
                                stripped_id = '0'
                            correspondence[stripped_id] = machine_id
                            
                            # 2. 添加前导零，填充到3位
                            padded_id = stereo_id.zfill(3)
                            correspondence[padded_id] = machine_id
                            
                            # 3. 如果是长编号，也存储后3位
                            if len(stereo_id) > 3:
                                short_id = stereo_id[-3:]
                                correspondence[short_id] = machine_id
                                # 后3位去除前导零的版本
                                correspondence[short_id.lstrip('0')] = machine_id
                            
                            # 4. 对于短编号，尝试各种前导零组合
                            if len(stereo_id) <= 2:
                                if len(stereo_id) == 1:
                                    correspondence['0' + stereo_id] = machine_id
                                    correspondence['00' + stereo_id] = machine_id
                                elif len(stereo_id) == 2:
                                    correspondence['0' + stereo_id] = machine_id
                            
                            # 5. 对于短编号，添加常见前缀
                            if len(stereo_id) <= 3:
                                for prefix in ["25042311", "30046S0518000"]:
                                    correspondence[prefix + padded_id] = machine_id
                                    # 还可以尝试添加不同长度的变体
                                    if len(stereo_id) <= 2:
                                        correspondence[prefix + stereo_id.zfill(2)] = machine_id
                                    correspondence[prefix + stripped_id] = machine_id
        
        print(f"从CSV文件中读取了 {len(correspondence)} 条整机-双目编号对应关系")
        return correspondence
    
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        sys.exit(1)

def normalize_stereo_id(folder_name):
    """尝试从文件夹名称中提取标准化的双目编号"""
    # 如果文件夹名称是纯数字，直接返回
    if folder_name.isdigit():
        return folder_name
    
    # 如果文件夹名称是类似"MK2-12-608"的格式，提取最后一部分
    if folder_name.startswith("MK2-") and "-" in folder_name[4:]:
        parts = folder_name.split("-")
        if len(parts) >= 3:
            return parts[-1]
    
    # 尝试从文件夹名称中提取数字部分
    digits_only = ''.join(c for c in folder_name if c.isdigit())
    if digits_only:
        # 如果提取的数字很长(>6位)，取最后3位
        if len(digits_only) > 6:
            return digits_only[-3:]
        # 如果数字长度是3-6位，直接返回
        elif 3 <= len(digits_only) <= 6:
            return digits_only
        # 如果数字长度小于3位，可能是短编号，需要处理前导零
        else:
            # 如果原始文件夹名有前导零，保留它们
            if folder_name.startswith("0"):
                return folder_name
            return digits_only
    
    return folder_name

def generate_report(report_path, processed_folders, unmatched_folders):
    """生成处理汇总报告"""
    try:
        with open(report_path, 'w', encoding='utf-8') as file:
            # 写入报告标题和时间戳
            file.write(f"# MK2样机双目相机数据处理报告\n\n")
            file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入处理统计
            file.write(f"## 处理统计\n\n")
            file.write(f"- 总处理文件夹数: {len(processed_folders)}\n")
            file.write(f"- 未匹配文件夹数: {len(unmatched_folders)}\n\n")
            
            # 写入已处理文件夹信息
            file.write(f"## 已处理文件夹\n\n")
            if processed_folders:
                file.write(f"| 原文件夹名 | 新文件夹名 | 生成的文件 |\n")
                file.write(f"|------------|------------|------------|\n")
                
                for folder_info in processed_folders:
                    original_name = folder_info['original_name']
                    new_name = folder_info['new_name']
                    files = ", ".join(folder_info['files'])
                    
                    file.write(f"| {original_name} | {new_name} | {files} |\n")
            else:
                file.write("无已处理文件夹\n\n")
            
            # 写入未匹配文件夹信息
            file.write(f"\n## 未匹配文件夹\n\n")
            if unmatched_folders:
                for folder in unmatched_folders:
                    file.write(f"- {folder}\n")
            else:
                file.write("无未匹配文件夹\n")
        
        print(f"已生成处理报告: {report_path}")
        return True
    
    except Exception as e:
        print(f"生成报告时出错: {e}")
        return False

def extract_stereo_id_from_filename(filename):
    """从文件名中提取双目编号"""
    # 文件名格式预期为 "模组编号_dualcamera_calibration.json"
    # 先移除文件扩展名
    filename_no_ext = filename.rsplit('.', 1)[0]
    # 提取前缀（模组编号）
    parts = filename_no_ext.split('_')
    if parts and parts[0]:
        return parts[0]
    return None

def process_root_json_files(root_dir, correspondence, format_type, sensor_template, okvis_template, keep_original):
    """处理根目录下的JSON文件，创建对应的子文件夹并生成YAML文件"""
    root_path = Path(root_dir)
    
    # 确保根目录存在
    if not root_path.exists():
        print(f"错误: 根目录 {root_dir} 不存在")
        return [], []
    
    # 获取根目录下的所有JSON文件，匹配格式"模组编号_dualcamera_calibration.json"
    json_pattern = re.compile(r'^(\d+)_dualcamera_calibration\.json$')
    json_files = [f for f in root_path.iterdir() if f.is_file() and json_pattern.match(f.name)]
    
    if not json_files:
        print(f"在根目录 {root_dir} 中未找到匹配格式的JSON文件")
        return [], []
    
    # 用于存储处理成功的文件夹信息和无法匹配的文件
    processed_folders = []
    unmatched_files = []
    
    # 处理每个JSON文件
    for json_file in json_files:
        # 提取模组编号
        stereo_id = extract_stereo_id_from_filename(json_file.name)
        if not stereo_id:
            print(f"无法从文件名 {json_file.name} 中提取双目编号，跳过")
            unmatched_files.append(json_file.name)
            continue
        
        # 检查是否能找到对应的整机编号
        machine_id = None
        
        # 直接尝试查找标准化后的ID
        if stereo_id in correspondence:
            machine_id = correspondence[stereo_id]
        else:
            # 尝试一些常见的变体形式
            variants = [
                stereo_id.lstrip('0'),  # 去除前导零
                stereo_id.zfill(3),     # 填充到3位
                '0' + stereo_id if len(stereo_id) <= 2 else stereo_id,  # 添加一个前导零
                '00' + stereo_id if len(stereo_id) == 1 else stereo_id   # 添加两个前导零
            ]
            
            # 尝试查找变体
            for variant in variants:
                if variant in correspondence:
                    machine_id = correspondence[variant]
                    print(f"找到编号变体匹配: {stereo_id} -> {variant} -> {machine_id}")
                    stereo_id = variant  # 使用找到匹配的变体
                    break
        
        if machine_id:
            # 处理machine_id中可能包含的MK2前缀
            prefix = "MK2-"
            if machine_id.startswith(prefix):
                # 如果machine_id已经包含MK2前缀，则不再添加
                new_folder_name = f"{machine_id}-{stereo_id}"
            else:
                # 如果machine_id不包含MK2前缀，则添加
                new_folder_name = f"{prefix}{machine_id}-{stereo_id}"
            
            # 创建新文件夹
            new_folder_path = root_path / new_folder_name
            
            # 确保目标文件夹不存在
            if new_folder_path.exists():
                print(f"警告: 目标文件夹 {new_folder_name} 已存在，跳过处理 {json_file.name}")
                continue
            
            # 创建新文件夹
            try:
                new_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"创建文件夹: {new_folder_name}")
                
                # 复制JSON文件到新文件夹并重命名
                target_json_path = new_folder_path / "dualcamera_calibration.json"
                shutil.copy2(json_file, target_json_path)
                print(f"复制JSON文件: {json_file.name} -> {target_json_path}")
                
                # 用于存储此文件夹处理的文件列表
                generated_files = []
                
                # 确定输出文件的前缀
                output_prefix = str(new_folder_path / new_folder_name)
                
                # 根据格式类型生成相应的YAML文件
                if format_type == 'sensor' or format_type == 'all':
                    # 生成sensor.yaml文件
                    sensor_yaml_path = f"{output_prefix}_sensor.yaml"
                    cmd_sensor = [
                        "python3",
                        os.path.join("convert_dualcamera_json_to_yaml.py"),
                        "--input", str(target_json_path),
                        "--output", output_prefix,
                        "--format", "sensor"
                    ]
                    
                    if sensor_template:
                        cmd_sensor.extend(["--sensor_template", sensor_template])
                    
                    print(f"生成sensor.yaml: {sensor_yaml_path}")
                    try:
                        result = subprocess.run(cmd_sensor, check=True)
                        if result.returncode == 0:
                            generated_files.append(f"{new_folder_name}_sensor.yaml")
                    except subprocess.CalledProcessError as e:
                        print(f"生成sensor.yaml时出错: {e}")
                
                # 如果是okvis格式或all格式，且提供了okvis模板，生成okvis.yaml文件
                if (format_type == 'okvis' or format_type == 'all') and okvis_template:
                    okvis_yaml_path = f"{output_prefix}_okvis.yaml"
                    cmd_okvis = [
                        "python3",
                        os.path.join("convert_dualcamera_json_to_yaml.py"),
                        "--input", str(target_json_path),
                        "--output", output_prefix,
                        "--format", "okvis",
                        "--okvis_template", okvis_template
                    ]
                    
                    print(f"生成okvis.yaml: {okvis_yaml_path}")
                    try:
                        result = subprocess.run(cmd_okvis, check=True)
                        if result.returncode == 0:
                            generated_files.append(f"{new_folder_name}_okvis.yaml")
                    except subprocess.CalledProcessError as e:
                        print(f"生成okvis.yaml时出错: {e}")
                
                # 存储处理成功的文件夹信息
                processed_folders.append({
                    'original_name': json_file.name,
                    'new_name': new_folder_name,
                    'files': generated_files
                })
                
                # 如果不保留原始文件，删除根目录下的JSON文件
                if not keep_original:
                    os.remove(json_file)
                    print(f"删除原始JSON文件: {json_file}")
                
            except Exception as e:
                print(f"处理文件 {json_file.name} 时出错: {e}")
                unmatched_files.append(json_file.name)
        else:
            print(f"无法找到匹配: 文件 {json_file.name}, 模组编号 {stereo_id}")
            unmatched_files.append(json_file.name)
    
    # 报告未匹配的文件
    if unmatched_files:
        print("\n以下JSON文件在CSV文件中未找到对应的整机编号:")
        for file in unmatched_files:
            print(f"  - {file}")
    
    return processed_folders, unmatched_files

def process_folders(root_dir, correspondence, format_type, sensor_template, okvis_template, keep_original, report_path=None):
    """处理根目录下的所有子文件夹"""
    root_path = Path(root_dir)
    
    # 确保根目录存在
    if not root_path.exists():
        print(f"错误: 根目录 {root_dir} 不存在")
        sys.exit(1)
    
    # 获取所有子文件夹
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print(f"警告: 在 {root_dir} 中未找到子文件夹")
        return [], []
    
    # 用于存储无法匹配的文件夹
    unmatched_folders = []
    # 用于存储处理成功的文件夹信息
    processed_folders = []
    
    # 处理每个子文件夹
    for folder in subfolders:
        original_folder_name = folder.name
        stereo_id = normalize_stereo_id(original_folder_name)
        
        # 检查是否能找到对应的整机编号
        machine_id = None
        
        # 直接尝试查找标准化后的ID
        if stereo_id in correspondence:
            machine_id = correspondence[stereo_id]
        else:
            # 尝试一些常见的变体形式
            variants = [
                stereo_id.lstrip('0'),  # 去除前导零
                stereo_id.zfill(3),     # 填充到3位
                '0' + stereo_id if len(stereo_id) <= 2 else stereo_id,  # 添加一个前导零
                '00' + stereo_id if len(stereo_id) == 1 else stereo_id   # 添加两个前导零
            ]
            
            # 尝试查找变体
            for variant in variants:
                if variant in correspondence:
                    machine_id = correspondence[variant]
                    print(f"找到编号变体匹配: {stereo_id} -> {variant} -> {machine_id}")
                    stereo_id = variant  # 使用找到匹配的变体
                    break
        
        if machine_id:
            # 处理machine_id中可能包含的MK2前缀
            prefix = "MK2-"
            if machine_id.startswith(prefix):
                # 如果machine_id已经包含MK2前缀，则不再添加
                new_folder_name = f"{machine_id}-{stereo_id}"
            else:
                # 如果machine_id不包含MK2前缀，则添加
                new_folder_name = f"{prefix}{machine_id}-{stereo_id}"
            
            new_folder_path = folder.parent / new_folder_name
            
            # 检查是否需要改名（如果已经是正确格式就不需要）
            if original_folder_name == new_folder_name:
                print(f"文件夹 {original_folder_name} 已经是正确格式，无需重命名")
                process_folder = folder
            else:
                # 检查json文件是否存在
                json_file = folder / "dualcamera_calibration.json"
                
                if json_file.exists():
                    # 确保目标文件夹不存在
                    if new_folder_path.exists():
                        print(f"警告: 目标文件夹 {new_folder_name} 已存在，跳过处理 {original_folder_name}")
                        continue
                    
                    # 重命名文件夹或复制一个新的（如果需要保留原始文件夹）
                    if keep_original:
                        print(f"复制文件夹 {original_folder_name} 到 {new_folder_name}")
                        shutil.copytree(folder, new_folder_path)
                    else:
                        print(f"重命名文件夹 {original_folder_name} 为 {new_folder_name}")
                        folder.rename(new_folder_path)
                    
                    process_folder = new_folder_path
                else:
                    print(f"警告: 在文件夹 {original_folder_name} 中未找到 dualcamera_calibration.json 文件")
                    continue
            
            # 用于存储此文件夹处理的文件列表
            generated_files = []
            
            # 确定输出文件的前缀，从文件夹名处理为正确格式
            base_folder_name = process_folder.name
            # 去除可能重复的MK2前缀
            if base_folder_name.startswith("MK2-MK2-"):
                base_folder_name = "MK2-" + base_folder_name.split("MK2-MK2-")[1]
                print(f"修正重复的MK2前缀: {process_folder.name} -> {base_folder_name}")
            
            output_prefix = str(process_folder / base_folder_name)
            
            # 根据格式类型生成相应的YAML文件
            if format_type == 'sensor' or format_type == 'all':
                # 生成sensor.yaml文件
                sensor_yaml_path = f"{output_prefix}_sensor.yaml"
                cmd_sensor = [
                    "python3",
                    os.path.join("convert_dualcamera_json_to_yaml.py"),
                    "--input", str(process_folder / "dualcamera_calibration.json"),
                    "--output", output_prefix,
                    "--format", "sensor"
                ]
                
                if sensor_template:
                    cmd_sensor.extend(["--sensor_template", sensor_template])
                
                print(f"生成sensor.yaml: {sensor_yaml_path}")
                try:
                    result = subprocess.run(cmd_sensor, check=True)
                    if result.returncode == 0:
                        generated_files.append(f"{base_folder_name}_sensor.yaml")
                except subprocess.CalledProcessError as e:
                    print(f"生成sensor.yaml时出错: {e}")
            
            # 如果是okvis格式或all格式，且提供了okvis模板，生成okvis.yaml文件
            if (format_type == 'okvis' or format_type == 'all') and okvis_template:
                okvis_yaml_path = f"{output_prefix}_okvis.yaml"
                cmd_okvis = [
                    "python3",
                    os.path.join("convert_dualcamera_json_to_yaml.py"),
                    "--input", str(process_folder / "dualcamera_calibration.json"),
                    "--output", output_prefix,
                    "--format", "okvis",
                    "--okvis_template", okvis_template
                ]
                
                print(f"生成okvis.yaml: {okvis_yaml_path}")
                try:
                    result = subprocess.run(cmd_okvis, check=True)
                    if result.returncode == 0:
                        generated_files.append(f"{base_folder_name}_okvis.yaml")
                except subprocess.CalledProcessError as e:
                    print(f"生成okvis.yaml时出错: {e}")
            
            # 存储处理成功的文件夹信息
            processed_folders.append({
                'original_name': original_folder_name,
                'new_name': new_folder_name,
                'files': generated_files
            })
        else:
            print(f"无法找到匹配: 文件夹 {original_folder_name}, 标准化ID {stereo_id}")
            unmatched_folders.append(original_folder_name)
    
    # 报告未匹配的文件夹
    if unmatched_folders:
        print("\n以下文件夹在CSV文件中未找到对应的整机编号:")
        for folder in unmatched_folders:
            print(f"  - {folder}")
    
    return processed_folders, unmatched_folders

def main():
    args = parse_args()
    
    # 读取CSV文件获取对应关系
    correspondence = read_correspondence_from_csv(args.csv)
    
    # 首先处理根目录下的JSON文件(情况二)
    root_processed_folders, unmatched_files = process_root_json_files(
        args.root,
        correspondence,
        args.format,
        args.sensor_template,
        args.okvis_template,
        args.keep_original
    )
    
    # 然后处理子文件夹(情况一)
    folder_processed_folders, unmatched_folders = process_folders(
        args.root, 
        correspondence, 
        args.format,
        args.sensor_template, 
        args.okvis_template,
        args.keep_original,
        None  # 不在这里生成报告，而是在后面合并结果后生成
    )
    
    # 合并处理结果
    all_processed_folders = root_processed_folders + folder_processed_folders
    
    # 打印处理结果统计
    print(f"\n处理完成!")
    print(f"根目录下的JSON文件: 成功处理 {len(root_processed_folders)} 个，无法匹配 {len(unmatched_files)} 个。")
    print(f"子文件夹: 成功处理 {len(folder_processed_folders)} 个，无法匹配 {len(unmatched_folders)} 个。")
    print(f"总计: 成功处理 {len(all_processed_folders)} 个。")
    
    # 如果指定报告路径，生成处理报告
    if args.report:
        # 在报告中合并两种情况的结果
        generate_report(args.report, all_processed_folders, unmatched_folders + unmatched_files)
    # 如果没有指定报告路径但有处理结果，可以自动生成一个报告
    elif all_processed_folders:
        default_report_path = f"MK2_处理报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        print(f"生成默认处理报告: {default_report_path}")
        generate_report(default_report_path, all_processed_folders, unmatched_folders + unmatched_files)

if __name__ == "__main__":
    main() 