#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import argparse
import subprocess
import tarfile
import cv2
import numpy as np

# 执行命令函数
def execute_command(command, cwd=None):
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, check=True)
        print(f"命令执行成功: {command}")
    except subprocess.CalledProcessError as e:
        print(f"执行命令出错: {command}. {e}")
        raise

# 合并日志文件
def merge_logs(source_dir):
    print(f"开始合并日志文件... (输入目录: {source_dir})")
    
    # 检查输入目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误: 目录不存在 - {source_dir}")
        return False
    
    # 检查是否有tar文件
    tar_files = [f for f in os.listdir(source_dir) if f.endswith('.tar')]
    if not tar_files:
        print(f"警告: 未找到任何 .tar 文件在 {source_dir}")
        return False
    
    # 遍历目录中的每个文件
    for file_path in os.listdir(source_dir):
        full_path = os.path.join(source_dir, file_path)
        
        # 检查文件是否为压缩文件
        if file_path.endswith('.tar'):
            # 获取文件名（不带路径和扩展名）
            filename = os.path.splitext(file_path)[0]
            
            # 创建解压目标目录
            target_dir = os.path.join(source_dir, filename)
            
            # 创建目录
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                # 使用Python的tarfile模块解压文件到目标目录
                print(f"正在解压 {full_path} 到 {target_dir}...")
                with tarfile.open(full_path, 'r') as tar:
                    # 提取所有文件，但忽略绝对路径
                    for member in tar.getmembers():
                        # 修改成员路径，确保它是相对路径
                        if member.name.startswith('/'):
                            member.name = member.name.lstrip('/')
                        
                        # 跳过可能导致问题的路径
                        if '..' in member.name or member.name.startswith('/'):
                            print(f"跳过可能不安全的路径: {member.name}")
                            continue
                        
                        try:
                            tar.extract(member, path=target_dir)
                        except Exception as e:
                            print(f"提取 {member.name} 时出错: {e}")
                
                # 合并日志文件
                log_xz_path = os.path.join(source_dir, f"{filename}.log.xz")
                
                # 使用glob查找所有匹配的.xz文件
                devtest_path = os.path.join(target_dir, "mnt", "data", "rockrobo", "devtest")
                if os.path.exists(devtest_path):
                    # 创建输出文件
                    with open(log_xz_path, 'wb') as outfile:
                        # 查找所有.xz文件并合并
                        for root, dirs, files in os.walk(devtest_path):
                            for f in files:
                                if f.endswith('.xz'):
                                    xz_file_path = os.path.join(root, f)
                                    try:
                                        with open(xz_file_path, 'rb') as infile:
                                            outfile.write(infile.read())
                                            print(f"合并日志文件: {xz_file_path}")
                                    except Exception as e:
                                        print(f"读取 {xz_file_path} 时出错: {e}")
                    
                    # 解压缩 .xz 文件
                    if os.path.exists(log_xz_path) and os.path.getsize(log_xz_path) > 0:
                        try:
                            execute_command(f"xz -d {filename}.log.xz", source_dir)
                            print(f"成功解压 {filename}.log.xz")
                        except Exception as e:
                            print(f"解压 {filename}.log.xz 时出错: {e}")
                    else:
                        print(f"警告: 日志文件 {log_xz_path} 为空或不存在")
                else:
                    print(f"警告: 路径不存在: {devtest_path}")
                
                # 清理临时文件
                try:
                    shutil.rmtree(target_dir)
                    print(f"已删除临时目录: {target_dir}")
                except Exception as e:
                    print(f"删除临时目录 {target_dir} 时出错: {e}")
                
                if os.path.exists(log_xz_path):
                    try:
                        os.remove(log_xz_path)
                        print(f"已删除临时文件: {log_xz_path}")
                    except Exception as e:
                        print(f"删除临时文件 {log_xz_path} 时出错: {e}")
                
                print(f"已提取 '{full_path}' 到 '{target_dir}'")
            except Exception as e:
                print(f"处理 '{full_path}' 时出错: {e}")
                # 清理临时文件
                if os.path.exists(target_dir):
                    try:
                        shutil.rmtree(target_dir)
                    except Exception as e2:
                        print(f"清理临时目录 {target_dir} 时出错: {e2}")
                
                log_xz_path = os.path.join(source_dir, f"{filename}.log.xz")
                if os.path.exists(log_xz_path):
                    try:
                        os.remove(log_xz_path)
                    except Exception as e2:
                        print(f"清理临时文件 {log_xz_path} 时出错: {e2}")
    
    print("日志合并完成")
    return True

# 创建目标文件夹
def create_directories(output_dir, left_camera_id=0, right_camera_id=1):
    """创建相机目录结构，返回左右相机的路径"""
    left_camera_dir = os.path.join(output_dir, f"camera/camera{left_camera_id}")
    right_camera_dir = os.path.join(output_dir, f"camera/camera{right_camera_id}")
    
    # 创建相机目录
    os.makedirs(left_camera_dir, exist_ok=True)
    os.makedirs(right_camera_dir, exist_ok=True)
    
    return left_camera_dir, right_camera_dir

# 处理图像文件
def organize_images(source_dir, output_dir, left_camera_id=0, right_camera_id=1):
    print(f"开始整理图像文件... (图像目录: {source_dir}, 输出目录: {output_dir})")
    
    # 检查输入目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误: 图像目录不存在 - {source_dir}")
        return False
        
    # 检查输出目录是否存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建目标文件夹
    try:
        left_camera_dir, right_camera_dir = create_directories(output_dir, left_camera_id, right_camera_id)
    except Exception as e:
        print(f"创建相机目录时出错: {e}")
        return False
    
    # 创建相机映射
    camera_map = {
        'L': f"camera{left_camera_id}",
        'R': f"camera{right_camera_id}"
    }
    
    print(f"相机映射配置: L -> camera{left_camera_id}, R -> camera{right_camera_id}")
    
    # 检查是否需要解压图像
    decompress_dir = os.path.join(source_dir, 'decompress')
    using_decompressed = False
    
    # 复制png_decoder到源目录并执行
    png_decoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'png_decoder')
    if os.path.exists(png_decoder_path):
        try:
            # 复制png_decoder到源目录
            dest_decoder_path = os.path.join(source_dir, 'png_decoder')
            shutil.copy2(png_decoder_path, dest_decoder_path)
            # 确保文件有执行权限
            os.chmod(dest_decoder_path, 0o755)
            # 创建解压目录
            os.makedirs(decompress_dir, exist_ok=True)
            # 在源目录中执行png_decoder
            execute_command('./png_decoder ./decompress', source_dir)
            # 使用解压后的目录作为源目录
            source_dir = decompress_dir
            using_decompressed = True
            print(f"图像已解压到 {decompress_dir}")
        except Exception as e:
            print(f"解压图像时出错: {e}")
            print(f"将使用原始图像目录: {source_dir}")
    else:
        print(f"警告: png_decoder不存在于 {png_decoder_path}，跳过解压步骤")
    
    # 获取源目录中的所有文件
    files = []
    try:
        for root, _, filenames in os.walk(source_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    except Exception as e:
        print(f"扫描图像文件时出错: {e}")
        return False
    
    if not files:
        print(f"警告: 在目录 {source_dir} 中未找到任何文件")
        # 清理临时文件
        if using_decompressed and os.path.exists(decompress_dir):
            try:
                shutil.rmtree(decompress_dir)
                print(f"清理临时目录: {decompress_dir}")
            except Exception as e:
                print(f"清理临时目录时出错: {e}")
        return False
    
    # 计数器
    count_l = 0
    count_r = 0
    count_yaml = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # 处理YAML文件
        if filename.endswith('.yaml'):
            try:
                shutil.copy(file_path, output_dir)
                count_yaml += 1
                continue
            except Exception as e:
                print(f"复制YAML文件时出错 {file_path}: {e}")
                continue
        
        # 处理图像文件
        if filename.endswith('.png'):
            # 检查是否为butchart格式
            parts = filename.split('_')
            if len(parts) >= 3 and (parts[0] == 'L' or parts[0] == 'R'):
                # butchart格式
                camera_folder = camera_map.get(parts[0])
                timestamp = parts[1]
                dest_folder = os.path.join(output_dir, 'camera', camera_folder)
                dest_file = os.path.join(dest_folder, f"{timestamp}.png")
                
                if parts[0] == 'L':
                    count_l += 1
                else:
                    count_r += 1
            else:
                # 使用正则表达式提取时间戳
                timestamp_match = re.search(r'(\d+)', filename)
                if not timestamp_match:
                    continue
                
                timestamp = timestamp_match.group(1)
                
                # 确定目标文件夹
                if 'L' in filename:
                    dest_folder = os.path.join(output_dir, 'camera', camera_map['L'])
                    count_l += 1
                elif 'R' in filename:
                    dest_folder = os.path.join(output_dir, 'camera', camera_map['R'])
                    count_r += 1
                else:
                    # 如果文件名中既没有L也没有R，则跳过
                    continue
                
                dest_file = os.path.join(dest_folder, f"{timestamp}.png")
            
            # 读取图像并转换为灰度图
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    # 转换为灰度图
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 保存灰度图
                    cv2.imwrite(dest_file, gray_img)
                    print(f"转换并移动 {file_path} 到 {dest_file} (灰度图)")
                else:
                    # 如果无法读取图像，则直接复制
                    shutil.copy2(file_path, dest_file)
                    print(f"无法读取图像，直接复制 {file_path} 到 {dest_file}")
            except Exception as e:
                print(f"处理图像时出错 {file_path}: {e}")
                # 出错时直接复制
                try:
                    shutil.copy2(file_path, dest_file)
                    print(f"直接复制 {file_path} 到 {dest_file}")
                except Exception as e2:
                    print(f"复制图像时出错 {file_path}: {e2}")
    
    # 清理临时文件
    if using_decompressed and os.path.exists(decompress_dir):
        try:
            shutil.rmtree(decompress_dir)
            print(f"清理临时目录: {decompress_dir}")
        except Exception as e:
            print(f"清理临时目录时出错: {e}")
    
    print(f"处理完成: {count_l}个L文件移动到{camera_map['L']}, {count_r}个R文件移动到{camera_map['R']}, {count_yaml}个YAML文件")
    
    # 返回值表示是否成功找到并处理了图像
    return count_l > 0 or count_r > 0 or count_yaml > 0

# 处理日志文件
def process_logs(input_dir, output_dir):
    print(f"开始处理日志文件... (输入目录: {input_dir}, 输出目录: {output_dir})")
    
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return False
        
    # 检查输出目录是否存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制Bin2ToText到输入目录并执行
    bin2totext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Bin2ToText')
    if os.path.exists(bin2totext_path):
        # 复制Bin2ToText到输入目录
        dest_bin2totext_path = os.path.join(input_dir, 'Bin2ToText')
        shutil.copy2(bin2totext_path, dest_bin2totext_path)
        # 确保文件有执行权限
        os.chmod(dest_bin2totext_path, 0o755)
        # 在输入目录中执行Bin2ToText
        try:
            execute_command('./Bin2ToText', input_dir)
        except Exception as e:
            print(f"执行Bin2ToText时出错: {e}")
            return False
    else:
        print(f"警告: Bin2ToText不存在于 {bin2totext_path}，跳过日志处理")
        return False
    
    success = True
    
    # 移动处理后的日志文件，并排序
    source_log = os.path.join(input_dir, 'Sensor_fprintf.log')
    dest_log = os.path.join(output_dir, 'RRLDR_fprintf.log')
    
    if os.path.exists(source_log):
        try:
            # 读取文件内容
            with open(source_log, 'r') as f:
                lines = f.readlines()
            
            # 解析时间戳
            parsed_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    # 尝试解析第一列为时间戳
                    timestamp = int(parts[0])
                    parsed_lines.append((timestamp, line))
                except (ValueError, IndexError):
                    # 如果无法解析时间戳，使用最大值确保排在末尾
                    parsed_lines.append((float('inf'), line))
            
            # 按时间戳排序
            parsed_lines.sort(key=lambda x: x[0])
            
            # 写入排序后的内容到目标文件
            with open(dest_log, 'w') as f:
                for _, line in parsed_lines:
                    f.write(line)
            
            print(f"移动并按时间戳排序日志文件: {source_log} -> {dest_log}")
        except Exception as e:
            print(f"排序和移动日志文件时出错: {e}")
            # 如果排序失败，直接复制文件
            try:
                shutil.copy2(source_log, dest_log)
                print(f"直接复制日志文件(未排序): {source_log} -> {dest_log}")
            except Exception as e2:
                print(f"复制日志文件时出错: {e2}")
                success = False
    else:
        print(f"警告: 日志文件 {source_log} 不存在，无法移动")
        success = False
    
    # 查找并复制SLAM_fprintf.log文件
    slam_logs = []
    for file in os.listdir(input_dir):
        if "SLAM_fprintf" in file and file.endswith(".log"):
            slam_logs.append(file)
    
    if slam_logs:
        for slam_log in slam_logs:
            source_slam_log = os.path.join(input_dir, slam_log)
            dest_slam_log = os.path.join(output_dir, slam_log)
            try:
                shutil.copy2(source_slam_log, dest_slam_log)
                print(f"复制SLAM日志文件: {source_slam_log} -> {dest_slam_log}")
            except Exception as e:
                print(f"复制SLAM日志文件时出错 {slam_log}: {e}")
                success = False
    else:
        print("未找到SLAM_fprintf.log文件")
    
    return success

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='智能SLAM数据处理工具')
    parser.add_argument('--input', '-i', type=str, default=os.getcwd(), 
                        help='输入目录 (默认: 当前目录)')
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(os.path.dirname(os.getcwd()), 
                                            os.path.basename(os.getcwd()).split('.')[0] + '_converted_log'),
                        help='输出目录 (默认: ../当前目录名_converted_log)')
    parser.add_argument('--cameras', '-c', type=str, 
                        help='图像目录 (可选，默认与输入目录相同)')
    parser.add_argument('--merge-logs', '-m', action='store_true',
                        help='合并日志文件')
    parser.add_argument('--process-logs', '-p', action='store_true',
                        help='处理日志文件')
    parser.add_argument('--organize-images', '-g', action='store_true',
                        help='整理图像文件')
    parser.add_argument('--left-camera-id', '-l', type=int, default=0,
                        help='左相机(L)对应的相机ID (默认: 0)')
    parser.add_argument('--right-camera-id', '-r', type=int, default=1,
                        help='右相机(R)对应的相机ID (默认: 1)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输入目录存在
    if not os.path.exists(args.input):
        print(f"错误: 输入目录 '{args.input}' 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    
    # 如果没有指定相机目录，则使用输入目录
    if not args.cameras:
        args.cameras = args.input
        print(f"未指定相机图像目录，使用输入目录: {args.cameras}")
    elif not os.path.exists(args.cameras):
        print(f"错误: 相机图像目录 '{args.cameras}' 不存在")
        return
    
    # 如果没有指定任何操作，则执行所有操作
    if not (args.merge_logs or args.process_logs or args.organize_images):
        print("未指定具体操作，将执行完整处理（合并日志、处理日志、整理图像）")
        args.merge_logs = True
        args.process_logs = True
        args.organize_images = True
    
    # 执行操作
    status = {}
    
    if args.merge_logs:
        status['merge_logs'] = merge_logs(args.input)
    
    if args.process_logs:
        status['process_logs'] = process_logs(args.input, args.output)
    
    if args.organize_images:
        try:
            organize_images(args.cameras, args.output, args.left_camera_id, args.right_camera_id)
            status['organize_images'] = True
        except Exception as e:
            print(f"整理图像时出错: {e}")
            status['organize_images'] = False
    
    # 打印操作状态摘要
    print("\n处理摘要:")
    for operation, success in status.items():
        status_text = "成功" if success else "失败"
        print(f"- {operation}: {status_text}")
    
    print("\n数据处理完成！")

if __name__ == "__main__":
    main() 


# 使用示例:

# # 完整处理（合并日志、处理日志、整理图像）- 不指定任何操作标志时的默认行为
# python organize_cameras.py -i /path/to/input -o /path/to/output -c /path/to/images

# # 显式指定执行完整处理
# python organize_cameras.py -i /path/to/input -o /path/to/output -c /path/to/images -m -p -g

# # 只整理图像
# python organize_cameras.py -i /path/to/input -o /path/to/output -c /path/to/images -g

# # 只合并和处理日志
# python organize_cameras.py -i /path/to/input -o /path/to/output -m -p

# # 设置左右相机ID映射（将L映射到camera1，R映射到camera0）
# python organize_cameras.py -i /path/to/input -o /path/to/output -c /path/to/images -l 1 -r 0

# # 设置左右相机ID映射（将L映射到camera2，R映射到camera3）
# python organize_cameras.py -i /path/to/input -o /path/to/output -c /path/to/images -l 2 -r 3