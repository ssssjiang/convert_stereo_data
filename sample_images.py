#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的图像采样脚本：按步长采样图像并复制到新文件夹

Usage:
    python3 sample_images_simple.py -i /path/to/images -o /path/to/output -s 5
"""

import os
import sys
import argparse
import shutil


def sample_images(input_folder, output_folder, stride=1):
    """
    按步长采样图像并复制到输出文件夹
    
    Args:
        input_folder: 输入图像文件夹路径
        output_folder: 输出文件夹路径
        stride: 采样步长，每 stride 张取 1 张
    """
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在: {input_folder}")
        return
    
    # 获取所有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for file in os.listdir(input_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"错误: 在 {input_folder} 中未找到图像文件")
        return
    
    # 按文件名排序
    image_files.sort()
    
    # 按步长采样
    sampled_files = image_files[::stride]
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    print("="*60)
    print("图像采样")
    print("="*60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"总图像数量: {len(image_files)}")
    print(f"采样步长: {stride}")
    print(f"采样后数量: {len(sampled_files)}")
    print(f"采样率: {100*len(sampled_files)/len(image_files):.1f}%")
    print("="*60)
    
    # 复制文件
    success_count = 0
    for idx, filename in enumerate(sampled_files, 1):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            success_count += 1
            
            if idx % 50 == 0:
                print(f"进度: {idx}/{len(sampled_files)} ({100*idx//len(sampled_files)}%)")
                
        except Exception as e:
            print(f"警告: 复制 {filename} 失败: {e}")
    
    print("="*60)
    print(f"完成! 成功复制 {success_count}/{len(sampled_files)} 张图像")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="按步长采样图像并复制到新文件夹"
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="输入图像文件夹路径")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="输出文件夹路径")
    parser.add_argument('-s', '--stride', type=int, default=1,
                        help="采样步长，每 N 张取 1 张 (默认: 1，不采样)")
    
    args = parser.parse_args()
    
    sample_images(args.input, args.output, args.stride)


if __name__ == "__main__":
    main()

