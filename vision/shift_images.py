#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像y轴平移工具
功能：读取指定目录下的每张图像，沿y轴下移两个像素，保持原始图像尺寸和文件名不变
"""

import cv2
import numpy as np
import argparse
import os
import glob
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将指定目录下的图像沿y轴下移两个像素")
    parser.add_argument('--input_dir', type=str, required=True, 
                       help="输入图像目录路径")
    parser.add_argument('--output_dir', type=str, required=True, 
                       help="输出图像目录路径")
    parser.add_argument('--shift_pixels', type=int, default=2, 
                       help="沿y轴下移的像素数 (默认: 2)")
    parser.add_argument('--supported_formats', type=str, nargs='+', 
                       default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
                       help="支持的图像格式 (默认: jpg jpeg png bmp tiff tif)")
    parser.add_argument('--fill_value', type=int, default=0,
                       help="顶部填充像素的值 (默认: 0 黑色)")
    parser.add_argument('--verbose', action='store_true',
                       help="显示详细处理信息")
    
    return parser.parse_args()


def shift_image_y_axis(image, shift_pixels, fill_value=0):
    """
    将图像沿y轴下移指定像素
    
    Args:
        image: 输入图像 (numpy数组)
        shift_pixels: 下移的像素数
        fill_value: 顶部填充的像素值
    
    Returns:
        shifted_image: 移动后的图像
    """
    height, width = image.shape[:2]
    
    # 创建与原图相同尺寸的输出图像
    if len(image.shape) == 3:  # 彩色图像
        shifted_image = np.full((height, width, image.shape[2]), fill_value, dtype=image.dtype)
    else:  # 灰度图像
        shifted_image = np.full((height, width), fill_value, dtype=image.dtype)
    
    # 将原图像向下移动shift_pixels个像素
    if shift_pixels > 0 and shift_pixels < height:
        shifted_image[shift_pixels:, :] = image[:-shift_pixels, :]
    elif shift_pixels == 0:
        shifted_image = image.copy()
    elif shift_pixels >= height:
        # 如果移动距离大于等于图像高度，整个图像都被移出，返回填充图像
        pass  # shifted_image已经是填充值
    
    return shifted_image


def get_image_files(input_dir, supported_formats):
    """
    获取指定目录下所有支持格式的图像文件
    
    Args:
        input_dir: 输入目录路径
        supported_formats: 支持的图像格式列表
    
    Returns:
        image_files: 图像文件路径列表
    """
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    for fmt in supported_formats:
        # 支持大小写变体
        pattern_lower = f"*.{fmt.lower()}"
        pattern_upper = f"*.{fmt.upper()}"
        
        image_files.extend(glob.glob(os.path.join(input_dir, pattern_lower)))
        image_files.extend(glob.glob(os.path.join(input_dir, pattern_upper)))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    return image_files


def process_images(input_dir, output_dir, shift_pixels, supported_formats, fill_value, verbose):
    """
    处理指定目录下的所有图像
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        shift_pixels: 下移像素数
        supported_formats: 支持的图像格式
        fill_value: 填充值
        verbose: 是否显示详细信息
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = get_image_files(input_dir, supported_formats)
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到支持格式的图像文件")
        print(f"支持的格式: {supported_formats}")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"处理设置: 沿y轴下移 {shift_pixels} 个像素, 填充值: {fill_value}")
    
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files):
        try:
            if verbose:
                print(f"[{i+1}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
            
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"错误: 无法读取图像 {image_path}")
                error_count += 1
                continue
            
            # 获取原始图像信息
            original_height, original_width = image.shape[:2]
            
            # 执行y轴平移
            shifted_image = shift_image_y_axis(image, shift_pixels, fill_value)
            
            # 构建输出文件路径，保持原文件名
            input_filename = os.path.basename(image_path)
            output_file_path = os.path.join(output_dir, input_filename)
            
            # 保存处理后的图像
            success = cv2.imwrite(output_file_path, shifted_image)
            if success:
                success_count += 1
                if verbose:
                    print(f"    成功保存到: {output_file_path}")
                    print(f"    图像尺寸: {original_width}x{original_height}")
            else:
                print(f"错误: 无法保存图像到 {output_file_path}")
                error_count += 1
                
        except Exception as e:
            print(f"错误: 处理图像 {image_path} 时发生异常: {str(e)}")
            error_count += 1
    
    # 输出处理结果统计
    print(f"\n处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")


def main():
    """主函数"""
    args = parse_args()
    
    try:
        process_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            shift_pixels=args.shift_pixels,
            supported_formats=args.supported_formats,
            fill_value=args.fill_value,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"程序执行错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 