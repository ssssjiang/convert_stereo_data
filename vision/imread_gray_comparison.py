#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse


def compare_imread_methods(image_path, output_dir='imread_comparison'):
    """
    比较两种读取灰度图的方式：
    1. 读取彩色图像然后转换为灰度: imread(...) -> cvtColor(BGR2GRAY)
    2. 直接读取为灰度图: imread(..., IMREAD_GRAYSCALE)
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 方法1: 读取RGB然后转换为灰度
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray_from_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # 方法2: 直接读取为灰度
    gray_direct = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 保存结果
    method1_output = os.path.join(output_dir, "gray_from_color.png")
    method2_output = os.path.join(output_dir, "gray_direct.png")
    
    cv2.imwrite(method1_output, gray_from_color)
    cv2.imwrite(method2_output, gray_direct)
    
    # 比较结果
    are_equal = np.array_equal(gray_from_color, gray_direct)
    
    print(f"两种读取方式结果是否相同: {'✅ 相同' if are_equal else '❌ 不同'}")
    
    if not are_equal:
        # 计算差异
        diff = cv2.absdiff(gray_from_color, gray_direct)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        non_zero_diff = np.count_nonzero(diff)
        diff_percent = (non_zero_diff / float(gray_from_color.size)) * 100
        
        print(f"最大差异: {max_diff}")
        print(f"平均差异: {mean_diff:.4f}")
        print(f"差异标准差: {std_diff:.4f}")
        print(f"有差异的像素数: {non_zero_diff} ({diff_percent:.2f}%)")
        
        # 保存差异图像
        diff_scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_file = os.path.join(output_dir, "difference.png")
        cv2.imwrite(diff_file, diff_scaled)
        
        # 增强差异可视化 - 使用热力图
        diff_heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
        diff_heatmap_file = os.path.join(output_dir, "difference_heatmap.png")
        cv2.imwrite(diff_heatmap_file, diff_heatmap)
        
        # 像素值分布比较
        print("\n灰度值分布比较:")
        hist1 = cv2.calcHist([gray_from_color], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray_direct], [0], None, [256], [0, 256])
        
        hist_diff = np.sum(np.abs(hist1 - hist2)) / np.sum(hist1)
        print(f"直方图差异: {hist_diff:.4f}")
    
    return are_equal, output_dir


def test_multiple_formats(output_dir='imread_format_test'):
    """测试不同格式图像的读取差异"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 创建渐变图像
    for i in range(256):
        img[:, i, 0] = i  # 蓝色通道
        img[i, :, 1] = i  # 绿色通道
        img[i, 255-i, 2] = i  # 红色通道
    
    # 添加一些特征
    cv2.circle(img, (128, 128), 64, (255, 0, 0), -1)  # 蓝色圆
    cv2.rectangle(img, (32, 32), (96, 96), (0, 255, 0), -1)  # 绿色矩形
    cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5)  # 红色线
    
    # 测试不同格式
    formats = ['png', 'jpg', 'bmp', 'tiff', 'webp']
    qualities = [50, 95]
    
    results = []
    
    for fmt in formats:
        for q in qualities:
            format_dir = os.path.join(output_dir, f"{fmt}_q{q}")
            os.makedirs(format_dir, exist_ok=True)
            
            # 保存测试图像
            if fmt in ['jpg', 'jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, q]
            elif fmt == 'webp':
                params = [cv2.IMWRITE_WEBP_QUALITY, q]
            elif fmt == 'png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, 9 - q // 10))]
            else:
                params = []
            
            test_file = os.path.join(format_dir, f"test.{fmt}")
            cv2.imwrite(test_file, img, params)
            
            # 比较两种读取方式
            are_equal, _ = compare_imread_methods(test_file, format_dir)
            
            results.append({
                'format': fmt,
                'quality': q,
                'are_equal': are_equal
            })
    
    # 显示结果摘要
    print("\n不同格式读取差异摘要:")
    for r in results:
        print(f"格式: {r['format']}, 质量: {r['quality']}, 结果: {'✅ 相同' if r['are_equal'] else '❌ 不同'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='比较不同imread灰度读取方式的差异')
    parser.add_argument('--image', help='输入图像路径(可选)')
    parser.add_argument('--test-formats', action='store_true', help='测试多种格式')
    parser.add_argument('--output', default='imread_test', help='输出目录')
    
    args = parser.parse_args()
    
    if args.test_formats:
        print("测试不同图像格式的读取差异...")
        test_multiple_formats(args.output)
    elif args.image:
        print(f"比较图像 {args.image} 的两种读取方式...")
        compare_imread_methods(args.image, args.output)
    else:
        print("请提供输入图像路径或使用--test-formats参数测试多种格式")
        parser.print_help()


if __name__ == "__main__":
    main() 