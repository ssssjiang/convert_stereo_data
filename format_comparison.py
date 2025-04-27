#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from tabulate import tabulate


def create_test_image(width=640, height=480):
    """创建一个测试图像，具有渐变和各种色彩模式"""
    # 创建基础渐变图像
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # 创建不同模式的RGB通道
    r_channel = np.uint8((np.sin(xv * 10) + 1) * 127.5)
    g_channel = np.uint8((np.cos(yv * 10) + 1) * 127.5)
    b_channel = np.uint8(xv * yv * 255)
    
    # 组合通道
    image = np.stack([b_channel, g_channel, r_channel], axis=2)  # OpenCV使用BGR顺序
    
    # 添加一些形状
    cv2.circle(image, (width // 4, height // 4), 50, (255, 0, 0), -1)  # 蓝色圆形
    cv2.rectangle(image, (width // 2, height // 2), (width // 2 + 100, height // 2 + 100), (0, 255, 0), -1)  # 绿色矩形
    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5)  # 红色线条
    
    return image


def test_conversion_path(rgb_image, img_format='png', quality=95, output_dir='format_test'):
    """测试不同格式和质量参数下的两种转换路径
    
    路径1: RGB->格式文件->BGR->灰度
    路径2: RGB->灰度->格式文件->灰度
    
    参数:
        rgb_image: RGB图像
        img_format: 图像格式 ('png', 'jpg', 'webp', 等)
        quality: 压缩质量 (对PNG无效)
        output_dir: 输出目录
    
    返回:
        比较结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建格式特定的子目录
    format_dir = os.path.join(output_dir, f"{img_format}_q{quality}")
    os.makedirs(format_dir, exist_ok=True)
    
    # 路径1: RGB->格式文件->BGR->灰度
    path1_color_file = os.path.join(format_dir, f"path1_color.{img_format}")
    
    # 设置imwrite参数
    params = []
    if img_format.lower() == 'jpg' or img_format.lower() == 'jpeg':
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_format.lower() == 'webp':
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif img_format.lower() == 'png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, 9 - quality // 10))]
    
    # 保存彩色图像
    cv2.imwrite(path1_color_file, rgb_image, params)
    
    # 读取彩色图像并转为灰度
    path1_read_color = cv2.imread(path1_color_file)
    path1_gray = cv2.cvtColor(path1_read_color, cv2.COLOR_BGR2GRAY)
    path1_gray_file = os.path.join(format_dir, f"path1_gray.{img_format}")
    cv2.imwrite(path1_gray_file, path1_gray, params)
    
    # 保存路径1的最终灰度图像用于比较
    path1_final_gray_file = os.path.join(format_dir, f"path1_final_gray.png")
    cv2.imwrite(path1_final_gray_file, path1_gray)
    
    # 路径2: RGB->灰度->格式文件->灰度
    path2_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    path2_gray_file = os.path.join(format_dir, f"path2_gray.{img_format}")
    cv2.imwrite(path2_gray_file, path2_gray, params)
    path2_read_gray = cv2.imread(path2_gray_file, cv2.IMREAD_GRAYSCALE)
    
    # 保存路径2的最终灰度图像用于比较
    path2_final_gray_file = os.path.join(format_dir, f"path2_final_gray.png")
    cv2.imwrite(path2_final_gray_file, path2_read_gray)
    
    # 比较两个结果
    are_equal = np.array_equal(path1_gray, path2_read_gray)
    
    result = {
        'format': img_format,
        'quality': quality,
        'are_equal': are_equal
    }
    
    if not are_equal:
        # 计算差异
        diff = cv2.absdiff(path1_gray, path2_read_gray)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        non_zero_diff = np.count_nonzero(diff)
        diff_percent = (non_zero_diff / float(path1_gray.size)) * 100
        
        # 保存差异图像
        diff_scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_file = os.path.join(format_dir, "difference.png")
        cv2.imwrite(diff_file, diff_scaled)
        
        # 增强差异可视化 - 使用热力图
        diff_heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
        diff_heatmap_file = os.path.join(format_dir, "difference_heatmap.png")
        cv2.imwrite(diff_heatmap_file, diff_heatmap)
        
        result.update({
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'non_zero_diff': non_zero_diff,
            'diff_percent': diff_percent
        })
    
    return result


def main():
    parser = argparse.ArgumentParser(description='比较不同图像格式的转换路径')
    parser.add_argument('--width', type=int, default=640, help='测试图像宽度')
    parser.add_argument('--height', type=int, default=480, help='测试图像高度')
    parser.add_argument('--output', default='format_test', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建测试图像
    test_image = create_test_image(args.width, args.height)
    
    # 定义要测试的格式和质量
    formats_to_test = [
        {'format': 'png', 'qualities': [0, 50, 100]},
        {'format': 'jpg', 'qualities': [10, 50, 75, 90, 100]},
        {'format': 'webp', 'qualities': [10, 50, 75, 90, 100]}
    ]
    
    results = []
    
    for format_info in formats_to_test:
        img_format = format_info['format']
        for quality in format_info['qualities']:
            print(f"测试格式: {img_format}, 质量: {quality}")
            result = test_conversion_path(
                test_image, img_format, quality, args.output
            )
            results.append(result)
    
    # 按格式分组显示结果
    print("\n转换路径比较结果:")
    
    table_data = []
    for r in results:
        equal_status = "✅ 相同" if r['are_equal'] else "❌ 不同"
        
        if r['are_equal']:
            row = [r['format'], r['quality'], equal_status, "-", "-", "-", "-"]
        else:
            row = [
                r['format'], 
                r['quality'], 
                equal_status,
                f"{r['max_diff']}",
                f"{r['mean_diff']:.4f}",
                f"{r['std_diff']:.4f}",
                f"{r['diff_percent']:.2f}%"
            ]
        table_data.append(row)
    
    headers = ["格式", "质量", "结果", "最大差异", "平均差异", "标准差", "差异像素占比"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存结果到文本文件
    result_file = os.path.join(args.output, "comparison_results.txt")
    with open(result_file, 'w') as f:
        f.write("图像格式转换路径比较结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write("\n\n")
        f.write("路径1: RGB->格式文件->BGR->灰度\n")
        f.write("路径2: RGB->灰度->格式文件->灰度\n")
    
    print(f"\n结果已保存到: {result_file}")
    print(f"图像输出目录: {args.output}")


if __name__ == "__main__":
    main() 