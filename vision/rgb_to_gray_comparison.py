#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
import hashlib  # 添加hashlib模块导入
import matplotlib.pyplot as plt  # 添加用于绘制直方图


def compare_rgb_to_gray(rgb_image_path, gray_image_path, output_dir="comparison_results"):
    """
    将RGB图像通过 RGB->PNG->BGR->Gray 路径转换为灰度图，然后与输入的灰度图比较。
    
    Args:
        rgb_image_path: RGB图像路径
        gray_image_path: 灰度图像路径
        output_dir: 输出目录，用于保存比较结果
    
    Returns:
        比较结果和差异统计信息
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取RGB图像
    rgb_img = cv2.imread(rgb_image_path)
    if rgb_img is None:
        raise ValueError(f"无法读取RGB图像: {rgb_image_path}")
    
    # 读取灰度图像
    gray_img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise ValueError(f"无法读取灰度图像: {gray_image_path}")
    
    # 保存原始灰度图像副本
    original_gray_path = os.path.join(output_dir, "original_gray.png")
    cv2.imwrite(original_gray_path, gray_img)
    
    height, width = rgb_img.shape[:2]
    
    # 转换路径: RGB->PNG->BGR->Gray
    temp_rgb_png = os.path.join(output_dir, "temp_rgb.png")
    cv2.imwrite(temp_rgb_png, rgb_img)
    
    # 读取PNG (将得到BGR格式)
    bgr_from_png = cv2.imread(temp_rgb_png)
    
    # 转换为灰度图
    converted_gray = cv2.cvtColor(bgr_from_png, cv2.COLOR_BGR2GRAY)
    
    # 保存转换后的灰度图
    converted_gray_path = os.path.join(output_dir, "converted_gray.png")
    cv2.imwrite(converted_gray_path, converted_gray)
    
    # 确保两个灰度图尺寸一致
    if gray_img.shape != converted_gray.shape:
        print(f"警告: 图像尺寸不匹配。原始灰度图: {gray_img.shape}, 转换后灰度图: {converted_gray.shape}")
        # 调整尺寸以便比较
        if gray_img.shape[0] > converted_gray.shape[0] or gray_img.shape[1] > converted_gray.shape[1]:
            gray_img = cv2.resize(gray_img, (converted_gray.shape[1], converted_gray.shape[0]))
        else:
            converted_gray = cv2.resize(converted_gray, (gray_img.shape[1], gray_img.shape[0]))
    
    # 比较结果
    are_equal = np.array_equal(converted_gray, gray_img)
    
    # 使用MD5哈希值比较
    gray_img_hash = hashlib.md5(gray_img.tobytes()).hexdigest()
    converted_gray_hash = hashlib.md5(converted_gray.tobytes()).hexdigest()
    hash_equal = gray_img_hash == converted_gray_hash
    
    results = {
        "pixel_equal": are_equal,
        "hash_equal": hash_equal,
        "original_hash": gray_img_hash,
        "converted_hash": converted_gray_hash
    }
    
    if not are_equal:
        # 计算差异
        diff = cv2.absdiff(converted_gray, gray_img)
        diff_scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_path = os.path.join(output_dir, "difference.png")
        cv2.imwrite(diff_path, diff_scaled)
        
        # 创建差异热力图以便更好地可视化
        diff_heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
        diff_heatmap_path = os.path.join(output_dir, "difference_heatmap.png")
        cv2.imwrite(diff_heatmap_path, diff_heatmap)
        
        # 分析差异值分布
        unique_diffs, diff_counts = np.unique(diff, return_counts=True)
        diff_distribution = {int(val): int(count) for val, count in zip(unique_diffs, diff_counts)}
        
        # 保存差异值分布数据
        diff_distribution_file = os.path.join(output_dir, "difference_distribution.txt")
        with open(diff_distribution_file, 'w') as f:
            f.write("差异值分布:\n")
            for diff_val, count in sorted(diff_distribution.items()):
                percentage = (count / diff.size) * 100
                f.write(f"差异值 {diff_val}: {count} 像素 ({percentage:.4f}%)\n")
        
        # 绘制差异值分布直方图
        plt.figure(figsize=(10, 6))
        plt.bar(diff_distribution.keys(), diff_distribution.values(), color='blue')
        plt.title('像素值差异分布')
        plt.xlabel('差异值')
        plt.ylabel('像素数量')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "difference_histogram.png"))
        plt.close()
        
        # 计算统计信息
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        num_diff_pixels = np.count_nonzero(diff)
        percent_diff = (num_diff_pixels / (gray_img.shape[0] * gray_img.shape[1])) * 100
        
        results.update({
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "num_diff_pixels": num_diff_pixels,
            "percent_diff": percent_diff,
            "diff_distribution": diff_distribution
        })
    
    return are_equal and hash_equal, results


def main():
    parser = argparse.ArgumentParser(description='比较RGB图像转灰度与输入灰度图是否一致')
    parser.add_argument('--rgb_image', required=True, help='RGB图像路径')
    parser.add_argument('--gray_image', required=True, help='灰度图像路径')
    parser.add_argument('--output_dir', default='comparison_results', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"比较RGB图像 {args.rgb_image} 转换为灰度图后与灰度图像 {args.gray_image} 的差异")
    
    are_equal, results = compare_rgb_to_gray(
        args.rgb_image, args.gray_image, args.output_dir
    )
    
    print(f"像素值比较结果: {'✅ 相同' if results['pixel_equal'] else '❌ 不同'}")
    print(f"MD5哈希比较结果: {'✅ 相同' if results['hash_equal'] else '❌ 不同'}")
    print(f"原始灰度图哈希值: {results['original_hash']}")
    print(f"转换后灰度图哈希值: {results['converted_hash']}")
    
    if are_equal:
        print("\n✅ 总体结果: 转换后的灰度图与输入的灰度图完全一致。")
    else:
        print("\n❌ 总体结果: 转换后的灰度图与输入的灰度图存在差异。")
        
        if 'max_diff' in results:
            print("\n差异统计:")
            print(f"  最大差异: {results['max_diff']}")
            print(f"  平均差异: {results['mean_diff']:.4f}")
            print(f"  标准差: {results['std_diff']:.4f}")
            print(f"  不同像素数量: {results['num_diff_pixels']} " 
                f"({results['percent_diff']:.2f}% 的图像)")
            
            # 显示差异值分布情况
            print("\n差异值分布:")
            total_pixels = results['num_diff_pixels'] / (results['percent_diff'] / 100) if results['percent_diff'] > 0 else 1
            for diff_val, count in sorted(results["diff_distribution"].items()):
                if count > 0:  # 只显示存在的差异值
                    percentage = (count / total_pixels) * 100
                    print(f"  差异值 {diff_val}: {count} 像素 ({percentage:.4f}%)")
            
            if 1 in results["diff_distribution"] and results["diff_distribution"][1] > 0:
                print("\n注意: 发现大量差异值为1的像素，这通常是由图像转换过程中的舍入误差导致的。")
                print("这种微小差异在视觉上通常不可见，但会导致像素级比较和哈希值不匹配。")
        
        print(f"\n查看 '{args.output_dir}' 目录获取可视化比较结果和详细分析。")


if __name__ == "__main__":
    main() 