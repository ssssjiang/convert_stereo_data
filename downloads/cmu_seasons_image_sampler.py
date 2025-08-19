#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CMU-Seasons数据集图像采样工具

功能：
1. 扫描CMU-Seasons数据集目录结构
2. 收集所有图像的绝对路径
3. 按指定步长抽样图像
4. 将抽样后的前100张图像复制到新文件夹预览效果
5. 生成图像路径列表文档

使用示例：
python cmu_seasons_image_sampler.py -d /mnt/data/roborock/datasets/CMU-Seasons -s 10 -o sampled_images -n 100
"""

import os
import argparse
import shutil
from pathlib import Path
import re
from typing import List, Tuple


def scan_image_directory(dataset_path: str) -> Tuple[List[str], dict]:
    """
    扫描CMU-Seasons数据集目录，收集所有图像路径
    
    Args:
        dataset_path: 数据集根目录路径
        
    Returns:
        tuple: (所有图像路径列表, 统计信息字典)
    """
    print("正在扫描数据集目录结构...")
    
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"未找到images目录: {images_dir}")
    
    all_image_paths = []
    stats = {
        'total_images': 0,
        'slices': {},
        'cameras': {'c0': 0, 'c1': 0},
        'image_types': {'database': 0, 'query': 0}
    }
    
    # 扫描所有slice目录
    slice_dirs = [d for d in os.listdir(images_dir) if d.startswith('slice')]
    slice_dirs.sort(key=lambda x: int(x.replace('slice', '')))
    
    for slice_dir in slice_dirs:
        slice_path = os.path.join(images_dir, slice_dir)
        if not os.path.isdir(slice_path):
            continue
            
        print(f"  扫描 {slice_dir}...")
        stats['slices'][slice_dir] = {'database': 0, 'query': 0, 'total': 0}
        
        # 扫描database和query子目录
        for sub_dir in ['database', 'query']:
            sub_path = os.path.join(slice_path, sub_dir)
            if not os.path.exists(sub_path):
                continue
                
            # 收集该目录下的所有jpg图像
            image_files = [f for f in os.listdir(sub_path) if f.endswith('.jpg')]
            image_files.sort()  # 按文件名排序，确保时间顺序
            
            for image_file in image_files:
                full_path = os.path.join(sub_path, image_file)
                all_image_paths.append(full_path)
                
                # 统计信息
                stats['slices'][slice_dir][sub_dir] += 1
                stats['slices'][slice_dir]['total'] += 1
                stats['image_types'][sub_dir] += 1
                
                # 统计相机信息
                if '_c0_' in image_file:
                    stats['cameras']['c0'] += 1
                elif '_c1_' in image_file:
                    stats['cameras']['c1'] += 1
    
    stats['total_images'] = len(all_image_paths)
    
    print(f"扫描完成！共找到 {stats['total_images']} 张图像")
    return all_image_paths, stats


def extract_timestamp_from_filename(filename: str) -> int:
    """
    从文件名中提取时间戳
    
    Args:
        filename: 图像文件名
        
    Returns:
        int: 时间戳（微秒）
    """
    # 匹配格式：img_00274_c1_1287503834167650us_rect.jpg
    pattern = r'img_\d+_c[01]_(\d+)us_rect\.jpg'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0


def sample_images_by_step(image_paths: List[str], step: int) -> List[str]:
    """
    按步长抽样图像
    
    Args:
        image_paths: 所有图像路径列表
        step: 抽样步长
        
    Returns:
        List[str]: 抽样后的图像路径列表
    """
    print(f"按步长 {step} 抽样图像...")
    
    # 按文件名中的时间戳排序
    print("  按时间戳排序图像...")
    sorted_paths = sorted(image_paths, key=lambda x: extract_timestamp_from_filename(os.path.basename(x)))
    
    # 按步长抽样
    sampled_paths = sorted_paths[::step]
    
    print(f"抽样完成！从 {len(image_paths)} 张图像中抽取了 {len(sampled_paths)} 张")
    return sampled_paths


def save_image_paths_list(image_paths: List[str], output_file: str, title: str = "图像路径列表"):
    """
    保存图像路径列表到文件
    
    Args:
        image_paths: 图像路径列表
        output_file: 输出文件路径
        title: 列表标题
    """
    print(f"保存图像路径列表到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n")
        f.write(f"# 总数: {len(image_paths)} 张图像\n")
        f.write(f"# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, path in enumerate(image_paths, 1):
            f.write(f"{i:6d}: {path}\n")
    
    print(f"已保存 {len(image_paths)} 条图像路径")


def copy_preview_images(sampled_paths: List[str], output_dir: str, max_images: int = 100):
    """
    复制抽样后的图像到预览文件夹
    
    Args:
        sampled_paths: 抽样后的图像路径列表
        output_dir: 输出目录
        max_images: 最大复制图像数量
    """
    print(f"复制前 {max_images} 张抽样图像到预览文件夹: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制图像
    copied_count = 0
    for i, src_path in enumerate(sampled_paths[:max_images]):
        if not os.path.exists(src_path):
            print(f"警告：源文件不存在，跳过: {src_path}")
            continue
            
        # 生成新的文件名，包含序号
        filename = os.path.basename(src_path)
        new_filename = f"{i+1:03d}_{filename}"
        dst_path = os.path.join(output_dir, new_filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            if (i + 1) % 10 == 0:
                print(f"  已复制 {i + 1} 张图像...")
        except Exception as e:
            print(f"警告：复制文件失败: {src_path} -> {dst_path}, 错误: {e}")
    
    print(f"复制完成！共复制了 {copied_count} 张图像到 {output_dir}")


def print_statistics(stats: dict):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
    """
    print("\n=== 数据集统计信息 ===")
    print(f"总图像数量: {stats['total_images']}")
    print(f"相机统计: C0={stats['cameras']['c0']}, C1={stats['cameras']['c1']}")
    print(f"类型统计: Database={stats['image_types']['database']}, Query={stats['image_types']['query']}")
    
    print("\n各Slice图像数量:")
    for slice_name, slice_stats in sorted(stats['slices'].items(), key=lambda x: int(x[0].replace('slice', ''))):
        print(f"  {slice_name}: Database={slice_stats['database']}, Query={slice_stats['query']}, Total={slice_stats['total']}")


def main():
    parser = argparse.ArgumentParser(
        description="CMU-Seasons数据集图像采样工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法：步长10，复制100张预览图像
  python cmu_seasons_image_sampler.py -d /mnt/data/roborock/datasets/CMU-Seasons -s 10 -o sampled_images -n 100
  
  # 更大步长：步长50，复制50张预览图像
  python cmu_seasons_image_sampler.py -d /mnt/data/roborock/datasets/CMU-Seasons -s 50 -o preview_step50 -n 50
  
  # 只生成路径列表，不复制图像
  python cmu_seasons_image_sampler.py -d /mnt/data/roborock/datasets/CMU-Seasons -s 20 --no-copy
        """
    )
    
    parser.add_argument('-d', '--dataset', required=True,
                       help='CMU-Seasons数据集根目录路径')
    parser.add_argument('-s', '--step', type=int, default=10,
                       help='抽样步长 (默认: 10)')
    parser.add_argument('-o', '--output', default='sampled_images',
                       help='输出目录名称 (默认: sampled_images)')
    parser.add_argument('-n', '--num-preview', type=int, default=100,
                       help='复制的预览图像数量 (默认: 100)')
    parser.add_argument('--no-copy', action='store_true',
                       help='只生成路径列表，不复制预览图像')
    
    args = parser.parse_args()
    
    # 检查数据集路径
    if not os.path.exists(args.dataset):
        print(f"错误：数据集路径不存在: {args.dataset}")
        return 1
    
    print("=== CMU-Seasons 图像采样工具 ===")
    print(f"数据集路径: {args.dataset}")
    print(f"抽样步长: {args.step}")
    print(f"输出目录: {args.output}")
    print(f"预览图像数量: {args.num_preview}")
    print()
    
    try:
        # 步骤1：扫描数据集，收集所有图像路径
        all_image_paths, stats = scan_image_directory(args.dataset)
        
        # 打印统计信息
        print_statistics(stats)
        
        # 步骤2：保存完整的图像路径列表
        all_paths_file = f"{args.output}_all_images.txt"
        save_image_paths_list(all_image_paths, all_paths_file, "所有图像路径列表")
        print()
        
        # 步骤3：按步长抽样
        sampled_paths = sample_images_by_step(all_image_paths, args.step)
        
        # 步骤4：保存抽样后的图像路径列表
        sampled_paths_file = f"{args.output}_sampled_step{args.step}.txt"
        save_image_paths_list(sampled_paths, sampled_paths_file, 
                            f"抽样图像路径列表 (步长={args.step})")
        print()
        
        # 步骤5：复制预览图像（如果需要）
        if not args.no_copy:
            copy_preview_images(sampled_paths, args.output, args.num_preview)
        else:
            print("跳过复制预览图像 (--no-copy)")
        
        print("\n=== 处理完成 ===")
        print(f"生成的文件:")
        print(f"  - 完整路径列表: {all_paths_file}")
        print(f"  - 抽样路径列表: {sampled_paths_file}")
        if not args.no_copy:
            print(f"  - 预览图像目录: {args.output}/")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 