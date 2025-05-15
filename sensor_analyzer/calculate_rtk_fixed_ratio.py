#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算多个数据集的RTK固定解比例

此脚本遍历根目录下的所有数据集子目录，
计算每个子目录中rtk_pose.txt（固定解）与rtk_pose_full.txt（所有RTK数据）的行数比例
"""

import os
import glob
import sys
from prettytable import PrettyTable

def count_file_lines(file_path):
    """计算文件的行数，忽略以#开头的注释行"""
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r') as f:
        # 忽略以#开头的注释行
        return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))

def calculate_rtk_ratio(dataset_dir):
    """计算单个数据集的RTK固定解比例"""
    rtk_pose_path = os.path.join(dataset_dir, "rtk_pose.txt")
    rtk_pose_full_path = os.path.join(dataset_dir, "rtk_pose_full.txt")
    
    # 如果任一文件不存在，返回None
    if not os.path.exists(rtk_pose_path) or not os.path.exists(rtk_pose_full_path):
        return None
    
    # 计算文件行数
    fixed_count = count_file_lines(rtk_pose_path)
    total_count = count_file_lines(rtk_pose_full_path)
    
    # 计算比例，避免除以零
    if total_count == 0:
        ratio = 0
    else:
        ratio = fixed_count / total_count
    
    return {
        "dataset": os.path.basename(dataset_dir),
        "fixed_count": fixed_count,
        "total_count": total_count,
        "ratio": ratio
    }

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    print(f"正在搜索目录: {root_dir}")
    
    # 寻找所有可能包含RTK数据的子目录
    # 假设每个数据集目录都至少有rtk_pose.txt或rtk_pose_full.txt文件
    rtk_file_paths = glob.glob(os.path.join(root_dir, "**/rtk_pose*.txt"), recursive=True)
    
    if not rtk_file_paths:
        print("未找到任何包含RTK数据的目录!")
        return
    
    # 获取所有包含RTK数据的唯一目录
    dataset_dirs = set(os.path.dirname(path) for path in rtk_file_paths)
    
    # 计算每个数据集的RTK固定解比例
    results = []
    for dataset_dir in dataset_dirs:
        result = calculate_rtk_ratio(dataset_dir)
        if result:
            results.append(result)
    
    if not results:
        print("未找到任何有效的RTK数据！")
        return
    
    # 按比例降序排序结果
    results.sort(key=lambda x: x["ratio"], reverse=True)
    
    # 创建美观的表格显示结果
    table = PrettyTable()
    table.field_names = ["数据集", "固定解数量", "总RTK数量", "固定解比例"]
    
    for result in results:
        table.add_row([
            result["dataset"],
            result["fixed_count"],
            result["total_count"],
            f"{result['ratio']:.2%}"
        ])
    
    print(table)
    
    # 计算平均固定解比例
    avg_ratio = sum(r["ratio"] for r in results) / len(results)
    print(f"\n平均固定解比例: {avg_ratio:.2%}")
    
    # 输出一个简单的总结
    print(f"\n总共分析了 {len(results)} 个数据集")
    print(f"最高固定解比例: {max(r['ratio'] for r in results):.2%}")
    print(f"最低固定解比例: {min(r['ratio'] for r in results):.2%}")

if __name__ == "__main__":
    main() 