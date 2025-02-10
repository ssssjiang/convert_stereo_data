#!/usr/bin/env python3
import os
import argparse
import json
import zipfile
from tabulate import tabulate
from typing import List, Dict
from datetime import datetime


def find_test_dirs(root_dir: str) -> List[str]:
    """查找所有包含gt子文件夹的测试目录"""
    test_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'gt' in dirnames:  # 如果当前目录包含gt子文件夹
            test_dirs.append(dirpath)
    return test_dirs


def find_latest_eval_dir(res_dir: str) -> str:
    """查找最新的评估结果目录"""
    eval_dirs = [d for d in os.listdir(res_dir) if d.startswith('eval_')]
    if not eval_dirs:
        return ""
    return sorted(eval_dirs)[-1]


def parse_stats_file(zip_path: str) -> dict:
    """解析统计结果文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('stats.json') as f:
                stats = json.load(f)
                return {
                    'rmse': stats['rmse'],
                    'mean': stats['mean']
                }
    except Exception as e:
        print(f"解析统计文件失败 {zip_path}: {str(e)}")
        return {'rmse': float('nan'), 'mean': float('nan')}


def collect_results(root_dir: str, res_name: str = "res0") -> Dict:
    """收集所有测试目录的评估结果"""
    results = {}
    test_dirs = find_test_dirs(root_dir)

    if not test_dirs:
        print(f"在{root_dir}下未找到任何包含gt子文件夹的测试目录")
        return results

    for test_dir in test_dirs:
        test_name = os.path.relpath(test_dir, root_dir)  # 使用相对路径作为测试名称
        res_dir = os.path.join(test_dir, res_name)

        if not os.path.isdir(res_dir):
            print(f"未找到结果目录: {res_dir}")
            continue

        eval_dir_name = find_latest_eval_dir(res_dir)
        if not eval_dir_name:
            print(f"未找到评估结果: {res_dir}")
            continue

        eval_dir = os.path.join(res_dir, eval_dir_name)

        # 解析APE结果
        ape_stats = parse_stats_file(os.path.join(eval_dir, "ape_stats.zip"))
        # 解析RPE结果
        rpe_stats = parse_stats_file(os.path.join(eval_dir, "rpe_stats.zip"))

        results[test_name] = {
            'ape_rmse': ape_stats['rmse'],
            'ape_mean': ape_stats['mean'],
            'rpe_rmse': rpe_stats['rmse'],
            'rpe_mean': rpe_stats['mean']
        }

    return results


def print_results_table(results: Dict, root_dir: str, res_name: str):
    """打印评估结果表格"""
    if not results:
        print("没有可用的评估结果")
        return

    # 准备表格数据
    headers = ['测试目录', 'APE RMSE', 'APE Mean', 'RPE RMSE', 'RPE Mean']
    table_data = []

    for test_name, stats in results.items():
        row = [
            test_name,
            f"{stats['ape_rmse']:.4f}",
            f"{stats['ape_mean']:.4f}",
            f"{stats['rpe_rmse']:.4f}",
            f"{stats['rpe_mean']:.4f}"
        ]
        table_data.append(row)

    # 按测试目录名称排序
    table_data.sort(key=lambda x: x[0])

    # 打印表格
    print("\n评估结果统计：")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # 保存到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"trajectory_evaluation_{res_name}_{timestamp}.csv"
    csv_path = os.path.join(root_dir, csv_filename)

    # 创建markdown格式的表格文件
    md_filename = f"trajectory_evaluation_{res_name}_{timestamp}.md"
    md_path = os.path.join(root_dir, md_filename)

    # 保存CSV
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table_data:
            f.write(','.join(row) + '\n')

    # 保存Markdown
    with open(md_path, 'w') as f:
        f.write(f"# 轨迹评估结果统计 ({res_name})\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='pipe'))

    print(f"\n结果已保存到：")
    print(f"- CSV: {csv_path}")
    print(f"- Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="汇总SLAM轨迹评估结果"
    )
    parser.add_argument(
        "root_dir",
        help="数据集根目录，将递归搜索其下所有包含gt子文件夹的测试目录"
    )
    parser.add_argument(
        "--res-name",
        default="res0",
        help="要评估的结果目录名称（默认：res0）"
    )

    args = parser.parse_args()
    results = collect_results(args.root_dir, args.res_name)
    print_results_table(results, args.root_dir, args.res_name)


if __name__ == "__main__":
    main()