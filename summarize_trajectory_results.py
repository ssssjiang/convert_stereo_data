#!/usr/bin/env python3
import os
import argparse
import json
import zipfile
from tabulate import tabulate
from typing import List, Dict
from datetime import datetime
import yaml


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


def parse_statistics_file(yaml_path: str) -> dict:
    """解析统计文件"""
    try:
        # 首先尝试直接读取文件内容
        with open(yaml_path, 'r') as f:
            content = f.read()

        # 预处理内容，确保正确的YAML格式
        processed_content = {}
        current_key = None
        current_data = {}

        for line in content.split('\n'):
            line = line.rstrip()
            if not line:
                continue

            # 处理主键
            if not line.startswith(' '):
                if current_key and current_data:
                    processed_content[current_key] = current_data.copy()
                    current_data.clear()

                if ':' in line:
                    current_key = line.split(':')[0].strip()
                continue

            # 处理值
            parts = [p.strip() for p in line.strip().split(':')]
            if len(parts) == 2:
                key, value = parts
                try:
                    current_data[key] = float(value)
                except ValueError:
                    current_data[key] = value

        # 添加最后一组数据
        if current_key and current_data:
            processed_content[current_key] = current_data.copy()

        # 处理结果
        results = {}
        for key, value in processed_content.items():
            if isinstance(value, dict) and 'samples' in value and 'stddev' in value:
                try:
                    stddev = float(value['stddev'])
                    if abs(stddev) < 1e-10:  # 标准差接近0
                        results[key] = int(float(value['samples']))
                    else:
                        results[key] = float(value['mean'])
                except (ValueError, TypeError) as e:
                    print(f"处理数据时出错 - 键: {key}, 值: {value}, 错误: {str(e)}")

        # 调试信息
        if not results:
            print(f"警告: 未能从文件中提取任何有效数据: {yaml_path}")
            print("处理后的内容:")
            for k, v in processed_content.items():
                print(f"  {k}: {v}")

        return results

    except Exception as e:
        print(f"解析统计文件失败 {yaml_path}: {str(e)}")
        print(f"错误发生在处理以下内容时:\n{content[:500]}...")  # 打印更多内容以帮助调试
        return {}


def collect_results(root_dir: str, res_name: str = "res0") -> Dict:
    """收集所有测试目录的评估结果"""
    trajectory_results = {}
    statistics_results = {}
    test_dirs = find_test_dirs(root_dir)

    if not test_dirs:
        print(f"在{root_dir}下未找到任何包含gt子文件夹的测试目录")
        return trajectory_results, statistics_results

    for test_dir in test_dirs:
        test_name = os.path.relpath(test_dir, root_dir)
        res_dir = os.path.join(test_dir, res_name)

        if not os.path.isdir(res_dir):
            print(f"未找到结果目录: {res_dir}")
            continue

        # 处理轨迹评估结果
        eval_dir_name = find_latest_eval_dir(res_dir)
        if eval_dir_name:
            eval_dir = os.path.join(res_dir, eval_dir_name)
            ape_stats = parse_stats_file(os.path.join(eval_dir, "ape_stats.zip"))
            rpe_stats = parse_stats_file(os.path.join(eval_dir, "rpe_stats.zip"))

            trajectory_results[test_name] = {
                'ape_rmse': ape_stats['rmse'],
                'ape_mean': ape_stats['mean'],
                'rpe_rmse': rpe_stats['rmse'],
                'rpe_mean': rpe_stats['mean']
            }

        # 处理statistics.yaml
        statistics_path = os.path.join(res_dir, "statistics.yaml")
        if os.path.isfile(statistics_path):
            stats = parse_statistics_file(statistics_path)
            if stats:
                statistics_results[test_name] = stats

    return trajectory_results, statistics_results


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


def print_statistics_table(results: Dict, root_dir: str, res_name: str):
    """打印统计数据表格"""
    if not results:
        print("没有可用的统计数据")
        return

    # 定义需要保留的指标和其显示名称
    metrics_to_keep = {
        '[Final Frame0] unconverged seed num': 'Unconverged Seeds',
        '[Final Frame0] feature num': 'Features',
        '[Final Frame0] landmark num': 'Landmarks',
        '[Ransac] ransac angle diff': 'RANSAC Angle Diff',
        '[Ransac] ransac trans diff': 'RANSAC Trans Diff',
        '[Tracking] after_remove_outliers': 'Track After',
        '[Tracking] before_remove_outliers': 'Track Before'
    }

    # 准备表格数据
    headers = ['测试目录'] + list(metrics_to_keep.values())
    table_data = []

    for test_name, stats in sorted(results.items()):
        row = [test_name]
        for full_name, _ in metrics_to_keep.items():
            value = stats.get(full_name, float('nan'))
            if isinstance(value, (int, float)):
                row.append(f"{value:.2f}" if isinstance(value, float) else str(value))
            else:
                row.append(str(value))
        table_data.append(row)

    # 打印表格
    print("\n运行统计数据：")
    print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))

    # 保存到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"statistics_{res_name}_{timestamp}.csv"
    csv_path = os.path.join(root_dir, csv_filename)

    # 创建markdown格式的表格文件
    md_filename = f"statistics_{res_name}_{timestamp}.md"
    md_path = os.path.join(root_dir, md_filename)

    # 保存CSV
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table_data:
            f.write(','.join(str(x) for x in row) + '\n')

    # 保存Markdown
    with open(md_path, 'w') as f:
        f.write(f"# SLAM运行统计数据 ({res_name})\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='pipe'))

    print(f"\n统计数据已保存到：")
    print(f"- CSV: {csv_path}")
    print(f"- Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="汇总SLAM轨迹评估结果和运行统计数据"
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
    trajectory_results, statistics_results = collect_results(args.root_dir, args.res_name)
    print_results_table(trajectory_results, args.root_dir, args.res_name)
    print_statistics_table(statistics_results, args.root_dir, args.res_name)


if __name__ == "__main__":
    main()