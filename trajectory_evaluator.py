#!/usr/bin/env python3
import os
import argparse
import subprocess
from typing import List, Optional
from datetime import datetime
import sys
import json
import zipfile
from tabulate import tabulate  # 需要安装：pip install tabulate


class TrajectoryEvaluator:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}  # 存储评估结果

    def log(self, message: str):
        """输出日志信息"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

    def run_evo_command(self, cmd: List[str], save_path: str) -> bool:
        """
        执行evo命令并保存结果

        Args:
            cmd: evo命令及参数列表
            save_path: 结果保存路径

        Returns:
            bool: 命令执行是否成功
        """
        self.log(f"执行命令: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            output = result.stdout

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(output)
            self.log(f"评估结果已保存到: {save_path}")
            return True

        except subprocess.CalledProcessError as e:
            self.log(f"命令执行失败: {' '.join(cmd)}")
            self.log(f"错误信息: {e.stdout}")
            return False
        except Exception as e:
            self.log(f"发生未知错误: {str(e)}")
            return False

    def parse_stats_file(self, zip_path: str) -> dict:
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
            self.log(f"解析统计文件失败 {zip_path}: {str(e)}")
            return {'rmse': float('nan'), 'mean': float('nan')}

    def process_directory(self, test_dir: str, evo_args: Optional[List[str]] = None) -> bool:
        """
        评估单个测试目录的轨迹数据

        Args:
            test_dir: 测试目录路径
            evo_args: 额外的evo参数

        Returns:
            bool: 处理是否成功
        """
        gt_path = os.path.join(test_dir, "gt", "tof_pose.txt")

        # 查找所有结果目录
        result_dirs = []
        i = 0
        while True:
            res_dir = os.path.join(test_dir, f"res{i}")
            if not os.path.isdir(res_dir):
                break
            result_dirs.append(res_dir)
            i += 1

        if not result_dirs:
            self.log(f"未找到任何结果目录(res0, res1, ...): {test_dir}")
            return False

        # 检查ground truth文件
        if not os.path.isfile(gt_path):
            self.log(f"未找到ground truth文件: {gt_path}")
            return False

        success = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for res_dir in result_dirs:
            res_path = os.path.join(res_dir, "pose.txt")
            if not os.path.isfile(res_path):
                self.log(f"未找到结果文件: {res_path}")
                continue

            # 将评估结果保存在对应的结果目录下
            res_name = os.path.basename(res_dir)
            eval_dir = os.path.join(res_dir, f"eval_{timestamp}")
            os.makedirs(eval_dir, exist_ok=True)

            # 执行APE评估并保存数据
            ape_prefix = os.path.join(eval_dir, "ape")
            ape_cmd = [
                "evo_ape",
                "tum",
                "-v",
                "-a",
                "--save_results", f"{ape_prefix}_stats.zip",
                gt_path,
                res_path
            ]
            if evo_args:
                ape_cmd.extend(evo_args)
            ape_success = self.run_evo_command(ape_cmd, f"{ape_prefix}_log.txt")

            # 生成APE俯视图
            ape_cmd = [
                "evo_ape",
                "tum",
                "-v",
                "-a",
                "--plot_mode", "yx",
                "--save_plot", f"{ape_prefix}_top.pdf",
                gt_path,
                res_path
            ]
            ape_top_success = self.run_evo_command(ape_cmd, f"{ape_prefix}_top_log.txt")

            # 生成APE侧视图
            ape_cmd = [
                "evo_ape",
                "tum",
                "-v",
                "-a",
                "--plot_mode", "yz",
                "--save_plot", f"{ape_prefix}_side.pdf",
                gt_path,
                res_path
            ]
            ape_side_success = self.run_evo_command(ape_cmd, f"{ape_prefix}_side_log.txt")

            # 执行RPE评估并保存数据
            rpe_prefix = os.path.join(eval_dir, "rpe")
            rpe_cmd = [
                "evo_rpe",
                "tum",
                "-v",
                "-a",
                "--pose_relation", "trans_part",
                "--delta", "1",
                "--delta_unit", "m",
                "--all_pair",
                "--save_results", f"{rpe_prefix}_stats.zip",
                gt_path,
                res_path
            ]
            if evo_args:
                rpe_cmd.extend(evo_args)
            rpe_success = self.run_evo_command(rpe_cmd, f"{rpe_prefix}_log.txt")

            # 生成RPE俯视图
            rpe_cmd = [
                "evo_rpe",
                "tum",
                "-v",
                "-a",
                "--pose_relation", "trans_part",
                "--delta", "1",
                "--delta_unit", "m",
                "--all_pair",
                "--plot_mode", "yx",
                "--save_plot", f"{rpe_prefix}_top.pdf",
                gt_path,
                res_path
            ]
            rpe_top_success = self.run_evo_command(rpe_cmd, f"{rpe_prefix}_top_log.txt")

            # 生成RPE侧视图
            rpe_cmd = [
                "evo_rpe",
                "tum",
                "-v",
                "-a",
                "--pose_relation", "trans_part",
                "--delta", "1",
                "--delta_unit", "m",
                "--all_pair",
                "--plot_mode", "yz",
                "--save_plot", f"{rpe_prefix}_side.pdf",
                gt_path,
                res_path
            ]
            rpe_side_success = self.run_evo_command(rpe_cmd, f"{rpe_prefix}_side_log.txt")

            # 生成轨迹俯视图
            traj_cmd = [
                "evo_traj",
                "tum",
                "--ref", gt_path,
                "--plot_mode", "yx",
                "--save_plot", os.path.join(eval_dir, "trajectories_top.pdf"),
                res_path
            ]
            traj_top_success = self.run_evo_command(traj_cmd, os.path.join(eval_dir, "trajectories_top_log.txt"))

            # 生成轨迹侧视图
            traj_cmd = [
                "evo_traj",
                "tum",
                "--ref", gt_path,
                "--plot_mode", "yz",
                "--save_plot", os.path.join(eval_dir, "trajectories_side.pdf"),
                res_path
            ]
            traj_side_success = self.run_evo_command(traj_cmd, os.path.join(eval_dir, "trajectories_side_log.txt"))

            success = success and ape_success and ape_top_success and ape_side_success and \
                      rpe_success and rpe_top_success and rpe_side_success and \
                      traj_top_success and traj_side_success

            # 保存评估结果
            test_name = os.path.basename(test_dir)

            # 解析APE结果
            ape_stats = self.parse_stats_file(f"{ape_prefix}_stats.zip")
            # 解析RPE结果
            rpe_stats = self.parse_stats_file(f"{rpe_prefix}_stats.zip")

            # 存储结果
            if test_name not in self.results:
                self.results[test_name] = {}
            self.results[test_name][res_name] = {
                'ape_rmse': ape_stats['rmse'],
                'ape_mean': ape_stats['mean'],
                'rpe_rmse': rpe_stats['rmse'],
                'rpe_mean': rpe_stats['mean']
            }

        return success

    def print_results_table(self):
        """打印评估结果表格"""
        if not self.results:
            self.log("没有可用的评估结果")
            return

        # 准备表格数据
        headers = ['测试目录', '结果目录', 'APE RMSE', 'APE Mean', 'RPE RMSE', 'RPE Mean']
        table_data = []

        for test_name, res_dict in self.results.items():
            for res_name, stats in res_dict.items():
                row = [
                    test_name,
                    res_name,
                    f"{stats['ape_rmse']:.4f}",
                    f"{stats['ape_mean']:.4f}",
                    f"{stats['rpe_rmse']:.4f}",
                    f"{stats['rpe_mean']:.4f}"
                ]
                table_data.append(row)

        # 打印表格
        self.log("\n评估结果统计：")
        self.log(tabulate(table_data, headers=headers, tablefmt='grid'))

        # 保存到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"trajectory_evaluation_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write(','.join(headers) + '\n')
            for row in table_data:
                f.write(','.join(row) + '\n')
        self.log(f"\n结果已保存到：{csv_file}")


def find_test_dirs(root_dir: str) -> List[str]:
    """查找所有包含gt子文件夹的测试目录"""
    test_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'gt' in dirnames:  # 如果当前目录包含gt子文件夹
            test_dirs.append(dirpath)
    return test_dirs


def main():
    parser = argparse.ArgumentParser(
        description="批量评估SLAM轨迹数据，使用evo工具计算APE和RPE指标"
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
    parser.add_argument(
        "--evo-args",
        nargs="*",
        help="传递给evo的额外参数"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少输出信息"
    )

    args = parser.parse_args()
    evaluator = TrajectoryEvaluator(verbose=not args.quiet)

    test_dirs = find_test_dirs(args.root_dir)
    if not test_dirs:
        evaluator.log(f"在{args.root_dir}下未找到任何包含gt子文件夹的测试目录")
        sys.exit(1)

    total = len(test_dirs)
    success = 0

    for i, test_dir in enumerate(sorted(test_dirs), 1):
        rel_path = os.path.relpath(test_dir, args.root_dir)
        evaluator.log(f"处理目录 [{i}/{total}]: {rel_path}")

        res_dir = os.path.join(test_dir, args.res_name)
        if not os.path.isdir(res_dir):
            evaluator.log(f"结果目录不存在: {res_dir}")
            continue

        if evaluator.process_directory(test_dir, args.evo_args):
            success += 1

    evaluator.log(f"评估完成: {success}/{total} 个目录处理成功")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()