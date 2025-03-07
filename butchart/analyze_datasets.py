#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
import shutil
from analyze_frame_rate import analyze_frame_rate

def setup_logging(log_file=None):
    """设置日志记录"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # 创建控制台处理器
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    
    # 获取根日志记录器并添加处理器
    logger = logging.getLogger()
    logger.handlers = []  # 清除现有处理器
    logger.addHandler(console)
    
    # 如果提供了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

def analyze_dataset(dataset_path, logger):
    """分析单个数据集的相机时间戳"""
    logger.info(f"开始分析数据集: {dataset_path}")
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    # 检查camera目录是否存在
    camera_dir = os.path.join(dataset_path, "camera")
    if not os.path.exists(camera_dir):
        logger.error(f"相机目录不存在: {camera_dir}")
        return False
    
    # 创建分析结果目录
    analysis_dir = os.path.join(dataset_path, "timestamp_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 设置分析日志文件
    analysis_log_file = os.path.join(analysis_dir, "timestamp_analysis.log")
    
    # 重定向标准输出到日志文件
    original_stdout = sys.stdout
    with open(analysis_log_file, 'w') as f:
        sys.stdout = f
        
        # 执行时间戳分析
        try:
            # 分析相机时间戳
            analyze_frame_rate(image_dir=camera_dir, plot=True, save_drops=True)
        except Exception as e:
            logger.error(f"分析数据集时出错: {str(e)}")
            sys.stdout = original_stdout
            return False
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout
    
    # 移动生成的图像文件到分析目录
    current_dir = os.getcwd()
    for file in os.listdir(current_dir):
        if file.endswith('.png') and (file.startswith('Camera') or file.startswith('camera')):
            src_file = os.path.join(current_dir, file)
            dst_file = os.path.join(analysis_dir, file)
            shutil.move(src_file, dst_file)
    
    logger.info(f"数据集分析完成，结果保存在: {analysis_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='分析多个数据集的相机时间戳')
    parser.add_argument('--yaml', required=True, help='包含数据集列表的YAML文件路径')
    parser.add_argument('--log', help='日志文件路径')
    args = parser.parse_args()
    
    # 设置日志记录
    logger = setup_logging(args.log)
    
    # 读取YAML文件
    try:
        with open(args.yaml, 'r') as f:
            datasets = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"读取YAML文件时出错: {str(e)}")
        return 1
    
    if not datasets or not isinstance(datasets, list):
        logger.error("YAML文件格式错误或不包含数据集列表")
        return 1
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始批量分析 {len(datasets)} 个数据集，时间: {start_time}")
    
    # 分析每个数据集
    success_count = 0
    for i, dataset_path in enumerate(datasets):
        logger.info(f"处理数据集 {i+1}/{len(datasets)}: {dataset_path}")
        if analyze_dataset(dataset_path, logger):
            success_count += 1
    
    # 记录结束时间和统计信息
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"批量分析完成，时间: {end_time}")
    logger.info(f"总耗时: {duration}")
    logger.info(f"成功分析: {success_count}/{len(datasets)} 个数据集")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 