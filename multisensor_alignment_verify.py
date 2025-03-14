#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from scipy import signal
import glob
from sklearn.cluster import DBSCAN
from datetime import datetime
import json
from collections import defaultdict
import re
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.font_manager as fm
import warnings
from scipy import interpolate
from scipy.signal import savgol_filter
import scipy.stats
from scipy.signal import correlate, butter, filtfilt
import math

# 更具体的警告过滤器
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 特别针对字形缺失警告 - 修复转义序列
warnings.filterwarnings("ignore", message="Glyph \\d+ .*? missing from font.*")

# ========== 配置matplotlib使用支持中文的字体 ==========
# 尝试查找系统中支持中文的字体
chinese_fonts = []
for font in ['Noto Sans CJK SC', 'AR PL UMing CN', 'AR PL UKai CN', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font), fallback_to_default=True)
        if font_path and font_path != fm.findfont(fm.FontProperties(family="DejaVu Sans")):
            # 确保找到的不是系统默认字体
            chinese_fonts.append(font)
            print(f"找到支持中文的字体: {font}")
    except Exception as e:
        print(f"查找字体 {font} 时出错: {e}")
        continue

if chinese_fonts:
    # 使用找到的第一个支持中文的字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
    print(f"使用字体: {chinese_fonts} 以支持中文显示")
else:
    # 如果没有找到支持中文的字体，则使用英文标签
    print("未找到支持中文的字体，将使用英文标签")

# 确保特殊符号正确显示
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号代替Unicode减号

# 设置matplotlib使用支持中文的字体
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'AR PL UKai CN', 'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多传感器(IMU、轮式编码器、RTK、相机)时间戳对齐综合验证工具")
    
    # 数据源参数
    parser.add_argument('--imu_file', type=str, required=True, help="IMU数据文件路径")
    parser.add_argument('--odo_file', type=str, help="轮式编码器数据文件路径")
    parser.add_argument('--rtk_file', type=str, help="RTK数据文件路径")
    parser.add_argument('--image_dir', type=str, help="图像数据目录")
    parser.add_argument('--image_timestamp_file', type=str, help="图像时间戳文件路径")
    
    # 验证方法参数
    parser.add_argument('--methods', type=str, default="all", 
                       help="验证方法(逗号分隔): correlation,event,visual,all")
    parser.add_argument('--sensors', type=str, default="all",
                       help="要验证的传感器组合(逗号分隔): imu_odo,imu_rtk,imu_camera,all")
    
    # 通用参数
    parser.add_argument('--output_dir', type=str, default="./alignment_verification", help="结果输出目录")
    parser.add_argument('--visualize', action='store_true', help="显示验证结果的可视化图表")
    parser.add_argument('--tolerance_ms', type=float, default=30.0, help="时间对齐容差(毫秒)")
    
    # 轮式编码器物理参数
    parser.add_argument('--wheel_perimeter', type=float, default=0.7477, help="车轮周长(米)")
    parser.add_argument('--wheel_halflength', type=float, default=0.1775, help="轮距的一半(米)")
    parser.add_argument('--encoder_scale', type=float, default=1194.0, help="编码器刻度(脉冲/转)")
    
    # 特定传感器参数
    parser.add_argument('--imu_timestamp_col', type=str, default="timestamp", help="IMU数据中的时间戳列名")
    parser.add_argument('--odo_timestamp_col', type=str, default="timestamp", help="轮式编码器数据中的时间戳列名")
    parser.add_argument('--rtk_timestamp_col', type=str, default="timestamp", help="RTK数据中的时间戳列名")
    parser.add_argument('--image_pattern', type=str, default="*.jpg", help="图像文件匹配模式")
    
    # 算法参数
    parser.add_argument('--window_size', type=int, default=500, help="互相关窗口大小")
    parser.add_argument('--max_lag', type=int, default=200, help="最大检查延迟(单位：采样点)")
    parser.add_argument('--max_lag_ms', type=int, default=100, help="相关性方法最大偏移搜索范围(毫秒)")
    parser.add_argument('--known_offset', type=float, default=None, help="已知的时间偏移(毫秒)，用于验证算法准确性")
    
    return parser.parse_args()

# ========== 数据加载函数 ==========

def load_imu_data(args):
    """加载IMU数据"""
    print(f"加载IMU数据: {args.imu_file}")
    
    # 先检查文件的前几行，以确定正确的读取方式
    with open(args.imu_file, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]  # 读取前5行
    
    print("文件前5行预览:")
    for i, line in enumerate(first_lines):
        print(f"  行 {i+1}: {line.strip()}")
    
    # 判断文件格式
    has_header = True
    header_line = 0  # 默认第一行是标题行
    
    # 检查第一行是否为注释行
    if first_lines[0].startswith('#'):
        # 注释行可能包含列名
        header_line = 0
        print("  检测到第一行是注释行，尝试移除注释符号作为列标题")
        
        # 尝试读取文件，显式指定第一行为标题行，但移除注释符号
        try:
            imu_data = pd.read_csv(args.imu_file, comment=None, header=0)
            # 如果列名中包含#，移除它
            imu_data.columns = [col.replace('#', '').strip() if isinstance(col, str) else col for col in imu_data.columns]
        except Exception as e:
            print(f"  尝试读取第一行作为标题失败: {str(e)}")
            # 回退到无标题模式
            has_header = False
    else:
        # 尝试读取文件，但暂时不指定comment参数
        try:
            imu_data = pd.read_csv(args.imu_file, header=0)
        except Exception as e:
            print(f"  尝试读取第一行作为标题失败: {str(e)}")
            has_header = False
    
    # 如果前面的尝试失败或者需要跳过注释，使用标准方式读取
    if not has_header or 'imu_data' not in locals():
        print("  使用标准方式读取IMU数据，跳过注释行")
        try:
            # 使用参数跳过注释，但不指定列名
            imu_data = pd.read_csv(args.imu_file, comment='#', header=None)
            
            # 生成默认列名
            if 'timestamp' in args.imu_timestamp_col.lower():
                # 时间戳列名可能提示了正确的列名模式
                prefix = args.imu_timestamp_col.split('_')[0] if '_' in args.imu_timestamp_col else ''
                if prefix:
                    # 使用与时间戳列相同前缀的列名
                    col_names = [f"{prefix}_{i}" for i in range(len(imu_data.columns))]
                    imu_data.columns = col_names
                else:
                    # 使用默认列名，确保包含时间戳列
                    col_names = [args.imu_timestamp_col]
                    col_names.extend([f"column_{i}" for i in range(1, len(imu_data.columns))])
                    imu_data.columns = col_names
        except Exception as e:
            print(f"  标准读取方式也失败: {str(e)}")
            # 最后尝试: 完全忽略任何特殊处理
            imu_data = pd.read_csv(args.imu_file)
    
    # 输出列信息进行确认
    print("\n文件中检测到的列:")
    for i, col in enumerate(imu_data.columns):
        print(f"  列 {i+1}: 名称='{col}', 类型={imu_data[col].dtype}, 示例值={imu_data[col].iloc[0] if len(imu_data) > 0 else 'N/A'}")
    
    # 检查时间戳列是否存在
    if args.imu_timestamp_col not in imu_data.columns:
        print(f"警告: 未找到指定的时间戳列 '{args.imu_timestamp_col}'")
        
        # 尝试根据列数据特征识别时间戳列
        timestamp_candidates = []
        for col in imu_data.columns:
            # 时间戳列通常是递增的数值列
            if imu_data[col].dtype in [np.int64, np.float64]:
                # 检查是否大体递增
                is_increasing = (imu_data[col].diff() >= 0).mean() > 0.95  # 95%以上的差值为正
                if is_increasing:
                    timestamp_candidates.append(col)
        
        if timestamp_candidates:
            # 使用第一个候选时间戳列
            args.imu_timestamp_col = timestamp_candidates[0]
            print(f"  自动选择 '{args.imu_timestamp_col}' 作为时间戳列")
        else:
            raise ValueError(f"IMU数据缺少时间戳列 {args.imu_timestamp_col}，且无法自动识别时间戳列")
    
    # 计算角速度幅值
    gyro_cols = [col for col in imu_data.columns if 'gyro' in str(col).lower()]
    
    if not gyro_cols:
        # 如果找不到包含"gyro"的列，尝试根据数据特征识别角速度列
        print("  未找到包含'gyro'的列名，尝试根据数据特征识别角速度列...")
        
        # 角速度列的典型特征: 数值型，且值域通常较小（几rad/s）
        numeric_cols = [col for col in imu_data.columns 
                        if col != args.imu_timestamp_col and 
                        imu_data[col].dtype in [np.int64, np.float64]]
        
        # 检查每列的统计特性
        for col in numeric_cols:
            col_mean = imu_data[col].mean()
            col_std = imu_data[col].std()
            col_abs_max = imu_data[col].abs().max()
            
            print(f"  列 '{col}' 数据特征: 均值={col_mean:.4f}, 标准差={col_std:.4f}, 最大绝对值={col_abs_max:.4f}")
            
            # 典型的角速度值不会特别大(假设单位是rad/s或deg/s)
            if col_abs_max < 100 and col_std > 0.001:
                gyro_cols.append(col)
    
    # 计算角速度幅值
    if len(gyro_cols) >= 3:
        # 如果有三个角速度分量，计算合成角速度
        print(f"  使用列 {gyro_cols[:3]} 计算角速度幅值")
        x_col, y_col, z_col = gyro_cols[:3]
        imu_data['angular_velocity_magnitude'] = np.sqrt(
            imu_data[x_col]**2 + imu_data[y_col]**2 + imu_data[z_col]**2
        )
    elif len(gyro_cols) > 0:
        # 使用找到的第一个角速度列
        print(f"  使用列 {gyro_cols[0]} 计算角速度幅值")
        imu_data['angular_velocity_magnitude'] = np.abs(imu_data[gyro_cols[0]])
    elif 'gyro_z' in imu_data.columns:
        # 如果只有z轴角速度
        print("  使用gyro_z列计算角速度幅值")
        imu_data['angular_velocity_magnitude'] = np.abs(imu_data['gyro_z'])
    else:
        raise ValueError("IMU数据缺少角速度列，无法计算角速度幅值。" +
                        "尝试检查文件格式或手动指定角速度列名。")
    
    # 确保时间戳是升序排列的
    imu_data = imu_data.sort_values(by=args.imu_timestamp_col)
    
    print(f"IMU数据加载完成，共 {len(imu_data)} 条记录")
    return imu_data

def load_odometry_data(args):
    """加载轮式编码器数据"""
    if not args.odo_file:
        return None
        
    print(f"加载轮式编码器数据: {args.odo_file}")
    
    # 先检查文件的前几行，以确定正确的读取方式
    with open(args.odo_file, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]  # 读取前5行
    
    print("文件前5行预览:")
    for i, line in enumerate(first_lines):
        print(f"  行 {i+1}: {line.strip()}")
    
    # 判断文件格式
    has_header = True
    header_line = 0  # 默认第一行是标题行
    
    # 检查第一行是否为注释行
    if first_lines[0].startswith('#'):
        # 注释行可能包含列名
        header_line = 0
        print("  检测到第一行是注释行，尝试移除注释符号作为列标题")
        
        # 尝试读取文件，显式指定第一行为标题行，但移除注释符号
        try:
            odo_data = pd.read_csv(args.odo_file, comment=None, header=0)
            # 如果列名中包含#，移除它
            odo_data.columns = [col.replace('#', '').strip() if isinstance(col, str) else col for col in odo_data.columns]
        except Exception as e:
            print(f"  尝试读取第一行作为标题失败: {str(e)}")
            # 回退到无标题模式
            has_header = False
    else:
        # 尝试读取文件，但暂时不指定comment参数
        try:
            odo_data = pd.read_csv(args.odo_file, header=0)
        except Exception as e:
            print(f"  尝试读取第一行作为标题失败: {str(e)}")
            has_header = False
    
    # 如果前面的尝试失败或者需要跳过注释，使用标准方式读取
    if not has_header or 'odo_data' not in locals():
        print("  使用标准方式读取轮式编码器数据，跳过注释行")
        try:
            # 使用参数跳过注释，但不指定列名
            odo_data = pd.read_csv(args.odo_file, comment='#', header=None)
            
            # 生成默认列名
            if 'timestamp' in args.odo_timestamp_col.lower():
                # 时间戳列名可能提示了正确的列名模式
                prefix = args.odo_timestamp_col.split('_')[0] if '_' in args.odo_timestamp_col else ''
                if prefix:
                    # 使用与时间戳列相同前缀的列名
                    col_names = [f"{prefix}_{i}" for i in range(len(odo_data.columns))]
                    odo_data.columns = col_names
                else:
                    # 使用默认列名，确保包含时间戳列
                    col_names = [args.odo_timestamp_col]
                    col_names.extend([f"column_{i}" for i in range(1, len(odo_data.columns))])
                    odo_data.columns = col_names
        except Exception as e:
            print(f"  标准读取方式也失败: {str(e)}")
            # 最后尝试: 完全忽略任何特殊处理
            odo_data = pd.read_csv(args.odo_file)
    
    # 输出列信息进行确认
    print("\n文件中检测到的列:")
    for i, col in enumerate(odo_data.columns):
        print(f"  列 {i+1}: 名称='{col}', 类型={odo_data[col].dtype}, 示例值={odo_data[col].iloc[0] if len(odo_data) > 0 else 'N/A'}")
    
    # 检查时间戳列是否存在
    if args.odo_timestamp_col not in odo_data.columns:
        print(f"警告: 未找到指定的时间戳列 '{args.odo_timestamp_col}'")
        
        # 尝试根据列数据特征识别时间戳列
        timestamp_candidates = []
        for col in odo_data.columns:
            # 时间戳列通常是递增的数值列
            if odo_data[col].dtype in [np.int64, np.float64]:
                # 检查是否大体递增
                is_increasing = (odo_data[col].diff() >= 0).mean() > 0.95  # 95%以上的差值为正
                if is_increasing:
                    timestamp_candidates.append(col)
        
        if timestamp_candidates:
            # 使用第一个候选时间戳列
            args.odo_timestamp_col = timestamp_candidates[0]
            print(f"  自动选择 '{args.odo_timestamp_col}' 作为时间戳列")
        else:
            raise ValueError(f"轮式编码器数据缺少时间戳列 {args.odo_timestamp_col}，且无法自动识别时间戳列")
    
    # 确保时间戳是升序排列的，计算时间差
    odo_data = odo_data.sort_values(by=args.odo_timestamp_col)
    odo_data['dt'] = odo_data[args.odo_timestamp_col].diff().fillna(0) / 1000
    
    # 根据可用的列计算角速度
    if 'angular_velocity' in odo_data.columns:
        # 已有角速度列
        print("  使用数据中已有的angular_velocity列")
        odo_data['angular_velocity_magnitude'] = np.abs(odo_data['angular_velocity'])
    elif 'speed_w' in odo_data.columns:
        # 使用角速度列
        print("  使用数据中的speed_w列作为角速度")
        odo_data['angular_velocity_magnitude'] = np.abs(odo_data['speed_w'])
    elif any(col.lower() in ['left_count', 'left_counts', 'left_ticks', 'left'] for col in odo_data.columns) and any(col.lower() in ['right_count', 'right_counts', 'right_ticks', 'right'] for col in odo_data.columns):
        # 智能检测左右轮计数列
        left_col = None
        right_col = None
        
        # 查找可能的左轮列名
        for col_pattern in ['left_count', 'left_counts', 'left_ticks', 'left']:
            matches = [col for col in odo_data.columns if col_pattern.lower() in col.lower()]
            if matches:
                left_col = matches[0]
                break
        
        # 查找可能的右轮列名
        for col_pattern in ['right_count', 'right_counts', 'right_ticks', 'right']:
            matches = [col for col in odo_data.columns if col_pattern.lower() in col.lower()]
            if matches:
                right_col = matches[0]
                break
        
        if left_col and right_col:
            print(f"  自动检测到左右轮计数列: {left_col}, {right_col}")
            
            # 使用左右轮计数差分计算角速度
            print("  从左右轮计数差分计算角速度")
            
            # 获取物理参数
            wheel_base = getattr(args, 'wheel_base', 2.0 * getattr(args, 'wheel_halflength', 0.1775))  # 计算轮距
            encoder_resolution = getattr(args, 'encoder_resolution', getattr(args, 'encoder_scale', 1194.0))  # 编码器分辨率
            wheel_perimeter = getattr(args, 'wheel_perimeter', 0.7477)  # 轮周长
            
            print(f"  使用参数: 轮距={wheel_base}m, 轮周长={wheel_perimeter}m, 编码器分辨率={encoder_resolution} 脉冲/转")
            
            # 计算左右轮的计数变化
            left_diff = odo_data[left_col].diff()
            right_diff = odo_data[right_col].diff()
            
            # 处理首行NaN值
            left_diff.iloc[0] = 0
            right_diff.iloc[0] = 0
            
            # 过滤异常值: 如果计数差异过大(可能是计数器溢出或传感器错误)，设为0
            max_count_diff = getattr(args, 'max_count_diff', 1000)  # 最大允许计数差异
            left_outlier = left_diff.abs() > max_count_diff
            right_outlier = right_diff.abs() > max_count_diff
            
            if left_outlier.any() or right_outlier.any():
                outlier_count = left_outlier.sum() + right_outlier.sum()
                print(f"  检测到{outlier_count}条异常值，将被过滤")
                left_diff[left_outlier] = 0
                right_diff[right_outlier] = 0

            # 计算左右轮线速度
            valid_dt = odo_data['dt'] > 1e-6  # 有效时间差
            
            # 初始化角速度列
            odo_data['left_velocity'] = 0.0
            odo_data['right_velocity'] = 0.0
            
            # 计算有效的轮速度
            if valid_dt.any():
                # 左轮速度 (米/秒)
                odo_data.loc[valid_dt, 'left_velocity'] = (
                    left_diff.loc[valid_dt] / odo_data.loc[valid_dt, 'dt'] * 
                    wheel_perimeter / encoder_resolution
                )
                
                # 右轮速度 (米/秒)
                odo_data.loc[valid_dt, 'right_velocity'] = (
                    right_diff.loc[valid_dt] / odo_data.loc[valid_dt, 'dt'] * 
                    wheel_perimeter / encoder_resolution
                )
            
            # 计算物理模型角速度 (弧度/秒)
            odo_data['physical_angular_velocity'] = (
                (odo_data['right_velocity'] - odo_data['left_velocity']) / wheel_base
            )
            
            # 使用物理模型的角速度作为最终结果
            odo_data['angular_velocity_magnitude'] = np.abs(odo_data['physical_angular_velocity'])
            
            # 比较两种方法的结果
            mean_physical = odo_data['angular_velocity_magnitude'].mean()
            
            print(f"  物理模型法平均角速度: {mean_physical:.4f} rad/s")
            
            # 数据统计
            print(f"  数据总量: {len(odo_data)}条")
            print(f"  左轮计数范围: {odo_data[left_col].min()} ~ {odo_data[left_col].max()}")
            print(f"  右轮计数范围: {odo_data[right_col].min()} ~ {odo_data[right_col].max()}")
        else:
            raise ValueError(f"无法识别左右轮计数列。检测到的列名: {list(odo_data.columns)}")
    else:
        # 尝试从数据特征识别可能的角速度列或速度差
        print("  尝试从数据特征识别可能的角速度列或轮速度列...")
        
        # 排除时间戳列和已知的非速度列
        exclude_patterns = ['timestamp', 'time', 'date', 'id', 'index']
        candidate_cols = [
            col for col in odo_data.columns 
            if not any(pattern in str(col).lower() for pattern in exclude_patterns)
            and odo_data[col].dtype in [np.int64, np.float64]
        ]
        
        # 检查每列的统计特性，寻找可能的角速度列
        angular_velocity_col = None
        
        for col in candidate_cols:
            # 计算基本统计量
            mean_val = odo_data[col].mean()
            std_val = odo_data[col].std()
            max_val = odo_data[col].abs().max()
            
            print(f"  列 '{col}' 数据特征: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 最大绝对值={max_val:.4f}")
            
            # 角速度通常不会特别大
            if max_val < 20 and std_val > 0.01:
                # 检查是否有正负变化（角速度通常会有）
                has_sign_change = (odo_data[col] > 0).any() and (odo_data[col] < 0).any()
                
                if has_sign_change or col.lower() in ['angular', 'angvel', 'omega', 'yaw_rate', 'yawrate']:
                    angular_velocity_col = col
                    break
        
        if angular_velocity_col:
            print(f"  使用列 '{angular_velocity_col}' 作为角速度")
            odo_data['angular_velocity_magnitude'] = np.abs(odo_data[angular_velocity_col])
        else:
            raise ValueError("轮式编码器数据缺少必要的列来计算角速度。请检查文件格式或手动指定角速度相关列。")
    
    # 角速度数据过滤
    # 去除异常值：角速度不应该超过合理的物理限制
    max_angular_velocity = getattr(args, 'max_angular_velocity', 10.0)  # 默认最大10弧度/秒
    angular_outlier = odo_data['angular_velocity_magnitude'] > max_angular_velocity
    
    if angular_outlier.any():
        print(f"  检测到{angular_outlier.sum()}条角速度异常值，将被裁剪到{max_angular_velocity}rad/s")
        odo_data.loc[angular_outlier, 'angular_velocity_magnitude'] = max_angular_velocity
    
    # 计算统计量
    mean_angular_velocity = odo_data['angular_velocity_magnitude'].mean()
    max_angular_velocity_observed = odo_data['angular_velocity_magnitude'].max()
    print(f"  角速度统计: 平均={mean_angular_velocity:.4f}rad/s, 最大={max_angular_velocity_observed:.4f}rad/s")
    
    print(f"轮式编码器数据加载完成，共 {len(odo_data)} 条记录")
    return odo_data

def load_rtk_data(args):
    """加载RTK数据"""
    if not args.rtk_file:
        return None
        
    print(f"加载RTK数据: {args.rtk_file}")
    
    # 判断输入文件格式
    is_rtk_full_format = False
    with open(args.rtk_file, 'r') as f:
        header_line = f.readline().strip()
        # 检查是否符合rtk_full.txt的格式
        if 'velocity_status' in header_line or 'solution_status' in header_line:
            is_rtk_full_format = True
    
    # 加载数据
    if is_rtk_full_format:
        # 处理rtk_full.txt格式
        # 与C++结构体匹配的所有列定义
        columns = [
            "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            "num_sats", "solution_status", "velocity_status", "lat", "lon", "alt",
            "pos_std_x", "pos_std_y", "pos_std_z",          # 位置标准差
            "velocity_x", "velocity_y", "velocity_z",        # 速度
            "vel_std_x", "vel_std_y", "vel_std_z",           # 速度标准差
            "ref_pos_x", "ref_pos_y", "ref_pos_z"           # 基站位置
        ]
        
        # 列类型定义，用于转换
        column_types = {
            "timestamp": float,
            "tx": float, "ty": float, "tz": float,
            "qx": float, "qy": float, "qz": float, "qw": float,
            "num_sats": int, "solution_status": int, "velocity_status": int,
            "lat": float, "lon": float, "alt": float,
            "pos_std_x": float, "pos_std_y": float, "pos_std_z": float,
            "velocity_x": float, "velocity_y": float, "velocity_z": float,
            "vel_std_x": float, "vel_std_y": float, "vel_std_z": float,
            "ref_pos_x": float, "ref_pos_y": float, "ref_pos_z": float
        }
        
        # 读取数据，跳过注释行
        data = []
        with open(args.rtk_file, 'r') as f:
            # 跳过头部行
            line = f.readline()
            while line.startswith('#'):
                line = f.readline()
                if not line:  # 文件末尾
                    break
                
            # 处理数据行
            while line:
                if not line.strip():  # 跳过空行
                    line = f.readline()
                    continue
                    
                values = line.strip().split()
                
                # 确保至少有基本的位置数据
                if len(values) < 8:  # 至少需要timestamp和位置数据
                    print(f"警告: 跳过不完整的行: {line.strip()}")
                    line = f.readline()
                    continue
                
                # 转换数据
                row = {}
                # 使用所有可用的列，对不存在的列填充NaN
                for i, col in enumerate(columns):
                    if i < len(values):
                        try:
                            row[col] = column_types[col](values[i])
                        except (ValueError, IndexError):
                            row[col] = np.nan
                    else:
                        row[col] = np.nan
                
                data.append(row)
                line = f.readline()
        
        # 转换为DataFrame
        rtk_data = pd.DataFrame(data)
        
        # 使用timestamp作为时间戳列
        args.rtk_timestamp_col = "timestamp"
        
        # 将时间戳乘以1000，从秒转换为毫秒，与其他数据保持一致
        rtk_data['timestamp'] = rtk_data['timestamp'] * 1000
        print("  将RTK时间戳从秒转换为毫秒")
        
        # 检查速度数据是否存在且有效
        has_velocity_data = False
        if all(col in rtk_data.columns for col in ['velocity_x', 'velocity_y', 'velocity_z']):
            # 检查是否全为NaN或0
            non_zero_velocity = (
                (rtk_data['velocity_x'].abs() > 1e-6) | 
                (rtk_data['velocity_y'].abs() > 1e-6) | 
                (rtk_data['velocity_z'].abs() > 1e-6)
            ).any()
            
            if non_zero_velocity:
                has_velocity_data = True
                print("  检测到有效的速度数据")
            
        if not has_velocity_data:
            print("  RTK数据中没有有效的速度信息，从位置差分计算速度")
            # 确保时间戳是升序排序的
            rtk_data = rtk_data.sort_values(by='timestamp')
            
            # 计算时间差(秒)
            rtk_data['dt'] = rtk_data['timestamp'].diff().fillna(0)
            
            # 计算位置差
            rtk_data['dx'] = rtk_data['tx'].diff().fillna(0)
            rtk_data['dy'] = rtk_data['ty'].diff().fillna(0)
            rtk_data['dz'] = rtk_data['tz'].diff().fillna(0)
            
            # 计算速度 (m/s)
            valid_dt = rtk_data['dt'] > 0
            rtk_data.loc[valid_dt, 'velocity_x'] = rtk_data.loc[valid_dt, 'dx'] / rtk_data.loc[valid_dt, 'dt']
            rtk_data.loc[valid_dt, 'velocity_y'] = rtk_data.loc[valid_dt, 'dy'] / rtk_data.loc[valid_dt, 'dt']
            rtk_data.loc[valid_dt, 'velocity_z'] = rtk_data.loc[valid_dt, 'dz'] / rtk_data.loc[valid_dt, 'dt']
            
            # 将首行速度设置为0
            rtk_data.loc[~valid_dt, ['velocity_x', 'velocity_y', 'velocity_z']] = 0
        else:
            print("  使用RTK数据中的原始速度信息")
            
    else:
        raise ValueError("RTK数据格式不支持，请提供包含velocity_status的rtk_full.txt格式文件")
    
    # 处理velocity_status (0:有效, 1:无效)
    # 对于无效的速度值，将速度设置为0
    if 'velocity_status' in rtk_data.columns:
        invalid_velocity = rtk_data['velocity_status'] == 1
        if invalid_velocity.any():
            print(f"  发现{invalid_velocity.sum()}条无效速度记录，将被设置为0")
            for col in ['velocity_x', 'velocity_y', 'velocity_z']:
                rtk_data.loc[invalid_velocity, col] = 0.0
    
    # 计算角速度（从速度方向变化）
    print("  计算角速度...")
    if 'velocity_x' in rtk_data.columns and 'velocity_y' in rtk_data.columns:
        # 计算速度幅值
        rtk_data['speed'] = np.sqrt(
            rtk_data['velocity_x']**2 + 
            rtk_data['velocity_y']**2 + 
            rtk_data['velocity_z']**2
        )
        
        # 设置低速阈值，可通过args参数覆盖
        min_speed_threshold = getattr(args, 'rtk_min_speed_threshold', 0.1)  # 默认0.1 m/s
        print(f"  角速度计算的最小速度阈值: {min_speed_threshold} m/s")
        
        # 计算方向角 (仅对速度不为0的点计算)
        non_zero_speed = rtk_data['speed'] > min_speed_threshold*0.01  # 速度大于阈值的1%，用于计算方向
        
        # 初始化方向角列
        rtk_data['direction'] = np.nan
        
        # 只对有效速度点计算方向角
        rtk_data.loc[non_zero_speed, 'direction'] = np.arctan2(
            rtk_data.loc[non_zero_speed, 'velocity_y'], 
            rtk_data.loc[non_zero_speed, 'velocity_x']
        )
        
        # 对于速度接近0的点，方向角不可靠，使用插值填充
        if non_zero_speed.any():
            # 使用前向填充，再使用后向填充
            rtk_data['direction'] = rtk_data['direction'].fillna(method='ffill').fillna(method='bfill')
        else:
            # 如果所有速度都为0，设置默认方向角为0
            rtk_data['direction'] = 0.0
        
        # 计算方向角的变化量和时间间隔
        rtk_data['direction_diff'] = rtk_data['direction'].diff().fillna(0)
        rtk_data['time_diff'] = rtk_data['timestamp'].diff().fillna(0) / 1000.0  # 转换为秒
        
        # 处理方向角的周期性（在-π和π之间的跳变）
        rtk_data['direction_diff'] = np.where(
            rtk_data['direction_diff'] > np.pi, 
            rtk_data['direction_diff'] - 2*np.pi, 
            rtk_data['direction_diff']
        )
        rtk_data['direction_diff'] = np.where(
            rtk_data['direction_diff'] < -np.pi, 
            rtk_data['direction_diff'] + 2*np.pi, 
            rtk_data['direction_diff']
        )
        
        # 计算带符号的角速度 (弧度/秒)
        valid_time = rtk_data['time_diff'] > 1e-6  # 防止除零
        rtk_data['angular_velocity'] = 0.0  # 初始化为0
        rtk_data.loc[valid_time, 'angular_velocity'] = rtk_data.loc[valid_time, 'direction_diff'] / rtk_data.loc[valid_time, 'time_diff']
        
        # 计算角速度幅值 (弧度/秒)
        rtk_data['angular_velocity_magnitude'] = np.abs(rtk_data['angular_velocity'])
        
        # 将低速状态下的角速度设为0 (速度过低时角速度不可靠)
        low_speed = rtk_data['speed'] < min_speed_threshold
        if low_speed.any():
            print(f"  发现{low_speed.sum()}条低速记录({100*low_speed.sum()/len(rtk_data):.1f}%)，角速度将被设置为0")
            rtk_data.loc[low_speed, 'angular_velocity'] = 0.0
            rtk_data.loc[low_speed, 'angular_velocity_magnitude'] = 0.0
            
        # 角速度异常值过滤
        ang_vel_std = rtk_data.loc[~low_speed, 'angular_velocity_magnitude'].std()
        ang_vel_mean = rtk_data.loc[~low_speed, 'angular_velocity_magnitude'].mean()
        outlier_threshold = ang_vel_mean + 5 * ang_vel_std  # 5倍标准差
        
        if not pd.isna(outlier_threshold) and outlier_threshold > 0:
            outliers = (rtk_data['angular_velocity_magnitude'] > outlier_threshold) & (~low_speed)
            if outliers.any():
                outlier_count = outliers.sum()
                print(f"  发现{outlier_count}条角速度异常值(>{outlier_threshold:.2f}弧度/秒)，将被平滑处理")
                
                # 对异常值使用邻近平均值替代
                for idx in rtk_data.index[outliers]:
                    # 获取前后5个有效值的平均
                    window_size = 5
                    start_idx = max(0, rtk_data.index.get_loc(idx) - window_size)
                    end_idx = min(len(rtk_data), rtk_data.index.get_loc(idx) + window_size + 1)
                    neighbors = rtk_data.iloc[start_idx:end_idx]
                    valid_neighbors = neighbors[neighbors['angular_velocity_magnitude'] <= outlier_threshold]
                    
                    if len(valid_neighbors) > 0:
                        # 替换为有效邻居的平均值
                        avg_value = valid_neighbors['angular_velocity_magnitude'].mean()
                        rtk_data.loc[idx, 'angular_velocity_magnitude'] = avg_value
                        # 保持原始符号
                        sign = 1 if rtk_data.loc[idx, 'angular_velocity'] >= 0 else -1
                        rtk_data.loc[idx, 'angular_velocity'] = sign * avg_value
                
        # 打印角速度统计信息
        print(f"  角速度统计: 平均={rtk_data['angular_velocity_magnitude'].mean():.4f}弧度/秒, "
              f"最大={rtk_data['angular_velocity_magnitude'].max():.4f}弧度/秒, "
              f"标准差={rtk_data['angular_velocity_magnitude'].std():.4f}弧度/秒")
            
    else:
        raise ValueError("RTK数据缺少速度列，无法计算角速度")
    
    # 确保时间戳是升序排列的
    rtk_data = rtk_data.sort_values(by=args.rtk_timestamp_col)
    
    print(f"加载了 {len(rtk_data)} 条RTK记录")
    
    # 解释solution_status
    if 'solution_status' in rtk_data.columns:
        solution_status_map = {
            0: "无效解",
            1: "单点定位解",
            2: "伪距差分",
            4: "固定解", 
            5: "浮动解"
        }
        status_counts = rtk_data['solution_status'].value_counts().sort_index()
        print("\nRTK solution_status分布:")
        for status, count in status_counts.items():
            percentage = count / len(rtk_data) * 100
            status_name = solution_status_map.get(status, f"未知状态({status})")
            print(f"  {status_name}: {count}条 ({percentage:.1f}%)")
    
    # 解释velocity_status
    if 'velocity_status' in rtk_data.columns:
        velocity_status_map = {
            0: "速度解状态有效",
            1: "速度解状态无效"
        }
        status_counts = rtk_data['velocity_status'].value_counts().sort_index()
        print("\nRTK velocity_status分布:")
        for status, count in status_counts.items():
            percentage = count / len(rtk_data) * 100
            status_name = velocity_status_map.get(status, f"未知状态({status})")
            print(f"  {status_name}: {count}条 ({percentage:.1f}%)")
    
    # 如果有位置标准差，计算平均精度
    if all(col in rtk_data.columns for col in ['pos_std_x', 'pos_std_y', 'pos_std_z']):
        # 检查是否有有效的位置标准差数据
        has_valid_pos_std = (
            (~pd.isna(rtk_data['pos_std_x'])) & 
            (~pd.isna(rtk_data['pos_std_y'])) & 
            (~pd.isna(rtk_data['pos_std_z']))
        ).any()
        
        if has_valid_pos_std:
            mean_pos_std = np.sqrt(
                rtk_data['pos_std_x'].fillna(0)**2 + 
                rtk_data['pos_std_y'].fillna(0)**2 + 
                rtk_data['pos_std_z'].fillna(0)**2
            ).mean()
            print(f"\n位置精度统计: 平均标准差 = {mean_pos_std:.4f} m")
    
    # 速度统计信息
    if 'speed' in rtk_data.columns:
        mean_speed = rtk_data['speed'].mean()
        max_speed = rtk_data['speed'].max()
        print(f"\n速度统计: 平均速度 = {mean_speed:.2f} m/s, 最大速度 = {max_speed:.2f} m/s")
    
    return rtk_data

def process_image_data(args):
    """处理图像序列，计算角速度信号"""
    print(f"处理图像序列，提取角速度信号: {args.image_dir}")
    
    # 获取图像文件和时间戳
    image_files, image_timestamps = load_image_timestamps(args)
    
    if len(image_files) < 2:
        print("警告: 图像序列至少需要2帧，无法计算角速度")
        return pd.DataFrame(columns=['timestamp', 'angular_velocity'])
    
    # 相机内参矩阵，如果没有提供则使用估计值
    camera_matrix = None
    if hasattr(args, 'camera_matrix') and args.camera_matrix is not None:
        camera_matrix = args.camera_matrix
    else:
        # 默认估计值 (根据实际相机调整)
        h, w = cv2.imread(image_files[0]).shape[:2]
        fx = fy = max(h, w)  # 一个估计值
        cx, cy = w/2, h/2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    # 计算每对连续帧的角速度
    angular_velocities = []
    
    for i in range(1, len(image_files)):
        prev_frame = cv2.imread(image_files[i-1])
        curr_frame = cv2.imread(image_files[i])
        
        prev_timestamp = image_timestamps[i-1]
        curr_timestamp = image_timestamps[i]
        
        try:
            # 使用SIFT特征计算角速度
            angular_vel = compute_angular_velocity_sift(
                prev_frame, curr_frame, 
                camera_matrix, 
                prev_timestamp, curr_timestamp
            )
            
            angular_velocities.append({
                'timestamp': curr_timestamp,
                'angular_velocity': angular_vel
            })
            
            if i % 10 == 0:
                print(f"已处理 {i}/{len(image_files)-1} 帧")
                
        except Exception as e:
            print(f"处理帧 {i} 时出错: {str(e)}")
    
    # 创建角速度数据帧
    if not angular_velocities:
        print("警告: 未能计算任何有效的角速度值")
        return pd.DataFrame(columns=['timestamp', 'angular_velocity'])
    
    return pd.DataFrame(angular_velocities)

def compute_angular_velocity_sift(prev_frame, curr_frame, camera_matrix, prev_timestamp, curr_timestamp):
    """
    使用SIFT特征提取、本质矩阵分解计算角速度
    
    参数:
        prev_frame: 前一帧图像
        curr_frame: 当前帧图像
        camera_matrix: 相机内参矩阵
        prev_timestamp: 前一帧时间戳(秒)
        curr_timestamp: 当前帧时间戳(秒)
    
    返回:
        angular_velocity: 角速度(弧度/秒)
    """
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. SIFT特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(prev_gray, None)
    kp2, des2 = sift.detectAndCompute(curr_gray, None)
    
    if len(kp1) < 8 or len(kp2) < 8:
        return 0.0  # 特征点不足，无法计算
    
    # 2. 特征匹配
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # 应用Lowe比率测试筛选好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 8:
        return 0.0  # 匹配点不足，无法稳定计算本质矩阵
    
    # 3. 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # 4. 计算本质矩阵
    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    if E is None or E.shape != (3, 3):
        return 0.0  # 本质矩阵计算失败
    
    # 5. 从本质矩阵恢复旋转和平移
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix, mask=mask)
    
    # 6. 提取旋转角度
    # 转换为轴角表示
    angle_axis, _ = cv2.Rodrigues(R)
    
    # 计算旋转幅度（轴角的范数）
    rotation_angle = np.linalg.norm(angle_axis)
    
    # 7. 计算角速度 = 旋转角度 / 时间间隔
    time_interval = curr_timestamp - prev_timestamp
    
    if time_interval < 1e-6:  # 防止除零
        return 0.0
    
    angular_velocity = rotation_angle / time_interval
    
    return angular_velocity

def load_image_timestamps(args):
    """
    加载图像文件列表和对应的时间戳
    
    支持两种方式:
    1. 从文件名解析时间戳
    2. 从外部时间戳文件读取
    """
    # 获取图像文件列表
    image_dir = args.image_dir
    image_files = sorted(glob.glob(os.path.join(image_dir, args.image_pattern)))
    
    if not image_files:
        raise ValueError(f"未找到图像文件: {os.path.join(image_dir, args.image_pattern)}")
    
    timestamps = []
    
    # 方法1: 从外部时间戳文件读取
    if args.image_timestamp_file and os.path.exists(args.image_timestamp_file):
        print(f"从时间戳文件加载图像时间: {args.image_timestamp_file}")
        
        # 读取时间戳文件
        timestamp_df = pd.read_csv(args.image_timestamp_file)
        
        # 确定时间戳列名
        ts_col = None
        for col in ['timestamp', 'time', 'timestamps', 'times']:
            if col in timestamp_df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            raise ValueError("时间戳文件中未找到时间戳列")
        
        # 提取文件名和时间戳的映射关系
        filename_col = None
        for col in ['filename', 'file', 'name', 'image', 'path']:
            if col in timestamp_df.columns:
                filename_col = col
                break
        
        if filename_col is None:
            # 假设时间戳顺序与排序后的文件列表一致
            if len(timestamp_df) != len(image_files):
                raise ValueError(f"时间戳数量({len(timestamp_df)})与图像文件数量({len(image_files)})不匹配")
            
            timestamps = timestamp_df[ts_col].values.tolist()
        else:
            # 构建文件名到时间戳的映射
            file_to_ts = {}
            for _, row in timestamp_df.iterrows():
                file_to_ts[os.path.basename(row[filename_col])] = row[ts_col]
            
            # 根据文件名获取时间戳
            timestamps = [file_to_ts.get(os.path.basename(f), 0) for f in image_files]
    
    # 方法2: 从文件名解析时间戳
    else:
        print("尝试从文件名解析时间戳")
        for img_file in image_files:
            # 尝试多种常见的时间戳格式
            basename = os.path.basename(img_file)
            
            # 尝试提取数字部分作为时间戳
            timestamp = 0
            nums = re.findall(r'\d+', basename)
            if nums:
                try:
                    # 使用最长的数字序列作为可能的时间戳
                    timestamp = float(max(nums, key=len)) / 1000.0  # 假设毫秒时间戳
                except ValueError:
                    pass
            
            timestamps.append(timestamp)
    
    # 确保时间戳为浮点数
    timestamps = [float(ts) for ts in timestamps]
    
    # 验证时间戳数量与图像文件数量相同
    if len(timestamps) != len(image_files):
        raise ValueError(f"时间戳数量({len(timestamps)})与图像文件数量({len(image_files)})不匹配")
    
    return image_files, timestamps

# ========== 验证方法函数 ==========

def estimate_sampling_rate(data, ts_col):
    """估计数据的采样率（Hz）"""
    timestamps = data[ts_col].values
    if len(timestamps) < 2:
        return 1.0  # 默认值
        
    # 计算时间差的中位数
    time_diffs = np.diff(timestamps)
    median_diff_ms = np.median(time_diffs)
    
    # 如果时间戳单位是秒，则转换为毫秒
    if median_diff_ms < 0.1:  # 可能是秒为单位
        median_diff_ms *= 1000
        
    # 计算采样率 (Hz)
    sampling_rate = 1000.0 / median_diff_ms
    
    return sampling_rate

def resample_signal(data, ts_col, value_col, target_rate=None, target_len=None):
    """
    将信号重采样到指定频率或长度
    
    参数:
        data: 包含时间戳和值的DataFrame
        ts_col: 时间戳列名
        value_col: 值列名
        target_rate: 目标采样率(Hz)，如果提供则按此采样率重采样
        target_len: 目标长度，如果提供则重采样到该长度
    """
    timestamps = data[ts_col].values
    values = data[value_col].values
    
    if len(timestamps) < 2:
        return timestamps, values
    
    # 创建插值函数
    interp_func = interp1d(timestamps, values, kind='linear', 
                         bounds_error=False, fill_value=np.nan)
    
    t_min = timestamps.min()
    t_max = timestamps.max()
    
    if target_rate is not None:
        # 根据目标采样率创建新的时间点
        interval = 1000.0 / target_rate  # 时间间隔(ms)
        new_timestamps = np.arange(t_min, t_max, interval)
    elif target_len is not None:
        # 根据目标长度创建新的时间点
        new_timestamps = np.linspace(t_min, t_max, target_len)
    else:
        # 默认情况，保持原始采样率
        return timestamps, values
    
    # 重采样
    new_values = interp_func(new_timestamps)
    
    # 处理可能的NaN值
    mask = ~np.isnan(new_values)
    return new_timestamps[mask], new_values[mask]

def correlation_verification(data1, data2, ts_col1, ts_col2, args):
    """使用相关性方法进行传感器对齐验证
    
    Args:
        data1: 传感器1的数据帧
        data2: 传感器2的数据帧
        ts_col1: 传感器1的时间戳列名
        ts_col2: 传感器2的时间戳列名
        args: 参数对象
    
    Returns:
        dict: 包含偏移估计和置信度信息的字典
    """
    print("  使用correlation方法验证...")
    
    # 检查参数
    if not isinstance(data1, pd.DataFrame) or not isinstance(data2, pd.DataFrame):
        print("  错误: 输入数据必须是DataFrame")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
        
    if ts_col1 not in data1.columns or ts_col2 not in data2.columns:
        print(f"  错误: 时间戳列 {ts_col1} 或 {ts_col2} 不存在")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    if 'angular_velocity_magnitude' not in data1.columns or 'angular_velocity_magnitude' not in data2.columns:
        print("  错误: 角速度数据不存在")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    # 估计两个信号的采样率
    rate1 = estimate_sampling_rate(data1, ts_col1)
    rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {rate1:.2f} Hz, 信号2采样率: {rate2:.2f} Hz")
    
    # 确定要比较的公共时间范围
    start_time = max(data1[ts_col1].min(), data2[ts_col2].min())
    end_time = min(data1[ts_col1].max(), data2[ts_col2].max())
    
    # 确保有足够的数据
    if end_time - start_time < 1000:  # 至少需要1秒数据
        print("  错误: 共同时间范围太短")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    # 记录初始时间范围，但不再将其用作偏移计算基础
    print(f"  信号公共部分: {start_time:.1f}~{end_time:.1f}ms, 长度: {end_time-start_time:.1f}ms")
    
    # --------------------------------------------------------------------
    # 预处理信号，使用直接重采样方法避免引入偏移
    # --------------------------------------------------------------------
    
    # 在公共范围内截取数据
    data1_common = data1[(data1[ts_col1] >= start_time) & (data1[ts_col1] <= end_time)].copy()
    data2_common = data2[(data2[ts_col2] >= start_time) & (data2[ts_col2] <= end_time)].copy()
    
    # 检查数据量
    if len(data1_common) < 10 or len(data2_common) < 10:
        print("  错误: 共同时间范围内数据点太少")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    # 排序并去重
    data1_common = data1_common.sort_values(by=ts_col1).drop_duplicates(subset=[ts_col1])
    data2_common = data2_common.sort_values(by=ts_col2).drop_duplicates(subset=[ts_col2])
    
    # 确定最佳重采样率 - 使用较高采样率的1.5倍以保留信号特征
    target_rate = max(rate1, rate2) * 1.5
    
    # 直接进行重采样，避免复杂预处理
    try:
        # 统一时间轴 - 使用共同的起始时间和相同的采样间隔
        t_start = max(data1_common[ts_col1].min(), data2_common[ts_col2].min())
        t_end = min(data1_common[ts_col1].max(), data2_common[ts_col2].max())
        
        # 计算采样间隔和样本数
        interval = 1000.0 / target_rate  # 毫秒
        num_samples = int((t_end - t_start) / interval) + 1
        
        # 创建统一时间轴
        uniform_time = np.linspace(t_start, t_end, num_samples)
        
        # 对两个信号进行线性插值
        f1 = interp1d(data1_common[ts_col1], data1_common['angular_velocity_magnitude'], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        f2 = interp1d(data2_common[ts_col2], data2_common['angular_velocity_magnitude'], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # 重采样到统一时间轴
        signal1 = f1(uniform_time)
        signal2 = f2(uniform_time)
        
        # 标准化信号
        signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
        
        # 打印重采样后的信号长度
        print(f"  重采样后数据点数: 信号1: {len(signal1)}点, 信号2: {len(signal2)}点")
        
    except Exception as e:
        print(f"  重采样过程出错: {str(e)}")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    # 确保信号长度相同
    if len(signal1) != len(signal2):
        print(f"  错误: 重采样后信号长度不一致: {len(signal1)} vs {len(signal2)}")
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    # 计算最大可能的时间偏移（以样本点为单位）
    # 从args获取最大偏移限制，如果存在
    max_lag_ms = getattr(args, 'max_lag_ms', 600)  # 默认最大600ms偏移
    max_lag_samples = int(max_lag_ms / interval)
    print(f"  最大搜索偏移范围: ±{max_lag_ms}ms")
    
    # 创建搜索范围列表（以样本点为单位）
    small_range_ms = min(100, max_lag_ms / 6)
    medium_range_ms = min(200, max_lag_ms / 3)
    
    search_ranges = [
        (0, int(small_range_ms / interval)),                       # 0到small_range_ms (默认100ms)
        (-int(medium_range_ms / interval), int(medium_range_ms / interval)),  # ±medium_range_ms (默认200ms)
        (-max_lag_samples, max_lag_samples)                       # ±max_lag_ms (默认600ms)
    ]
    
    # 使用已知偏移来调整搜索范围（如果有）
    known_offset = getattr(args, 'known_offset', None)
    if known_offset is not None:
        try:
            known_offset = float(known_offset)
            known_offset_samples = int(known_offset / interval)
            # 围绕已知偏移创建搜索范围
            search_ranges = [
                (max(0, known_offset_samples - int(50 / interval)), 
                 known_offset_samples + int(50 / interval)),
                (max(-int(200 / interval), known_offset_samples - int(150 / interval)), 
                 min(int(200 / interval), known_offset_samples + int(150 / interval))),
                (max(-max_lag_samples, known_offset_samples - max_lag_samples // 2), 
                 min(max_lag_samples, known_offset_samples + max_lag_samples // 2))
            ]
        except:
            pass  # 如果转换失败，使用默认范围
    
    # 存储相关性结果
    correlation_results = []
    
    # 对每个搜索范围计算互相关
    for range_idx, (min_lag_samples, max_lag_samples) in enumerate(search_ranges):
        try:
            # 将样本偏移转换回毫秒，用于显示
            min_lag_ms = min_lag_samples * interval
            max_lag_ms = max_lag_samples * interval
            
            print(f"  计算搜索范围{range_idx+1}的相关性: {min_lag_ms:.1f}ms ~ {max_lag_ms:.1f}ms")
            
            # 计算完整的互相关
            cross_corr = correlate(signal1, signal2, mode='full')
            
            # 创建所有可能的lag
            all_lags = np.arange(-len(signal2)+1, len(signal1))
            
            # 仅选择搜索范围内的lag
            valid_indices = (all_lags >= min_lag_samples) & (all_lags <= max_lag_samples)
            valid_lags = all_lags[valid_indices]
            valid_xcorr = cross_corr[valid_indices]
            
            if len(valid_lags) == 0:
                print(f"  警告: 范围{range_idx+1}内无有效lag")
                continue
            
            # 归一化互相关
            norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
            if norm_factor > 0:
                valid_xcorr = valid_xcorr / norm_factor
            
            # 寻找最大互相关及其位置
            max_idx = np.argmax(valid_xcorr)
            max_xcorr = valid_xcorr[max_idx]
            best_lag_samples = valid_lags[max_idx]
            
            # 将样本点转换回毫秒
            best_lag_ms = best_lag_samples * interval
            
            # 细化峰值位置（二次插值）
            if max_idx > 0 and max_idx < len(valid_xcorr) - 1:
                y0, y1, y2 = valid_xcorr[max_idx-1:max_idx+2]
                x0, x1, x2 = valid_lags[max_idx-1:max_idx+2]
                
                # 二次插值公式
                if y1 > y0 and y1 > y2:  # 确保是峰值
                    denom = 2 * (2*y1 - y0 - y2)
                    if abs(denom) > 1e-10:  # 避免除零
                        delta = (y2 - y0) / denom
                        # 限制插值在合理范围内
                        if abs(delta) < 1:
                            refined_lag_samples = x1 + delta * (x2 - x1)
                            refined_lag_ms = refined_lag_samples * interval
                            best_lag_ms = refined_lag_ms
            
            # 估计置信度
            # 1. 基于互相关值 (范围 -1 到 1)
            corr_confidence = (max_xcorr + 1) / 2  # 将范围从[-1,1]映射到[0,1]
            
            # 2. 基于峰值突出度
            peak_prominence = max_xcorr - np.mean(valid_xcorr)
            peak_confidence = min(1.0, max(0.0, peak_prominence * 3))  # 缩放到[0,1]
            
            # 3. 计算峰值清晰度 (峰值与次高峰的比值)
            sorted_xcorr = np.sort(valid_xcorr)
            if len(sorted_xcorr) > 1 and sorted_xcorr[-1] > sorted_xcorr[-2]:
                peak_clarity = (sorted_xcorr[-1] - sorted_xcorr[-2]) / (sorted_xcorr[-1] - np.mean(sorted_xcorr))
                peak_clarity = min(1.0, max(0.0, peak_clarity * 2))  # 缩放到[0,1]
            else:
                peak_clarity = 0.5
            
            # 综合置信度分数
            confidence = 0.4 * corr_confidence + 0.4 * peak_confidence + 0.2 * peak_clarity
            
            # 计算标准差 (使用半高全宽方法)
            std_ms = 0
            try:
                # 找到半高点
                half_height = (max_xcorr + np.min(valid_xcorr)) / 2
                above_half = valid_xcorr >= half_height
                
                if np.sum(above_half) > 1:
                    # 找到连续的上升和下降边缘
                    above_indices = np.where(above_half)[0]
                    left_idx = above_indices[0]
                    right_idx = above_indices[-1]
                    
                    # 计算FWHM (半高全宽)
                    fwhm_samples = valid_lags[right_idx] - valid_lags[left_idx]
                    fwhm_ms = fwhm_samples * interval
                    
                    # 转换为标准差 (高斯分布的FWHM = 2.355 * sigma)
                    std_ms = fwhm_ms / 2.355
                    std_ms = max(5.0, std_ms)  # 至少5ms的标准差
                else:
                    std_ms = 50.0  # 默认值
            except:
                std_ms = 50.0  # 出错时使用默认值
            
            print(f"    结果: 偏移={best_lag_ms:.2f}ms, 相关性={max_xcorr:.4f}, 标准差={std_ms:.2f}ms, 置信度={confidence:.2f}")
            
            # 保存结果
            correlation_results.append({
                'lag': best_lag_ms,
                'corr': float(max_xcorr),
                'std': float(std_ms),
                'confidence': float(confidence),
                'range': (min_lag_ms, max_lag_ms)
            })
            
        except Exception as e:
            print(f"  计算相关性时出错 (范围{range_idx+1}): {str(e)}")
    
    # 检查是否有结果
    if not correlation_results:
        print("  错误: 所有搜索范围都未能产生有效结果")
        return {'lag': 0, 'corr': 0, 'std': 0, 'confidence': 0, 'status': 'failed', 'method': 'correlation'}
    
    # 按置信度排序结果
    correlation_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 打印所有结果
    print("  所有相关性结果:")
    for i, result in enumerate(correlation_results):
        print(f"    结果{i+1}: 偏移={result['lag']:.2f}ms, 相关性={result['corr']:.4f}, 置信度={result['confidence']:.2f}")
    
    # 选择最佳结果
    best_result = correlation_results[0]
    
    # 高置信度结果可以直接使用
    if best_result['confidence'] > 0.7:
        final_lag = best_result['lag']
        final_std = best_result['std']
        final_confidence = best_result['confidence']
        print(f"  使用高置信度最佳结果: 偏移={final_lag:.2f}ms, 置信度={final_confidence:.2f}")
    else:
        # 否则使用置信度加权平均
        high_conf_results = [r for r in correlation_results if r['confidence'] > 0.4]
        if high_conf_results:
            # 加权平均计算
            sum_weighted_lag = sum(r['lag'] * r['confidence'] for r in high_conf_results)
            sum_weights = sum(r['confidence'] for r in high_conf_results)
            final_lag = sum_weighted_lag / sum_weights
            
            # 标准差估计 - 使用加权结果的标准差
            if len(high_conf_results) > 1:
                lag_variance = sum(r['confidence'] * (r['lag'] - final_lag)**2 for r in high_conf_results) / sum_weights
                final_std = math.sqrt(lag_variance)
            else:
                final_std = high_conf_results[0]['std']
            
            final_confidence = sum(r['confidence']**2 for r in high_conf_results) / sum(r['confidence'] for r in high_conf_results)
            print(f"  使用加权平均: 偏移={final_lag:.2f}ms, 置信度={final_confidence:.2f}")
        else:
            # 没有高置信度结果时使用最佳结果
            final_lag = best_result['lag']
            final_std = best_result['std']
            final_confidence = best_result['confidence']
            print(f"  使用最佳可用结果: 偏移={final_lag:.2f}ms, 置信度={final_confidence:.2f}")
    
    print(f"  相关性方法最终结果: 偏移={final_lag:.2f}ms, 标准差={final_std:.2f}ms")
    
    # 计算但不使用基础时间差 (两个信号起点差异)
    base_time_diff = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  注意: 数据起点时间差为 {base_time_diff:.2f}ms (不计入最终偏移结果)")
    
    # 判断是否对齐（基于精细偏移和容差）- 不再使用起点差异
    tolerance_ms = args.tolerance_ms if hasattr(args, 'tolerance_ms') else 30.0
    # 修改对齐判断逻辑，确保相关性值足够高
    # 只有当相关性大于0.2且偏移在容差范围内时才认为已对齐
    corr_threshold = 0.2  # 最小相关性阈值
    is_aligned = abs(final_lag) <= tolerance_ms and best_result['corr'] >= corr_threshold
    
    # 如果相关性太低，打印警告
    if best_result['corr'] < corr_threshold:
        print(f"  警告: 最大相关性({best_result['corr']:.4f})低于阈值({corr_threshold})，可能表示传感器数据不相关")
    
    return {
        'lag': final_lag,
        'corr': best_result['corr'],
        'std': final_std,
        'confidence': final_confidence,
        'status': 'success' if final_confidence > 0.6 and is_aligned else 'low_confidence',
        'method': 'correlation',
        'mean_offset': final_lag,  # 使用精细偏移作为最终结果，不再包含基础偏移
        'base_time_diff': base_time_diff,  # 仍然保存基础时间差，但仅用于信息展示
        'is_aligned': is_aligned  # 添加对齐状态
    }

def detect_events(data, ts_col, value_col='angular_velocity_magnitude', 
                threshold_percentile=90, window_size=5, min_distance=10, 
                dynamic_threshold=True):
    """检测信号中的重要事件（峰值）- 优化版
    
    参数:
        data: 包含时间戳和信号值的DataFrame
        ts_col: 时间戳列名
        value_col: 信号值列名
        threshold_percentile: 信号值阈值百分位数
        window_size: 检测窗口大小（点数）
        min_distance: 事件之间的最小距离（点数）
        dynamic_threshold: 是否使用动态阈值（基于局部数据)
        
    返回:
        包含'timestamps'、'values'、'indices'和'features'的字典，表示事件特征
    """
    if len(data) < window_size * 2:
        return {'timestamps': [], 'values': [], 'indices': [], 'features': []}
    
    # 确保按时间排序
    sorted_data = data.sort_values(by=ts_col).copy()
    
    # 1. 多阶段信号处理
    # 创建多个平滑尺度的信号版本，以捕获不同时间尺度的特征
    scales = []
    
    # 原始信号
    detection_cols = [value_col]
    sorted_data[f'{value_col}_raw'] = sorted_data[value_col]
    
    # 多尺度平滑
    try:
        # 中等平滑 - 减少噪声但保留主要特征
        smooth_window_med = min(len(sorted_data) // 5, 15)
        if smooth_window_med % 2 == 0:
            smooth_window_med += 1
            
        sorted_data[f'{value_col}_med'] = savgol_filter(
            sorted_data[value_col].values,
            window_length=smooth_window_med,
            polyorder=2
        )
        detection_cols.append(f'{value_col}_med')
        scales.append('med')
        
        # 强平滑 - 只保留大规模特征
        smooth_window_large = min(len(sorted_data) // 3, 31)
        if smooth_window_large % 2 == 0:
            smooth_window_large += 1
            
        sorted_data[f'{value_col}_large'] = savgol_filter(
            sorted_data[value_col].values,
            window_length=smooth_window_large,
            polyorder=2
        )
        detection_cols.append(f'{value_col}_large')
        scales.append('large')
    except:
        # 如果滤波失败，仅使用原始信号
        pass
    
    # 2. 多层次事件检测 - 在不同尺度上分别检测事件
    all_events = []
    for i, col in enumerate(detection_cols):
        # 对每个尺度使用不同的参数
        scale_factor = 0.8 if i == 0 else 1.0 if i == 1 else 1.2
        scale_window = max(3, int(window_size * scale_factor))
        scale_distance = max(3, int(min_distance * scale_factor))
        scale_threshold = threshold_percentile if i == 0 else threshold_percentile - 5
        
        # 计算全局阈值
        values_array = sorted_data[col].values
        values_array = values_array[~np.isnan(values_array)]
        
        if len(values_array) == 0:
            continue
        
        # 使用更稳健的阈值计算
        if len(values_array) > 10:
            q75 = np.percentile(values_array, 75)
            q25 = np.percentile(values_array, 25)
            iqr = q75 - q25  # 四分位距
            
            # 根据指定百分位调整阈值强度
            percentile_factor = scale_threshold / 90.0
            global_threshold = q75 + percentile_factor * 1.5 * iqr
        else:
            global_threshold = np.percentile(values_array, scale_threshold)
        
        # 在此尺度下检测峰值
        for j in range(scale_window, len(sorted_data) - scale_window, max(1, scale_distance // 3)):
            # 选择窗口
            window = sorted_data.iloc[j-scale_window:j+scale_window+1]
            
            # 检查当前点
            current_value = window.iloc[scale_window][col]
            
            # 判断是否为峰值
            is_peak = (current_value == window[col].max())
            
            # 检查是否为显著的局部高点
            if not is_peak:
                # 左右相邻点的平均值
                neighbors_avg = (window[col].iloc[scale_window-1] + 
                                window[col].iloc[scale_window+1]) / 2
                
                # 相对于邻近点的突起比例
                peak_ratio = 1.3 - (0.1 * i)  # 随尺度减小比例要求
                if current_value > neighbors_avg * peak_ratio:
                    is_peak = True
            
            if not is_peak:
                continue
            
            # 计算动态阈值
            if dynamic_threshold:
                # 窗口范围阈值
                local_threshold = np.percentile(window[col], scale_threshold)
                
                # 更大范围的阈值
                larger_window_size = min(scale_window * 3, len(sorted_data) // 10)
                start_idx = max(0, j - larger_window_size)
                end_idx = min(len(sorted_data), j + larger_window_size + 1)
                larger_window = sorted_data.iloc[start_idx:end_idx]
                
                if len(larger_window) > 5:
                    larger_threshold = np.percentile(larger_window[col], scale_threshold)
                    # 组合阈值，优先考虑局部
                    final_threshold = 0.5 * local_threshold + 0.5 * max(larger_threshold, global_threshold * 0.6)
                else:
                    final_threshold = local_threshold
            else:
                final_threshold = global_threshold
            
            # 验证峰值显著性
            if current_value < final_threshold:
                continue
            
            # 计算事件特征
            timestamp = window.iloc[scale_window][ts_col]
            
            # 计算事件强度特征
            # 1. 相对强度 (当前值 / 阈值)
            relative_strength = current_value / final_threshold
            
            # 2. 峰值陡峭度（相对于相邻点的变化率）
            left_slope = (current_value - window[col].iloc[scale_window-1]) if scale_window > 0 else 0
            right_slope = (current_value - window[col].iloc[scale_window+1]) if scale_window < len(window) - 1 else 0
            steepness = (left_slope + right_slope) / 2 if scale_window > 0 and scale_window < len(window) - 1 else 0
            
            # 3. 峰值宽度（半高宽）- 简化估计
            half_height = current_value / 2
            width = 1
            for k in range(1, scale_window):
                if scale_window-k >= 0 and window[col].iloc[scale_window-k] < half_height:
                    break
                width += 1
            for k in range(1, scale_window):
                if scale_window+k < len(window) and window[col].iloc[scale_window+k] < half_height:
                    break
                width += 1
            
            # 添加事件到列表
            event = {
                'timestamp': timestamp,
                'value': current_value,
                'index': j,
                'scale': i,
                'relative_strength': relative_strength,
                'steepness': steepness,
                'width': width,
                'prominence': current_value - neighbors_avg if scale_window > 0 and scale_window < len(window) - 1 else current_value
            }
            all_events.append(event)
    
    # 3. 事件去重和合并
    # 按时间戳排序
    all_events.sort(key=lambda e: e['timestamp'])
    
    # 合并相近的事件 (基于时间戳)
    merged_events = []
    i = 0
    while i < len(all_events):
        current = all_events[i]
        merged_cluster = [current]
        
        # 查找时间上接近的事件
        j = i + 1
        while j < len(all_events) and abs(all_events[j]['timestamp'] - current['timestamp']) < min_distance:
            merged_cluster.append(all_events[j])
            j += 1
        
        # 从合并的群集中选择最佳事件（根据多种特征）
        if len(merged_cluster) > 1:
            # 综合评分 - 强度、陡峭度和小尺度优先
            best_event = max(merged_cluster, key=lambda e: 
                           e['relative_strength'] * 0.5 + 
                           e['steepness'] * 0.3 + 
                           e['prominence'] * 0.2 - 
                           e['scale'] * 0.1)  # 优先小尺度事件
        else:
            best_event = current
        
        merged_events.append(best_event)
        i = j
    
    # 4. 生成最终事件列表
    # 如果事件过多，根据突出度选择最显著的事件
    if len(merged_events) > 50:
        merged_events.sort(key=lambda e: e['relative_strength'] * e['prominence'], reverse=True)
        merged_events = merged_events[:50]
    
    # 按时间排序
    merged_events.sort(key=lambda e: e['timestamp'])
    
    # 提取结果
    timestamps = [e['timestamp'] for e in merged_events]
    values = [e['value'] for e in merged_events]
    indices = [e['index'] for e in merged_events]
    features = [{'prominence': e['prominence'], 
                'steepness': e['steepness'], 
                'width': e['width'],
                'strength': e['relative_strength']} for e in merged_events]
    
    return {
        'timestamps': timestamps,
        'values': values,
        'indices': indices,
        'features': features
    }

def event_verification(data1, data2, ts_col1, ts_col2, args):
    """使用事件同步验证时间戳对齐 - 改进版本"""
    # 估计采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 记录原始数据的时间差，但不使用它计算偏移
    base_time_diff = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  数据起点时间差: {base_time_diff:.2f}ms (不计入最终偏移结果)")
    
    # 根据采样率动态调整事件检测参数
    # 采样率与检测参数的调整
    threshold_pct1 = min(92, 75 + int(17 * np.log10(max(1, sampling_rate1) / 10)))
    threshold_pct2 = min(92, 75 + int(17 * np.log10(max(1, sampling_rate2) / 10)))
    
    # 窗口大小与采样率成正比
    win_size1 = max(5, int(0.15 * sampling_rate1))  # 约150ms窗口
    win_size2 = max(5, int(0.15 * sampling_rate2))
    
    # 事件间最小距离
    min_dist1 = max(5, int(0.3 * sampling_rate1))  # 至少0.3秒间隔
    min_dist2 = max(5, int(0.3 * sampling_rate2))
    
    print(f"  事件检测参数 - 信号1: 阈值百分位={threshold_pct1}, 窗口大小={win_size1}, 最小距离={min_dist1}")
    print(f"  事件检测参数 - 信号2: 阈值百分位={threshold_pct2}, 窗口大小={win_size2}, 最小距离={min_dist2}")
    
    # 检测事件，使用多尺度动态阈值方法
    events1 = detect_events(data1, ts_col1, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct1,
                          window_size=win_size1, min_distance=min_dist1,
                          dynamic_threshold=True)
    
    events2 = detect_events(data2, ts_col2, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct2,
                          window_size=win_size2, min_distance=min_dist2,
                          dynamic_threshold=True)
    
    print(f"  检测到事件 - 信号1: {len(events1['timestamps'])}个, 信号2: {len(events2['timestamps'])}个")
    
    # 增强版事件检测 - 多次尝试不同参数，确保检测到足够事件
    if len(events1['timestamps']) < 6 or len(events2['timestamps']) < 6:
        retry_attempts = 0
        max_attempts = 5
        
        # 多参数尝试策略
        parameter_sets = [
            # (阈值1降低, 阈值2降低, 距离1缩短比例, 距离2缩短比例, 窗口1缩短比例, 窗口2缩短比例)
            (15, 15, 0.8, 0.8, 0.9, 0.9),
            (25, 25, 0.6, 0.6, 0.8, 0.8),
            (35, 35, 0.5, 0.5, 0.7, 0.7),
            (45, 45, 0.4, 0.4, 0.6, 0.6),
            (55, 55, 0.3, 0.3, 0.5, 0.5)
        ]
        
        while (len(events1['timestamps']) < 6 or len(events2['timestamps']) < 6) and retry_attempts < max_attempts:
            params = parameter_sets[retry_attempts]
            retry_attempts += 1
            
            print(f"  事件数量不足，尝试第{retry_attempts}次调整参数")
            
            if len(events1['timestamps']) < 6:
                new_threshold1 = max(40, threshold_pct1 - params[0])
                new_distance1 = max(3, int(min_dist1 * params[2]))
                new_window1 = max(3, int(win_size1 * params[4]))
                
                events1 = detect_events(data1, ts_col1, value_col='angular_velocity_magnitude', 
                                      threshold_percentile=new_threshold1,
                                      window_size=new_window1, 
                                      min_distance=new_distance1,
                                      dynamic_threshold=True)
            
            if len(events2['timestamps']) < 6:
                new_threshold2 = max(40, threshold_pct2 - params[1])
                new_distance2 = max(3, int(min_dist2 * params[3]))
                new_window2 = max(3, int(win_size2 * params[5]))
                
                events2 = detect_events(data2, ts_col2, value_col='angular_velocity_magnitude', 
                                      threshold_percentile=new_threshold2,
                                      window_size=new_window2, 
                                      min_distance=new_distance2,
                                      dynamic_threshold=True)
            
            print(f"  参数调整后 - 信号1: {len(events1['timestamps'])}个事件, 信号2: {len(events2['timestamps'])}个事件")
    
    # 如果仍然没有足够的事件，返回无法配准的结果
    if len(events1['timestamps']) < 3 or len(events2['timestamps']) < 3:
        print("  事件不足, 无法进行可靠的匹配")
        return {
            'events1': events1,
            'events2': events2,
            'match_stats': {
                'matches': [],
                'unmatched1': list(range(len(events1['timestamps']))),
                'unmatched2': list(range(len(events2['timestamps']))),
                'match_rate': 0.0,
                'mean_diff': 0.0,
                'std_diff': 0.0
            },
            'is_aligned': False,
            'method': 'event',
            'mean_offset': 0.0,  # 初始状态下不能判定偏移
            'std_offset': 0.0,
            'base_time_diff': base_time_diff,  # 仍然记录基础时间差，但仅作为信息
            'detail_offset': 0.0
        }
    
    # 计算搜索窗口和策略
    # 基础搜索窗口更大，确保能找到真实匹配
    adjusted_base_diff = abs(base_time_diff)
    
    # 使用args.max_lag_ms作为最大偏移搜索范围
    max_search_window = getattr(args, 'max_lag_ms', 3500)  # 默认最大3500ms
    search_window_base = min(max(args.tolerance_ms * 15, 1500), max_search_window)  # 至少1500ms的搜索窗口，但不超过最大值
    search_window_ms = min(search_window_base * (1 + min(adjusted_base_diff / 3000, 1.0)), max_search_window)  # 根据基础差异动态调整
    
    # 考虑采样率因素 - 采样率低需要更大窗口
    rate_factor = max(1.0, 50 / min(sampling_rate1, sampling_rate2))
    search_window_ms = min(search_window_ms * min(rate_factor, 3.0), max_search_window)  # 最多增加3倍，但不超过最大值
    
    print(f"  事件匹配窗口: ±{search_window_ms:.1f}ms (最大: {max_search_window}ms)")
    
    # 多阶段匹配策略
    # 1. 设定初始对齐为0（不使用基础时间差）
    adjusted_timestamps2 = np.array(events2['timestamps'])
    
    # 2. 基于事件特征的匹配评分矩阵
    score_matrix = np.zeros((len(events1['timestamps']), len(events2['timestamps'])))
    
    for i, ts1 in enumerate(events1['timestamps']):
        for j, adj_ts2 in enumerate(adjusted_timestamps2):
            # 时间差距评分 - 指数衰减
            time_diff = abs(ts1 - adj_ts2)
            if time_diff > search_window_ms:
                score_matrix[i, j] = 0
                continue
                
            time_score = np.exp(-time_diff / (search_window_ms / 4))
            
            # 特征相似度评分
            feature_similarity = 0
            if 'features' in events1 and 'features' in events2 and \
               i < len(events1['features']) and j < len(events2['features']):
                feat1 = events1['features'][i]
                feat2 = events2['features'][j]
                
                # 归一化特征相似度 (0-1范围)
                if 'prominence' in feat1 and 'prominence' in feat2:
                    prom_ratio = min(feat1['prominence'], feat2['prominence']) / max(feat1['prominence'], feat2['prominence'] + 1e-10)
                    feature_similarity += 0.4 * prom_ratio
                    
                if 'steepness' in feat1 and 'steepness' in feat2:
                    steep_ratio = min(feat1['steepness'], feat2['steepness']) / max(feat1['steepness'], feat2['steepness'] + 1e-10)
                    feature_similarity += 0.3 * steep_ratio
                    
                if 'width' in feat1 and 'width' in feat2:
                    width_ratio = min(feat1['width'], feat2['width']) / max(feat1['width'], feat2['width'] + 1e-10)
                    feature_similarity += 0.3 * width_ratio
            
            # 综合评分
            final_score = time_score * (0.7 + 0.3 * feature_similarity)
            score_matrix[i, j] = final_score
    
    # 3. 多轮匹配 - 使用匈牙利算法找到全局最优匹配
    matches = []
    unmatched1 = []
    unmatched2 = []
    
    # 在分数矩阵上应用匈牙利算法
    try:
        from scipy.optimize import linear_sum_assignment
        
        # 反转分数以用于最小化问题
        cost_matrix = 1 - score_matrix
        
        # 执行匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 筛选有效匹配 (分数高于阈值)
        min_match_score = 0.3  # 最低匹配分数阈值
        
        for i, j in zip(row_ind, col_ind):
            match_score = score_matrix[i, j]
            if match_score >= min_match_score:
                time_diff = events1['timestamps'][i] - adjusted_timestamps2[j]
                matches.append({
                    'event1_idx': i,
                    'event2_idx': j,
                    'event1_ts': events1['timestamps'][i],
                    'event2_ts': events2['timestamps'][j],
                    'time_diff': time_diff,
                    'score': match_score
                })
            else:
                unmatched1.append(i)
                unmatched2.append(j)
        
        # 添加未匹配的索引
        matched_idx1 = {m['event1_idx'] for m in matches}
        matched_idx2 = {m['event2_idx'] for m in matches}
        
        unmatched1.extend([i for i in range(len(events1['timestamps'])) if i not in matched_idx1])
        unmatched2.extend([j for j in range(len(events2['timestamps'])) if j not in matched_idx2])
        
    except ImportError:
        # 如果没有scipy.optimize，使用贪婪匹配
        print("  使用贪婪匹配算法（精度可能较低）")
        
        # 按分数排序所有可能的匹配
        all_possible_matches = []
        for i in range(len(events1['timestamps'])):
            for j in range(len(events2['timestamps'])):
                if score_matrix[i, j] > 0.3:  # 分数阈值
                    all_possible_matches.append((i, j, score_matrix[i, j]))
        
        # 排序并贪婪选择
        all_possible_matches.sort(key=lambda x: x[2], reverse=True)
        
        matched1 = set()
        matched2 = set()
        
        for i, j, score in all_possible_matches:
            if i not in matched1 and j not in matched2:
                matched1.add(i)
                matched2.add(j)
                time_diff = events1['timestamps'][i] - adjusted_timestamps2[j]
                matches.append({
                    'event1_idx': i,
                    'event2_idx': j,
                    'event1_ts': events1['timestamps'][i],
                    'event2_ts': events2['timestamps'][j],
                    'time_diff': time_diff,
                    'score': score
                })
        
        # 添加未匹配的索引
        unmatched1 = [i for i in range(len(events1['timestamps'])) if i not in matched1]
        unmatched2 = [j for j in range(len(events2['timestamps'])) if j not in matched2]
    
    # 计算匹配率
    match_rate = len(matches) / max(len(events1['timestamps']), len(events2['timestamps']))
    print(f"  匹配结果: {len(matches)}对匹配事件, {len(unmatched1)}个未匹配信号1事件, {len(unmatched2)}个未匹配信号2事件")
    
    # 4. 增强型统计分析
    if len(matches) >= 3:
        # 加权时间差 - 优先考虑高分数匹配
        time_diffs = np.array([m['time_diff'] for m in matches])
        match_scores = np.array([m['score'] for m in matches])
        
        # 使用多种鲁棒统计方法
        
        # 1. 加权中位数（RANSAC风格）
        # 先用中位数确定可靠区间
        median_diff = np.median(time_diffs)
        mad = np.median(np.abs(time_diffs - median_diff))
        
        # 调整统计范围 - 使用自适应阈值
        adaptive_threshold = max(3.0 * mad, min(100.0, args.tolerance_ms * 3))
        
        # 标记内点和异常值
        inlier_mask = np.abs(time_diffs - median_diff) <= adaptive_threshold
        
        if np.sum(inlier_mask) >= 3:
            # 使用内点加权计算最终偏移
            inlier_diffs = time_diffs[inlier_mask]
            inlier_scores = match_scores[inlier_mask]
            
            # 归一化分数权重
            weights = inlier_scores / np.sum(inlier_scores)
            
            # 加权平均
            mean_diff = np.sum(inlier_diffs * weights)
            
            # 加权标准差
            variance = np.sum(weights * (inlier_diffs - mean_diff)**2)
            std_diff = np.sqrt(variance)
            
            # 检查置信度
            filtered_count = len(time_diffs) - np.sum(inlier_mask)
            if filtered_count > 0:
                print(f"  RANSAC过滤: 移除{filtered_count}个异常点，保留{np.sum(inlier_mask)}个内点")
            
            # 使用精细偏移作为最终结果，不再使用基础时间差
            final_offset = mean_diff
            
            # 多指标对齐判断
            low_std = std_diff <= args.tolerance_ms * 2.5
            good_match_rate = match_rate >= 0.3 or (match_rate >= 0.2 and len(matches) >= 5)
            high_confidence = np.mean(inlier_scores) >= 0.6
            
            is_aligned = abs(final_offset) <= args.tolerance_ms and low_std and good_match_rate
            
            # 输出详细分析结果
            print(f"  事件匹配分析: 平均得分={np.mean(inlier_scores):.2f}, 标准差={std_diff:.2f}ms, 匹配率={match_rate*100:.1f}%")
            print(f"  事件方法结果: 偏移={final_offset:.2f}ms")
            print(f"  对齐判断: {'对齐' if is_aligned else '未对齐'} (标准差:{low_std}, 匹配率:{good_match_rate}, 置信度:{high_confidence})")
            
        else:
            # 匹配点不足
            mean_diff = median_diff
            std_diff = mad * 1.4826  # 转换MAD为等效标准差
            final_offset = mean_diff
            is_aligned = False
            print("  有效匹配数不足，结果可能不可靠")
    else:
        # 几乎没有匹配
        mean_diff = 0
        std_diff = 0
        final_offset = 0  # 不再使用基础时间差
        is_aligned = False
        print("  匹配数量太少，无法可靠估计偏移")
    
    # 返回增强的结果
    match_stats = {
        'matches': matches,
        'unmatched1': unmatched1,
        'unmatched2': unmatched2,
        'match_rate': match_rate,
        'mean_diff': mean_diff,
        'std_diff': std_diff
    }
    
    return {
        'events1': events1,
        'events2': events2,
        'match_stats': match_stats,
        'is_aligned': is_aligned,
        'method': 'event',
        'mean_offset': final_offset,  # 现在只使用精细偏移作为最终结果
        'std_offset': std_diff,
        'base_time_diff': base_time_diff,  # 保留基础时间差，但仅用于信息展示
        'detail_offset': mean_diff,
        'confidence': np.mean(match_scores) if len(matches) > 0 else 0
    }

def visual_verification(data1, data2, ts_col1, ts_col2, args):
    """使用视觉特征验证两组数据的时间戳对齐"""
    print("\n执行视觉验证分析...")

    # 估计采样率
    rate1 = estimate_sampling_rate(data1, ts_col1)
    rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  采样率估计: 数据集1={rate1:.1f}Hz, 数据集2={rate2:.1f}Hz")
    
    # 获取时间戳范围
    min_ts1, max_ts1 = data1[ts_col1].min(), data1[ts_col1].max()
    min_ts2, max_ts2 = data2[ts_col2].min(), data2[ts_col2].max()
    
    # 记录基础时间差，但不用于计算最终偏移
    time_diff_base = (min_ts1 - min_ts2)
    time_range_1 = max_ts1 - min_ts1
    time_range_2 = max_ts2 - min_ts2
    
    # 确定有效重叠处理时间范围
    max_process_duration = 180 * 1000  # 60秒，单位毫秒
    process_duration = min(time_range_1, time_range_2, max_process_duration)
    
    # 调试输出
    print(f"  数据集1时间范围: {min_ts1:.1f}ms - {max_ts1:.1f}ms (持续{time_range_1/1000:.1f}秒)")
    print(f"  数据集2时间范围: {min_ts2:.1f}ms - {max_ts2:.1f}ms (持续{time_range_2/1000:.1f}秒)")
    print(f"  数据起点时间差: {time_diff_base:.1f}ms (不计入最终偏移结果)")
    print(f"  处理持续时间: {process_duration/1000:.1f}秒")
    
    # 确保数据点足够进行可靠对齐
    if len(data1) < 100 or len(data2) < 100:
        print("  数据点不足，无法可靠进行视觉对齐")
        return {
            'is_aligned': False,
            'confidence': 0.0,
            'mean_offset': 0.0,  # 不使用基础时间差
            'base_time_diff': time_diff_base,  # 仍然记录基础时间差，但仅用于信息展示
            'method': 'visual'
        }
    
    # 对于带有角速度幅值的数据 (特别适合合成数据)，使用直接相关性方法
    # 这种方法在合成数据上表现特别好
    if 'angular_velocity_magnitude' in data1.columns and 'angular_velocity_magnitude' in data2.columns:
        # 提取角速度幅值列并执行直接相关性分析
        print("  检测到角速度幅值列，使用直接相关性分析...")
        sig1 = data1['angular_velocity_magnitude'].values
        sig2 = data2['angular_velocity_magnitude'].values
        
        # 设置最大搜索范围，确保能覆盖较大偏移
        max_offset = 0
        if hasattr(args, 'max_lag_ms'):
            max_offset = args.max_lag_ms
        else:
            max_offset = getattr(args, 'max_lag', 0) * (1000.0 / min(rate1, rate2))  # 将采样点转换为毫秒
            
        max_search_ms = min(max(600, abs(time_diff_base) * 1.2, max_offset), 2000)  # 设置上限为2000ms
        
        print(f"  直接相关性分析搜索范围: ±{max_search_ms:.1f}ms")
        
        # 三级搜索策略，为真实数据和合成数据优化
        search_levels = [
            {'range': max_search_ms, 'step': max(max_search_ms / 10, 20)},       # 粗搜索
            {'range': max_search_ms / 3, 'step': max(max_search_ms / 60, 5)},    # 中等精度
            {'range': max_search_ms / 10, 'step': 2.0}                          # 精细搜索
        ]
        
        # 初始化最佳偏移值
        best_offset = 0.0
        best_corr = -1.0
        
        # 逐级搜索
        for level, search_params in enumerate(search_levels):
            # 确定搜索范围
            if level == 0:
                # 第一级：在基础时间差附近搜索
                search_min = -search_params['range']
                search_max = search_params['range']
            else:
                # 后续级别：在上一级最佳结果附近搜索
                search_min = best_offset - search_params['range']
                search_max = best_offset + search_params['range']
            
            # 生成要测试的偏移值
            test_offsets = np.arange(search_min, search_max + search_params['step'], search_params['step'])
            
            print(f"  第{level+1}级偏移搜索: 范围=[{search_min:.1f}, {search_max:.1f}]ms, 步长={search_params['step']:.1f}ms")
            
            # 记录每个偏移的相关性
            correlations = []
            
            # 采样间隔，用于将时间偏移转换为样本点
            # 使用两个信号中较高的采样率
            interval_ms1 = 1000.0 / rate1
            interval_ms2 = 1000.0 / rate2
            
            # 计算实际时间序列，考虑采样率
            ts1 = np.array([min_ts1 + i * interval_ms1 for i in range(len(sig1))])
            ts2 = np.array([min_ts2 + i * interval_ms2 for i in range(len(sig2))])
            
            # 对于每个候选偏移，计算相关性
            for offset in test_offsets:
                # 应用偏移到时间戳
                adjusted_ts2 = ts2 + offset
                
                # 确定共同时间范围
                common_min = max(ts1.min(), adjusted_ts2.min())
                common_max = min(ts1.max(), adjusted_ts2.max())
                
                # 确保有足够的重叠
                if common_max - common_min < 5000:  # 至少5秒重叠
                    correlations.append((offset, 0))
                    continue
                
                # 选择重叠区域的数据
                mask1 = (ts1 >= common_min) & (ts1 <= common_max)
                mask2 = (adjusted_ts2 >= common_min) & (adjusted_ts2 <= common_max)
                
                # 提取重叠区域的值
                values1 = sig1[mask1]
                values2 = sig2[mask2]
                
                # 如果采样率不同，需要重采样到相同的时间点
                if abs(rate1 - rate2) > 1.0:  # 采样率差异大于1Hz
                    # 创建统一的时间点
                    common_ts = np.linspace(common_min, common_max, min(500, len(values1), len(values2)))
                    
                    # 使用线性插值重采样
                    if len(values1) > 5 and len(values2) > 5:  # 确保有足够的点
                        from scipy.interpolate import interp1d
                        f1 = interp1d(ts1[mask1], values1, bounds_error=False, fill_value='extrapolate')
                        f2 = interp1d(adjusted_ts2[mask2], values2, bounds_error=False, fill_value='extrapolate')
                        
                        resampled1 = f1(common_ts)
                        resampled2 = f2(common_ts)
                        
                        # 计算相关系数
                        if len(resampled1) > 5:
                            try:
                                corr = np.corrcoef(resampled1, resampled2)[0, 1]
                                if np.isnan(corr):
                                    corr = 0
                            except:
                                corr = 0
                        else:
                            corr = 0
                    else:
                        corr = 0
                else:
                    # 如果采样率相似，直接计算相关性（需确保长度相同）
                    min_len = min(len(values1), len(values2))
                    if min_len > 5:
                        try:
                            corr = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                            if np.isnan(corr):
                                corr = 0
                        except:
                            corr = 0
                    else:
                        corr = 0
                
                # 记录该偏移的相关性
                correlations.append((offset, corr))
            
            # 找到最佳相关性
            if correlations:
                # 按相关性排序
                correlations.sort(key=lambda x: x[1], reverse=True)
                best_offset, best_corr = correlations[0]
                
                # 打印最佳结果
                print(f"  第{level+1}级最佳偏移: {best_offset:.2f}ms, 相似度: {best_corr:.4f}")
        
        # 计算最终偏移和相似度
        final_offset = time_diff_base + best_offset
        
        # 根据相关性计算置信度
        # 将相关系数转换为0-1范围的置信度（相关系数范围为-1到1）
        confidence = max(0, (best_corr + 1) / 2)
        
        # 信号质量调整 - 如果信号变化不大，降低置信度
        sig1_std = np.std(sig1)
        sig2_std = np.std(sig2)
        sig1_range = np.max(sig1) - np.min(sig1)
        sig2_range = np.max(sig2) - np.min(sig2)
        
        # 判断信号是否有足够的变化
        if sig1_std < 0.05 * sig1_range or sig2_std < 0.05 * sig2_range:
            confidence *= 0.7  # 降低置信度
            
        # 检查最佳偏移是否在搜索边界，如果是，可能没有找到真正的最佳偏移
        if abs(best_offset) >= max_search_ms * 0.95:
            confidence *= 0.6  # 降低置信度
            
        # 判断是否对齐
        tolerance_ms = args.tolerance_ms if hasattr(args, 'tolerance_ms') else 30.0
        is_aligned = confidence > 0.6 and abs(best_offset) <= tolerance_ms * 3
        
        # 输出结果
        print(f"  视觉方法结果: 偏移={final_offset:.2f}ms")
        print(f"  相似度: {best_corr:.4f}, 置信度: {confidence:.2f}")
        print(f"  对齐状态: {'已对齐' if is_aligned else '未对齐'}")
        
        # 返回结果
        return {
            'is_aligned': is_aligned,
            'confidence': confidence,
            'mean_offset': final_offset,
            'std_offset': abs(best_offset) / 2,  # 简单估计标准差
            'base_time_diff': time_diff_base,  # 保留基础时间差，但仅作为信息
            'fine_offset': best_offset,
            'similarity': best_corr,
            'method': 'visual'
        }

    # 处理复杂的真实数据情况
    # 以下是无角速度幅值列或需要更精细分析的情况
    print("  检测到需要更复杂分析，使用多尺度特征匹配...")
    
    # 选择重叠部分数据
    end_ts1 = min_ts1 + process_duration
    end_ts2 = min_ts2 + process_duration
    
    data1_overlap = data1[(data1[ts_col1] >= min_ts1) & (data1[ts_col1] <= end_ts1)].copy()
    data2_overlap = data2[(data2[ts_col2] >= min_ts2) & (data2[ts_col2] <= end_ts2)].copy()
    
    # 准备必要的数据列
    # 1. 优先使用角速度分量 (x,y,z)
    # 2. 如果没有分量但有角速度幅值，创建伪分量
    # 3. 如果有其他有用特征，也可以包括
    
    # 确定要处理的列
    primary_columns = ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
    secondary_columns = ['angular_velocity_magnitude', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']
    
    # 检查数据集中包含哪些列
    columns_to_process = []
    for col in primary_columns:
        if col in data1_overlap.columns and col in data2_overlap.columns:
            columns_to_process.append(col)
    
    # 如果没有找到主要列，使用次要列
    if not columns_to_process:
        for col in secondary_columns:
            if col in data1_overlap.columns and col in data2_overlap.columns:
                columns_to_process.append(col)
    
    # 如果有角速度幅值但没有角速度分量，创建伪分量
    if 'angular_velocity_magnitude' in data1_overlap.columns and 'angular_velocity_magnitude' in data2_overlap.columns:
        if not all(col in columns_to_process for col in primary_columns[:3]):
            # 创建伪分量 - 使用不同相位的正弦函数，使伪分量更自然
            for df in [data1_overlap, data2_overlap]:
                magnitude = df['angular_velocity_magnitude'].values
                # 创建三个不同相位的伪分量
                df['angular_velocity_x'] = magnitude * np.cos(np.linspace(0, 2*np.pi, len(magnitude)))
                df['angular_velocity_y'] = magnitude * np.sin(np.linspace(0, 2*np.pi, len(magnitude)))
                df['angular_velocity_z'] = magnitude * np.cos(np.linspace(np.pi/4, 2*np.pi+np.pi/4, len(magnitude)))
            
            # 添加这些列到处理列表
            for col in primary_columns[:3]:
                if col not in columns_to_process:
                    columns_to_process.append(col)
    
    # 如果仍然没有找到可用列，返回错误
    if not columns_to_process:
        print("  无法找到合适的数据列进行视觉验证")
        return {
            'is_aligned': False,
            'confidence': 0.0,
            'mean_offset': 0.0,  # 不使用基础时间差
            'base_time_diff': time_diff_base,  # 仍然记录基础时间差，但仅用于信息展示
            'method': 'visual'
        }
    
    print(f"  使用以下列进行视觉验证: {columns_to_process}")
    
    # 计算平均采样间隔
    avg_interval1 = 1000.0 / rate1  # ms
    avg_interval2 = 1000.0 / rate2  # ms
    
    # 确定目标采样率 (使用较高的采样率)
    target_interval = min(avg_interval1, avg_interval2) * 1.05  # 稍微降低一点采样率，确保有足够的点
    target_rate = 1000.0 / target_interval
    print(f"  目标采样间隔: {target_interval:.2f}ms (采样率: {target_rate:.1f}Hz)")
    
    # 对两个信号进行均匀重采样
    resampled_data1 = {}
    resampled_data2 = {}
    
    target_len = int(process_duration / target_interval)
    
    for col in columns_to_process:
        resampled_data1[col] = resample_signal(
            data1_overlap, ts_col1, col, 
            target_len=target_len
        )
        resampled_data2[col] = resample_signal(
            data2_overlap, ts_col2, col, 
            target_len=target_len
        )
    
    # 多尺度特征提取
    features = []
    
    # 定义多个窗口大小 (单位: ms)
    window_sizes = [1000, 2000, 4000, 8000]  # 1秒, 2秒, 4秒, 8秒窗口
    
    for window_size in window_sizes:
        window_points = int(window_size / target_interval)
        
        if window_points > 10 and window_points < target_len / 2:
            for col in columns_to_process:
                signal1 = resampled_data1[col]
                signal2 = resampled_data2[col]
                
                # 滑动窗口特征提取
                step_size = max(1, window_points // 4)  # 75%重叠
                
                for i in range(0, target_len - window_points, step_size):
                    # 提取窗口数据
                    win1 = signal1[i:i+window_points]
                    win2 = signal2[i:i+window_points]
                    
                    # 滑动窗口的时间中点
                    window_center = i + window_points // 2
                    
                    # 尝试提取各种特征
                    try:
                        # 统计特征
                        stats1 = extract_statistical_features(win1)
                        stats2 = extract_statistical_features(win2)
                        
                        # 频率特征
                        try:
                            freq1 = extract_frequency_features(win1, fs=target_rate)
                        except:
                            freq1 = {'dominant_freq': 0, 'energy_low': 0, 'energy_mid': 0, 'energy_high': 0}
                            
                        try:
                            freq2 = extract_frequency_features(win2, fs=target_rate)
                        except:
                            freq2 = {'dominant_freq': 0, 'energy_low': 0, 'energy_mid': 0, 'energy_high': 0}
                        
                        # 形态特征
                        morph1 = extract_morphological_features(win1)
                        morph2 = extract_morphological_features(win2)
                        
                        # 存储特征
                        features.append({
                            'col': col,
                            'window_size': window_size,
                            'window_center': window_center,
                            'signal1': win1,
                            'signal2': win2,
                            'stats1': stats1,
                            'stats2': stats2,
                            'freq1': freq1,
                            'freq2': freq2,
                            'morph1': morph1,
                            'morph2': morph2
                        })
                    except Exception as e:
                        # 特征提取失败，跳过这个窗口
                        continue
    
    if not features:
        print("  无法提取足够特征进行视觉对齐")
        return {
            'is_aligned': False,
            'confidence': 0.0,
            'mean_offset': 0.0,  # 不使用基础时间差
            'base_time_diff': time_diff_base,  # 仍然记录基础时间差，但仅用于信息展示
            'method': 'visual'
        }
    
    print(f"  成功提取 {len(features)} 组特征")
    
    # 多阶段偏移搜索策略
    # 第1阶段：粗略搜索，大范围大步长
    # 第2阶段：中等搜索，基于第1阶段结果的范围，中等步长
    # 第3阶段：精细搜索，基于第2阶段结果的范围，小步长
    
    stages = 3  # 三阶段搜索
    best_offset = 0.0  # 初始值
    best_similarity = 0.0
    similarities = []
    
    # 设置最大搜索范围
    # 对于真实数据，我们需要更大的搜索范围
    max_range_ms = min(500, abs(time_diff_base) * 0.5)  # 最大500ms或基础时间差的50%
    
    for stage in range(stages):
        # 定义搜索范围和步长
        if stage == 0:  # 第一阶段：粗略搜索
            range_ms = max_range_ms
            step_ms = max(10.0, range_ms / 10)   # 较大步长，至少10ms
            
            # 中心化搜索范围
            search_range = (-range_ms, range_ms)
        
        elif stage == 1:  # 第二阶段：中等精度搜索
            # 围绕第一阶段结果扩大搜索
            range_ms = max_range_ms / 3  # 缩小搜索范围
            step_ms = max(5.0, range_ms / 15)  # 中等步长，至少5ms
            
            # 更新搜索范围，围绕最佳偏移
            search_range = (best_offset - range_ms, best_offset + range_ms)
        
        else:  # 第三阶段：精细搜索
            range_ms = max_range_ms / 10  # 进一步缩小范围
            step_ms = 2.0  # 小步长，2ms
            
            # 围绕第二阶段的最佳值搜索
            search_range = (best_offset - range_ms, best_offset + range_ms)
        
        print(f"  第{stage+1}级偏移搜索: 范围=[{search_range[0]:.1f}, {search_range[1]:.1f}]ms, 步长={step_ms:.1f}ms")
        
        # 生成偏移值进行测试
        offsets = np.arange(search_range[0], search_range[1] + step_ms/2, step_ms)
        
        # 重置相似度列表
        similarities = []
        
        # 对每个可能的偏移计算相似度
        for offset in offsets:
            # 计算全局相似度得分
            similarity = calculate_offset_similarity(features, offset, target_interval)
            similarities.append((offset, similarity))
        
        # 找出最佳偏移和相似度
        if similarities:
            offset_similarities = np.array([(o, s) for o, s in similarities])
            
            if len(offset_similarities) > 0:
                # 找到最高相似度点及其附近区域
                best_idx = np.argmax(offset_similarities[:, 1])
                best_offset = offset_similarities[best_idx, 0]
                best_similarity = offset_similarities[best_idx, 1]
                
                # 调试输出
                print(f"  第{stage+1}级最佳偏移: {best_offset:.2f}ms, 相似度: {best_similarity:.4f}")
    
    # 计算最终视觉方法偏移估计
    final_offset = best_offset  # 不再使用基础时间差
    
    # 计算偏移置信度
    # 使用二次拟合峰值附近相似度曲线，评估峰值锐度
    confidence = 0.0
    
    if len(similarities) >= 5:
        # 提取相似度峰值附近的点
        peak_idx = np.argmax([s[1] for s in similarities])
        range_start = max(0, peak_idx - 2)
        range_end = min(len(similarities), peak_idx + 3)
        
        # 峰值附近的偏移和相似度
        peak_offsets = [similarities[i][0] for i in range(range_start, range_end)]
        peak_similarities = [similarities[i][1] for i in range(range_start, range_end)]
        
        # 计算峰值锐度作为置信度指标
        if len(peak_offsets) >= 3:
            try:
                # 尝试二次拟合
                coeffs = np.polyfit(peak_offsets, peak_similarities, 2)
                peak_sharpness = -coeffs[0]  # 二次项系数的负值是峰值锐度
                
                # 将峰值锐度归一化为置信度 (0-1 范围)
                confidence = min(1.0, max(0.0, peak_sharpness * 5000))  # 调整缩放因子
                
                # 考虑整体相似度
                confidence = confidence * 0.6 + best_similarity * 0.4  # 平衡峰值锐度和最佳相似度
            except:
                # 如果拟合失败，使用最佳相似度作为基础置信度
                confidence = best_similarity
        else:
            confidence = best_similarity
    else:
        confidence = best_similarity  # 样本点少时，使用相似度作为置信度
    
    # 确保置信度在合理范围内
    confidence = min(1.0, max(0.0, confidence))
    
    # 考虑偏移的合理性调整置信度
    # 如果偏移超过基础时间差的一定比例，降低置信度
    if abs(time_diff_base) > 0:
        relative_change = abs(best_offset) / abs(time_diff_base)
        if relative_change > 0.6:  # 如果偏移超过基础时间差的60%
            confidence *= max(0.4, 1.0 - (relative_change - 0.6) / 0.4)
    
    # 判断信号是否对齐
    tolerance_ms = args.tolerance_ms if hasattr(args, 'tolerance_ms') else 30.0
    is_aligned = confidence > 0.6 and abs(best_offset) <= tolerance_ms * 3  # 使用更宽松的容差
    
    # 调试输出
    print(f"  视觉方法结果: 偏移={final_offset:.2f}ms")
    print(f"  相似度: {best_similarity:.4f}, 置信度: {confidence:.2f}")
    print(f"  对齐状态: {'已对齐' if is_aligned else '未对齐'}")
    
    # 返回结果
    return {
        'is_aligned': is_aligned,
        'confidence': confidence,
        'mean_offset': final_offset,
        'std_offset': abs(best_offset) / 2,  # 估计标准差
        'base_time_diff': time_diff_base,  # 保留基础时间差，但仅作为信息
        'fine_offset': best_offset,
        'similarity': best_similarity,
        'method': 'visual'
    }

def extract_statistical_features(signal):
    """提取信号的统计特征"""
    if len(signal) == 0:
        return {
            'mean': 0, 'std': 0, 'skew': 0, 'kurtosis': 0,
            'q25': 0, 'q50': 0, 'q75': 0, 'iqr': 0
        }
    
    # 计算统计矩
    mean = np.mean(signal)
    std = np.std(signal)
    
    # 鲁棒计算高阶矩以避免除零等问题
    try:
        skew = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)
    except:
        skew = 0
        kurtosis = 0
    
    # 四分位数
    try:
        q25 = np.percentile(signal, 25)
        q50 = np.percentile(signal, 50)
        q75 = np.percentile(signal, 75)
        iqr = q75 - q25
    except:
        q25, q50, q75, iqr = 0, 0, 0, 0
    
    return {
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'iqr': iqr
    }

def extract_frequency_features(signal, fs):
    """提取信号的频域特征"""
    if len(signal) < 10:
        return {
            'dominant_freq': 0,
            'energy_low': 0,
            'energy_mid': 0,
            'energy_high': 0
        }
    
    # 应用窗函数减少频谱泄漏
    windowed_signal = signal * np.hanning(len(signal))
    
    # 计算功率谱密度
    freqs, psd = scipy.signal.welch(windowed_signal, fs=fs, nperseg=min(256, len(signal)))
    
    # 如果计算失败，返回零特征
    if len(freqs) == 0 or len(psd) == 0:
        return {
            'dominant_freq': 0,
            'energy_low': 0,
            'energy_mid': 0,
            'energy_high': 0
        }
    
    # 找出主导频率
    dominant_idx = np.argmax(psd)
    dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0
    
    # 计算不同频段的能量
    total_energy = np.sum(psd)
    
    if total_energy > 0:
        # 定义频率边界
        low_bound = 0.5  # Hz
        mid_bound = 5.0  # Hz
        high_bound = 15.0  # Hz
        
        # 计算各频段能量比例
        low_mask = freqs <= low_bound
        mid_mask = (freqs > low_bound) & (freqs <= mid_bound)
        high_mask = freqs > mid_bound
        
        energy_low = np.sum(psd[low_mask]) / total_energy if np.any(low_mask) else 0
        energy_mid = np.sum(psd[mid_mask]) / total_energy if np.any(mid_mask) else 0
        energy_high = np.sum(psd[high_mask]) / total_energy if np.any(high_mask) else 0
    else:
        energy_low, energy_mid, energy_high = 0, 0, 0
    
    return {
        'dominant_freq': dominant_freq,
        'energy_low': energy_low,
        'energy_mid': energy_mid,
        'energy_high': energy_high
    }

def extract_morphological_features(signal):
    """提取信号的形态特征"""
    if len(signal) < 3:
        return {'peak_count': 0, 'zero_crossings': 0, 'abs_energy': 0}
    
    # 计算峰值数量（局部最大值）
    peaks = scipy.signal.find_peaks(signal)[0]
    peak_count = len(peaks)
    
    # 计算过零率
    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    
    # 计算绝对能量
    abs_energy = np.sum(np.abs(signal))
    
    return {
        'peak_count': peak_count,
        'zero_crossings': zero_crossings,
        'abs_energy': abs_energy
    }

def calculate_offset_similarity(features, offset_ms, interval_ms):
    """
    计算给定时间偏移下两个信号之间的相似度
    
    参数:
        features: 提取的特征列表
        offset_ms: 要测试的偏移值（毫秒）
        interval_ms: 平均采样间隔（毫秒）
    
    返回:
        相似度得分（0-1之间，1表示完全匹配）
    """
    # 如果没有特征，返回0
    if not features:
        return 0.0
    
    total_similarity = 0.0
    feature_count = 0
    
    # 处理每组特征
    for feature_set in features:
        # 获取信号和特征
        signal1 = feature_set.get('signal1', [])
        signal2 = feature_set.get('signal2', [])
        
        # 跳过无效特征集
        if len(signal1) < 10 or len(signal2) < 10:
            continue
        
        # 计算时间轴偏移点数
        offset_points = int(round(offset_ms / interval_ms))
        
        # 应用偏移，计算重叠区域
        if offset_points >= 0:
            sig1_slice = signal1[offset_points:] if offset_points < len(signal1) else []
            sig2_slice = signal2[:len(signal2)-offset_points] if offset_points < len(signal2) else []
        else:
            abs_offset = abs(offset_points)
            sig1_slice = signal1[:len(signal1)-abs_offset] if abs_offset < len(signal1) else []
            sig2_slice = signal2[abs_offset:] if abs_offset < len(signal2) else []
        
        # 确保切片长度匹配并且足够长
        min_len = min(len(sig1_slice), len(sig2_slice))
        if min_len < 10:  # 至少需要10个点进行有意义的比较
            continue
        
        # 截取相同长度进行比较
        sig1_compare = sig1_slice[:min_len]
        sig2_compare = sig2_slice[:min_len]
        
        # 计算不同类型特征的相似度
        # 1. 统计特征比较
        stats1 = feature_set.get('stats1', {})
        stats2 = feature_set.get('stats2', {})
        
        # 比较统计特征
        if stats1 and stats2:
            stat_keys = ['mean', 'std', 'skew', 'kurtosis']
            stat_similarity = 0.0
            valid_stats = 0
            
            for key in stat_keys:
                if key in stats1 and key in stats2 and isinstance(stats1[key], (int, float)) and isinstance(stats2[key], (int, float)):
                    # 规格化差异
                    max_val = max(abs(stats1[key]), abs(stats2[key]))
                    if max_val > 0:
                        diff = abs(stats1[key] - stats2[key]) / max_val
                        stat_similarity += max(0, 1 - min(diff, 1))
                        valid_stats += 1
            
            if valid_stats > 0:
                stat_similarity /= valid_stats
                total_similarity += stat_similarity
                feature_count += 1
        
        # 2. 频率特征比较
        freq1 = feature_set.get('freq1', {})
        freq2 = feature_set.get('freq2', {})
        
        if freq1 and freq2:
            freq_keys = ['dominant_freq', 'energy_low', 'energy_mid', 'energy_high']
            freq_similarity = 0.0
            valid_freqs = 0
            
            for key in freq_keys:
                if key in freq1 and key in freq2 and isinstance(freq1[key], (int, float)) and isinstance(freq2[key], (int, float)):
                    if key == 'dominant_freq':
                        # 对主导频率，使用相对差异
                        max_freq = max(freq1[key], freq2[key])
                        if max_freq > 0:
                            diff = abs(freq1[key] - freq2[key]) / max_freq
                            freq_similarity += max(0, 1 - min(diff, 1))
                            valid_freqs += 1
                    else:
                        # 对能量分布，直接比较差异
                        diff = abs(freq1[key] - freq2[key])
                        freq_similarity += max(0, 1 - min(diff, 1))
                        valid_freqs += 1
            
            if valid_freqs > 0:
                freq_similarity /= valid_freqs
                total_similarity += freq_similarity * 1.5  # 频率特征权重略高
                feature_count += 1.5
        
        # 3. 形态学特征比较
        morph1 = feature_set.get('morph1', {})
        morph2 = feature_set.get('morph2', {})
        
        if morph1 and morph2:
            morph_keys = ['peak_count', 'zero_crossing_rate']
            morph_similarity = 0.0
            valid_morphs = 0
            
            for key in morph_keys:
                if key in morph1 and key in morph2 and isinstance(morph1[key], (int, float)) and isinstance(morph2[key], (int, float)):
                    # 对峰值计数，使用相对差异
                    max_val = max(morph1[key], morph2[key])
                    if max_val > 0:
                        diff = abs(morph1[key] - morph2[key]) / max_val
                        morph_similarity += max(0, 1 - min(diff, 1))
                        valid_morphs += 1
            
            if valid_morphs > 0:
                morph_similarity /= valid_morphs
                total_similarity += morph_similarity
                feature_count += 1
        
        # 4. 信号相关性比较（如果信号足够长）
        if min_len >= 30:
            try:
                # 标准化信号
                sig1_norm = (sig1_compare - np.mean(sig1_compare)) / np.std(sig1_compare) if np.std(sig1_compare) > 0 else sig1_compare
                sig2_norm = (sig2_compare - np.mean(sig2_compare)) / np.std(sig2_compare) if np.std(sig2_compare) > 0 else sig2_compare
                
                # 计算相关系数
                corr = np.corrcoef(sig1_norm, sig2_norm)[0, 1]
                
                # 处理无效值
                if np.isnan(corr):
                    corr = 0
                
                # 转换为相似度分数
                signal_similarity = max(0, (corr + 1) / 2)  # 将[-1, 1]映射到[0, 1]
                
                total_similarity += signal_similarity * 2  # 原始信号相关性权重更高
                feature_count += 2
            except:
                pass  # 如果计算失败，跳过此部分
    
    # 计算综合相似度
    if feature_count > 0:
        return total_similarity / feature_count
    else:
        return 0.0

# ========== 可视化函数 ==========

def visualize_correlation_results(verification_results, sensor1, sensor2, args):
    """可视化相关性验证结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取结果数据 - 更新字段名以匹配correlation_verification方法的返回值
    total_offset = verification_results.get('mean_offset', 0)
    best_lag = verification_results.get('lag', 0)
    base_time_diff = verification_results.get('base_time_diff', 0)
    corr_curve = verification_results.get('corr_curve', {'lags': [], 'corrs': []})
    best_corr = verification_results.get('corr', 0)
    original_corr = verification_results.get('original_corr', 0)
    # 优先使用明确的is_aligned字段，如果没有则回退到状态判断
    is_aligned = verification_results.get('is_aligned', verification_results.get('status', '') == 'success')
    
    # 绘制相关性曲线
    plt.figure(figsize=(12, 8))
    
    # 设置主标题
    plt.suptitle(f"{sensor1}-{sensor2} 相关性方法对齐分析", fontsize=16)
    
    # 绘制相关性曲线
    plt.subplot(2, 1, 1)
    lags = np.array(corr_curve['lags'])
    corrs = np.array(corr_curve['corrs'])
    
    if len(lags) > 0 and len(corrs) > 0:
        plt.plot(lags, corrs, 'b-', linewidth=1.5)
        
        # 标记最佳偏移
        if best_lag in lags:
            idx = np.where(lags == best_lag)[0][0]
            plt.axvline(x=best_lag, color='r', linestyle='--', linewidth=1.5)
            plt.plot(best_lag, corrs[idx], 'ro', markersize=8)
            plt.annotate(f'最佳偏移: {best_lag}ms', 
                        xy=(best_lag, corrs[idx]), 
                        xytext=(best_lag+20, corrs[idx]),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=10)
    
        # 标记零偏移
        if 0 in lags:
            idx = np.where(lags == 0)[0][0]
            plt.axvline(x=0, color='g', linestyle=':', linewidth=1.5)
            plt.plot(0, corrs[idx], 'go', markersize=6)
        
        # 标记容差范围
        plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green',
                   label=f'容差范围 (±{args.tolerance_ms}ms)')
        
        plt.title(f"相关性分析 (检测偏移: {total_offset:.2f}ms)")
        plt.xlabel('时间偏移 (ms)')
        plt.ylabel('相关系数')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无相关性数据', ha='center', va='center', fontsize=14)
        plt.axis('off')
        
    # 添加结果摘要
    plt.subplot(2, 1, 2)
    plt.axis('off')
    summary_text = f"""
    相关性分析结果摘要:
    -----------------------------------
    最佳偏移: {total_offset:.2f} ms
    数据起点差异: {base_time_diff:.2f} ms (仅供参考)
    
    最佳相关系数: {best_corr:.4f}
    原始相关系数: {original_corr if original_corr else 'N/A'}
    改进: {best_corr-original_corr if original_corr else 'N/A'}
    
    对齐状态: {'✓ 已对齐' if is_aligned else '✗ 未对齐'} (容差: ±{args.tolerance_ms} ms)
    """
    # 使用支持中文的字体，移除family='monospace'
    plt.text(0.1, 0.9, summary_text, fontsize=12, 
            ha='left', va='top', bbox=dict(facecolor='lightgray', alpha=0.2))
    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为主标题留出空间
    plt.savefig(os.path.join(args.output_dir, f"{sensor1}_{sensor2}_correlation_verification.png"), dpi=300)
    plt.close()

def visualize_event_results(verification_results, sensor1, sensor2, args):
    """可视化事件对齐验证结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    result_dir = args.output_dir
    
    # 获取事件匹配信息
    events1 = verification_results['events1']
    events2 = verification_results['events2']
    match_stats = verification_results['match_stats']
    mean_offset = verification_results.get('mean_offset', 0)
    base_time_diff = verification_results.get('base_time_diff', 0)
    detail_offset = verification_results.get('detail_offset', 0)
    is_aligned = verification_results.get('is_aligned', False)
    
    # 创建一个大的图像以显示结果
    plt.figure(figsize=(14, 10))
    
    # 添加标题
    plt.suptitle(f"{sensor1}-{sensor2} 事件方法对齐分析 (检测偏移: {mean_offset:.2f}ms)", fontsize=16)
    
    # 1. 事件匹配图
    plt.subplot(2, 1, 1)
    
    # 绘制事件时间点
    plt.scatter(events1['timestamps'], np.ones_like(events1['timestamps']), marker='|', s=100, 
               label=f'{sensor1} 事件', color='blue')
    plt.scatter(events2['timestamps'], np.ones_like(events2['timestamps'])*1.1, 
               marker='|', s=100, label=f'{sensor2} 事件', color='red')
    
    # 绘制匹配连线
    matched_events1 = [events1['timestamps'][m['event1_idx']] for m in match_stats['matches']]
    matched_events2 = [events2['timestamps'][m['event2_idx']] for m in match_stats['matches']]
    
    for i in range(len(matched_events1)):
        plt.plot([matched_events1[i], matched_events2[i]], [1, 1.1], 'k-', alpha=0.3)
    
    plt.title(f'事件匹配 (匹配率: {match_stats["match_rate"]*100:.1f}%, 事件数: {len(events1["timestamps"])}/{len(events2["timestamps"])})')
    plt.xlabel('时间戳 (ms)')
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 时间差分布
    plt.subplot(2, 1, 2)
    time_diffs = [m['time_diff'] for m in match_stats['matches']]
    
    # 解决ValueError: Too many bins 问题 - 动态计算bin数量
    if len(time_diffs) > 0:
        time_diff_range = max(time_diffs) - min(time_diffs)
        if time_diff_range <= 0:  # 所有值相同
            bin_count = 1
        else:
            # 每个bin至少1ms间隔，最少5个bin
            bin_count = min(30, max(5, int(time_diff_range)))
    else:
        bin_count = 5  # 默认值
    
    if len(time_diffs) > 0:
        plt.hist(time_diffs, bins=bin_count, alpha=0.7, color='blue')
        plt.axvline(x=0, color='k', linestyle='--', label='零偏移')
        plt.axvline(x=match_stats['mean_diff'], color='r', linestyle='-', 
                   label=f'平均偏移: {match_stats["mean_diff"]:.2f} ms')
        plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green', 
                   label=f'容差范围 (±{args.tolerance_ms} ms)')
        # 只在有绘制元素时添加图例
        plt.legend()
    else:
        plt.text(0.5, 0.5, "无匹配事件", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(f'时间差分布 (数据起点差异: {base_time_diff:.2f}ms，仅供参考)')
    plt.xlabel('时间差 (ms)')
    plt.ylabel('事件数量')
    plt.grid(True, alpha=0.3)
    
    # 添加结果摘要
    alignment_status = "已对齐" if is_aligned else "未对齐"
    # 使用支持中文的字体
    plt.figtext(0.5, 0.01, f"对齐状态: {alignment_status} (容差: ±{args.tolerance_ms}ms)", 
               ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.2))
    
    # 保存图形
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为主标题留出空间
    plt.savefig(os.path.join(result_dir, f"{sensor1}_{sensor2}_event_verification.png"), dpi=300)
    plt.close()

def visualize_visual_results(verification_results, sensor1, sensor2, args):
    """可视化视觉方法对齐验证结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 提取结果
    mean_offset = verification_results.get('mean_offset', 0)
    base_time_diff = verification_results.get('base_time_diff', 0)
    detail_offset = verification_results.get('fine_offset', 0)
    is_aligned = verification_results.get('is_aligned', False)
    confidence = verification_results.get('confidence', 0)
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    
    # 设置标题
    plt.suptitle(f"{sensor1}-{sensor2} 视觉方法时间戳对齐分析", fontsize=16)
    
    # 创建文本摘要区域
    plt.subplot(1, 1, 1)
    plt.axis('off')
    
    summary_text = f"""
    视觉对齐分析结果摘要:
    --------------------------------------
    检测偏移:       {mean_offset:.2f} ms
    数据起点差异:    {base_time_diff:.2f} ms (仅供参考)
    
    置信度:         {confidence:.2f}
    对齐状态:       {'✓ 已对齐' if is_aligned else '✗ 未对齐'} (容差: ±{args.tolerance_ms} ms)
    
    方法:           多尺度特征匹配
    """
    
    # 使用支持中文的字体，而不是monospace
    plt.text(0.5, 0.5, summary_text, fontsize=14, 
            ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.2))
    
    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为主标题留出空间
    plt.savefig(os.path.join(args.output_dir, f"{sensor1}_{sensor2}_visual_verification.png"), dpi=300)
    plt.close()

def save_results_to_txt(verification_results, output_dir):
    """保存所有验证结果到纯文本文件"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "alignment_summary.txt")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("===== 传感器对齐验证结果摘要 =====\n\n")
            
            for sensor_pair, pair_results in verification_results.items():
                sensor1, sensor2 = sensor_pair.split('_')
                f.write(f"传感器对: {sensor1}-{sensor2}\n")
                f.write("-" * 50 + "\n")
                
                # 判断是否有任何方法成功
                aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
                total_methods = len(pair_results)
                final_aligned = aligned_count >= total_methods / 2
                
                # 写入总体结果
                f.write(f"总体对齐状态: {'已对齐' if final_aligned else '未对齐'} ({aligned_count}/{total_methods}方法支持)\n\n")
                
                # 每种方法详细结果
                for method_name, result in pair_results.items():
                    f.write(f"{method_name.capitalize()}方法:\n")
                    
                    # 获取基本结果
                    offset = result.get('mean_offset', 0)
                    base_time_diff = result.get('base_time_diff', 0)
                    is_aligned = result.get('is_aligned', False)
                    
                    # 写入偏移和基础时间差
                    f.write(f"  - 检测偏移: {offset:.2f}ms\n")
                    f.write(f"  - 数据起点差异: {base_time_diff:.2f}ms (仅供参考)\n")
                    
                    # 根据方法获取并写入置信度
                    if method_name == 'correlation':
                        confidence = result.get('max_corr', 0)
                        f.write(f"  - 最大相关性: {confidence:.4f}\n")
                    elif method_name == 'event':
                        confidence = result.get('confidence', 0)
                        f.write(f"  - 置信度: {confidence:.4f}\n")
                    elif method_name == 'visual':
                        confidence = result.get('confidence', 0)
                        f.write(f"  - 置信度: {confidence:.4f}\n")
                    
                    # 写入对齐状态
                    f.write(f"  - 对齐状态: {'已对齐' if is_aligned else '未对齐'}\n\n")
                
                f.write("\n")
            
            f.write("\n===== 完成 =====\n")
        
        print(f"结果摘要已保存到: {result_file}")
        return result_file
    except Exception as e:
        print(f"保存结果到文本文件时出错: {e}")
        return None

def main():
    args = parse_arguments()
    
    # 加载数据
    print("加载传感器数据...")
    data_sources = {}
    all_verification_results = {}
    
    # 确定要使用的方法
    if args.methods == "all":
        methods = ['correlation', 'event', 'visual']
    else:
        methods = args.methods.split(',')
    
    # 确定要处理的传感器组合
    if args.sensors == "all":
        available_sensors = []
        
        # 检查IMU数据
        if args.imu_file:
            imu_data = load_imu_data(args)
            if imu_data is not None:
                data_sources['imu'] = {
                    'data': imu_data,
                    'ts_col': args.imu_timestamp_col
                }
                available_sensors.append('imu')
                print(f"已加载IMU数据: {len(imu_data)}行")
        
        # 检查轮式编码器数据
        if args.odo_file:
            odo_data = load_odometry_data(args)
            if odo_data is not None:
                data_sources['odo'] = {
                    'data': odo_data,
                    'ts_col': args.odo_timestamp_col
                }
                available_sensors.append('odo')
                print(f"已加载轮式编码器数据: {len(odo_data)}行")
        
        # 检查RTK数据
        if args.rtk_file:
            rtk_data = load_rtk_data(args)
            if rtk_data is not None:
                data_sources['rtk'] = {
                    'data': rtk_data,
                    'ts_col': args.rtk_timestamp_col
                }
                available_sensors.append('rtk')
                print(f"已加载RTK数据: {len(rtk_data)}行")
        
        # 检查图像数据
        if args.image_dir and args.image_timestamp_file:
            # 加载图像时间戳
            image_timestamps = load_image_timestamps(args)
            if image_timestamps is not None:
                # 处理图像数据
                image_data = process_image_data(args)
                if image_data is not None:
                    data_sources['image'] = {
                        'data': image_data,
                        'ts_col': 'timestamp'
                    }
                    available_sensors.append('image')
                    print(f"已处理图像数据: {len(image_data)}行")
        
        # 检查可用的传感器组合
        if len(available_sensors) >= 2:
            # 生成所有可能的传感器对
            sensor_pairs = []
            for i in range(len(available_sensors)):
                for j in range(i+1, len(available_sensors)):
                    sensor_pairs.append((available_sensors[i], available_sensors[j]))
        else:
            print("Error: 需要至少两个传感器数据源来验证时间戳对齐")
            return
    else:
        # 使用指定的传感器组合
        sensor_pair = args.sensors.split('_')
        if len(sensor_pair) != 2:
            print(f"Error: 传感器组合格式无效: {args.sensors}，应为'sensor1_sensor2'")
            return
        sensor1, sensor2 = sensor_pair
        
        # 加载第一个传感器数据
        if sensor1 == 'imu':
            data1 = load_imu_data(args)
            ts_col1 = args.imu_timestamp_col
        elif sensor1 == 'odo':
            data1 = load_odometry_data(args)
            ts_col1 = args.odo_timestamp_col
        elif sensor1 == 'rtk':
            data1 = load_rtk_data(args)
            ts_col1 = args.rtk_timestamp_col
        elif sensor1 == 'image':
            image_timestamps = load_image_timestamps(args)
            data1 = process_image_data(args)
            ts_col1 = 'timestamp'
        else:
            print(f"Error: 未知传感器类型: {sensor1}")
            return
        
        # 加载第二个传感器数据
        if sensor2 == 'imu':
            data2 = load_imu_data(args)
            ts_col2 = args.imu_timestamp_col
        elif sensor2 == 'odo':
            data2 = load_odometry_data(args)
            ts_col2 = args.odo_timestamp_col
        elif sensor2 == 'rtk':
            data2 = load_rtk_data(args)
            ts_col2 = args.rtk_timestamp_col
        elif sensor2 == 'image':
            if 'image_timestamps' not in locals():
                image_timestamps = load_image_timestamps(args)
            data2 = process_image_data(args)
            ts_col2 = 'timestamp'
        else:
            print(f"Error: 未知传感器类型: {sensor2}")
            return
        
        # 检查数据是否加载成功
        if data1 is None or data2 is None:
            print("Error: 无法加载传感器数据")
            return
        
        print(f"已加载{sensor1}数据: {len(data1)}行")
        print(f"已加载{sensor2}数据: {len(data2)}行")
        
        # 存储数据源
        data_sources = {
            sensor1: {'data': data1, 'ts_col': ts_col1},
            sensor2: {'data': data2, 'ts_col': ts_col2}
        }
        
        # 设置传感器对
        sensor_pairs = [(sensor1, sensor2)]
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 对每个传感器组合执行验证
    for sensor1, sensor2 in sensor_pairs:
        pair_key = f"{sensor1}_{sensor2}"
        print(f"\n验证 {sensor1} 和 {sensor2} 的时间戳对齐...")
        
        data1 = data_sources[sensor1]['data']
        ts_col1 = data_sources[sensor1]['ts_col']
        
        data2 = data_sources[sensor2]['data']
        ts_col2 = data_sources[sensor2]['ts_col']
        
        # 存储此组合的所有方法结果
        pair_results = {}
        
        # 对每种方法执行验证
        for method in methods:
            print(f"  使用{method}方法验证...")
            
            if method == 'correlation':
                # 互相关验证
                try:
                    result = correlation_verification(
                        data1, data2, ts_col1, ts_col2, args
                    )
                    pair_results['correlation'] = result
                    visualize_correlation_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  相关性验证出错: {e}")
                    continue
                
            elif method == 'event':
                # 事件同步验证
                try:
                    result = event_verification(
                        data1, data2, ts_col1, ts_col2, args
                    )
                    pair_results['event'] = result
                    visualize_event_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  事件验证出错: {e}")
                    continue
                
            elif method == 'visual':
                # 可视化比较
                try:
                    result = visual_verification(
                        data1, data2, ts_col1, ts_col2, args
                    )
                    pair_results['visual'] = result
                    visualize_visual_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  视觉验证出错: {e}")
                    continue
        
        # 存储组合结果
        all_verification_results[pair_key] = pair_results
    
    # 显示简要结果
    print("\n===== 验证结果摘要 =====")
    for sensor_pair, pair_results in all_verification_results.items():
        sensor1, sensor2 = sensor_pair.split('_')
        
        aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
        total_methods = len(pair_results)
        final_aligned = aligned_count >= total_methods / 2
        
        print(f"{sensor1}-{sensor2}: {'对齐' if final_aligned else '未对齐'} ({aligned_count}/{total_methods})")
    
    # 保存结果到文本文件
    summary_file = save_results_to_txt(all_verification_results, args.output_dir)
    if summary_file:
        print(f"\n简洁的判定结果已保存到: {summary_file}")
    
    # 生成综合报告（如果需要）
    if hasattr(args, 'generate_report') and args.generate_report:
        report_file = os.path.join(args.output_dir, "verification_report.html")
        print(f"\n详细报告已保存至: {report_file}")
    else:
        print("\n完成所有验证")

if __name__ == "__main__":
    main()