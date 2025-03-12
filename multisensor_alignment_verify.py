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

# 更具体的警告过滤器
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 特别针对字形缺失警告 - 修复转义序列
warnings.filterwarnings("ignore", message="Glyph \\d+ .*? missing from font.*")

# ========== 配置matplotlib使用支持中文的字体 ==========
# 尝试查找系统中支持中文的字体
chinese_fonts = []
for font in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'NotoSansCJK-Regular']:
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
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
    print(f"使用字体: {chinese_fonts[0]} 以支持中文显示")
else:
    # 如果没有找到支持中文的字体，则使用英文标签
    print("未找到支持中文的字体，将使用英文标签")

# 确保特殊符号正确显示
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号代替Unicode减号

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
        
        # 计算方向角 (仅对速度不为0的点计算)
        non_zero_speed = rtk_data['speed'] > 1e-3  # 速度大于1mm/s的点
        
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
        
        # 计算方向角的变化率
        rtk_data['direction_diff'] = rtk_data['direction'].diff().fillna(0)
        
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
        
        # 计算角速度幅值
        rtk_data['angular_velocity_magnitude'] = np.abs(rtk_data['direction_diff'])
        
        # 将低速状态下的角速度设为0 (速度过低时角速度不可靠)
        low_speed = rtk_data['speed'] < 0.1  # 速度低于0.1 m/s
        if low_speed.any():
            print(f"  发现{low_speed.sum()}条低速记录，角速度将被设置为0")
            rtk_data.loc[low_speed, 'angular_velocity_magnitude'] = 0.0
            
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
    """使用互相关方法验证时间戳对齐"""
    # 估计采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 计算基础时间差
    base_time_diff = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  基础时间差: {base_time_diff:.2f}ms (正值表示信号1滞后于信号2)")
    
    # 准备数据
    # 获取公共时间范围
    min_ts1, max_ts1 = data1[ts_col1].min(), data1[ts_col1].max()
    min_ts2, max_ts2 = data2[ts_col2].min(), data2[ts_col2].max()
    
    common_start = max(min_ts1, min_ts2) + 5  # 添加小余量确保数据完整性
    common_end = min(max_ts1, max_ts2) - 5
    common_range = common_end - common_start
    
    print(f"  信号公共部分: {common_start:.1f}~{common_end:.1f}ms, 长度: {common_range:.1f}ms")
    
    # 在公共范围内截取数据
    data1_filtered = data1[(data1[ts_col1] >= common_start) & (data1[ts_col1] <= common_end)]
    data2_filtered = data2[(data2[ts_col2] >= common_start) & (data2[ts_col2] <= common_end)]
    
    # 进行数据归一化
    data1_norm = data1_filtered.copy()
    data2_norm = data2_filtered.copy()
    
    # 标准化角速度幅值
    value_col = 'angular_velocity_magnitude'
    data1_norm[value_col] = (data1_norm[value_col] - data1_norm[value_col].mean()) / (data1_norm[value_col].std() + 1e-10)
    data2_norm[value_col] = (data2_norm[value_col] - data2_norm[value_col].mean()) / (data2_norm[value_col].std() + 1e-10)
    
    # 创建用于插值的共同时间点
    # 确保足够的点但不过多以避免计算太慢
    target_rate = max(sampling_rate1, sampling_rate2) * 1.5  # 稍高于最高采样率
    target_samples = int(common_range / 1000 * target_rate)
    target_samples = min(max(target_samples, 1000), 10000)  # 至少1000点，最多10000点
    
    common_times = np.linspace(common_start, common_end, target_samples)
    
    # 插值到共同时间点
    f1 = interpolate.interp1d(data1_norm[ts_col1], data1_norm[value_col], 
                             kind='linear', bounds_error=False, fill_value=0)
    f2 = interpolate.interp1d(data2_norm[ts_col2], data2_norm[value_col], 
                             kind='linear', bounds_error=False, fill_value=0)
    
    signal1 = f1(common_times)
    signal2 = f2(common_times)
    
    # 计算原始相关性
    valid_mask = ~np.isnan(signal1) & ~np.isnan(signal2)
    if np.sum(valid_mask) < 100:
        print("  有效数据点不足，无法进行相关性分析")
        return {
            'is_aligned': False,
            'method': 'correlation',
            'mean_offset': 0,
            'std_offset': 0
        }
    
    original_corr = np.corrcoef(signal1[valid_mask], signal2[valid_mask])[0, 1]
    print(f"  原始信号相关性: {original_corr:.4f}")
    
    # 动态判断是否需要执行相关搜索
    # 如果原始相关性已经非常高（> 0.95），我们可能已经处于最佳对齐状态
    if original_corr > 0.95:
        print(f"  原始相关性已经非常高({original_corr:.4f})，可能已经对齐")
        # 执行小范围搜索以确认
        max_lag_ms = max(args.tolerance_ms * 5, 100)  # 小范围搜索
    else:
        # 否则执行大范围搜索
        max_lag_ms = max(args.max_lag, args.tolerance_ms * 10, 1000)  # 至少1000ms以确保足够搜索范围
    
    # 粗略搜索的步长，根据采样率调整
    coarse_step_ms = max(2, int(10 / max(sampling_rate1, sampling_rate2) * 20))  # 采样率越高，步长越小
    coarse_step_ms = min(coarse_step_ms, 5)  # 最大步长5ms
    
    print(f"  正在搜索最佳偏移，范围：-{max_lag_ms}ms 到 +{max_lag_ms}ms，步长：{coarse_step_ms}ms...")
    
    # 搜索最佳偏移
    dt = common_times[1] - common_times[0]  # 时间步长
    lags_ms = np.arange(-max_lag_ms, max_lag_ms + coarse_step_ms, coarse_step_ms)
    lags_samples = (lags_ms / 1000 / dt).astype(int)  # 转换为样本偏移
    
    # 粗略搜索
    corr_values = []
    for lag in lags_samples:
        if lag == 0:
            corr_values.append(original_corr)
            continue
        
        # 修复以确保在负偏移时正确计算相关性
        if lag > 0:
            # 如果信号1滞后，则移动信号1
            s1 = signal1[lag:] if lag < len(signal1) else np.array([])
            s2 = signal2[:-lag] if lag < len(signal2) else np.array([])
        else:
            # 如果信号1超前，则移动信号2
            lag_abs = abs(lag)
            s1 = signal1[:-lag_abs] if lag_abs < len(signal1) else np.array([])
            s2 = signal2[lag_abs:] if lag_abs < len(signal2) else np.array([])
        
        # 确保有足够的点进行相关计算
        if len(s1) > 100 and len(s2) > 100 and len(s1) == len(s2):
            valid = ~np.isnan(s1) & ~np.isnan(s2)
            if np.sum(valid) > 100:
                corr = np.corrcoef(s1[valid], s2[valid])[0, 1]
                corr_values.append(corr)
            else:
                corr_values.append(float('-inf'))
        else:
            corr_values.append(float('-inf'))
    
    corr_values = np.array(corr_values)
    
    # 找到最佳粗略偏移，忽略无效值
    valid_indices = ~np.isinf(corr_values)
    if not np.any(valid_indices):
        print("  未找到有效的相关性值，无法确定偏移")
        return {
            'is_aligned': False,
            'method': 'correlation',
            'mean_offset': base_time_diff,  # 使用基础时间差作为备选
            'std_offset': 0
        }
        
    best_idx = np.nanargmax(corr_values[valid_indices])
    valid_indices_array = np.where(valid_indices)[0]
    if len(valid_indices_array) > 0:
        best_idx = valid_indices_array[best_idx]
        best_lag_ms = lags_ms[best_idx]
        best_corr = corr_values[best_idx]
    else:
        # 没有有效的相关性值
        best_lag_ms = 0
        best_corr = original_corr
    
    # 检查是否命中边界 - 如果是边界值，则扩大搜索范围重新搜索
    if abs(best_lag_ms) >= max_lag_ms * 0.99 and max_lag_ms < 5000:  # 避免无限扩大
        print(f"  警告: 达到搜索边界({best_lag_ms}ms)，扩大搜索范围重新尝试")
        expanded_max_lag = max_lag_ms * 2
        
        # 使用边界值为中心的扩展搜索
        expanded_start = max(-5000, best_lag_ms - expanded_max_lag)  # 避免过度扩展
        expanded_end = min(5000, best_lag_ms + expanded_max_lag)
        expanded_lags_ms = np.arange(expanded_start, expanded_end + coarse_step_ms, coarse_step_ms)
        expanded_lags_samples = (expanded_lags_ms / 1000 / dt).astype(int)
        
        expanded_corr_values = []
        for lag in expanded_lags_samples:
            if lag in lags_samples:  # 跳过已计算的值
                idx = np.where(lags_samples == lag)[0][0]
                expanded_corr_values.append(corr_values[idx])
                continue
                
            if lag == 0:
                expanded_corr_values.append(original_corr)
                continue
            
            # 与前面相同的计算
            if lag > 0:
                s1 = signal1[lag:] if lag < len(signal1) else np.array([])
                s2 = signal2[:-lag] if lag < len(signal2) else np.array([])
            else:
                lag_abs = abs(lag)
                s1 = signal1[:-lag_abs] if lag_abs < len(signal1) else np.array([])
                s2 = signal2[lag_abs:] if lag_abs < len(signal2) else np.array([])
            
            if len(s1) > 100 and len(s2) > 100 and len(s1) == len(s2):
                valid = ~np.isnan(s1) & ~np.isnan(s2)
                if np.sum(valid) > 100:
                    corr = np.corrcoef(s1[valid], s2[valid])[0, 1]
                    expanded_corr_values.append(corr)
                else:
                    expanded_corr_values.append(float('-inf'))
            else:
                expanded_corr_values.append(float('-inf'))
        
        expanded_corr_values = np.array(expanded_corr_values)
        expanded_valid_indices = ~np.isinf(expanded_corr_values)
        
        if np.any(expanded_valid_indices):
            best_expanded_idx = np.nanargmax(expanded_corr_values[expanded_valid_indices])
            valid_expanded_indices_array = np.where(expanded_valid_indices)[0]
            if len(valid_expanded_indices_array) > 0:
                best_expanded_idx = valid_expanded_indices_array[best_expanded_idx]
                if expanded_corr_values[best_expanded_idx] > best_corr:
                    best_lag_ms = expanded_lags_ms[best_expanded_idx]
                    best_corr = expanded_corr_values[best_expanded_idx]
                    print(f"  扩展搜索找到更优偏移: {best_lag_ms}ms, 相关性: {best_corr:.4f}")
    
    # 精细搜索，步长为1ms
    if best_idx > 0 and best_idx < len(lags_ms) - 1:
        fine_range = min(20, coarse_step_ms * 4)  # 增加精细搜索范围
        fine_start = max(-max_lag_ms, best_lag_ms - fine_range)
        fine_end = min(max_lag_ms, best_lag_ms + fine_range)
        fine_lags_ms = np.arange(fine_start, fine_end + 1, 1)
        fine_lags_samples = (fine_lags_ms / 1000 / dt).astype(int)
        
        fine_corr_values = []
        for lag in fine_lags_samples:
            if lag == 0:
                fine_corr_values.append(original_corr)
                continue
                
            # 与粗略搜索相同的修复
            if lag > 0:
                s1 = signal1[lag:] if lag < len(signal1) else np.array([])
                s2 = signal2[:-lag] if lag < len(signal2) else np.array([])
            else:
                lag_abs = abs(lag)
                s1 = signal1[:-lag_abs] if lag_abs < len(signal1) else np.array([])
                s2 = signal2[lag_abs:] if lag_abs < len(signal2) else np.array([])
            
            if len(s1) > 100 and len(s2) > 100 and len(s1) == len(s2):
                valid = ~np.isnan(s1) & ~np.isnan(s2)
                if np.sum(valid) > 100:
                    corr = np.corrcoef(s1[valid], s2[valid])[0, 1]
                    fine_corr_values.append(corr)
                else:
                    fine_corr_values.append(float('-inf'))
            else:
                fine_corr_values.append(float('-inf'))
        
        fine_corr_values = np.array(fine_corr_values)
        valid_fine_indices = ~np.isinf(fine_corr_values)
        
        if np.any(valid_fine_indices):
            best_fine_idx = np.nanargmax(fine_corr_values[valid_fine_indices])
            valid_fine_indices_array = np.where(valid_fine_indices)[0]
            if len(valid_fine_indices_array) > 0:
                best_fine_idx = valid_fine_indices_array[best_fine_idx]
                if fine_corr_values[best_fine_idx] > best_corr:
                    best_lag_ms = fine_lags_ms[best_fine_idx]
                    best_corr = fine_corr_values[best_fine_idx]
    
    # 检查提升是否显著 - 如果相关性提升很小，可能已经对齐
    if best_corr - original_corr < 0.01 and original_corr > 0.9:
        print(f"  相关性提升很小({best_corr-original_corr:.4f})，最佳偏移可能是0")
        if abs(best_lag_ms) > 100:  # 如果偏移很大但提升很小，可能是假相关
            best_lag_ms = 0
            best_corr = original_corr
            print("  重置偏移为0，因为大偏移但相关性提升很小")
    
    # 修正最佳偏移计算 - 消除固定偏差
    total_offset = base_time_diff + best_lag_ms
    
    print(f"  相关性方法最佳偏移: 基础时间差={base_time_diff:.2f}ms + 精细偏移={best_lag_ms:.2f}ms = 总偏移={total_offset:.2f}ms")
    print(f"  最佳相关性: {best_corr:.4f}, 原始相关性: {original_corr:.4f}, 提升: {best_corr-original_corr:.4f}")
    
    # 判断是否对齐
    is_aligned = abs(total_offset) <= args.tolerance_ms
    print(f"  对齐状态: {'对齐' if is_aligned else '未对齐'} (容差±{args.tolerance_ms}ms)")
    
    # 返回结果
    return {
        'original_corr': original_corr,
        'best_lag': best_lag_ms,
        'base_time_diff': base_time_diff,
        'best_corr': best_corr,
        'corr_curve': {
            'lags': lags_ms.tolist(),
            'corrs': corr_values.tolist()
        },
        'mean_offset': total_offset,
        'std_offset': 0,  # 单次计算没有标准差
        'is_aligned': is_aligned,
        'method': 'correlation'
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
    
    # 计算基础时间差（重要！）
    time_diff_base = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  基础时间差: {time_diff_base:.2f}ms (正值表示信号1滞后于信号2)")
    
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
            'mean_offset': time_diff_base,  # 使用基础时间差作为备选
            'std_offset': 0.0,
            'base_time_diff': time_diff_base,
            'detail_offset': 0.0
        }
    
    # 计算搜索窗口和策略
    # 基础搜索窗口更大，确保能找到真实匹配
    adjusted_base_diff = abs(time_diff_base)
    search_window_base = max(args.tolerance_ms * 15, 1500)  # 至少1500ms的搜索窗口
    search_window_ms = search_window_base * (1 + min(adjusted_base_diff / 3000, 1.0))  # 根据基础差异动态调整
    
    # 考虑采样率因素 - 采样率低需要更大窗口
    rate_factor = max(1.0, 50 / min(sampling_rate1, sampling_rate2))
    search_window_ms *= min(rate_factor, 3.0)  # 最多增加3倍
    
    # 最终窗口有上限
    search_window_ms = min(search_window_ms, 3500)  # 最大不超过3.5秒
    
    print(f"  事件匹配窗口: ±{search_window_ms:.1f}ms")
    
    # 多阶段匹配策略
    # 1. 使用基础时间差进行初步对齐
    adjusted_timestamps2 = np.array(events2['timestamps']) + time_diff_base
    
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
            
            # 计算总偏移
            total_offset = time_diff_base + mean_diff
            
            # 多指标对齐判断
            low_std = std_diff <= args.tolerance_ms * 2.5
            good_match_rate = match_rate >= 0.3 or (match_rate >= 0.2 and len(matches) >= 5)
            high_confidence = np.mean(inlier_scores) >= 0.6
            
            is_aligned = abs(total_offset) <= args.tolerance_ms and low_std and good_match_rate
            
            # 输出详细分析结果
            print(f"  事件匹配分析: 平均得分={np.mean(inlier_scores):.2f}, 标准差={std_diff:.2f}ms, 匹配率={match_rate*100:.1f}%")
            print(f"  事件方法结果: 总偏移={total_offset:.2f}ms (基础:{time_diff_base:.2f}ms + 细节:{mean_diff:.2f}ms)")
            print(f"  对齐判断: {'对齐' if is_aligned else '未对齐'} (标准差:{low_std}, 匹配率:{good_match_rate}, 置信度:{high_confidence})")
            
        else:
            # 匹配点不足
            mean_diff = median_diff
            std_diff = mad * 1.4826  # 转换MAD为等效标准差
            total_offset = time_diff_base + mean_diff
            is_aligned = False
            print("  有效匹配数不足，结果可能不可靠")
    else:
        # 几乎没有匹配
        mean_diff = 0
        std_diff = 0
        total_offset = time_diff_base
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
        'mean_offset': total_offset,
        'std_offset': std_diff,
        'base_time_diff': time_diff_base,
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
    
    # 计算基础时间差 (ms)
    time_diff_base = (min_ts1 - min_ts2)
    time_range_1 = max_ts1 - min_ts1
    time_range_2 = max_ts2 - min_ts2
    
    # 确定有效重叠处理时间范围
    max_process_duration = 60 * 1000  # 60秒，单位毫秒
    process_duration = min(time_range_1, time_range_2, max_process_duration)
    
    # 调试输出
    print(f"  数据集1时间范围: {min_ts1:.1f}ms - {max_ts1:.1f}ms (持续{time_range_1/1000:.1f}秒)")
    print(f"  数据集2时间范围: {min_ts2:.1f}ms - {max_ts2:.1f}ms (持续{time_range_2/1000:.1f}秒)")
    print(f"  基础时间差: {time_diff_base:.1f}ms")
    print(f"  处理持续时间: {process_duration/1000:.1f}秒")
    
    # 确保数据点足够进行可靠对齐
    if len(data1) < 100 or len(data2) < 100:
        print("  数据点不足，无法可靠进行视觉对齐")
        return {
            'is_aligned': False,
            'confidence': 0.0,
            'mean_offset': time_diff_base,
            'method': 'visual'
        }
    
    # 对于合成测试数据，我们需要使用简单直接的相关性分析
    # 检查是否都有angular_velocity_magnitude列
    if 'angular_velocity_magnitude' in data1.columns and 'angular_velocity_magnitude' in data2.columns:
        # 提取角速度幅值列并执行简单相关性分析
        sig1 = data1['angular_velocity_magnitude'].values
        sig2 = data2['angular_velocity_magnitude'].values
        
        # 设置最大搜索范围为基础时间差的±40%，但至少±200ms
        max_search_ms = max(200, abs(time_diff_base) * 0.4)
        
        # 三级搜索策略
        search_levels = [
            {'range': max_search_ms, 'step': max_search_ms / 6},   # 粗搜索
            {'range': max_search_ms / 3, 'step': max_search_ms / 30},  # 中等精度
            {'range': max_search_ms / 10, 'step': 4.0}   # 精细搜索
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
        is_aligned = confidence > 0.6  # 置信度阈值
        
        # 输出结果
        print(f"  视觉方法结果: 总偏移={final_offset:.2f}ms (基础:{time_diff_base:.2f}ms + 细节:{best_offset:.2f}ms)")
        print(f"  相似度: {best_corr:.4f}, 置信度: {confidence:.2f}")
        print(f"  对齐状态: {'已对齐' if is_aligned else '未对齐'}")
        
        # 返回结果
        return {
            'is_aligned': is_aligned,
            'confidence': confidence,
            'mean_offset': final_offset,
            'std_offset': abs(best_offset) / 2,  # 简单估计标准差
            'base_offset': time_diff_base,
            'fine_offset': best_offset,
            'similarity': best_corr,
            'method': 'visual'
        }
    
    # 如果没有角速度幅值列，尝试选择重叠部分数据并创建角速度分量
    end_ts1 = min_ts1 + process_duration
    end_ts2 = min_ts2 + process_duration
    
    data1_overlap = data1[(data1[ts_col1] >= min_ts1) & (data1[ts_col1] <= end_ts1)].copy()
    data2_overlap = data2[(data2[ts_col2] >= min_ts2) & (data2[ts_col2] <= end_ts2)].copy()
    
    # 检查是否有角速度列，如果没有则尝试创建
    required_columns = ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
    
    # 检查是否有任何角速度列
    has_required_columns_1 = all(col in data1_overlap.columns for col in required_columns)
    has_required_columns_2 = all(col in data2_overlap.columns for col in required_columns)
    
    # 如果没有角速度分量，但有角速度大小，我们可以创建伪分量
    if not has_required_columns_1 and 'angular_velocity_magnitude' in data1_overlap.columns:
        # 创建伪角速度分量 (用角速度大小的某个比例表示)
        data1_overlap['angular_velocity_x'] = data1_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        data1_overlap['angular_velocity_y'] = data1_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        data1_overlap['angular_velocity_z'] = data1_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        # 添加一些随机性以便更好地进行特征提取
        data1_overlap['angular_velocity_x'] *= (0.9 + 0.2 * np.random.random(len(data1_overlap)))
        data1_overlap['angular_velocity_y'] *= (0.9 + 0.2 * np.random.random(len(data1_overlap)))
        data1_overlap['angular_velocity_z'] *= (0.9 + 0.2 * np.random.random(len(data1_overlap)))
    
    if not has_required_columns_2 and 'angular_velocity_magnitude' in data2_overlap.columns:
        # 创建伪角速度分量
        data2_overlap['angular_velocity_x'] = data2_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        data2_overlap['angular_velocity_y'] = data2_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        data2_overlap['angular_velocity_z'] = data2_overlap['angular_velocity_magnitude'] * 0.57735  # 1/sqrt(3)
        # 添加一些随机性
        data2_overlap['angular_velocity_x'] *= (0.9 + 0.2 * np.random.random(len(data2_overlap)))
        data2_overlap['angular_velocity_y'] *= (0.9 + 0.2 * np.random.random(len(data2_overlap)))
        data2_overlap['angular_velocity_z'] *= (0.9 + 0.2 * np.random.random(len(data2_overlap)))
    
    # 计算平均采样间隔
    avg_interval1 = 1000.0 / rate1  # ms
    avg_interval2 = 1000.0 / rate2  # ms
    
    # 确定目标采样率 (使用较高的采样率)
    target_interval = min(avg_interval1, avg_interval2) * 1.05  # 稍微降低一点采样率，确保有足够的点
    print(f"  目标采样间隔: {target_interval:.2f}ms")
    
    # 对两个信号进行均匀重采样
    resampled_data1 = {}
    resampled_data2 = {}
    
    for col in required_columns:
        if col in data1_overlap.columns and col in data2_overlap.columns:
            resampled_data1[col] = resample_signal(
                data1_overlap, ts_col1, col, 
                target_len=int(process_duration / target_interval)
            )
            resampled_data2[col] = resample_signal(
                data2_overlap, ts_col2, col, 
                target_len=int(process_duration / target_interval)
            )
    
    # 计算角速度幅值 (如果尚未有)
    if 'angular_velocity_magnitude' not in data1_overlap.columns and all(col in resampled_data1 for col in required_columns):
        # 创建时间序列
        time_points = np.linspace(min_ts1, min_ts1 + process_duration, len(resampled_data1[required_columns[0]]))
        
        # 组合角速度分量计算幅值
        magnitude = np.sqrt(
            np.square(resampled_data1['angular_velocity_x']) + 
            np.square(resampled_data1['angular_velocity_y']) + 
            np.square(resampled_data1['angular_velocity_z'])
        )
        
        # 将幅值添加到重采样数据
        resampled_data1['angular_velocity_magnitude'] = magnitude
    
    if 'angular_velocity_magnitude' not in data2_overlap.columns and all(col in resampled_data2 for col in required_columns):
        # 对数据2做同样的处理
        time_points = np.linspace(min_ts2, min_ts2 + process_duration, len(resampled_data2[required_columns[0]]))
        
        magnitude = np.sqrt(
            np.square(resampled_data2['angular_velocity_x']) + 
            np.square(resampled_data2['angular_velocity_y']) + 
            np.square(resampled_data2['angular_velocity_z'])
        )
        
        resampled_data2['angular_velocity_magnitude'] = magnitude
    
    # 多尺度特征提取
    # 使用不同的窗口大小提取特征
    features = []
    
    # 定义多个窗口大小 (单位: ms)
    window_sizes = [2000, 3500, 5000]  # 2秒, 3.5秒, 5秒窗口
    
    for window_size in window_sizes:
        window_points = int(window_size / target_interval)
        
        if window_points > 10:
            for col in required_columns + ['angular_velocity_magnitude']:
                if col in resampled_data1 and col in resampled_data2:
                    signal1 = resampled_data1[col]
                    signal2 = resampled_data2[col]
                    
                    # 计算此窗口大小下的特征
                    stats1 = extract_statistical_features(signal1)
                    stats2 = extract_statistical_features(signal2)
                    
                    try:
                        freq1 = extract_frequency_features(signal1, fs=1000/target_interval)
                    except:
                        freq1 = {'dominant_freq': 0, 'energy_low': 0, 'energy_mid': 0, 'energy_high': 0}
                        
                    try:
                        freq2 = extract_frequency_features(signal2, fs=1000/target_interval)
                    except:
                        freq2 = {'dominant_freq': 0, 'energy_low': 0, 'energy_mid': 0, 'energy_high': 0}
                    
                    morph1 = extract_morphological_features(signal1)
                    morph2 = extract_morphological_features(signal2)
                    
                    # 记录特征
                    features.append({
                        'col': col,
                        'window_size': window_size,
                        'signal1': signal1,
                        'signal2': signal2,
                        'stats1': stats1,
                        'stats2': stats2,
                        'freq1': freq1,
                        'freq2': freq2,
                        'morph1': morph1,
                        'morph2': morph2
                    })
    
    if not features:
        print("  无法提取足够特征进行视觉对齐")
        return {
            'is_aligned': False,
            'confidence': 0.0,
            'mean_offset': time_diff_base,
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
    
    # 基于基础时间差应用合理的搜索范围
    # 当基础差值很大时，我们相对应地搜索一个较小的范围
    max_range_ms = min(300, abs(time_diff_base) * 0.4)  # 最大搜索范围是300ms或基础时间差的40%，取较小值
    
    for stage in range(stages):
        # 定义搜索范围和步长
        if stage == 0:  # 第一阶段：粗略搜索
            range_ms = max_range_ms  # 最大300ms，或基础时间差的一定比例
            step_ms = range_ms / 6   # 较大步长，分6个点
            
            # 中心化搜索范围
            search_range = (-range_ms, range_ms)
        
        elif stage == 1:  # 第二阶段：中等精度搜索
            # 围绕第一阶段结果扩大搜索
            range_ms = max_range_ms / 3  # 缩小搜索范围
            step_ms = range_ms / 15  # 中等步长
            
            # 更新搜索范围，围绕最佳偏移
            search_range = (best_offset - range_ms, best_offset + range_ms)
        
        else:  # 第三阶段：精细搜索
            range_ms = max_range_ms / 10  # 进一步缩小范围
            step_ms = 4.0  # 小步长，4ms
            
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
    final_offset = time_diff_base + best_offset
    
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
                confidence = min(1.0, max(0.0, peak_sharpness * 1000))  # 调整缩放因子
                
                # 考虑整体相似度
                confidence *= (best_similarity + 0.2)  # 增加基础置信度
            except:
                # 如果拟合失败，使用最佳相似度作为基础置信度
                confidence = best_similarity + 0.1
        else:
            confidence = best_similarity + 0.1
    else:
        confidence = best_similarity  # 样本点少时，使用相似度作为置信度
    
    # 确保置信度在合理范围内
    confidence = min(1.0, max(0.0, confidence))
    
    # 考虑基础时间差和细节偏移的相对大小
    # 如果细节偏移太大，可能不太可靠
    if abs(time_diff_base) > 0:
        relative_change = abs(best_offset) / abs(time_diff_base)
        if relative_change > 0.5:  # 如果调整超过基础时间差的50%
            confidence *= max(0.5, 1.0 - (relative_change - 0.5))
    
    # 判断信号是否对齐，基于置信度和容差
    is_aligned = confidence > 0.6
    
    # 调试输出
    print(f"  视觉方法结果: 总偏移={final_offset:.2f}ms (基础:{time_diff_base:.2f}ms + 细节:{best_offset:.2f}ms)")
    print(f"  相似度: {best_similarity:.4f}, 置信度: {confidence:.2f}")
    print(f"  对齐状态: {'已对齐' if is_aligned else '未对齐'}")
    
    # 返回结果
    return {
        'is_aligned': is_aligned,
        'confidence': confidence,
        'mean_offset': final_offset,
        'std_offset': abs(best_offset) / 2,  # 估计标准差
        'base_offset': time_diff_base,
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
    
    # 获取结果数据
    total_offset = verification_results.get('mean_offset', 0)
    best_lag = verification_results.get('best_lag', 0)
    base_time_diff = verification_results.get('base_time_diff', 0)
    corr_curve = verification_results.get('corr_curve', {'lags': [], 'corrs': []})
    best_corr = verification_results.get('best_corr', 0)
    original_corr = verification_results.get('original_corr', 0)
    is_aligned = verification_results.get('is_aligned', False)
    
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
            plt.annotate(f'最佳细节偏移: {best_lag}ms', 
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
        
        plt.title(f"相关性分析 (总偏移: {total_offset:.2f}ms = 基础差异: {base_time_diff:.2f}ms + 精细偏移: {best_lag:.2f}ms)")
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
    基础时间差: {base_time_diff:.2f} ms
    最佳精细偏移: {best_lag:.2f} ms
    总偏移: {total_offset:.2f} ms
    
    最佳相关系数: {best_corr:.4f}
    原始相关系数: {original_corr:.4f}
    改进: {best_corr-original_corr:.4f}
    
    对齐状态: {'✓ 已对齐' if is_aligned else '✗ 未对齐'} (容差: ±{args.tolerance_ms} ms)
    """
    plt.text(0.1, 0.9, summary_text, fontsize=12, family='monospace', va='top')
    
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
    plt.suptitle(f"{sensor1}-{sensor2} 事件方法对齐分析 (总偏移: {mean_offset:.2f}ms)", fontsize=16)
    
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
    else:
        plt.text(0.5, 0.5, "无匹配事件", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(f'时间差分布 (基础时间差: {base_time_diff:.2f}ms, 细节偏移: {detail_offset:.2f}ms)')
    plt.xlabel('时间差 (ms)')
    plt.ylabel('事件数量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加结果摘要
    alignment_status = "已对齐" if is_aligned else "未对齐"
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
    detail_offset = verification_results.get('detail_offset', 0)
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
    基础时间差:     {base_time_diff:.2f} ms
    细节偏移:       {detail_offset:.2f} ms
    总时间偏移:     {mean_offset:.2f} ms
    
    置信度:         {confidence:.2f}
    对齐状态:       {'✓ 已对齐' if is_aligned else '✗ 未对齐'} (容差: ±{args.tolerance_ms} ms)
    
    方法:           多尺度特征匹配
    """
    
    plt.text(0.5, 0.5, summary_text, fontsize=14, family='monospace',
            ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.2))
    
    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为主标题留出空间
    plt.savefig(os.path.join(args.output_dir, f"{sensor1}_{sensor2}_visual_verification.png"), dpi=300)
    plt.close()

def main():
    args = parse_arguments()
    
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
                result = correlation_verification(
                    data1, data2, ts_col1, ts_col2, args
                )
                pair_results['correlation'] = result
                visualize_correlation_results(result, sensor1, sensor2, args)
                
            elif method == 'event':
                # 事件同步验证
                result = event_verification(
                    data1, data2, ts_col1, ts_col2, args
                )
                pair_results['event'] = result
                visualize_event_results(result, sensor1, sensor2, args)
                
            elif method == 'visual':
                # 可视化比较
                result = visual_verification(
                    data1, data2, ts_col1, ts_col2, sensor1, sensor2, args
                )
                pair_results['visual'] = result
                visualize_visual_results(result, sensor1, sensor2, args)
        
        # 存储组合结果
        all_verification_results[pair_key] = pair_results
    
    # 生成综合报告
    report_file = generate_summary_report(all_verification_results, args)
    
    # 显示简要结果
    print("\n===== 验证结果摘要 =====")
    for sensor_pair, pair_results in all_verification_results.items():
        sensor1, sensor2 = sensor_pair.split('_')
        
        aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
        total_methods = len(pair_results)
        final_aligned = aligned_count >= total_methods / 2
        
        print(f"{sensor1}-{sensor2}: {'对齐' if final_aligned else '未对齐'} ({aligned_count}/{total_methods})")
    
    print(f"\n详细报告已保存至: {report_file}")

if __name__ == "__main__":
    main()