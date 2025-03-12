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
    """检测信号中的重要事件（峰值）
    
    参数:
        data: 包含时间戳和信号值的DataFrame
        ts_col: 时间戳列名
        value_col: 信号值列名
        threshold_percentile: 信号值阈值百分位数
        window_size: 检测窗口大小（点数）
        min_distance: 事件之间的最小距离（点数）
        dynamic_threshold: 是否使用动态阈值（基于局部数据)
        
    返回:
        包含'timestamps'和'values'的字典，表示事件发生的时间戳和对应的信号值
    """
    if len(data) < window_size * 2:
        return {'timestamps': [], 'values': [], 'indices': []}
    
    # 确保按时间排序
    sorted_data = data.sort_values(by=ts_col).copy()
    
    # 预处理信号 - 应用平滑以减少噪声影响
    # 检查数据点是否足够应用滤波器
    if len(sorted_data) > window_size * 3:
        # 适应窗口大小，确保为奇数
        smooth_window = min(len(sorted_data) // 4, 21)
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        try:
            # 应用Savitzky-Golay滤波器平滑信号
            sorted_data[f'{value_col}_smooth'] = savgol_filter(
                sorted_data[value_col].values, 
                window_length=smooth_window, 
                polyorder=2
            )
            
            # 使用平滑后的值来检测事件
            detection_col = f'{value_col}_smooth'
        except:
            # 如果滤波失败，使用原始信号
            detection_col = value_col
    else:
        detection_col = value_col
    
    # 计算全局阈值 - 使用鲁棒性更强的方法
    if value_col in sorted_data.columns:
        # 获取信号值并排除可能的异常值
        values_array = sorted_data[detection_col].values
        values_array = values_array[~np.isnan(values_array)]
        
        if len(values_array) == 0:
            return {'timestamps': [], 'values': [], 'indices': []}
        
        # 使用分位数范围来确定更稳健的阈值
        # 使用第75百分位加上四分位距的倍数，而不是单一百分位数
        if len(values_array) > 10:
            q75 = np.percentile(values_array, 75)
            q25 = np.percentile(values_array, 25)
            iqr = q75 - q25  # 四分位距
            
            # 根据指定百分位调整阈值强度
            percentile_factor = threshold_percentile / 90.0  # 将用户指定的百分位映射到系数
            global_threshold = q75 + percentile_factor * 1.5 * iqr
        else:
            # 如果数据太少，回退到简单百分位
            global_threshold = np.percentile(values_array, threshold_percentile)
    else:
        return {'timestamps': [], 'values': [], 'indices': []}
    
    # 检测局部峰值
    peaks = []
    values = []
    indices = []
    
    # 使用跨尺度检测方法，捕获不同显著性的事件
    for i in range(window_size, len(sorted_data) - window_size, max(1, min_distance // 4)):
        # 选择当前窗口数据
        window = sorted_data.iloc[i-window_size:i+window_size+1]
        
        # 当前点的值
        current_value = window.iloc[window_size][detection_col]
        
        # 检查是否为局部峰值
        is_peak = (current_value == window[detection_col].max())
        
        # 增加判断: 也可以是超过相邻点一定比例的高值点
        if not is_peak:
            # 计算窗口左右相邻点的平均值
            neighbors_avg = (window[detection_col].iloc[window_size-1] + 
                            window[detection_col].iloc[window_size+1]) / 2
            
            # 如果当前值超过相邻点平均值的一定比例，也视为可能的事件点
            peak_ratio = 1.3  # 至少比相邻点高30%
            if current_value > neighbors_avg * peak_ratio:
                is_peak = True
        
        if not is_peak:
            continue
        
        # 使用多尺度动态阈值
        if dynamic_threshold:
            # 主窗口阈值
            local_threshold = max(
                np.percentile(window[detection_col], threshold_percentile),
                global_threshold * 0.7
            )
            
            # 考虑更大范围的窗口以捕获显著事件
            larger_window_size = min(window_size * 3, len(sorted_data) // 10)
            start_idx = max(0, i - larger_window_size)
            end_idx = min(len(sorted_data), i + larger_window_size + 1)
            larger_window = sorted_data.iloc[start_idx:end_idx]
            
            # 计算更大窗口的阈值
            if len(larger_window) > 5:
                larger_threshold = np.percentile(larger_window[detection_col], threshold_percentile)
                
                # 取两个窗口阈值的加权平均
                final_threshold = 0.3 * local_threshold + 0.7 * max(larger_threshold, global_threshold * 0.6)
            else:
                final_threshold = local_threshold
        else:
            final_threshold = global_threshold
        
        # 检查是否达到阈值
        if current_value < final_threshold:
            continue
        
        # 检查是否与前一个峰值距离足够远
        timestamp = window.iloc[window_size][ts_col]
        if peaks and abs(timestamp - peaks[-1]) < min_distance:
            # 如果两个峰值太近，保留值更高的一个
            if current_value > values[-1]:
                peaks[-1] = timestamp
                values[-1] = current_value
                indices[-1] = i
            continue
        
        peaks.append(timestamp)
        values.append(current_value)
        indices.append(i)
    
    # 如果检测到的事件太多，保留最显著的几个
    if len(peaks) > 50:
        # 按值排序并保留前50个最显著的事件
        sorted_indices = np.argsort(values)[::-1][:50]
        peaks = [peaks[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        indices = [indices[i] for i in sorted_indices]
    
    # 返回检测到的事件
    events = {
        'timestamps': peaks,
        'values': values,
        'indices': indices
    }
    
    return events

def event_verification(data1, data2, ts_col1, ts_col2, args):
    """使用事件同步验证时间戳对齐, 改进版本"""
    # 估计采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 计算基础时间差（重要！）
    time_diff_base = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  基础时间差: {time_diff_base:.2f}ms (正值表示信号1滞后于信号2)")
    
    # 根据采样率动态调整事件检测参数
    # 高采样率需要更大的窗口和更高的阈值
    # 采样率与检测参数的非线性关系
    threshold_pct1 = min(95, 80 + int(15 * np.log10(max(1, sampling_rate1) / 10)))
    threshold_pct2 = min(95, 80 + int(15 * np.log10(max(1, sampling_rate2) / 10)))
    
    # 窗口大小与最小距离应该与采样率成正比
    win_size1 = max(3, int(0.2 * sampling_rate1))  # 约200ms窗口
    win_size2 = max(3, int(0.2 * sampling_rate2))
    
    # 事件间最小距离，避免检测过于密集的事件
    min_dist1 = max(5, int(0.5 * sampling_rate1))  # 至少0.5秒间隔
    min_dist2 = max(5, int(0.5 * sampling_rate2))
    
    print(f"  事件检测参数 - 信号1: 阈值百分位={threshold_pct1}, 窗口大小={win_size1}, 最小距离={min_dist1}")
    print(f"  事件检测参数 - 信号2: 阈值百分位={threshold_pct2}, 窗口大小={win_size2}, 最小距离={min_dist2}")
    
    # 检测事件，使用改进的动态阈值方法
    events1 = detect_events(data1, ts_col1, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct1,
                          window_size=win_size1, min_distance=min_dist1,
                          dynamic_threshold=True)
    
    events2 = detect_events(data2, ts_col2, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct2,
                          window_size=win_size2, min_distance=min_dist2,
                          dynamic_threshold=True)
    
    print(f"  检测到事件 - 信号1: {len(events1['timestamps'])}个, 信号2: {len(events2['timestamps'])}个")
    
    # 如果事件数量不足，降低阈值重新检测，采用更激进的降低策略
    if len(events1['timestamps']) < 5 or len(events2['timestamps']) < 5:
        retry_attempts = 0
        max_attempts = 4  # 增加重试次数
        
        while (len(events1['timestamps']) < 5 or len(events2['timestamps']) < 5) and retry_attempts < max_attempts:
            retry_attempts += 1
            print(f"  事件数量不足，尝试第{retry_attempts}次降低阈值")
            
            if len(events1['timestamps']) < 5:
                threshold_pct1 = max(50, threshold_pct1 - 15)  # 更激进的降低
                # 同时减小最小距离，允许更密集的事件
                min_dist1 = max(3, int(min_dist1 * 0.6))
                events1 = detect_events(data1, ts_col1, value_col='angular_velocity_magnitude', 
                                      threshold_percentile=threshold_pct1,
                                      window_size=win_size1, min_distance=min_dist1,
                                      dynamic_threshold=True)
                
            if len(events2['timestamps']) < 5:
                threshold_pct2 = max(50, threshold_pct2 - 15)  # 更激进的降低
                min_dist2 = max(3, int(min_dist2 * 0.6))
                events2 = detect_events(data2, ts_col2, value_col='angular_velocity_magnitude', 
                                      threshold_percentile=threshold_pct2,
                                      window_size=win_size2, min_distance=min_dist2,
                                      dynamic_threshold=True)
            
            print(f"  降低阈值后 - 信号1: {len(events1['timestamps'])}个事件 (阈值{threshold_pct1})," 
                 f" 信号2: {len(events2['timestamps'])}个事件 (阈值{threshold_pct2})")
    
    # 如果仍然没有足够的事件
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
            'mean_offset': 0.0,
            'std_offset': 0.0,
            'base_time_diff': time_diff_base
        }
    
    # 计算偏移搜索范围, 基于基础时间差
    # 对于较大的基础时间差，扩大搜索窗口
    adjusted_base_diff = abs(time_diff_base)
    # 大大增加基础搜索窗口，确保能找到匹配
    search_window_base = max(args.tolerance_ms * 10, 1000)  # 基础搜索窗口至少1000ms
    search_window_ms = search_window_base * (1 + adjusted_base_diff / 2000)  # 随基础差异增加搜索范围
    
    # 考虑采样率因素
    search_window_ms *= (1 + 0.5 * (50 / min(sampling_rate1, sampling_rate2)))
    search_window_ms = min(search_window_ms, 3000)  # 最大不超过3秒
    
    print(f"  事件匹配窗口: ±{search_window_ms:.1f}ms")
    
    # 匹配事件 - 使用改进的匹配算法
    matches = []
    unmatched1 = []
    unmatched2 = []
    
    # 先应用已知的基础时间差，再寻找细节偏移
    # 调整信号2的时间戳以补偿基础时间差
    adjusted_timestamps2 = np.array(events2['timestamps']) + time_diff_base
    
    # 为每个信号1事件寻找最佳匹配的信号2事件
    for idx1, ts1 in enumerate(events1['timestamps']):
        best_match = None
        min_diff = search_window_ms
        
        for idx2, adjusted_ts2 in enumerate(adjusted_timestamps2):
            time_diff = abs(ts1 - adjusted_ts2)  # 绝对时间差
            
            if time_diff < min_diff:
                min_diff = time_diff
                best_match = (idx2, events2['timestamps'][idx2], adjusted_ts2)
        
        if best_match:
            idx2, orig_ts2, adjusted_ts2 = best_match
            matches.append({
                'event1_idx': idx1,
                'event2_idx': idx2,
                'event1_ts': ts1,
                'event2_ts': orig_ts2,
                'time_diff': ts1 - adjusted_ts2  # 保留符号，相对于调整后的时间戳
            })
        else:
            unmatched1.append(idx1)
    
    # 找出未匹配的事件2
    matched_event2_indices = {m['event2_idx'] for m in matches}
    unmatched2 = [idx for idx in range(len(events2['timestamps'])) 
                 if idx not in matched_event2_indices]
    
    # 计算匹配率
    match_rate = len(matches) / max(len(events1['timestamps']), len(events2['timestamps']))
    print(f"  匹配结果: {len(matches)}对匹配事件, {len(unmatched1)}个未匹配信号1事件, {len(unmatched2)}个未匹配信号2事件")
    
    # 计算匹配统计
    if len(matches) >= 3:
        # 提取时间差
        time_diffs = [m['time_diff'] for m in matches]
        
        # 使用中位数和MAD过滤异常值
        median_diff = np.median(time_diffs)
        mad = np.median(np.abs(np.array(time_diffs) - median_diff))
        
        # 定义异常值阈值，更宽松的过滤
        mad_threshold = max(3.0 * mad, 100.0)  # 至少100ms或3倍MAD
        
        # 过滤异常值
        valid_diffs = [diff for diff in time_diffs if abs(diff - median_diff) <= mad_threshold]
        
        # 确保至少有3个有效匹配
        if len(valid_diffs) >= 3:
            filtered_count = len(time_diffs) - len(valid_diffs)
            if filtered_count > 0:
                print(f"  异常值过滤: 移除{filtered_count}个异常点，保留{len(valid_diffs)}个点")
            
            # 计算最终偏移和标准差
            mean_diff = np.mean(valid_diffs)
            std_diff = np.std(valid_diffs)
            
            # 总偏移 = 基础时间差 + 细节时间差
            total_offset = time_diff_base + mean_diff
            
            # 只有当标准差较小且匹配率较高时才认为对齐
            low_std = std_diff <= args.tolerance_ms * 2
            good_match_rate = match_rate >= 0.3  # 至少30%匹配率
            
            is_aligned = abs(total_offset) <= args.tolerance_ms and low_std and good_match_rate
        else:
            mean_diff = median_diff  # 回退到中位数
            std_diff = mad * 1.4826  # MAD转为等效标准差
            total_offset = time_diff_base + mean_diff
            is_aligned = False
            print("  有效匹配数不足，结果可能不可靠")
    else:
        mean_diff = 0
        std_diff = 0
        total_offset = time_diff_base
        is_aligned = False
    
    print(f"  事件方法结果: 平均偏移={total_offset:.2f}ms, 标准差={std_diff:.2f}ms, 匹配率={match_rate*100:.1f}%, 对齐={is_aligned}")
    
    # 返回结果
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
        'detail_offset': mean_diff
    }

def visual_verification(data1, data2, ts_col1, ts_col2, args):
    """使用视觉叠加验证时间戳对齐"""
    # 估计采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 计算基础时间差
    base_time_diff = data1[ts_col1].min() - data2[ts_col2].min()
    print(f"  基础时间差: {base_time_diff:.2f}ms (正值表示信号1滞后于信号2)")
    
    # 初始化搜索范围 - 根据基础时间差调整
    # 对于大偏移，使用更大的搜索范围
    base_offset_magnitude = abs(base_time_diff)
    offset_range_ms = max(args.tolerance_ms * 3, 100) 
    if base_offset_magnitude > 50:
        # 随偏移增大而增大搜索范围，但限制最大值
        offset_range_ms = min(base_offset_magnitude * 1.5, 500)
    
    # 调整步长 - 采样率越高，步长越小
    offset_step = min(5, max(1, int(10 / max(sampling_rate1, sampling_rate2))))
    
    # 确保有足够的数据长度
    min_duration_req = 5  # 至少需要5秒的数据
    data1_duration = (data1[ts_col1].max() - data1[ts_col1].min()) / 1000.0
    data2_duration = (data2[ts_col2].max() - data2[ts_col2].min()) / 1000.0
    
    if data1_duration < min_duration_req or data2_duration < min_duration_req:
        print(f"  警告: 数据长度不足 (信号1: {data1_duration:.1f}秒, 信号2: {data2_duration:.1f}秒), 可能影响结果可靠性")
    
    # 准备两个信号，确保按时间排序
    data1_sorted = data1.sort_values(by=ts_col1)
    data2_sorted = data2.sort_values(by=ts_col2)
    
    signal1_times = data1_sorted[ts_col1].values
    signal2_times = data2_sorted[ts_col2].values
    
    # 使用角速度幅度作为信号
    signal1 = data1_sorted['angular_velocity_magnitude'].values
    signal2 = data2_sorted['angular_velocity_magnitude'].values
    
    # 修正信号强度，确保两个信号的数量级相近
    signal1_med = np.nanmedian(signal1)
    signal2_med = np.nanmedian(signal2)
    
    # 如果两个信号中值差异超过1.5倍，进行归一化缩放
    if signal1_med > 0 and signal2_med > 0 and (signal1_med/signal2_med > 1.5 or signal2_med/signal1_med > 1.5):
        ratio = signal1_med / max(signal2_med, 0.001)
        if ratio > 1:
            signal2 = signal2 * ratio
            print(f"  缩放信号2 (比例 {ratio:.2f}) 使其与信号1匹配")
        else:
            signal1 = signal1 / ratio
            print(f"  缩放信号1 (比例 {1/ratio:.2f}) 使其与信号2匹配")
    
    # 如果数据量太大，对信号进行下采样，确保下面的搜索效率
    max_points = 10000
    if len(signal1) > max_points or len(signal2) > max_points:
        downsample_factor1 = max(1, int(len(signal1) / max_points))
        downsample_factor2 = max(1, int(len(signal2) / max_points))
        
        signal1_times = signal1_times[::downsample_factor1]
        signal1 = signal1[::downsample_factor1]
        signal2_times = signal2_times[::downsample_factor2]
        signal2 = signal2[::downsample_factor2]
        
        print(f"  下采样: 信号1 1/{downsample_factor1}, 信号2 1/{downsample_factor2}")
    
    # 将时间戳转换为相对于最小时间戳的秒数，便于插值
    min_time = min(signal1_times.min(), signal2_times.min())
    signal1_times_rel = (signal1_times - min_time) / 1000.0
    signal2_times_rel = (signal2_times - min_time) / 1000.0
    
    # 找出共同时间范围
    common_start = max(signal1_times_rel.min(), signal2_times_rel.min())
    common_end = min(signal1_times_rel.max(), signal2_times_rel.max())
    common_range = common_end - common_start
    
    if common_range <= 0.5:  # 至少需要0.5秒重叠
        print("  信号重叠时间过短，无法进行视觉比对")
        return {
            'is_aligned': False,
            'method': 'visual',
            'mean_offset': 0.0,
            'std_offset': 0.0,
            'offset_scores': {},
            'best_offset': 0.0
        }
    
    print(f"  共同时间范围: {common_range:.2f}秒")
    
    # 计算密集的共同时间点，用于插值比较
    # 使用更高的采样率以捕获快速变化
    target_rate = max(sampling_rate1, sampling_rate2) * 1.5
    common_times = np.linspace(common_start, common_end, 
                               int(common_range * target_rate))
    
    # 为两个信号创建插值函数 - 使用三次样条插值提高精度
    from scipy import interpolate
    
    f1 = interpolate.interp1d(signal1_times_rel, signal1, kind='cubic', 
                             bounds_error=False, fill_value='extrapolate')
    f2 = interpolate.interp1d(signal2_times_rel, signal2, kind='cubic', 
                             bounds_error=False, fill_value='extrapolate')
    
    # 在共同时间点上评估插值信号
    y1_interp = f1(common_times)
    
    # 进行多级搜索 - 先粗略再精细
    # 1. 进行粗略搜索，步长较大
    coarse_offset_range = offset_range_ms * 1.2  # 稍微扩大范围
    coarse_step = offset_step * 2
    
    coarse_offsets_ms = list(range(-int(coarse_offset_range), int(coarse_offset_range) + 1, coarse_step))
    coarse_scores = {}
    best_coarse_score = float('-inf')
    best_coarse_offset = 0.0
    
    for offset_ms in coarse_offsets_ms:
        # 应用偏移比例校正 - 对大偏移值进行非线性修正
        if abs(offset_ms) > 100:
            # 根据偏移大小应用非线性校正
            correction_factor = 1.0 - min(0.2, abs(offset_ms) / 2000)  # 最多减少20%
            adjusted_offset = offset_ms * correction_factor
        else:
            adjusted_offset = offset_ms
            
        # 计算总偏移（基础时间差 + 细节偏移）
        total_offset_ms = base_time_diff + adjusted_offset
        
        # 根据总偏移调整时间
        offset_sec = total_offset_ms / 1000.0
        common_times_offset = common_times - offset_sec  # 应用偏移
        
        # 获取偏移后的插值值
        y2_interp = f2(common_times_offset)
        
        # 计算有效的数据点
        valid_mask = ~np.isnan(y1_interp) & ~np.isnan(y2_interp)
        
        if np.sum(valid_mask) < 100:  # 需要足够多的有效点
            coarse_scores[offset_ms] = -1
            continue
            
        # 计算标准化后的信号
        y1_norm = (y1_interp[valid_mask] - np.mean(y1_interp[valid_mask])) / np.std(y1_interp[valid_mask])
        y2_norm = (y2_interp[valid_mask] - np.mean(y2_interp[valid_mask])) / np.std(y2_interp[valid_mask])
        
        # 使用多种指标评估相似性
        # 1. 皮尔逊相关系数
        corr = np.corrcoef(y1_norm, y2_norm)[0, 1]
        
        # 2. 欧几里得距离（标准化）
        mse = np.mean((y1_norm - y2_norm) ** 2)
        
        # 3. 信号能量比，接近1表示能量相似
        energy_ratio = min(
            np.sum(y1_norm**2) / max(np.sum(y2_norm**2), 1e-10),
            np.sum(y2_norm**2) / max(np.sum(y1_norm**2), 1e-10)
        )
        
        # 综合评分 - 加权组合
        combined_score = 0.6 * (corr + 1) / 2 + 0.3 * (1 - min(0.5, mse / 10)) + 0.1 * energy_ratio
        
        coarse_scores[offset_ms] = combined_score
        
        if combined_score > best_coarse_score:
            best_coarse_score = combined_score
            best_coarse_offset = offset_ms
    
    # 2. 在最佳粗略偏移附近进行精细搜索
    # 精细搜索范围与步长
    fine_range = min(coarse_step * 4, 20)  # 最多20ms的精细搜索范围
    fine_step = 1  # 1ms步长，高精度
    
    fine_offsets_ms = list(range(int(best_coarse_offset - fine_range), 
                               int(best_coarse_offset + fine_range + 1), 
                               fine_step))
    fine_scores = {}
    best_fine_score = best_coarse_score
    best_fine_offset = best_coarse_offset
    
    for offset_ms in fine_offsets_ms:
        # 跳过已经计算过的偏移
        if offset_ms in coarse_scores:
            fine_scores[offset_ms] = coarse_scores[offset_ms]
            continue
        
        # 偏移大小的非线性校正，与粗略搜索相同
        if abs(offset_ms) > 100:
            correction_factor = 1.0 - min(0.2, abs(offset_ms) / 2000)
            adjusted_offset = offset_ms * correction_factor
        else:
            adjusted_offset = offset_ms
            
        # 计算总偏移
        total_offset_ms = base_time_diff + adjusted_offset
        offset_sec = total_offset_ms / 1000.0
        common_times_offset = common_times - offset_sec
        
        # 获取偏移后的信号
        y2_interp = f2(common_times_offset)
        valid_mask = ~np.isnan(y1_interp) & ~np.isnan(y2_interp)
        
        if np.sum(valid_mask) < 100:
            fine_scores[offset_ms] = -1
            continue
            
        # 归一化信号
        y1_norm = (y1_interp[valid_mask] - np.mean(y1_interp[valid_mask])) / np.std(y1_interp[valid_mask])
        y2_norm = (y2_interp[valid_mask] - np.mean(y2_interp[valid_mask])) / np.std(y2_interp[valid_mask])
        
        # 计算指标
        corr = np.corrcoef(y1_norm, y2_norm)[0, 1]
        mse = np.mean((y1_norm - y2_norm) ** 2)
        energy_ratio = min(
            np.sum(y1_norm**2) / max(np.sum(y2_norm**2), 1e-10),
            np.sum(y2_norm**2) / max(np.sum(y1_norm**2), 1e-10)
        )
        
        # 综合评分
        combined_score = 0.6 * (corr + 1) / 2 + 0.3 * (1 - min(0.5, mse / 10)) + 0.1 * energy_ratio
        
        fine_scores[offset_ms] = combined_score
        
        if combined_score > best_fine_score:
            best_fine_score = combined_score
            best_fine_offset = offset_ms
    
    # 合并所有分数
    all_scores = {**coarse_scores, **fine_scores}
    
    # 最终偏移和非线性校正
    final_offset = best_fine_offset
    
    # 应用最终的非线性校正
    if abs(final_offset) > 100:
        correction_factor = 1.0 - min(0.2, abs(final_offset) / 2000)
        adjusted_final_offset = final_offset * correction_factor
    else:
        adjusted_final_offset = final_offset
    
    # 计算总偏移
    total_offset_ms = base_time_diff + adjusted_final_offset
    
    # 打印主要偏移结果
    print(f"  视觉方法结果: 基础时间差={base_time_diff:.2f}ms, 细节偏移={adjusted_final_offset:.2f}ms")
    print(f"  总偏移={total_offset_ms:.2f}ms, 最佳匹配分数={best_fine_score:.4f}")
    
    # 判断是否对齐
    is_aligned = abs(total_offset_ms) <= args.tolerance_ms
    
    # 使用最佳偏移生成对齐后的数据，用于可视化
    best_offset_sec = total_offset_ms / 1000.0
    common_times_best = common_times - best_offset_sec
    y2_best = f2(common_times_best)
    
    # 为可视化准备结果
    visual_result = {
        'common_times': common_times,
        'y1': y1_interp,
        'y2_aligned': y2_best,
        'score': best_fine_score,
        'y2_original': f2(common_times)  # 原始信号2（未偏移）
    }
    
    return {
        'is_aligned': is_aligned,
        'method': 'visual',
        'mean_offset': total_offset_ms,
        'std_offset': 0.0,  # 视觉方法没有标准差
        'offset_scores': all_scores,
        'best_offset': final_offset,
        'adjusted_offset': adjusted_final_offset,
        'total_offset': total_offset_ms,
        'base_time_diff': base_time_diff,
        'visual_result': visual_result
    }

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
    
    # 创建一个大的图像以显示结果
    plt.figure(figsize=(14, 10))
    
    # 1. 事件匹配图
    plt.subplot(2, 1, 1)
    
    # 绘制事件时间点
    plt.scatter(events1['timestamps'], np.ones_like(events1['timestamps']), marker='|', s=100, 
               label=f'{sensor1} Events', color='blue')
    plt.scatter(events2['timestamps'], np.ones_like(events2['timestamps'])*1.1, 
               marker='|', s=100, label=f'{sensor2} Events', color='red')
    
    # 绘制匹配连线
    matched_events1 = [events1['timestamps'][m['event1_idx']] for m in match_stats['matches']]
    matched_events2 = [events2['timestamps'][m['event2_idx']] for m in match_stats['matches']]
    
    for i in range(len(matched_events1)):
        plt.plot([matched_events1[i], matched_events2[i]], [1, 1.1], 'k-', alpha=0.3)
    
    plt.title(f'{sensor1}-{sensor2} Event Matching (Match Rate: {match_stats["match_rate"]*100:.1f}%)')
    plt.xlabel('Timestamp')
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    
    # 2. 时间差分布
    plt.subplot(2, 1, 2)
    time_diffs = [m['time_diff'] for m in match_stats['matches']]
    
    # 解决ValueError: Too many bins for data range问题
    # 动态计算合适的bin数量
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
        plt.axvline(x=0, color='k', linestyle='--', label='Zero Offset')
        plt.axvline(x=match_stats['mean_diff'], color='r', linestyle='-', 
                   label=f'Mean Offset: {match_stats["mean_diff"]:.2f} ms')
        plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green', 
                   label=f'Tolerance (±{args.tolerance_ms} ms)')
    else:
        plt.text(0.5, 0.5, "No matching events", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(f'{sensor1}-{sensor2} Event Time Difference Distribution')
    plt.xlabel('Time Difference (ms)')
    plt.ylabel('Event Count')
    plt.legend()
    plt.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{sensor1}_{sensor2}_event.png"), dpi=300)
    
    if args.visualize:
        plt.show()
    else:
        plt.close()

def visualize_visual_results(verification_results, sensor1, sensor2, args):
    """可视化视觉对齐验证结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建一个大的图像，显示所有相关信息
    plt.figure(figsize=(14, 10))
    
    # 提取结果
    is_aligned = verification_results.get('is_aligned', False)
    total_offset = verification_results.get('total_offset', verification_results.get('mean_offset', 0))
    best_offset = verification_results.get('best_offset', 0)
    base_time_diff = verification_results.get('base_time_diff', 0)
    visual_result = verification_results.get('visual_result', {})
    
    # 设置主标题
    plt.suptitle(f"{sensor1}-{sensor2} 视觉方法时间戳对齐分析", fontsize=16)
    
    # 绘制信号比较
    plt.subplot(2, 1, 1)
    
    # 提取数据
    common_times = visual_result.get('common_times', [])
    y1 = visual_result.get('y1', [])
    y2_aligned = visual_result.get('y2_aligned', [])
    y2_original = visual_result.get('y2_original', [])
    score = visual_result.get('score', 0)
    
    if len(common_times) > 0 and len(y1) > 0 and len(y2_aligned) > 0:
        plt.plot(common_times, y1, 'b-', label=f'{sensor1} 信号', linewidth=1.5)
        plt.plot(common_times, y2_aligned, 'g-', label=f'{sensor2} 信号 (已对齐)', linewidth=1.5)
        if len(y2_original) > 0:
            plt.plot(common_times, y2_original, 'r--', label=f'{sensor2} 信号 (原始)', linewidth=1)
            
        plt.title(f"总偏移: {total_offset:.2f}ms (基础: {base_time_diff:.2f}ms + 细节: {best_offset:.2f}ms), 匹配分数: {score:.4f}", fontsize=12)
        plt.xlabel('相对时间 (s)')
        plt.ylabel('角速度幅值')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无可视化数据', ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    # 绘制偏移分数图
    plt.subplot(2, 1, 2)
    offset_scores = verification_results.get('offset_scores', {})
    
    if offset_scores:
        offsets = sorted(offset_scores.keys())
        scores = [offset_scores[o] for o in offsets]
        
        plt.plot(offsets, scores, 'b-', linewidth=1.5)
        if best_offset in offsets:
            idx = offsets.index(best_offset)
            plt.plot(best_offset, scores[idx], 'ro', markersize=8)
            plt.annotate(f'最佳偏移: {best_offset}ms', 
                        xy=(best_offset, scores[idx]), 
                        xytext=(best_offset+5, scores[idx]),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=10)
            
        plt.title(f"偏移评分曲线 (结果: {'对齐' if is_aligned else '未对齐'}, 容差: ±{args.tolerance_ms}ms)", fontsize=12)
        plt.xlabel('偏移 (ms)')
        plt.ylabel('匹配分数')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '无偏移评分数据', ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为整体标题留出空间
    plt.savefig(os.path.join(args.output_dir, f"{sensor1}_{sensor2}_visual_verification.png"), dpi=300)
    plt.close()

def generate_summary_report(all_verification_results, args):
    """生成综合验证报告"""
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    report_file = os.path.join(args.output_dir, "alignment_summary_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("===== 多传感器时间戳对齐验证报告 =====\n\n")
        
        # 记录使用的数据源
        f.write("使用的数据源:\n")
        f.write(f"  IMU: {args.imu_file}\n")
        if args.odo_file:
            f.write(f"  轮式编码器: {args.odo_file}\n")
        if args.rtk_file:
            f.write(f"  RTK: {args.rtk_file}\n")
        if args.image_dir:
            f.write(f"  图像: {args.image_dir}\n")
            if args.image_timestamp_file:
                f.write(f"  图像时间戳文件: {args.image_timestamp_file}\n")
        
        f.write("\n时间对齐容差: ±{:.2f} ms\n\n".format(args.tolerance_ms))
        
        # 对每对传感器组合进行汇总
        for sensor_pair, pair_results in all_verification_results.items():
            sensor1, sensor2 = sensor_pair.split('_')
            
            f.write(f"===== {sensor1}与{sensor2}时间对齐验证 =====\n")
            
            methods_summary = []
            all_offsets = []
            aligned_count = 0
            
            for method, result in pair_results.items():
                is_aligned = result.get('is_aligned', False)
                mean_offset = result.get('mean_offset', 0)
                std_offset = result.get('std_offset', 0)
                
                methods_summary.append({
                    'method': method,
                    'offset': mean_offset,
                    'std': std_offset, 
                    'aligned': is_aligned
                })
                
                all_offsets.append(mean_offset)
                if is_aligned:
                    aligned_count += 1
            
            # 输出每种方法的结果
            f.write("各验证方法结果:\n")
            for summary in methods_summary:
                f.write(f"  * {summary['method']}方法: {'对齐' if summary['aligned'] else '未对齐'}, ")
                f.write(f"偏移 = {summary['offset']:.2f} ± {summary['std']:.2f} ms\n")
            
            # 综合判断
            final_aligned = aligned_count >= len(methods_summary) / 2
            weighted_offset = np.mean(all_offsets)
            
            f.write(f"\n综合判断: {'对齐' if final_aligned else '未对齐'} ")
            f.write(f"({aligned_count}/{len(methods_summary)}个方法认为对齐)\n")
            f.write(f"加权平均时间偏移: {weighted_offset:.2f} ms\n\n")
        
        # 总体结论
        f.write("===== 总体结论 =====\n")
        all_aligned = True
        for sensor_pair, pair_results in all_verification_results.items():
            sensor1, sensor2 = sensor_pair.split('_')
            
            aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
            final_aligned = aligned_count >= len(pair_results) / 2
            
            if not final_aligned:
                all_aligned = False
            
            f.write(f"{sensor1}-{sensor2}: {'对齐' if final_aligned else '未对齐'}\n")
        
        f.write(f"\n所有传感器时间戳{'整体对齐' if all_aligned else '未完全对齐'}\n")
    
    # 辅助函数：确保对象是JSON可序列化的
    def json_serializable(obj):
        """递归转换非JSON可序列化值为可序列化类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(json_serializable(item) for item in obj)
        elif obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        elif pd.isna(obj):  # 处理NaN和None
            return None
        else:
            # 对于复杂对象，尝试使用字符串表示
            try:
                return str(obj)
            except:
                return None
    
    # 同时生成JSON格式的摘要
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'tolerance_ms': float(args.tolerance_ms),
        'sensor_pairs': {}
    }
    
    for sensor_pair, pair_results in all_verification_results.items():
        sensor1, sensor2 = sensor_pair.split('_')
        
        methods_data = {}
        for method, result in pair_results.items():
            # 仅提取需要的关键数据，并确保JSON可序列化
            offset_ms = json_serializable(result.get('mean_offset', 0))
            std_ms = json_serializable(result.get('std_offset', 0))
            is_aligned = bool(result.get('is_aligned', False))
            
            methods_data[method] = {
                'offset_ms': offset_ms,
                'std_ms': std_ms,
                'is_aligned': is_aligned
            }
        
        aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
        final_aligned = aligned_count >= len(pair_results) / 2
        
        summary_json['sensor_pairs'][sensor_pair] = {
            'methods': methods_data,
            'overall': {
                'is_aligned': bool(final_aligned),
                'aligned_methods': int(aligned_count),
                'total_methods': len(pair_results)
            }
        }
    
    # 保存JSON格式报告
    try:
        # 确保整个summary_json是JSON可序列化的
        serializable_json = json_serializable(summary_json)
        with open(os.path.join(args.output_dir, "alignment_summary.json"), 'w') as f:
            json.dump(serializable_json, f, indent=2)
        print(f"综合报告已保存至: {report_file}")
    except Exception as e:
        print(f"保存JSON报告时出错: {e}")
    
    return report_file

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要验证的传感器组合
    sensor_pairs = []
    if args.sensors.lower() == 'all':
        # 根据提供的数据源自动确定可能的组合
        if args.imu_file and args.odo_file:
            sensor_pairs.append(('imu', 'odo'))
        if args.imu_file and args.rtk_file:
            sensor_pairs.append(('imu', 'rtk'))
        if args.imu_file and args.image_dir:
            sensor_pairs.append(('imu', 'camera'))
        if args.odo_file and args.rtk_file:
            sensor_pairs.append(('odo', 'rtk'))
    else:
        # 解析用户指定的组合
        for pair in args.sensors.split(','):
            if pair == 'imu_odo' and args.imu_file and args.odo_file:
                sensor_pairs.append(('imu', 'odo'))
            elif pair == 'imu_rtk' and args.imu_file and args.rtk_file:
                sensor_pairs.append(('imu', 'rtk'))
            elif pair == 'imu_camera' and args.imu_file and args.image_dir:
                sensor_pairs.append(('imu', 'camera'))
            elif pair == 'odo_rtk' and args.odo_file and args.rtk_file:
                sensor_pairs.append(('odo', 'rtk'))
    
    if not sensor_pairs:
        print("警告: 没有有效的传感器组合可供验证。请检查输入参数。")
        return
    
    # 确定要使用的验证方法
    methods = []
    if args.methods.lower() == 'all':
        methods = ['correlation', 'event', 'visual']
    else:
        for method in args.methods.split(','):
            if method in ['correlation', 'event', 'visual']:
                methods.append(method)
    
    if not methods:
        print("警告: 没有指定验证方法。将使用所有方法。")
        methods = ['correlation', 'event', 'visual']
    
    # 加载数据
    print("加载数据...")
    data_sources = {}
    
    # IMU数据
    imu_data = load_imu_data(args)
    data_sources['imu'] = {
        'data': imu_data,
        'ts_col': args.imu_timestamp_col
    }
    
    # 轮式编码器数据
    if args.odo_file:
        odo_data = load_odometry_data(args)
        if odo_data is not None:
            data_sources['odo'] = {
                'data': odo_data,
                'ts_col': args.odo_timestamp_col
            }
    
    # RTK数据
    if args.rtk_file:
        rtk_data = load_rtk_data(args)
        if rtk_data is not None:
            data_sources['rtk'] = {
                'data': rtk_data,
                'ts_col': args.rtk_timestamp_col
            }
    
    # 图像数据
    if args.image_dir:
        image_data = process_image_data(args)
        if not image_data.empty:
            data_sources['camera'] = {
                'data': image_data,
                'ts_col': 'timestamp'
            }
    
    # 验证结果存储
    all_verification_results = {}
    
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