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
    imu_data = pd.read_csv(args.imu_file, comment='#')
    
    # 确保时间戳列存在
    if args.imu_timestamp_col not in imu_data.columns:
        raise ValueError(f"IMU数据缺少{args.imu_timestamp_col}列")
    
    # 计算角速度幅值
    gyro_cols = [col for col in imu_data.columns if 'gyro' in col.lower()]
    if len(gyro_cols) >= 3:
        # 如果有三个角速度分量，计算合成角速度
        x_col, y_col, z_col = gyro_cols[:3]
        imu_data['angular_velocity_magnitude'] = np.sqrt(
            imu_data[x_col]**2 + imu_data[y_col]**2 + imu_data[z_col]**2
        )
    elif 'gyro_z' in imu_data.columns:
        # 如果只有z轴角速度
        imu_data['angular_velocity_magnitude'] = np.abs(imu_data['gyro_z'])
    else:
        raise ValueError("IMU数据缺少角速度列")
    
    # 确保时间戳是升序排列的
    imu_data = imu_data.sort_values(by=args.imu_timestamp_col)
    
    return imu_data

def load_odometry_data(args):
    """加载轮式编码器数据"""
    if not args.odo_file:
        return None
        
    print(f"加载轮式编码器数据: {args.odo_file}")
    odo_data = pd.read_csv(args.odo_file, comment='#')
    
    # 确保时间戳列存在
    if args.odo_timestamp_col not in odo_data.columns:
        raise ValueError(f"轮式编码器数据缺少{args.odo_timestamp_col}列")
    
    # 根据可用的列计算角速度
    if 'angular_velocity' in odo_data.columns:
        # 已有角速度列
        odo_data['angular_velocity_magnitude'] = np.abs(odo_data['angular_velocity'])
    elif 'speed_w' in odo_data.columns:
        # 使用角速度列
        odo_data['angular_velocity_magnitude'] = np.abs(odo_data['speed_w'])
    elif 'left_count' in odo_data.columns and 'right_count' in odo_data.columns:
        # 使用左右轮计数差分计算角速度
        left_diff = odo_data['left_count'].diff().fillna(0)
        right_diff = odo_data['right_count'].diff().fillna(0)
        odo_data['angular_velocity_magnitude'] = np.abs(left_diff - right_diff)
    else:
        raise ValueError("轮式编码器数据缺少必要的列来计算角速度")
    
    # 确保时间戳是升序排列的
    odo_data = odo_data.sort_values(by=args.odo_timestamp_col)
    
    return odo_data

def load_rtk_data(args):
    """加载RTK数据"""
    if not args.rtk_file:
        return None
        
    print(f"加载RTK数据: {args.rtk_file}")
    rtk_data = pd.read_csv(args.rtk_file, comment='#')
    
    # 确保时间戳列存在
    if args.rtk_timestamp_col not in rtk_data.columns:
        raise ValueError(f"RTK数据缺少{args.rtk_timestamp_col}列")
    
    # 计算角速度（从速度方向变化）
    if 'velocity_x' in rtk_data.columns and 'velocity_y' in rtk_data.columns:
        # 计算方向角
        rtk_data['direction'] = np.arctan2(rtk_data['velocity_y'], rtk_data['velocity_x'])
        
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
    else:
        raise ValueError("RTK数据缺少速度列，无法计算角速度")
    
    # 确保时间戳是升序排列的
    rtk_data = rtk_data.sort_values(by=args.rtk_timestamp_col)
    
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
    """使用互相关分析验证时间戳对齐"""
    # 估计两个信号的采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 自适应调整窗口大小和时间偏移范围
    min_rate = min(sampling_rate1, sampling_rate2)
    window_size = int(args.window_size * (min_rate/50.0))  # 基于50Hz调整
    window_size = max(window_size, 50)  # 确保至少有50个样本
    
    # 调整最大lag为以毫秒为单位
    max_lag_ms = 500  # 最大500ms的偏移
    max_lag1 = int(max_lag_ms * sampling_rate1 / 1000)
    max_lag2 = int(max_lag_ms * sampling_rate2 / 1000)
    max_lag = min(max_lag1, max_lag2, args.max_lag)
    
    print(f"  调整后窗口大小: {window_size}, 最大延迟: {max_lag} 采样点")
    
    # 提取角速度数据
    df1 = data1.copy()
    df2 = data2.copy()
    
    # 归一化处理
    df1['angular_velocity_magnitude_norm'] = ((df1['angular_velocity_magnitude'] - 
                                            df1['angular_velocity_magnitude'].mean()) / 
                                            df1['angular_velocity_magnitude'].std())
    
    df2['angular_velocity_magnitude_norm'] = ((df2['angular_velocity_magnitude'] - 
                                            df2['angular_velocity_magnitude'].mean()) / 
                                            df2['angular_velocity_magnitude'].std())
    
    # 执行互相关验证
    results = []
    
    # 确定窗口数量
    num_windows = min(len(df1), len(df2)) // window_size - 1
    step = max(1, num_windows // 20)  # 约取20个窗口进行分析
    
    for i in range(0, num_windows, step):
        start_idx1 = i * window_size
        end_idx1 = start_idx1 + window_size
        
        if end_idx1 > len(df1):
            break
            
        window1 = df1.iloc[start_idx1:end_idx1]
        
        # 找到时间范围接近的窗口2
        window1_t_start = window1[ts_col1].min()
        window1_t_end = window1[ts_col1].max()
        
        # 扩大时间范围，确保包含可能的偏移
        window1_duration = window1_t_end - window1_t_start
        search_t_start = window1_t_start - window1_duration * 0.5
        search_t_end = window1_t_end + window1_duration * 0.5
        
        # 选择时间范围内的df2数据
        window2 = df2[(df2[ts_col2] >= search_t_start) & (df2[ts_col2] <= search_t_end)]
        
        if len(window2) < window_size / 2:
            # 窗口2中数据不足
            continue
        
        # 重采样到相同的长度
        target_len = 500  # 统一长度
        try:
            t1_new, sig1_new = resample_signal(
                window1, ts_col1, 'angular_velocity_magnitude_norm', target_len=target_len
            )
            t2_new, sig2_new = resample_signal(
                window2, ts_col2, 'angular_velocity_magnitude_norm', target_len=target_len
            )
            
            # 检查重采样后的数据是否足够
            if len(sig1_new) < target_len*0.8 or len(sig2_new) < target_len*0.8:
                continue
                
            # 计算互相关
            correlation = signal.correlate(sig1_new, sig2_new, mode='full')
            lags = signal.correlation_lags(len(sig1_new), len(sig2_new), mode='full')
            
            # 限制搜索范围
            valid_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
            valid_corr = correlation[valid_indices]
            valid_lags = lags[valid_indices]
            
            if len(valid_corr) == 0:
                continue
            
            # 找到最大互相关的位置
            max_idx = np.argmax(valid_corr)
            lag = valid_lags[max_idx]
            corr_value = valid_corr[max_idx]
            
            # 估计每个样本的实际时间间隔(毫秒)
            t1_interval = (t1_new[-1] - t1_new[0]) / (len(t1_new) - 1)
            
            # 转换为时间偏移（毫秒）
            offset_ms = lag * t1_interval
            
            # 判断是否在容差范围内
            is_aligned = abs(offset_ms) <= args.tolerance_ms
            
            results.append({
                'window_start': start_idx1,
                'lag': lag,
                'offset_ms': offset_ms,
                'corr_value': corr_value,
                'is_aligned': is_aligned
            })
        except Exception as e:
            print(f"    窗口处理失败: {str(e)}")
            continue
    
    # 分析结果
    if not results:
        return {
            'results': [],
            'alignment_rate': 0.0,
            'mean_offset': 0.0,
            'std_offset': 0.0,
            'is_aligned': False,
            'method': 'correlation'
        }
        
    alignment_rate = sum(1 for r in results if r['is_aligned']) / len(results)
    mean_offset = np.mean([r['offset_ms'] for r in results])
    std_offset = np.std([r['offset_ms'] for r in results])
    
    return {
        'results': results,
        'alignment_rate': alignment_rate,
        'mean_offset': mean_offset,
        'std_offset': std_offset,
        'is_aligned': alignment_rate >= 0.8,  # 如果80%以上的窗口都对齐，则认为整体对齐
        'method': 'correlation'
    }

def event_verification(data1, data2, ts_col1, ts_col2, args):
    """使用事件同步验证时间戳对齐"""
    # 估计采样率
    sampling_rate1 = estimate_sampling_rate(data1, ts_col1)
    sampling_rate2 = estimate_sampling_rate(data2, ts_col2)
    
    print(f"  信号1采样率: {sampling_rate1:.2f} Hz, 信号2采样率: {sampling_rate2:.2f} Hz")
    
    # 调整事件检测参数，考虑采样率
    threshold_pct1 = 85 + min(10, int(15 * sampling_rate1 / 50))  # 高频采样使用更高阈值
    threshold_pct2 = 85 + min(10, int(15 * sampling_rate2 / 50))
    
    # 为采样率低的信号调整窗口大小和间隔
    win_size1 = max(3, int(0.2 * sampling_rate1))  # 至少200ms
    win_size2 = max(3, int(0.2 * sampling_rate2))
    min_dist1 = max(2, int(0.5 * sampling_rate1))  # 至少500ms间隔
    min_dist2 = max(2, int(0.5 * sampling_rate2))
    
    # 检测事件
    events1 = detect_events(data1, ts_col1, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct1,
                          window_size=win_size1, min_distance=min_dist1)
    
    events2 = detect_events(data2, ts_col2, value_col='angular_velocity_magnitude', 
                          threshold_percentile=threshold_pct2,
                          window_size=win_size2, min_distance=min_dist2)
    
    # 低频传感器需要更宽松的匹配窗口
    low_rate = min(sampling_rate1, sampling_rate2)
    match_threshold_ms = args.tolerance_ms * (1 + 0.5 * (50 / low_rate))  # 低采样率使用更宽松阈值
    
    # 匹配事件
    matches = []
    unmatched1 = []
    unmatched2 = []
    
    # 遍历事件1
    for idx1, ts1 in enumerate(events1['timestamps']):
        best_match = None
        min_diff = float('inf')
        
        # 查找最接近的事件2
        for idx2, ts2 in enumerate(events2['timestamps']):
            diff = abs(ts1 - ts2)
            if diff < min_diff and diff <= match_threshold_ms:
                min_diff = diff
                best_match = (idx2, ts2, diff)
        
        if best_match:
            matches.append({
                'event1_idx': idx1,
                'event2_idx': best_match[0],
                'time_diff': best_match[2]  # ts1 - ts2，正值表示事件1晚于事件2
            })
        else:
            unmatched1.append(idx1)
    
    # 查找未匹配的事件2
    matched_event2_indices = {m['event2_idx'] for m in matches}
    unmatched2 = [idx for idx in range(len(events2['timestamps'])) 
                 if idx not in matched_event2_indices]
    
    # 计算匹配统计
    match_rate = len(matches) / max(len(events1['timestamps']), len(events2['timestamps']))
    
    if matches:
        time_diffs = [m['time_diff'] for m in matches]
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        # 判断是否对齐
        is_aligned = abs(mean_diff) <= args.tolerance_ms
    else:
        mean_diff = 0
        std_diff = 0
        is_aligned = False
    
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
        'method': 'event'
    }

def detect_events(data, ts_col, value_col='angular_velocity_magnitude', 
                threshold_percentile=90, window_size=5, min_distance=10):
    """
    检测传感器数据中的显著事件
    
    参数:
    - data: 包含时间戳和值的DataFrame
    - ts_col: 时间戳列名
    - value_col: 要分析的值列名
    - threshold_percentile: 阈值百分位数
    - window_size: 局部峰值检测窗口大小
    - min_distance: 峰值间最小距离
    """
    # 数据预处理和平滑
    values = data[value_col].values
    timestamps = data[ts_col].values
    
    # 应用中值滤波减少噪声
    if len(values) > 5:
        values_smooth = signal.medfilt(values, kernel_size=min(5, len(values)//2*2+1))
    else:
        values_smooth = values.copy()
    
    # 计算动态阈值
    threshold = np.percentile(values_smooth, threshold_percentile)
    
    # 检测局部峰值
    peaks, _ = signal.find_peaks(values_smooth, height=threshold, 
                               distance=min_distance)
    
    # 提取峰值特征
    peak_heights = values_smooth[peaks]
    peak_timestamps = timestamps[peaks]
    
    # 构建事件列表
    events = {
        'indices': peaks,
        'timestamps': peak_timestamps,
        'heights': peak_heights
    }
    
    return events

def visual_verification(data1, data2, ts_col1, ts_col2, sensor1, sensor2, args):
    """使用可视化方法验证时间戳对齐"""
    # 提取数据
    ts1 = data1[ts_col1].values
    signal1 = data1['angular_velocity_magnitude'].values
    
    ts2 = data2[ts_col2].values
    signal2 = data2['angular_velocity_magnitude'].values
    
    # 归一化处理
    signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # 查找最佳时间偏移
    max_lag_ms = 500  # 最大搜索范围(毫秒)
    step_ms = 5  # 搜索步长
    
    # 将时间戳转换为内部时间单位
    min_ts = max(np.min(ts1), np.min(ts2))
    max_ts = min(np.max(ts1), np.max(ts2))
    
    # 在共同时间范围内重采样数据
    common_ts = np.arange(min_ts, max_ts, step_ms)
    signal1_resampled = np.interp(common_ts, ts1, signal1_norm)
    signal2_resampled = np.interp(common_ts, ts2, signal2_norm)
    
    # 计算原始相关性
    original_corr = np.corrcoef(signal1_resampled, signal2_resampled)[0, 1]
    
    # 搜索最佳偏移
    lag_range = np.arange(-max_lag_ms, max_lag_ms + step_ms, step_ms)
    corr_values = []
    
    for lag in lag_range:
        # 应用偏移
        shifted_ts2 = ts2 + lag
        
        # 在共同时间范围内重采样偏移后的数据
        try:
            signal2_shifted = np.interp(common_ts, shifted_ts2, signal2_norm)
            corr = np.corrcoef(signal1_resampled, signal2_shifted)[0, 1]
            corr_values.append(corr)
        except:
            corr_values.append(np.nan)
    
    # 找到最佳偏移
    max_corr_idx = np.nanargmax(corr_values)
    best_lag = lag_range[max_corr_idx]
    best_corr = corr_values[max_corr_idx]
    
    # 判断是否对齐
    is_aligned = abs(best_lag) <= args.tolerance_ms
    
    return {
        'original_corr': original_corr,
        'best_lag': best_lag,
        'best_corr': best_corr,
        'corr_curve': {
            'lags': lag_range.tolist(),
            'corrs': corr_values
        },
        'mean_offset': best_lag,
        'std_offset': 0,  # 单次计算无标准差
        'is_aligned': is_aligned,
        'method': 'visual',
        'signals': {
            'common_ts': common_ts,
            'signal1': signal1_resampled,
            'signal2': signal2_resampled,
            'sensor1': sensor1,
            'sensor2': sensor2
        }
    }

# ========== 可视化函数 ==========

def visualize_correlation_results(verification_results, sensor1, sensor2, args):
    """可视化互相关验证结果"""
    # 创建结果目录
    result_dir = os.path.join(args.output_dir, f"{sensor1}_{sensor2}_correlation")
    os.makedirs(result_dir, exist_ok=True)
    
    # 提取结果
    results = verification_results['results']
    offsets = [r['offset_ms'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制偏移分布
    plt.hist(offsets, bins=30, color='blue', alpha=0.7)
    plt.axvline(x=0, color='k', linestyle='--', label='零偏移')
    plt.axvline(x=verification_results['mean_offset'], color='r', 
                linestyle='-', label=f'平均偏移: {verification_results["mean_offset"]:.2f} ms')
    plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green', 
                label=f'容差范围 (±{args.tolerance_ms} ms)')
    
    plt.title(f'{sensor1}-{sensor2}互相关偏移分布 (对齐率: {verification_results["alignment_rate"]*100:.1f}%)')
    plt.xlabel('时间偏移 (ms)')
    plt.ylabel('窗口数')
    plt.legend()
    plt.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{sensor1}_{sensor2}_correlation.png"), dpi=300)
    
    if args.visualize:
        plt.show()
    else:
        plt.close()

def visualize_event_results(verification_results, sensor1, sensor2, args):
    """可视化事件验证结果"""
    # 创建结果目录
    result_dir = os.path.join(args.output_dir, f"{sensor1}_{sensor2}_event")
    os.makedirs(result_dir, exist_ok=True)
    
    # 提取数据
    events1 = verification_results['events1']
    events2 = verification_results['events2']
    match_stats = verification_results['match_stats']
    
    plt.figure(figsize=(12, 10))
    
    # 1. 事件匹配图
    plt.subplot(2, 1, 1)
    # 绘制事件时间点
    plt.scatter(events1['timestamps'], np.ones_like(events1['timestamps']), marker='|', s=100, 
               label=f'{sensor1}事件', color='blue')
    plt.scatter(events2['timestamps'], np.ones_like(events2['timestamps'])*1.1, 
               marker='|', s=100, label=f'{sensor2}事件', color='red')
    
    # 绘制匹配连线
    matched_events1 = [events1['timestamps'][m['event1_idx']] for m in match_stats['matches']]
    matched_events2 = [events2['timestamps'][m['event2_idx']] for m in match_stats['matches']]
    
    for i in range(len(matched_events1)):
        plt.plot([matched_events1[i], 1], [1, 1.1], 'k-', alpha=0.3)
    
    plt.title(f'{sensor1}-{sensor2}事件匹配 (匹配率: {match_stats["match_rate"]*100:.1f}%)')
    plt.xlabel('时间戳')
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    
    # 2. 时间差分布
    plt.subplot(2, 1, 2)
    time_diffs = [m['time_diff'] for m in match_stats['matches']]
    plt.hist(time_diffs, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='k', linestyle='--', label='零时差')
    plt.axvline(x=match_stats['mean_diff'], color='r', linestyle='-', 
               label=f'平均时差: {match_stats["mean_diff"]:.2f} ms')
    plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green', 
               label=f'容差 (±{args.tolerance_ms} ms)')
    
    plt.title(f'{sensor1}-{sensor2}事件时间差分布')
    plt.xlabel('时间差 (ms)')
    plt.ylabel('事件数')
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
    """可视化signal比较结果"""
    # 创建结果目录
    result_dir = os.path.join(args.output_dir, f"{sensor1}_{sensor2}_visual")
    os.makedirs(result_dir, exist_ok=True)
    
    # 提取数据
    signals = verification_results['signals']
    common_ts = signals['common_ts']
    signal1 = signals['signal1']
    signal2 = signals['signal2']
    
    best_lag = verification_results['best_lag']
    best_corr = verification_results['best_corr']
    original_corr = verification_results['original_corr']
    
    # 使用偏移后的时间戳
    shifted_ts = common_ts + best_lag
    
    plt.figure(figsize=(12, 10))
    
    # 1. 原始信号对比
    plt.subplot(3, 1, 1)
    plt.plot(common_ts, signal1, 'b-', label=f'{sensor1}信号')
    plt.plot(common_ts, signal2, 'r-', label=f'{sensor2}信号')
    plt.title(f'原始信号对比 (相关性: {original_corr:.3f})')
    plt.xlabel('时间戳')
    plt.ylabel('归一化信号')
    plt.legend()
    plt.grid(True)
    
    # 2. 偏移后的信号对比
    plt.subplot(3, 1, 2)
    plt.plot(common_ts, signal1, 'b-', label=f'{sensor1}信号')
    plt.plot(shifted_ts, signal2, 'g-', label=f'{sensor2}信号(偏移{best_lag:.1f}ms)')
    plt.title(f'最佳偏移后的信号比较 (相关性: {best_corr:.3f})')
    plt.xlabel('时间戳')
    plt.ylabel('归一化信号')
    plt.legend()
    plt.grid(True)
    
    # 3. 相关性随偏移变化
    plt.subplot(3, 1, 3)
    lags = verification_results['corr_curve']['lags']
    corrs = verification_results['corr_curve']['corrs']
    
    plt.plot(lags, corrs, 'k-')
    plt.axvline(x=best_lag, color='r', linestyle='--', 
               label=f'最佳偏移: {best_lag:.1f} ms')
    plt.axvspan(-args.tolerance_ms, args.tolerance_ms, alpha=0.2, color='green',
               label=f'容差范围(±{args.tolerance_ms}ms)')
    plt.title('相关性随时间偏移变化')
    plt.xlabel('时间偏移 (ms)')
    plt.ylabel('相关系数')
    plt.legend()
    plt.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{sensor1}_{sensor2}_visual.png"), dpi=300)
    
    if args.visualize:
        plt.show()
    else:
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
    
    # 同时生成JSON格式的摘要
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'tolerance_ms': args.tolerance_ms,
        'sensor_pairs': {}
    }
    
    for sensor_pair, pair_results in all_verification_results.items():
        sensor1, sensor2 = sensor_pair.split('_')
        
        methods_data = {}
        for method, result in pair_results.items():
            methods_data[method] = {
                'offset_ms': result.get('mean_offset', 0),
                'std_ms': result.get('std_offset', 0),
                'is_aligned': result.get('is_aligned', False)
            }
        
        aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
        final_aligned = aligned_count >= len(pair_results) / 2
        
        summary_json['sensor_pairs'][sensor_pair] = {
            'methods': methods_data,
            'overall': {
                'is_aligned': final_aligned,
                'aligned_methods': aligned_count,
                'total_methods': len(pair_results)
            }
        }
    
    # 保存JSON格式报告
    with open(os.path.join(args.output_dir, "alignment_summary.json"), 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"综合报告已保存至: {report_file}")
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