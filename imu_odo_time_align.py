#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import os

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IMU、轮式编码器和RTK数据时间对齐工具")
    parser.add_argument('--imu_file', type=str, required=True, help="IMU数据文件路径")
    parser.add_argument('--odo_file', type=str, required=True, help="轮式编码器数据文件路径")
    parser.add_argument('--rtk_file', type=str, help="RTK数据文件路径")
    parser.add_argument('--output_file', type=str, default="aligned_data.csv", help="输出文件路径")
    parser.add_argument('--visualize', action='store_true', help="显示对齐结果的可视化图表")
    parser.add_argument('--window_size', type=int, default=500, help="计算互相关的窗口大小")
    parser.add_argument('--max_lag', type=int, default=200, help="最大时间滞后/超前(单位：采样点)")
    parser.add_argument('--search_step_size', type=int, default=50, help="搜索步长(单位：采样点)")
    parser.add_argument('--imu_timestamp_col', type=str, default="timestamp", help="IMU数据中的时间戳列名")
    parser.add_argument('--odo_timestamp_col', type=str, default="timestamp", help="轮式编码器数据中的时间戳列名")
    parser.add_argument('--rtk_timestamp_col', type=str, default="timestamp", help="RTK数据中的时间戳列名")
    parser.add_argument('--wheel_perimeter', type=float, default=0.22, help="轮子周长(米)")
    parser.add_argument('--wheel_halflength', type=float, default=0.1195, help="轮距的一半(米)")
    parser.add_argument('--encoder_scale', type=float, default=262, help="编码器计数比例因子")
    parser.add_argument('--align_rtk', action='store_true', help="是否包含RTK数据进行时间对齐")
    return parser.parse_args()

def load_imu_data(file_path):
    """加载IMU数据文件"""
    print(f"加载IMU数据: {file_path}")
    
    # 根据文件扩展名确定文件类型
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        try:
            # 先尝试带注释符号的读取
            imu_data = pd.read_csv(file_path, comment='#')
            # 检查列数以确定是否为标准IMU格式
            if len(imu_data.columns) == 7:
                print("检测到有注释的CSV格式")
                imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
            else:
                # 无列名读取
                imu_data = pd.read_csv(file_path, header=None)
                print("检测到无列名的CSV格式")
                # 根据列数确定合适的列名
                if len(imu_data.columns) == 7:
                    imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
                else:
                    print(f"警告: 不标准的IMU CSV格式，列数为 {len(imu_data.columns)}。请确认列名。")
                    # 可以根据实际情况设置默认列名
        except Exception as e:
            print(f"解析CSV文件时出错: {e}")
            raise
    else:
        # 尝试读取其他格式
        print(f"不支持的文件格式: {file_ext}")
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 确保时间戳是整数型
    if 'timestamp' in imu_data.columns:
        try:
            imu_data['timestamp'] = pd.to_numeric(imu_data['timestamp'])
        except:
            print("警告: 无法将时间戳转换为数值。保持原格式。")
    
    return imu_data

def load_odometer_data(file_path):
    """加载轮式编码器数据文件"""
    print(f"加载轮式编码器数据: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        try:
            # 尝试带注释符号读取
            odo_data = pd.read_csv(file_path, comment='#')
            # 检查是否包含左右轮编码器计数列
            required_cols = ['left_count', 'right_count']
            has_required_cols = all(col in odo_data.columns for col in required_cols)
            
            if not has_required_cols:
                # 无列名读取
                odo_data = pd.read_csv(file_path, header=None)
                print("检测到无列名的CSV格式")
                # 设置默认列名，根据实际数据格式调整
                if len(odo_data.columns) >= 3:  # 至少需要timestamp, left_count, right_count
                    # 这里我们至少需要时间戳和左右轮计数
                    col_names = ['timestamp', 'left_count', 'right_count']
                    # 如果有更多列，我们可以添加额外的列名
                    if len(odo_data.columns) > 3:
                        remaining_cols = [f'col_{i+3}' for i in range(len(odo_data.columns) - 3)]
                        col_names.extend(remaining_cols)
                    
                    odo_data.columns = col_names
                else:
                    print(f"警告: 编码器CSV格式不包含足够的列，列数为 {len(odo_data.columns)}。")
                    print("需要至少3列: timestamp, left_count, right_count")
                    raise ValueError("数据格式不兼容")
        except Exception as e:
            print(f"解析CSV文件时出错: {e}")
            raise
    else:
        print(f"不支持的文件格式: {file_ext}")
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 确保时间戳是整数型
    if 'timestamp' in odo_data.columns:
        try:
            odo_data['timestamp'] = pd.to_numeric(odo_data['timestamp'])
        except:
            print("警告: 无法将时间戳转换为数值。保持原格式。")
    
    return odo_data

def load_rtk_data(file_path):
    """加载RTK数据文件"""
    print(f"加载RTK数据: {file_path}")
    
    # 根据文件扩展名确定文件类型
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        try:
            # 先尝试带注释符号的读取
            rtk_data = pd.read_csv(file_path, comment='#')
            
            # 检查是否包含必要的列（时间戳、位置和线速度）
            required_cols = ['timestamp', 'velocity_x', 'velocity_y']
            has_required_cols = all(col in rtk_data.columns for col in required_cols)
            
            if not has_required_cols:
                # 无列名读取
                rtk_data = pd.read_csv(file_path, header=None)
                print("检测到无列名的CSV格式")
                
                # 根据列数设置默认列名
                # 假设RTK数据至少包含时间戳、位置(x,y,z)和线速度(velocity_x, velocity_y)
                if len(rtk_data.columns) >= 6:
                    rtk_data.columns = [
                        'timestamp', 'position_x', 'position_y', 'position_z',
                        'velocity_x', 'velocity_y'
                    ]
                    if len(rtk_data.columns) > 6:
                        # 如果有更多列，添加默认列名
                        remaining_cols = [f'col_{i+7}' for i in range(len(rtk_data.columns) - 6)]
                        rtk_data.columns = list(rtk_data.columns) + remaining_cols
                else:
                    print(f"警告: RTK CSV格式不含足够列，列数为 {len(rtk_data.columns)}。")
                    print("需要至少6列: timestamp, position_x, position_y, position_z, velocity_x, velocity_y")
                    raise ValueError("RTK数据格式不兼容")
                
        except Exception as e:
            print(f"解析RTK CSV文件时出错: {e}")
            raise
    else:
        print(f"不支持的文件格式: {file_ext}")
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 确保时间戳是整数型
    if 'timestamp' in rtk_data.columns:
        try:
            rtk_data['timestamp'] = pd.to_numeric(rtk_data['timestamp'])
        except:
            print("警告: 无法将RTK时间戳转换为数值。保持原格式。")
    
    return rtk_data

def calculate_angular_velocity_from_encoders(odo_data, args):
    """
    从左右轮编码器计数差分计算角速度
    注意：此函数完全不使用speed_w和speed_v，而是直接从编码器计数差分计算角速度
    
    参数:
        odo_data: DataFrame, 包含左右轮编码器计数的数据
        args: ArgumentParser对象，包含轮子周长，轮距和编码器比例因子
        
    返回:
        numpy.array: 计算得到的角速度序列
    """
    print("从编码器计数计算角速度...")
    
    # 确保数据是按时间戳排序的
    odo_data = odo_data.sort_values(by=args.odo_timestamp_col)
    
    # 计算左右轮计数的差分
    left_diff = odo_data['left_count'].diff().fillna(0)
    right_diff = odo_data['right_count'].diff().fillna(0)
    
    # 计算时间差（秒）
    time_diff = odo_data[args.odo_timestamp_col].diff().fillna(0) / 1000.0  # 假设时间戳单位是毫秒
    
    # 防止除以零
    time_diff = np.where(time_diff > 0, time_diff, np.inf)
    
    # 将编码器计数转换为距离（米）
    left_distance = (left_diff / args.encoder_scale) * args.wheel_perimeter
    right_distance = (right_diff / args.encoder_scale) * args.wheel_perimeter
    
    # 计算左右轮线速度（米/秒）
    left_speed = left_distance / time_diff
    right_speed = right_distance / time_diff
    
    # 计算角速度（弧度/秒）
    angular_velocity = (right_speed - left_speed) / (2 * args.wheel_halflength)
    
    # 将无穷大或NaN值替换为0
    angular_velocity = np.nan_to_num(angular_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.abs(angular_velocity)  # 返回角速度的绝对值

def calculate_angular_velocity_magnitude(data):
    """计算角速度的幅值"""
    # 确认数据包含角速度分量
    gyro_cols = []
    for col in data.columns:
        if 'gyro' in col.lower() and not 'yaw' in col.lower() and not 'pitch' in col.lower() and not 'roll' in col.lower():
            gyro_cols.append(col)
    
    if len(gyro_cols) >= 3:
        # 如果有三个角速度分量，计算欧几里得范数
        x_col, y_col, z_col = gyro_cols[:3]
        return np.sqrt(data[x_col]**2 + data[y_col]**2 + data[z_col]**2)
    elif 'gyro_yaw' in data.columns:
        # 如果有gyro_yaw列但没有三个角速度分量，使用yaw角速度绝对值
        return data['gyro_yaw'].abs()
    else:
        # 不再检查speed_w
        raise ValueError("数据中没有找到角速度相关列")

def calculate_angular_velocity_from_rtk(rtk_data, args):
    """
    从RTK数据的线速度计算角速度幅值
    
    参数:
        rtk_data: DataFrame, 包含RTK数据
        args: ArgumentParser对象
        
    返回:
        numpy.array: 计算得到的角速度幅值序列
    """
    print("从RTK线速度计算角速度幅值...")
    
    # 确保数据按时间戳排序
    rtk_data = rtk_data.sort_values(by=args.rtk_timestamp_col)
    
    # 计算位置的差分
    pos_x_diff = rtk_data['position_x'].diff().fillna(0)
    pos_y_diff = rtk_data['position_y'].diff().fillna(0)
    
    # 计算时间差（秒）
    time_diff = rtk_data[args.rtk_timestamp_col].diff().fillna(0) / 1000.0  # 假设时间戳单位是毫秒
    
    # 防止除以零
    time_diff = np.where(time_diff > 0, time_diff, np.inf)
    
    # 计算速度（如果RTK数据中已经包含速度，可以直接使用）
    if 'velocity_x' in rtk_data.columns and 'velocity_y' in rtk_data.columns:
        velocity_x = rtk_data['velocity_x'].values
        velocity_y = rtk_data['velocity_y'].values
    else:
        # 如果没有速度列，通过位置差分计算速度
        velocity_x = pos_x_diff / time_diff
        velocity_y = pos_y_diff / time_diff
    
    # 计算速度的角度变化（方向变化率）
    # 使用arctan2计算速度方向
    directions = np.arctan2(velocity_y, velocity_x)
    
    # 计算方向的变化率（角速度）
    direction_diff = np.diff(directions, prepend=directions[0])
    
    # 处理方向角突变（例如从359度到0度）
    # 如果角度变化超过π，则进行修正
    direction_diff = np.where(direction_diff > np.pi, direction_diff - 2*np.pi, direction_diff)
    direction_diff = np.where(direction_diff < -np.pi, direction_diff + 2*np.pi, direction_diff)
    
    # 计算角速度（方向变化率 / 时间间隔）
    angular_velocity = direction_diff / time_diff
    
    # 将无穷大或NaN值替换为0
    angular_velocity = np.nan_to_num(angular_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 返回角速度幅值
    return np.abs(angular_velocity)

def synchronize_using_angular_velocity(imu_data, odo_data, rtk_data, args):
    """使用角速度幅值对多个数据源进行时间同步"""
    print("计算角速度幅值...")
    
    # 计算IMU数据的角速度幅值
    try:
        imu_ang_vel_mag = calculate_angular_velocity_magnitude(imu_data)
    except ValueError as e:
        print(f"无法计算IMU角速度幅值: {e}")
        imu_ang_vel_mag = imu_data['gyro_z'].abs()  # 使用z轴角速度作为备选
        print("使用IMU z轴角速度作为备选")
    
    # 计算轮式编码器数据的角速度幅值
    # 使用编码器计数差分计算角速度
    odo_ang_vel_mag = calculate_angular_velocity_from_encoders(odo_data, args)
    print("使用从编码器计数差分计算的角速度")
    
    # 如果提供了RTK数据，计算RTK的角速度幅值
    rtk_ang_vel_mag = None
    rtk_norm = None
    if rtk_data is not None and args.align_rtk:
        print("包含RTK数据进行时间对齐")
        rtk_ang_vel_mag = calculate_angular_velocity_from_rtk(rtk_data, args)
        print("使用从RTK数据计算的角速度幅值")
    
    # 归一化处理，使信号有相似的尺度
    imu_norm = (imu_ang_vel_mag - imu_ang_vel_mag.mean()) / imu_ang_vel_mag.std()
    odo_norm = (odo_ang_vel_mag - odo_ang_vel_mag.mean()) / odo_ang_vel_mag.std()
    if rtk_data is not None and args.align_rtk:
        rtk_norm = (rtk_ang_vel_mag - rtk_ang_vel_mag.mean()) / rtk_ang_vel_mag.std()
    
    # 计算采样间隔
    imu_ts = imu_data[args.imu_timestamp_col].values
    odo_ts = odo_data[args.odo_timestamp_col].values
    imu_dt = np.median(np.diff(imu_ts))
    odo_dt = np.median(np.diff(odo_ts))
    
    print(f"IMU采样间隔: {imu_dt:.2f} ms")
    print(f"编码器采样间隔: {odo_dt:.2f} ms")
    
    if rtk_data is not None and args.align_rtk:
        rtk_ts = rtk_data[args.rtk_timestamp_col].values
        rtk_dt = np.median(np.diff(rtk_ts))
        print(f"RTK采样间隔: {rtk_dt:.2f} ms")
    
    # 滑动窗口计算互相关
    window_size = args.window_size
    search_step = args.search_step_size
    max_lag = args.max_lag
    
    # IMU-轮式编码器对齐
    print("开始IMU-轮式编码器对齐...")
    imu_to_odo_result = find_offset_between_signals(
        imu_norm, odo_norm, imu_ts, odo_ts, window_size, search_step, max_lag
    )
    
    # 如果有RTK数据，进行IMU-RTK对齐
    imu_to_rtk_result = None
    if rtk_data is not None and args.align_rtk:
        print("开始IMU-RTK对齐...")
        imu_to_rtk_result = find_offset_between_signals(
            imu_norm, rtk_norm, imu_ts, rtk_ts, window_size, search_step, max_lag
        )
    
    # 整合结果
    results = {
        'imu_ang_vel_mag': imu_ang_vel_mag,
        'odo_ang_vel_mag': odo_ang_vel_mag,
        'imu_norm': imu_norm,
        'odo_norm': odo_norm,
        'imu_to_odo_offset_ms': imu_to_odo_result['offset_ms'],
        'all_odo_offsets': imu_to_odo_result['all_offsets'],
    }
    
    if rtk_data is not None and args.align_rtk:
        results.update({
            'rtk_ang_vel_mag': rtk_ang_vel_mag,
            'rtk_norm': rtk_norm,
            'imu_to_rtk_offset_ms': imu_to_rtk_result['offset_ms'],
            'all_rtk_offsets': imu_to_rtk_result['all_offsets'],
        })
    
    return results

def find_offset_between_signals(signal1, signal2, ts1, ts2, window_size, search_step, max_lag):
    """计算两个信号之间的时间偏移"""
    best_lag = 0
    best_corr = -np.inf
    best_offset_ms = 0
    
    # 存储所有窗口的互相关结果
    all_corrs = []
    all_lags = []
    all_offsets = []
    
    # 计算时间间隔（使用第一个信号的采样间隔作为基准）
    dt = np.median(np.diff(ts1))
    
    # 为了效率，只在整个时间序列的一部分上进行搜索
    num_windows = min(len(signal1), len(signal2)) // window_size - 1
    
    for i in range(0, num_windows):
        start_idx = i * search_step
        if start_idx + window_size > min(len(signal1), len(signal2)):
            break
            
        window1 = signal1[start_idx:start_idx + window_size].values
        window2 = signal2[start_idx:start_idx + window_size].values
        
        # 计算互相关
        correlation = signal.correlate(window1, window2, mode='full')
        lags = signal.correlation_lags(len(window1), len(window2), mode='full')
        
        # 限制搜索范围在[-max_lag, max_lag]
        valid_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
        valid_corr = correlation[valid_indices]
        valid_lags = lags[valid_indices]
        
        # 找到最大互相关的位置
        max_idx = np.argmax(valid_corr)
        lag = valid_lags[max_idx]
        corr_value = valid_corr[max_idx]
        
        # 转换为时间偏移（毫秒）
        offset_ms = lag * dt
        
        all_corrs.append(corr_value)
        all_lags.append(lag)
        all_offsets.append(offset_ms)
        
        # 更新全局最佳匹配
        if corr_value > best_corr:
            best_corr = corr_value
            best_lag = lag
            best_offset_ms = offset_ms
    
    # 统计所有窗口的结果
    median_offset = np.median(all_offsets)
    mean_offset = np.mean(all_offsets)
    std_offset = np.std(all_offsets)
    
    print(f"最佳时间偏移: {best_offset_ms:.2f} ms (lag = {best_lag})")
    print(f"中位数偏移: {median_offset:.2f} ms")
    print(f"平均偏移: {mean_offset:.2f} ms")
    print(f"标准差: {std_offset:.2f} ms")
    
    # 选择中位数偏移作为最终结果，避免异常值的影响
    final_offset_ms = median_offset
    
    return {
        'offset_ms': final_offset_ms,
        'all_offsets': all_offsets,
        'best_corr': best_corr,
        'best_lag': best_lag
    }

def visualize_alignment(imu_data, odo_data, rtk_data, sync_results, args):
    """可视化时间对齐的结果"""
    print("可视化对齐结果...")
    
    # 提取同步结果
    imu_to_odo_offset_ms = sync_results['imu_to_odo_offset_ms']
    imu_to_rtk_offset_ms = sync_results.get('imu_to_rtk_offset_ms', 0) if rtk_data is not None else 0
    
    imu_ang_vel_mag = sync_results['imu_ang_vel_mag']
    odo_ang_vel_mag = sync_results['odo_ang_vel_mag']
    rtk_ang_vel_mag = sync_results.get('rtk_ang_vel_mag', None)
    
    imu_norm = sync_results['imu_norm']
    odo_norm = sync_results['odo_norm']
    rtk_norm = sync_results.get('rtk_norm', None)
    
    all_odo_offsets = sync_results['all_odo_offsets']
    all_rtk_offsets = sync_results.get('all_rtk_offsets', [])
    
    has_rtk = rtk_data is not None and 'rtk_ang_vel_mag' in sync_results
    
    # 创建图形
    n_rows = 4 if has_rtk else 3
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, n_rows*3))
    
    # 1. 原始角速度幅值
    axs[0].plot(imu_data[args.imu_timestamp_col], imu_ang_vel_mag, 'b-', label='IMU角速度幅值')
    axs[0].plot(odo_data[args.odo_timestamp_col], odo_ang_vel_mag, 'r-', label='轮式编码器角速度幅值')
    if has_rtk:
        axs[0].plot(rtk_data[args.rtk_timestamp_col], rtk_ang_vel_mag, 'g-', label='RTK角速度幅值')
    axs[0].set_title('原始角速度幅值')
    axs[0].set_xlabel('时间戳')
    axs[0].set_ylabel('角速度幅值')
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. IMU和轮式编码器的归一化角速度幅值
    # 使用修正后的时间戳绘制轮式编码器数据
    odo_time_adjusted = odo_data[args.odo_timestamp_col] + imu_to_odo_offset_ms
    
    # 选择一个更小的窗口来显示细节
    window_start = len(imu_norm) // 4
    window_end = min(window_start + 1000, len(imu_norm))
    
    axs[1].plot(imu_data[args.imu_timestamp_col][window_start:window_end], 
               imu_norm[window_start:window_end], 'b-', label='IMU (归一化)')
    axs[1].plot(odo_time_adjusted[window_start:window_end], 
               odo_norm[window_start:window_end], 'r-', label='轮式编码器 (归一化, 已对齐)')
    axs[1].set_title(f'IMU-轮式编码器归一化角速度幅值 (时间偏移 = {imu_to_odo_offset_ms:.2f} ms)')
    axs[1].set_xlabel('时间戳')
    axs[1].set_ylabel('归一化角速度')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. 轮式编码器偏移直方图
    axs[2].hist(all_odo_offsets, bins=30, color='red', alpha=0.7)
    axs[2].axvline(x=imu_to_odo_offset_ms, color='r', linestyle='--', 
                   label=f'IMU-编码器偏移 = {imu_to_odo_offset_ms:.2f} ms')
    axs[2].set_title('IMU-轮式编码器窗口时间偏移分布')
    axs[2].set_xlabel('时间偏移 (ms)')
    axs[2].set_ylabel('窗口数')
    axs[2].legend()
    axs[2].grid(True)
    
    # 4. 如果有RTK数据，显示IMU-RTK的对齐
    if has_rtk:
        # RTK时间调整
        rtk_time_adjusted = rtk_data[args.rtk_timestamp_col] + imu_to_rtk_offset_ms
        
        # IMU-RTK归一化对比
        axs[3].plot(imu_data[args.imu_timestamp_col][window_start:window_end], 
                   imu_norm[window_start:window_end], 'b-', label='IMU (归一化)')
        axs[3].plot(rtk_time_adjusted[window_start:window_end], 
                   rtk_norm[window_start:window_end], 'g-', label='RTK (归一化, 已对齐)')
        axs[3].set_title(f'IMU-RTK归一化角速度幅值 (时间偏移 = {imu_to_rtk_offset_ms:.2f} ms)')
        axs[3].set_xlabel('时间戳')
        axs[3].set_ylabel('归一化角速度')
        axs[3].legend()
        axs[3].grid(True)
        
        # 添加RTK偏移直方图
        if len(all_rtk_offsets) > 0:
            ax_rtk_hist = axs[3].twinx()
            ax_rtk_hist.hist(all_rtk_offsets, bins=30, color='green', alpha=0.3)
            ax_rtk_hist.axvline(x=imu_to_rtk_offset_ms, color='g', linestyle='--',
                               label=f'IMU-RTK偏移 = {imu_to_rtk_offset_ms:.2f} ms')
            ax_rtk_hist.set_ylabel('RTK窗口数', color='g')
            ax_rtk_hist.tick_params(axis='y', labelcolor='g')
    
    plt.tight_layout()
    
    # 保存图形到文件
    plt.savefig("time_alignment_results.png", dpi=300)
    print("图形已保存到 time_alignment_results.png")
    
    # 如果要求显示图形
    if args.visualize:
        plt.show()

def align_and_merge_data(imu_data, odo_data, rtk_data, sync_results, args):
    """根据同步结果对齐并合并多个数据源"""
    print("对齐并合并数据...")
    
    # 获取时间偏移（毫秒）
    imu_to_odo_offset_ms = sync_results['imu_to_odo_offset_ms']
    imu_to_rtk_offset_ms = sync_results.get('imu_to_rtk_offset_ms', 0) if rtk_data is not None else 0
    
    # 复制数据，避免修改原始数据
    imu_aligned = imu_data.copy()
    odo_aligned = odo_data.copy()
    rtk_aligned = rtk_data.copy() if rtk_data is not None else None
    
    # 调整时间戳（以IMU为基准）
    odo_aligned[args.odo_timestamp_col] = odo_aligned[args.odo_timestamp_col] + imu_to_odo_offset_ms
    if rtk_aligned is not None:
        rtk_aligned[args.rtk_timestamp_col] = rtk_aligned[args.rtk_timestamp_col] + imu_to_rtk_offset_ms
    
    # 将数据集转换为以时间戳为索引的时间序列
    imu_ts = imu_aligned.set_index(args.imu_timestamp_col)
    odo_ts = odo_aligned.set_index(args.odo_timestamp_col)
    rtk_ts = rtk_aligned.set_index(args.rtk_timestamp_col) if rtk_aligned is not None else None
    
    # 获取最小和最大共同时间戳
    min_timestamps = [imu_ts.index.min(), odo_ts.index.min()]
    max_timestamps = [imu_ts.index.max(), odo_ts.index.max()]
    
    if rtk_ts is not None:
        min_timestamps.append(rtk_ts.index.min())
        max_timestamps.append(rtk_ts.index.max())
    
    max_start = max(min_timestamps)
    min_end = min(max_timestamps)
    
    # 创建一个共同的时间索引 (每10ms采样一次)
    sample_freq_ms = 10  # 10ms
    common_timepoints = np.arange(max_start, min_end, sample_freq_ms)
    
    # 使用最近插值重采样数据集
    imu_resampled = imu_ts.reindex(
        index=imu_ts.index.union(common_timepoints)
    ).interpolate(method='nearest').reindex(common_timepoints)
    
    odo_resampled = odo_ts.reindex(
        index=odo_ts.index.union(common_timepoints)
    ).interpolate(method='nearest').reindex(common_timepoints)
    
    if rtk_ts is not None:
        rtk_resampled = rtk_ts.reindex(
            index=rtk_ts.index.union(common_timepoints)
        ).interpolate(method='nearest').reindex(common_timepoints)
    else:
        rtk_resampled = None
    
    # 重置索引，将时间戳作为列
    imu_resampled = imu_resampled.reset_index()
    odo_resampled = odo_resampled.reset_index()
    if rtk_resampled is not None:
        rtk_resampled = rtk_resampled.reset_index()
    
    # 重命名列，以避免重名
    imu_columns = {col: f'imu_{col}' for col in imu_resampled.columns if col != args.imu_timestamp_col}
    imu_resampled = imu_resampled.rename(columns=imu_columns)
    
    odo_columns = {col: f'odo_{col}' for col in odo_resampled.columns if col != args.odo_timestamp_col}
    odo_resampled = odo_resampled.rename(columns=odo_columns)
    
    # 合并IMU和轮式编码器数据
    merged_data = pd.merge(
        imu_resampled,
        odo_resampled,
        left_on=args.imu_timestamp_col,
        right_on=args.odo_timestamp_col,
        how='inner'
    )
    
    # 删除重复的时间戳列
    if args.imu_timestamp_col != args.odo_timestamp_col:
        merged_data = merged_data.drop(columns=[args.odo_timestamp_col])
    
    # 如果有RTK数据，合并它
    if rtk_resampled is not None:
        rtk_columns = {col: f'rtk_{col}' for col in rtk_resampled.columns if col != args.rtk_timestamp_col}
        rtk_resampled = rtk_resampled.rename(columns=rtk_columns)
        
        merged_data = pd.merge(
            merged_data,
            rtk_resampled,
            left_on=args.imu_timestamp_col,
            right_on=args.rtk_timestamp_col,
            how='inner'
        )
        
        # 删除重复的时间戳列
        if args.imu_timestamp_col != args.rtk_timestamp_col:
            merged_data = merged_data.drop(columns=[args.rtk_timestamp_col])
    
    # 添加相关元数据
    merged_data.attrs['imu_to_odo_offset_ms'] = imu_to_odo_offset_ms
    if rtk_data is not None:
        merged_data.attrs['imu_to_rtk_offset_ms'] = imu_to_rtk_offset_ms
    
    return merged_data

def main():
    """
    主函数
    此脚本使用IMU角速度数据、从轮式编码器计数(left_count, right_count)计算的角速度幅值，
    以及可选的RTK数据进行时间对齐。
    不使用speed_w和speed_v，而是直接从编码器计数差分结合轮子物理参数计算角速度。
    """
    args = parse_arguments()
    
    # 加载数据
    imu_data = load_imu_data(args.imu_file)
    odo_data = load_odometer_data(args.odo_file)
    
    print(f"IMU数据: {len(imu_data)} 行")
    print(f"轮式编码器数据: {len(odo_data)} 行")
    
    # 如果提供了RTK数据，加载它
    rtk_data = None
    if args.rtk_file and args.align_rtk:
        rtk_data = load_rtk_data(args.rtk_file)
        print(f"RTK数据: {len(rtk_data)} 行")
    
    # 执行时间同步
    sync_results = synchronize_using_angular_velocity(imu_data, odo_data, rtk_data, args)
    
    # 可视化对齐结果
    visualize_alignment(imu_data, odo_data, rtk_data, sync_results, args)
    
    # 对齐并合并数据
    merged_data = align_and_merge_data(imu_data, odo_data, rtk_data, sync_results, args)
    
    # 保存结果
    merged_data.to_csv(args.output_file, index=False)
    print(f"已将对齐的数据保存到 {args.output_file}")
    
    # 打印摘要
    print("\n==== 时间对齐摘要 ====")
    print(f"IMU数据文件: {args.imu_file}")
    print(f"轮式编码器数据文件: {args.odo_file}")
    print(f"IMU-轮式编码器时间偏移: {sync_results['imu_to_odo_offset_ms']:.2f} ms")
    
    if rtk_data is not None and args.align_rtk:
        print(f"RTK数据文件: {args.rtk_file}")
        print(f"IMU-RTK时间偏移: {sync_results['imu_to_rtk_offset_ms']:.2f} ms")
    
    print(f"合并后的数据行数: {len(merged_data)}")
    print(f"输出文件: {args.output_file}")
    print("====================\n")

if __name__ == "__main__":
    main() 


# python imu_odo_time_align.py --imu_file /path/to/imu_data.csv \
#                              --odo_file /path/to/odo_data.csv \
#                              --rtk_file /path/to/rtk_data.csv \
#                              --align_rtk \
#                              --visualize