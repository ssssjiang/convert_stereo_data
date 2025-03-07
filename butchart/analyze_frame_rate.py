#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os
import glob

def analyze_frame_rate(log_file=None, image_dir=None, plot=True, save_drops=True):
    """
    分析IMU、rawgyroodo数据以及图像的帧率和丢帧情况
    
    Args:
        log_file: 日志文件路径，可选
        image_dir: 图像目录路径，可选
        plot: 是否绘制和保存图像
        save_drops: 是否保存丢帧数据到CSV文件
    """
    # 设置matplotlib使用不依赖中文字体的配置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置最大打开图形数量的警告阈值
    plt.rcParams['figure.max_open_warning'] = 50
    
    # 存储时间戳
    imu_timestamps = []
    rawgyroodo_timestamps = []
    
    # 读取日志文件
    if log_file:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                try:
                    timestamp = int(parts[0])
                    data_type = parts[1]
                    
                    if data_type == 'imu':
                        imu_timestamps.append(timestamp)
                    elif data_type == 'rawgyroodo':
                        rawgyroodo_timestamps.append(timestamp)
                except (ValueError, IndexError):
                    continue
    
    # 计算帧率统计
    def calculate_stats(diffs, name, timestamps=None, plot=plot, save_drops=save_drops):
        if len(diffs) == 0:
            print(f"没有找到{name}数据")
            return None, None
            
        # 使用ASCII字符替代中文字符，避免字体问题
        safe_name = name.replace('段', 'segment')
            
        # 计算平均帧间隔（毫秒）
        mean_diff = np.mean(diffs)
        median_diff = np.median(diffs)
        std_diff = np.std(diffs)
        
        # 计算帧率（假设时间戳单位是毫秒）
        mean_fps = 1000 / mean_diff if mean_diff > 0 else 0
        
        print(f"\n{name}数据分析:")
        print(f"总帧数: {len(diffs) + 1}")
        print(f"平均帧间隔: {mean_diff:.2f} 时间单位")
        print(f"中位数帧间隔: {median_diff:.2f} 时间单位")
        print(f"帧间隔标准差: {std_diff:.2f} 时间单位")
        print(f"估计帧率: {mean_fps:.2f} Hz (假设时间单位为毫秒)")
        
        # 检测丢帧
        # 假设正常帧间隔是中位数的值，如果某个间隔超过中位数的1.5倍，则认为可能丢帧
        threshold = median_diff * 1.5
        dropped_indices = np.where(diffs > threshold)[0]
        
        # 增强的丢帧分析
        if len(dropped_indices) > 0:
            print(f"\n检测到可能的丢帧:")
            print(f"丢帧阈值: {threshold:.2f} 时间单位 (中位数的1.5倍)")
            print(f"可能丢帧的数量: {len(dropped_indices)}")
            
            # 找出极大值（最严重的丢帧）
            extreme_threshold = median_diff * 3  # 定义极大值为中位数的3倍
            extreme_indices = np.where(diffs > extreme_threshold)[0]
            
            if len(extreme_indices) > 0:
                max_diff_idx = np.argmax(diffs)
                max_diff = diffs[max_diff_idx]
                estimated_max_lost = int(max_diff / median_diff) - 1
                
                print(f"\n极大值丢帧分析:")
                print(f"极大值阈值: {extreme_threshold:.2f} 时间单位 (中位数的3倍)")
                print(f"极大值丢帧数量: {len(extreme_indices)}")
                print(f"最大帧间隔: {max_diff:.2f} 时间单位，估计丢失 {estimated_max_lost} 帧")
                
                # 显示前5个极大值丢帧
                print(f"\n前5个极大值丢帧:")
                sorted_extreme_indices = sorted(extreme_indices, key=lambda idx: diffs[idx], reverse=True)
                for i, idx in enumerate(sorted_extreme_indices[:5]):
                    if timestamps is not None:
                        ts1 = timestamps[idx]
                        ts2 = timestamps[idx + 1]
                        diff = diffs[idx]
                        estimated_lost = int(diff / median_diff) - 1
                        print(f"  极大值丢帧 {i+1}: 时间戳 {ts1} -> {ts2}, 间隔: {diff:.2f}, 估计丢失 {estimated_lost} 帧")
                    else:
                        diff = diffs[idx]
                        estimated_lost = int(diff / median_diff) - 1
                        print(f"  极大值丢帧 {i+1}: 索引 {idx}, 间隔: {diff:.2f}, 估计丢失 {estimated_lost} 帧")
            
            # 显示前10个丢帧情况
            for i, idx in enumerate(dropped_indices[:10]):
                if timestamps is not None:
                    ts1 = timestamps[idx]
                    ts2 = timestamps[idx + 1]
                    diff = diffs[idx]
                    estimated_lost = int(diff / median_diff) - 1
                    print(f"  丢帧 {i+1}: 时间戳 {ts1} -> {ts2}, 间隔: {diff:.2f}, 估计丢失 {estimated_lost} 帧")
                else:
                    diff = diffs[idx]
                    estimated_lost = int(diff / median_diff) - 1
                    print(f"  丢帧 {i+1}: 索引 {idx}, 间隔: {diff:.2f}, 估计丢失 {estimated_lost} 帧")
            
            if len(dropped_indices) > 10:
                print(f"  ... 还有 {len(dropped_indices) - 10} 个可能的丢帧 ...")
            
            # 将所有丢帧信息保存到文件
            drop_file_name = f'{safe_name}_frame_drops.csv'
            print(f"\n保存所有丢帧信息到文件: {drop_file_name}")
            
            if save_drops:
                with open(drop_file_name, 'w') as f:
                    # 写入CSV头
                    if timestamps is not None:
                        f.write("索引,时间戳1,时间戳2,帧间隔,估计丢失帧数,是否极大值\n")
                    else:
                        f.write("索引,帧间隔,估计丢失帧数,是否极大值\n")
                    
                    # 写入所有丢帧数据
                    for idx in dropped_indices:
                        diff = diffs[idx]
                        estimated_lost = int(diff / median_diff) - 1
                        is_extreme = "是" if diff > extreme_threshold else "否"
                        
                        if timestamps is not None:
                            ts1 = timestamps[idx]
                            ts2 = timestamps[idx + 1]
                            f.write(f"{idx},{ts1},{ts2},{diff:.2f},{estimated_lost},{is_extreme}\n")
                        else:
                            f.write(f"{idx},{diff:.2f},{estimated_lost},{is_extreme}\n")
            else:
                print("已禁用丢帧数据保存，跳过CSV文件生成")
            
            # 计算丢帧统计信息
            drop_diffs = diffs[dropped_indices]
            mean_drop_diff = np.mean(drop_diffs)
            median_drop_diff = np.median(drop_diffs)
            max_drop_diff = np.max(drop_diffs)
            min_drop_diff = np.min(drop_diffs)
            total_estimated_lost = sum(int(diff / median_diff) - 1 for diff in drop_diffs)
            
            # 计算丢帧比例
            drop_ratio = len(dropped_indices) / len(diffs) * 100
            estimated_total_frames = len(diffs) + 1 + total_estimated_lost
            estimated_drop_ratio = total_estimated_lost / estimated_total_frames * 100
            
            print(f"\n丢帧统计信息:")
            print(f"平均丢帧间隔: {mean_drop_diff:.2f} 时间单位")
            print(f"中位数丢帧间隔: {median_drop_diff:.2f} 时间单位")
            print(f"最大丢帧间隔: {max_drop_diff:.2f} 时间单位")
            print(f"最小丢帧间隔: {min_drop_diff:.2f} 时间单位")
            print(f"估计总丢失帧数: {total_estimated_lost} 帧")
            print(f"丢帧事件比例: {drop_ratio:.2f}% ({len(dropped_indices)}/{len(diffs)})")
            print(f"估计丢失帧比例: {estimated_drop_ratio:.2f}% ({total_estimated_lost}/{estimated_total_frames})")
            
            # 计算丢帧分布
            if len(dropped_indices) > 1:
                drop_intervals = np.diff(dropped_indices)
                mean_drop_interval = np.mean(drop_intervals)
                print(f"丢帧之间的平均间隔: {mean_drop_interval:.2f} 帧")
                
                # 检查丢帧是否集中在某些区域
                if np.std(drop_intervals) > mean_drop_interval:
                    print("丢帧分布不均匀，可能集中在某些区域")
                else:
                    print("丢帧分布相对均匀")
        
        # 绘制帧间隔直方图
        if plot:
            plt.figure(figsize=(12, 6))
            plt.hist(diffs, bins=50, alpha=0.7)
            plt.axvline(median_diff, color='g', linestyle='--', label=f'Median: {median_diff:.2f}')
            plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
            if len(dropped_indices) > 0:
                plt.axvline(extreme_threshold, color='m', linestyle='--', label=f'Extreme: {extreme_threshold:.2f}')
            # 使用ASCII字符替代中文字符，避免字体问题
            safe_name = name.replace('段', 'segment')
            plt.title(f'{safe_name} Frame Interval Distribution')
            plt.xlabel('Frame Interval (time units)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{safe_name}_frame_intervals.png')
            plt.close()  # 关闭图形，释放内存
        
        # 绘制帧间隔随时间变化图
        if plot:
            plt.figure(figsize=(12, 6))
            if len(dropped_indices) > 0:
                if timestamps is not None:
                    plt.plot(timestamps[:-1], diffs, 'b-', alpha=0.5)
                    plt.scatter(timestamps[dropped_indices], diffs[dropped_indices], color='r', s=30, label='Possible Drops')
                    # 标记极大值
                    if len(extreme_indices) > 0:
                        plt.scatter(timestamps[extreme_indices], diffs[extreme_indices], color='m', s=50, marker='*', label='Extreme Drops')
                else:
                    plt.plot(diffs, 'b-', alpha=0.5)
                    plt.scatter(dropped_indices, diffs[dropped_indices], color='r', s=30, label='Possible Drops')
                    # 标记极大值
                    if len(extreme_indices) > 0:
                        plt.scatter(extreme_indices, diffs[extreme_indices], color='m', s=50, marker='*', label='Extreme Drops')
            else:
                if timestamps is not None:
                    plt.plot(timestamps[:-1], diffs, 'b-', alpha=0.5)
                else:
                    plt.plot(diffs, 'b-', alpha=0.5)
            
            plt.axhline(median_diff, color='g', linestyle='--', label=f'Median: {median_diff:.2f}')
            plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
            if len(dropped_indices) > 0:
                plt.axhline(extreme_threshold, color='m', linestyle='--', label=f'Extreme: {extreme_threshold:.2f}')
            plt.title(f'{safe_name} Frame Interval vs Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Frame Interval (time units)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{safe_name}_frame_intervals_time.png')
            plt.close()  # 关闭图形，释放内存
        
        return median_diff, mean_fps
    
    # 提取所有数据段
    print("\n提取传感器数据段:")
    print("-" * 50)
    
    if log_file:
        print("IMU数据:")
        imu_segments = extract_segments(imu_timestamps)
        imu_segment_count = sum(len(segment) for segment in imu_segments)
        
        print("\nrawgyroodo数据:")
        rawgyroodo_segments = extract_segments(rawgyroodo_timestamps)
        rawgyroodo_segment_count = sum(len(segment) for segment in rawgyroodo_segments)
        
        # 分析每个数据段
        for i, (imu_segment, rawgyroodo_segment) in enumerate(zip(imu_segments, rawgyroodo_segments)):
            print(f"\n{'='*50}")
            print(f"分析数据段 {i+1}/{len(imu_segments)}")
            print(f"{'='*50}")
            
            # 转换为numpy数组
            imu_timestamps_segment = np.array(imu_segment)
            rawgyroodo_timestamps_segment = np.array(rawgyroodo_segment)
            
            # 计算时间差
            imu_diffs = np.diff(imu_timestamps_segment)
            rawgyroodo_diffs = np.diff(rawgyroodo_timestamps_segment)
            
            # 分析IMU和rawgyroodo数据
            print(f"\n分析数据段 {i+1} 的IMU数据:")
            imu_median_diff, imu_mean_fps = calculate_stats(imu_diffs, f'IMU_段{i+1}', imu_timestamps_segment, plot, save_drops)
            
            print(f"\n分析数据段 {i+1} 的rawgyroodo数据:")
            rawgyroodo_median_diff, rawgyroodo_mean_fps = calculate_stats(rawgyroodo_diffs, f'rawgyroodo_段{i+1}', rawgyroodo_timestamps_segment, plot, save_drops)
            
            # 分析IMU和rawgyroodo的时间同步情况
            print(f"\n分析数据段 {i+1} 的IMU和rawgyroodo的时间同步情况:")
            
            # 找出共同的时间范围
            min_time = max(min(imu_timestamps_segment), min(rawgyroodo_timestamps_segment))
            max_time = min(max(imu_timestamps_segment), max(rawgyroodo_timestamps_segment))
            
            print(f"共同时间范围: {min_time} - {max_time}")
            
            # 计算每个时间戳对应的另一个传感器最近的时间戳
            imu_in_range = imu_timestamps_segment[(imu_timestamps_segment >= min_time) & (imu_timestamps_segment <= max_time)]
            rawgyroodo_in_range = rawgyroodo_timestamps_segment[(rawgyroodo_timestamps_segment >= min_time) & (rawgyroodo_timestamps_segment <= max_time)]
            
            # 计算两个传感器的时间戳比例
            imu_count = len(imu_in_range)
            rawgyroodo_count = len(rawgyroodo_in_range)
            ratio = imu_count / rawgyroodo_count if rawgyroodo_count > 0 else 0
            
            print(f"IMU帧数: {imu_count}")
            print(f"rawgyroodo帧数: {rawgyroodo_count}")
            print(f"IMU/rawgyroodo帧数比例: {ratio:.2f}")
            
            # 如果比例接近1，可能是同步的
            if 0.9 <= ratio <= 1.1:
                print("IMU和rawgyroodo可能是同步的 (帧数比例接近1)")
            else:
                print("IMU和rawgyroodo可能不是同步的 (帧数比例不接近1)")
    else:
        imu_segments = []
        rawgyroodo_segments = []
        imu_segment_count = 0
        rawgyroodo_segment_count = 0
    
    # 分析图像时间戳
    camera0_segment_count = 0
    camera1_segment_count = 0
    
    if image_dir:
        if log_file:
            camera0_segment_count, camera1_segment_count = analyze_image_timestamps(image_dir, 
                                                                                       np.concatenate(imu_segments), 
                                                                                       np.concatenate(rawgyroodo_segments), 
                                                                                       calculate_stats,
                                                                                       plot,
                                                                                       save_drops)
        else:
            camera0_segment_count, camera1_segment_count = analyze_image_timestamps(image_dir, 
                                                                                       None, 
                                                                                       None, 
                                                                                       calculate_stats,
                                                                                       plot,
                                                                                       save_drops)
    
    # 打印最终汇总信息
    print("\n" + "="*50)
    print("所有数据段帧数汇总:")
    print("-"*50)
    if log_file:
        print(f"IMU数据:        {imu_segment_count} 帧")
        print(f"rawgyroodo数据: {rawgyroodo_segment_count} 帧")
    if image_dir:
        print(f"Camera0图像:    {camera0_segment_count} 帧")
        print(f"Camera1图像:    {camera1_segment_count} 帧")
    print("="*50)

def analyze_image_timestamps(image_dir, imu_timestamps=None, rawgyroodo_timestamps=None, calculate_stats=None, plot=True, save_drops=True):
    """
    分析图像时间戳的帧率和丢帧情况
    
    Args:
        image_dir: 图像目录路径
        imu_timestamps: IMU时间戳数组，用于比较
        rawgyroodo_timestamps: rawgyroodo时间戳数组，用于比较
        calculate_stats: 用于分析帧率的函数
        plot: 是否绘制和保存图像
        save_drops: 是否保存丢帧数据到CSV文件
        
    Returns:
        (camera0_count, camera1_count): 两个相机的帧数
    """
    print(f"\n分析图像时间戳:")
    
    # 检查camera0和camera1目录
    camera0_dir = os.path.join(image_dir, 'camera0')
    camera1_dir = os.path.join(image_dir, 'camera1')
    
    # 获取图像文件列表
    camera0_files = []
    camera1_files = []
    
    if os.path.exists(camera0_dir):
        camera0_files = sorted(glob.glob(os.path.join(camera0_dir, '*')))
        print(f"找到camera0图像: {len(camera0_files)}张")
    else:
        print(f"未找到camera0目录: {camera0_dir}")
    
    if os.path.exists(camera1_dir):
        camera1_files = sorted(glob.glob(os.path.join(camera1_dir, '*')))
        print(f"找到camera1图像: {len(camera1_files)}张")
    else:
        print(f"未找到camera1目录: {camera1_dir}")
    
    # 提取时间戳
    def extract_timestamps(files):
        timestamps = []
        for file_path in files:
            try:
                # 从文件名中提取时间戳
                filename = os.path.basename(file_path)
                # 假设文件名就是时间戳，移除扩展名
                timestamp = int(os.path.splitext(filename)[0])
                timestamps.append(timestamp)
            except (ValueError, IndexError) as e:
                print(f"无法从文件名提取时间戳: {file_path}, 错误: {str(e)}")
        
        if timestamps:
            # 对时间戳进行排序，确保严格递增
            print("对相机时间戳进行排序，确保严格递增...")
            timestamps = sorted(timestamps)
            
            # 移除重复的时间戳
            unique_timestamps = np.unique(timestamps)
            if len(unique_timestamps) < len(timestamps):
                print(f"移除了 {len(timestamps) - len(unique_timestamps)} 个重复的时间戳")
                timestamps = unique_timestamps
            
            timestamps = np.array(timestamps)
            
            # 打印时间戳范围和数量
            if len(timestamps) > 0:
                print(f"时间戳范围: {min(timestamps)} - {max(timestamps)}")
                if len(timestamps) > 1:
                    diffs = np.diff(timestamps)
                    print(f"平均帧间隔: {np.mean(diffs):.2f}")
                    print(f"中位数帧间隔: {np.median(diffs):.2f}")
                    print(f"最小帧间隔: {np.min(diffs)}")
                    print(f"最大帧间隔: {np.max(diffs)}")
        else:
            print("没有提取到有效的时间戳")
            timestamps = np.array([])
            
        return timestamps
    
    # 初始化相机帧数
    camera0_count = 0
    camera1_count = 0
    
    # 提取相机时间戳（只提取一次）
    camera0_timestamps = np.array([])
    camera1_timestamps = np.array([])
    
    if camera0_files:
        camera0_timestamps = extract_timestamps(camera0_files)
        camera0_count = len(camera0_timestamps)
    
    if camera1_files:
        camera1_timestamps = extract_timestamps(camera1_files)
        camera1_count = len(camera1_timestamps)
    
    # 分析camera0
    if len(camera0_timestamps) > 1:
        camera0_diffs = np.diff(camera0_timestamps)
        print("\n分析Camera0时间戳:")
        median_diff, frame_rate = calculate_stats(camera0_diffs, 'Camera0', camera0_timestamps, plot, save_drops)
        
        # 与IMU和rawgyroodo比较
        if imu_timestamps is not None and len(imu_timestamps) > 0:
            compare_timestamps(camera0_timestamps, imu_timestamps, 'Camera0', 'IMU', plot)
        
        if rawgyroodo_timestamps is not None and len(rawgyroodo_timestamps) > 0:
            compare_timestamps(camera0_timestamps, rawgyroodo_timestamps, 'Camera0', 'rawgyroodo', plot)
    elif len(camera0_timestamps) > 0:
        print("Camera0图像数量不足，无法分析帧率")
    
    # 分析camera1
    if len(camera1_timestamps) > 1:
        camera1_diffs = np.diff(camera1_timestamps)
        print("\n分析Camera1时间戳:")
        median_diff, frame_rate = calculate_stats(camera1_diffs, 'Camera1', camera1_timestamps, plot, save_drops)
        
        # 与IMU和rawgyroodo比较
        if imu_timestamps is not None and len(imu_timestamps) > 0:
            compare_timestamps(camera1_timestamps, imu_timestamps, 'Camera1', 'IMU', plot)
        
        if rawgyroodo_timestamps is not None and len(rawgyroodo_timestamps) > 0:
            compare_timestamps(camera1_timestamps, rawgyroodo_timestamps, 'Camera1', 'rawgyroodo', plot)
    elif len(camera1_timestamps) > 0:
        print("Camera1图像数量不足，无法分析帧率")
    
    # 比较camera0和camera1
    if len(camera0_timestamps) > 0 and len(camera1_timestamps) > 0:
        compare_timestamps(camera0_timestamps, camera1_timestamps, 'Camera0', 'Camera1', plot)
        
        # 绘制两个相机的时间戳对比图
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(range(min(100, len(camera0_timestamps))), camera0_timestamps[:100], label='Camera0')
            plt.plot(range(min(100, len(camera1_timestamps))), camera1_timestamps[:100], label='Camera1')
            plt.title('Camera0 vs Camera1 Timestamp Comparison')
            plt.xlabel('Frame Index')
            plt.ylabel('Timestamp')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('camera0_camera1_timestamps.png')
            plt.close()  # 关闭图形，释放内存
    
    # 突出显示相机帧数
    print(f"\n{'='*50}")
    print(f"Camera0 包含 {camera0_count} 帧图像")
    print(f"Camera1 包含 {camera1_count} 帧图像")
    print(f"{'='*50}\n")
    
    return camera0_count, camera1_count

def compare_timestamps(timestamps1, timestamps2, name1, name2, plot=True):
    """
    比较两组时间戳的同步情况
    
    Args:
        timestamps1: 第一组时间戳
        timestamps2: 第二组时间戳
        name1: 第一组时间戳的名称
        name2: 第二组时间戳的名称
        plot: 是否绘制和保存图像
    """
    print(f"\n比较{name1}和{name2}时间戳:")
    
    # 找出共同的时间范围
    min_time = max(min(timestamps1), min(timestamps2))
    max_time = min(max(timestamps1), max(timestamps2))
    
    print(f"共同时间范围: {min_time} - {max_time}")
    
    # 计算在共同时间范围内的时间戳数量
    ts1_in_range = timestamps1[(timestamps1 >= min_time) & (timestamps1 <= max_time)]
    ts2_in_range = timestamps2[(timestamps2 >= min_time) & (timestamps2 <= max_time)]
    
    # 计算两个传感器的时间戳比例
    count1 = len(ts1_in_range)
    count2 = len(ts2_in_range)
    ratio = count1 / count2 if count2 > 0 else 0
    
    print(f"{name1}帧数: {count1}")
    print(f"{name2}帧数: {count2}")
    print(f"{name1}/{name2}帧数比例: {ratio:.2f}")
    
    # 如果比例接近整数或整数倍数，可能是同步的
    if 0.9 <= ratio <= 1.1:
        print(f"{name1}和{name2}可能是1:1同步的")
    elif 1.9 <= ratio <= 2.1:
        print(f"{name1}和{name2}可能是2:1同步的")
    elif 0.45 <= ratio <= 0.55:
        print(f"{name1}和{name2}可能是1:2同步的")
    else:
        print(f"{name1}和{name2}可能不是简单的同步关系")
    
    # 检查时间戳的对齐情况
    # 取前100个时间戳进行分析
    limit = min(100, len(ts1_in_range), len(ts2_in_range))
    
    if limit > 0:
        # 找到最接近的时间戳对
        closest_pairs = []
        for ts1 in ts1_in_range[:limit]:
            # 找到ts2中最接近ts1的时间戳
            closest_idx = np.argmin(np.abs(ts2_in_range - ts1))
            closest_ts2 = ts2_in_range[closest_idx]
            closest_pairs.append((ts1, closest_ts2))
        
        # 计算时间差
        time_diffs = [abs(pair[0] - pair[1]) for pair in closest_pairs]
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        print(f"时间戳平均差异: {mean_diff:.2f} 时间单位")
        print(f"差异标准差: {std_diff:.2f} 时间单位")
        
        if std_diff < mean_diff * 0.1:  # 如果标准差小于平均差的10%
            print(f"{name1}和{name2}时间戳差异比较一致，可能是同步采集但有固定延迟")
        else:
            print(f"{name1}和{name2}时间戳差异变化较大，可能不是严格同步的")
        
        # 绘制时间戳差异图
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(time_diffs)), time_diffs, 'b-')
            plt.axhline(mean_diff, color='r', linestyle='--', label=f'Mean Diff: {mean_diff:.2f}')
            # 使用ASCII字符替代中文字符，避免字体问题
            safe_name1 = name1.replace('段', 'segment')
            safe_name2 = name2.replace('段', 'segment')
            plt.title(f'{safe_name1} vs {safe_name2} Timestamp Difference')
            plt.xlabel('Frame Index')
            plt.ylabel('Time Difference (time units)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{safe_name1}_{safe_name2}_timestamp_diff.png')
            plt.close()  # 关闭图形，释放内存

def extract_segments(timestamps, gap_threshold=10000):
    """
    提取时间戳数据的所有段
    当时间戳之间的间隔大于阈值时，认为是不同的数据段
    
    Args:
        timestamps: 时间戳列表
        gap_threshold: 分段阈值，默认为10000
        
    Returns:
        所有段的时间戳列表
    """
    if not timestamps:
        print("没有找到时间戳数据")
        return []
    
    print("注意: 每个数据段将被排序，确保时间戳严格递增")
    
    # 使用原始时间戳顺序，不进行排序
    timestamps_array = np.array(timestamps)
    
    # 计算时间戳之间的差值
    diffs = np.diff(timestamps_array)
    
    # 找出大于阈值的差值索引，这些是段的边界
    segment_boundaries = np.where(np.abs(diffs) > gap_threshold)[0]
    
    # 输出找到的段数
    total_segments = len(segment_boundaries) + 1
    
    # 如果没有找到段边界，则整个数据就是一段
    if len(segment_boundaries) == 0:
        last_segment = timestamps_array
        print(f"找到 1 个数据段，共 {len(last_segment)} 帧")
        
        # 对段内数据排序，确保严格递增
        last_segment = np.sort(last_segment)
        
        # 移除重复的时间戳
        unique_timestamps = np.unique(last_segment)
        if len(unique_timestamps) < len(last_segment):
            print(f"移除了 {len(last_segment) - len(unique_timestamps)} 个重复的时间戳")
            last_segment = unique_timestamps
            
        return [last_segment]
    else:
        print(f"找到 {total_segments} 个数据段 (间隔阈值: {gap_threshold})")
        
        # 输出各段的信息，简化输出
        segments_info = []
        segments = []
        
        # 第一段
        first_segment_end = segment_boundaries[0]
        first_segment = timestamps_array[:first_segment_end+1]
        
        # 对段内数据排序，确保严格递增
        first_segment = np.sort(first_segment)
        
        # 移除重复的时间戳
        unique_timestamps = np.unique(first_segment)
        if len(unique_timestamps) < len(first_segment):
            print(f"段1: 移除了 {len(first_segment) - len(unique_timestamps)} 个重复的时间戳")
            first_segment = unique_timestamps
            
        first_segment_frames = len(first_segment)
        segments.append(first_segment)
        segments_info.append(f"段1: {first_segment_frames}帧")
        
        # 中间段
        for i in range(len(segment_boundaries) - 1):
            segment_start = segment_boundaries[i] + 1
            segment_end = segment_boundaries[i + 1]
            segment = timestamps_array[segment_start:segment_end+1]
            
            # 对段内数据排序，确保严格递增
            segment = np.sort(segment)
            
            # 移除重复的时间戳
            unique_timestamps = np.unique(segment)
            if len(unique_timestamps) < len(segment):
                print(f"段{i+2}: 移除了 {len(segment) - len(unique_timestamps)} 个重复的时间戳")
                segment = unique_timestamps
                
            segment_frames = len(segment)
            segments.append(segment)
            segments_info.append(f"段{i+2}: {segment_frames}帧")
        
        # 最后一段
        last_segment_start = segment_boundaries[-1] + 1
        last_segment = timestamps_array[last_segment_start:]
        
        # 对段内数据排序，确保严格递增
        last_segment = np.sort(last_segment)
        
        # 移除重复的时间戳
        unique_timestamps = np.unique(last_segment)
        if len(unique_timestamps) < len(last_segment):
            print(f"段{total_segments}(最后): 移除了 {len(last_segment) - len(unique_timestamps)} 个重复的时间戳")
            last_segment = unique_timestamps
            
        segments.append(last_segment)
        segments_info.append(f"段{total_segments}(最后): {len(last_segment)}帧")
        
        # 打印段信息
        print("各段帧数: " + ", ".join(segments_info))
        
        # 输出每一段的时间范围
        for i, segment in enumerate(segments):
            if len(segment) > 0:
                start_time = segment[0]
                end_time = segment[-1]
                duration = end_time - start_time
                print(f"段{i+1}: {start_time} - {end_time} (持续: {duration})")
        
        return segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析IMU、rawgyroodo数据和图像的帧率和丢帧情况')
    parser.add_argument('--log_file', help='日志文件路径')
    parser.add_argument('--image_dir', help='图像目录路径，包含camera0和camera1子目录')
    parser.add_argument('--no-plot', action='store_true', help='不绘制和保存图像，仅输出分析结果')
    parser.add_argument('--no-save-drops', action='store_true', help='不保存丢帧数据到CSV文件')
    args = parser.parse_args()
    
    if not args.log_file and not args.image_dir:
        parser.error("至少需要提供 --log_file 或 --image_dir 参数之一")
    
    analyze_frame_rate(args.log_file, args.image_dir, not args.no_plot, not args.no_save_drops)
    
    if args.no_plot:
        print("\n分析完成，未保存图表。")
    else:
        print("\n分析完成，图表已保存。")
    
    # 确保所有图形都已关闭
    plt.close('all') 