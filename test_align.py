#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from scipy import signal
import matplotlib.font_manager as fm
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 尝试查找系统中支持中文的字体
chinese_fonts = []
for font in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'NotoSansCJK-Regular']:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
        if font_path:
            chinese_fonts.append(font)
    except:
        continue

if chinese_fonts:
    # 使用找到的第一个支持中文的字体
    plt.rcParams['font.family'] = chinese_fonts[0]
    print(f"使用字体: {chinese_fonts[0]} 以支持中文显示")
else:
    # 如果没有找到支持中文的字体，则使用英文标签
    print("未找到支持中文的字体，将使用英文标签")

# 配置字体回退
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [chinese_fonts[0], 'DejaVu Sans', 'Arial']

def generate_synthetic_data(known_offset_ms=150, duration_sec=60, imu_rate_hz=100, odo_rate_hz=20):
    """
    生成具有已知时间偏移的IMU和轮式编码器合成数据
    
    参数:
        known_offset_ms: IMU相对于ODO的已知时间偏移(毫秒)，正值表示IMU滞后于ODO
        duration_sec: 数据持续时间(秒)
        imu_rate_hz: IMU的采样率(Hz)
        odo_rate_hz: 轮式编码器的采样率(Hz)
    
    返回:
        imu_df: 包含IMU数据的DataFrame
        odo_df: 包含轮式编码器数据的DataFrame
    """
    # 创建时间戳序列
    imu_timestamps = np.arange(0, duration_sec, 1/imu_rate_hz) * 1000  # 毫秒
    odo_timestamps = np.arange(0, duration_sec, 1/odo_rate_hz) * 1000  # 毫秒
    
    # 应用已知偏移
    imu_timestamps = imu_timestamps + known_offset_ms
    
    # 生成基础信号 - 多个正弦波叠加，创建复杂但可辨识的模式
    t_base = np.arange(0, duration_sec, 0.01)  # 高分辨率基础时间
    
    # 创建有特征的角速度模式 - 包括静止段、匀速旋转段、和快速变化段
    angular_velocity_base = np.zeros_like(t_base)
    
    # 分段1: 静止 (0-10秒)
    segment1 = (t_base < 10)
    angular_velocity_base[segment1] = 0.05 * np.sin(2 * np.pi * 0.2 * t_base[segment1]) # 微小抖动
    
    # 分段2: 匀速旋转 (10-20秒)
    segment2 = (t_base >= 10) & (t_base < 20)
    angular_velocity_base[segment2] = 0.5 + 0.1 * np.sin(2 * np.pi * 0.5 * t_base[segment2])
    
    # 分段3: 快速变化 (20-30秒)
    segment3 = (t_base >= 20) & (t_base < 30)
    angular_velocity_base[segment3] = 2 * np.sin(2 * np.pi * 1.0 * t_base[segment3]) + 0.5 * np.sin(2 * np.pi * 2.5 * t_base[segment3])
    
    # 分段4: 明显事件 (30-40秒) - 几个短脉冲
    segment4 = (t_base >= 30) & (t_base < 40)
    angular_velocity_base[segment4] = 0.2 * np.sin(2 * np.pi * 0.3 * t_base[segment4])
    # 添加脉冲
    pulse_positions = [31, 33, 35, 37, 39]
    for pos in pulse_positions:
        pulse_mask = (t_base >= pos) & (t_base < pos + 0.5)
        angular_velocity_base[pulse_mask] += 3.0
    
    # 分段5: 随机振荡 (40-50秒)
    segment5 = (t_base >= 40) & (t_base < 50)
    t_segment5 = t_base[segment5] - 40
    angular_velocity_base[segment5] = 1.5 * np.sin(2 * np.pi * 0.8 * t_segment5) * np.sin(2 * np.pi * 0.2 * t_segment5)
    
    # 分段6: 最后静止 (50-60秒)
    segment6 = (t_base >= 50)
    angular_velocity_base[segment6] = 0.03 * np.random.randn(np.sum(segment6))
    
    # 在两个传感器上采样信号
    imu_angular_velocity = np.interp(imu_timestamps/1000, t_base, angular_velocity_base)
    odo_angular_velocity = np.interp(odo_timestamps/1000, t_base, angular_velocity_base)
    
    # 添加不同特性的传感器噪声
    imu_noise = 0.15 * np.random.randn(len(imu_timestamps))  # IMU高频噪声较小
    odo_noise = 0.25 * np.random.randn(len(odo_timestamps))  # ODO噪声较大
    
    # 添加噪声到信号
    imu_angular_velocity += imu_noise
    odo_angular_velocity += odo_noise
    
    # 创建DataFrame
    imu_df = pd.DataFrame({
        'timestamp': imu_timestamps,
        'angular_velocity_magnitude': np.abs(imu_angular_velocity)
    })
    
    # 模拟轮式编码器数据，包括左右轮计数
    wheel_base = 0.355  # 米
    wheel_perimeter = 0.7477  # 米
    encoder_resolution = 1194.0  # 脉冲/转
    
    dt_odo = 1000 / odo_rate_hz  # 采样间隔(毫秒)
    
    # 从角速度计算左右轮速度
    # 假设线速度基本保持在1m/s左右
    linear_velocity = 1.0 + 0.2 * np.sin(2 * np.pi * 0.05 * odo_timestamps/1000)
    
    # 根据公式: omega = (v_r - v_l) / wheel_base
    # 和 v = (v_r + v_l) / 2
    # 求解得: v_r = v + omega * wheel_base / 2, v_l = v - omega * wheel_base / 2
    right_velocity = linear_velocity + odo_angular_velocity * wheel_base / 2
    left_velocity = linear_velocity - odo_angular_velocity * wheel_base / 2
    
    # 转换为轮子计数增量
    # 计数 = 速度 * 时间 / 周长 * 分辨率
    right_count_increment = right_velocity * (dt_odo/1000) / wheel_perimeter * encoder_resolution
    left_count_increment = left_velocity * (dt_odo/1000) / wheel_perimeter * encoder_resolution
    
    # 累积计数
    right_count = np.cumsum(right_count_increment)
    left_count = np.cumsum(left_count_increment)
    
    # 创建ODO DataFrame
    odo_df = pd.DataFrame({
        'timestamp': odo_timestamps,
        'right_ticks': right_count,
        'left_ticks': left_count,
        'angular_velocity_magnitude': np.abs(odo_angular_velocity)
    })
    
    return imu_df, odo_df

# 保存数据为CSV文件
def save_synthetic_data(imu_df, odo_df, output_dir='./synthetic_data'):
    os.makedirs(output_dir, exist_ok=True)
    
    imu_file = os.path.join(output_dir, 'imu_synthetic.csv')
    odo_file = os.path.join(output_dir, 'odo_synthetic.csv')
    
    # 使用逗号作为分隔符，确保列之间正确分隔
    with open(imu_file, 'w') as f:
        f.write('# timestamp,angular_velocity_magnitude\n')
        for _, row in imu_df.iterrows():
            f.write(f"{row['timestamp']},{row['angular_velocity_magnitude']}\n")
    
    with open(odo_file, 'w') as f:
        f.write('# timestamp,left_ticks,right_ticks\n')
        for _, row in odo_df.iterrows():
            f.write(f"{row['timestamp']},{row['left_ticks']},{row['right_ticks']}\n")
    
    print(f"生成的IMU数据保存到: {imu_file}")
    print(f"生成的轮式编码器数据保存到: {odo_file}")
    
    return imu_file, odo_file

# 可视化生成的数据
def visualize_synthetic_data(imu_df, odo_df, known_offset_ms, output_dir='./synthetic_data'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # 原始数据对比
    plt.subplot(3, 1, 1)
    plt.plot(imu_df['timestamp'], imu_df['angular_velocity_magnitude'], 'b-', label='IMU角速度')
    plt.plot(odo_df['timestamp'], odo_df['angular_velocity_magnitude'], 'r-', label='ODO角速度')
    plt.title('合成数据对比 - 带有时间偏移')
    plt.xlabel('时间戳 (ms)')
    plt.ylabel('角速度幅值')
    plt.legend()
    plt.grid(True)
    
    # 修正偏移后的数据对比
    plt.subplot(3, 1, 2)
    plt.plot(imu_df['timestamp'], imu_df['angular_velocity_magnitude'], 'b-', label='IMU角速度')
    plt.plot(odo_df['timestamp'] + known_offset_ms, odo_df['angular_velocity_magnitude'], 'g-', 
             label=f'ODO角速度 (偏移校正 {known_offset_ms} ms)')
    plt.title('合成数据对比 - 校正偏移后')
    plt.xlabel('时间戳 (ms)')
    plt.ylabel('角速度幅值')
    plt.legend()
    plt.grid(True)
    
    # 局部放大区域
    plt.subplot(3, 1, 3)
    # 选择有显著特征的局部区域
    zoom_start = 30000
    zoom_end = 40000
    
    plt.plot(imu_df['timestamp'], imu_df['angular_velocity_magnitude'], 'b-', label='IMU角速度')
    plt.plot(odo_df['timestamp'] + known_offset_ms, odo_df['angular_velocity_magnitude'], 'g-', 
             label=f'ODO角速度 (偏移校正 {known_offset_ms} ms)')
    plt.xlim(zoom_start, zoom_end)
    plt.title('局部放大 - 校正偏移后')
    plt.xlabel('时间戳 (ms)')
    plt.ylabel('角速度幅值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_data_visualization.png'), dpi=300)
    plt.close()
    
    print(f"数据可视化结果保存到: {os.path.join(output_dir, 'synthetic_data_visualization.png')}")

def run_verification_with_args(imu_file, odo_file, known_offset_ms, output_dir='./synthetic_verification'):
    """运行验证程序并检查结果"""
    # 构建命令行参数
    class Args:
        def __init__(self):
            self.imu_file = imu_file
            self.odo_file = odo_file
            self.rtk_file = None
            self.image_dir = None
            self.image_timestamp_file = None
            self.methods = "all"
            self.sensors = "imu_odo"
            self.output_dir = output_dir
            self.visualize = False
            self.tolerance_ms = 30.0
            self.wheel_perimeter = 0.7477
            self.wheel_halflength = 0.1775
            self.encoder_scale = 1194.0
            self.imu_timestamp_col = "timestamp"
            self.odo_timestamp_col = "timestamp"
            self.rtk_timestamp_col = "timestamp"
            self.image_pattern = "*.jpg"
            self.window_size = 500
            self.max_lag = 200
    
    args = Args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 因为我们只分析imu和odo数据
    sensor_pairs = [('imu', 'odo')]
    methods = ['correlation', 'event', 'visual']
    
    # 加载数据
    print("加载合成数据...")
    
    # 从文件读取数据，指定逗号作为分隔符
    imu_data = pd.read_csv(imu_file, comment='#', names=['timestamp', 'angular_velocity_magnitude'], sep=',')
    odo_data = pd.read_csv(odo_file, comment='#', names=['timestamp', 'left_ticks', 'right_ticks'], sep=',')
    
    # 转换时间戳列为浮点数
    imu_data['timestamp'] = imu_data['timestamp'].astype(float)
    odo_data['timestamp'] = odo_data['timestamp'].astype(float)
    
    # 添加角速度计算
    # 确保时间戳是升序排列的，计算时间差
    odo_data = odo_data.sort_values(by='timestamp')
    odo_data['dt'] = odo_data['timestamp'].diff().fillna(0) / 1000
    
    # 获取物理参数
    wheel_base = 2.0 * args.wheel_halflength  # 计算轮距
    wheel_perimeter = args.wheel_perimeter  # 轮周长
    encoder_resolution = args.encoder_scale  # 编码器分辨率
    
    # 计算左右轮的计数变化
    odo_data['left_diff'] = odo_data['left_ticks'].diff().fillna(0)
    odo_data['right_diff'] = odo_data['right_ticks'].diff().fillna(0)
    
    # 计算左右轮线速度
    valid_dt = odo_data['dt'] > 1e-6  # 有效时间差
    
    # 初始化角速度列
    odo_data['left_velocity'] = 0.0
    odo_data['right_velocity'] = 0.0
    
    # 计算有效的轮速度
    if valid_dt.any():
        # 左轮速度 (米/秒)
        odo_data.loc[valid_dt, 'left_velocity'] = (
            odo_data.loc[valid_dt, 'left_diff'] / odo_data.loc[valid_dt, 'dt'] * 
            wheel_perimeter / encoder_resolution
        )
        
        # 右轮速度 (米/秒)
        odo_data.loc[valid_dt, 'right_velocity'] = (
            odo_data.loc[valid_dt, 'right_diff'] / odo_data.loc[valid_dt, 'dt'] * 
            wheel_perimeter / encoder_resolution
        )
    
    # 计算线速度 (米/秒) v = 0.5 * (v_r + v_l)
    odo_data['linear_velocity'] = 0.5 * (odo_data['right_velocity'] + odo_data['left_velocity'])
    
    # 计算物理模型角速度 (弧度/秒) omega = (v_r - v_l) / wheel_base
    odo_data['physical_angular_velocity'] = (
        (odo_data['right_velocity'] - odo_data['left_velocity']) / wheel_base
    )
    
    # 使用物理模型的角速度作为最终结果
    odo_data['angular_velocity_magnitude'] = np.abs(odo_data['physical_angular_velocity'])
    
    # 准备数据源
    data_sources = {
        'imu': {
            'data': imu_data,
            'ts_col': 'timestamp'
        },
        'odo': {
            'data': odo_data,
            'ts_col': 'timestamp'
        }
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
        from multisensor_alignment_verify import correlation_verification, event_verification, visual_verification
        
        # 尝试导入可视化函数，但如果失败则使用占位函数
        try:
            from multisensor_alignment_verify import visualize_correlation_results, visualize_event_results, visualize_visual_results
        except ImportError:
            # 定义占位可视化函数
            def visualize_correlation_results(result, sensor1, sensor2, args):
                print(f"  警告: 相关性结果可视化功能不可用")
            
            def visualize_event_results(result, sensor1, sensor2, args):
                print(f"  警告: 事件结果可视化功能不可用")
                
            def visualize_visual_results(result, sensor1, sensor2, args):
                print(f"  警告: 视觉结果可视化功能不可用")
        
        for method in methods:
            print(f"  使用{method}方法验证...")
            
            if method == 'correlation':
                # 互相关验证
                result = correlation_verification(
                    data1, data2, ts_col1, ts_col2, args
                )
                pair_results['correlation'] = result
                try:
                    visualize_correlation_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  可视化相关性结果时出错: {e}")
                
            elif method == 'event':
                # 事件同步验证
                result = event_verification(
                    data1, data2, ts_col1, ts_col2, args
                )
                pair_results['event'] = result
                try:
                    visualize_event_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  可视化事件结果时出错: {e}")
                
            elif method == 'visual':
                # 可视化比较
                result = visual_verification(
                    data1, data2, ts_col1, ts_col2, args
                )
                pair_results['visual'] = result
                try:
                    visualize_visual_results(result, sensor1, sensor2, args)
                except Exception as e:
                    print(f"  可视化视觉结果时出错: {e}")
        
        # 存储组合结果
        all_verification_results[pair_key] = pair_results
    
    # 显示简要结果
    print("\n===== 验证结果摘要 =====")
    for sensor_pair, pair_results in all_verification_results.items():
        sensor1, sensor2 = sensor_pair.split('_')
        
        print(f"已知时间偏移: {known_offset_ms} ms")
        for method, result in pair_results.items():
            offset_ms = result.get('mean_offset', 0)
            is_aligned = result.get('is_aligned', False)
            error_ms = abs(offset_ms - known_offset_ms)
            
            print(f"{method}方法: 检测偏移 = {offset_ms:.2f} ms, 误差 = {error_ms:.2f} ms, 对齐判断 = {is_aligned}")
        
        aligned_count = sum(1 for result in pair_results.values() if result.get('is_aligned', False))
        total_methods = len(pair_results)
        final_aligned = aligned_count >= total_methods / 2
        
        print(f"{sensor1}-{sensor2}: {'对齐' if final_aligned else '未对齐'} ({aligned_count}/{total_methods})")
    
    return all_verification_results

# 主函数
def main():
    # 1. 生成多个不同偏移值的合成数据
    offsets_to_test = [0, 15, 30, 50, 100, 150, 200, 300, 500]  # 增加更大的偏移值测试
    results = []
    
    for known_offset_ms in offsets_to_test:
        print(f"\n===== 测试已知偏移: {known_offset_ms} ms =====")
        output_dir = f"./synthetic_test_{known_offset_ms}ms"
        
        # 生成数据
        imu_df, odo_df = generate_synthetic_data(known_offset_ms=known_offset_ms)
        
        # 保存数据
        imu_file, odo_file = save_synthetic_data(imu_df, odo_df, output_dir=output_dir)
        
        try:
            # 可视化数据
            visualize_synthetic_data(imu_df, odo_df, known_offset_ms, output_dir=output_dir)
        except Exception as e:
            print(f"可视化数据时出错: {e}")
        
        try:
            # 运行验证
            verification_results = run_verification_with_args(imu_file, odo_file, known_offset_ms, output_dir=output_dir)
            
            # 收集结果
            methods_results = {}
            for method, result in verification_results['imu_odo'].items():
                methods_results[method] = {
                    'offset_ms': result.get('mean_offset', 0),
                    'is_aligned': result.get('is_aligned', False),
                    'confidence': result.get('confidence', 0),
                    'std_offset': result.get('std_offset', 0)
                }
            
            results.append({
                'known_offset_ms': known_offset_ms,
                'methods': methods_results
            })
        except Exception as e:
            print(f"运行验证时出错: {e}")
            # 继续测试下一个偏移值
            continue
    
    # 分析结果
    if results:
        print("\n\n===== 综合分析结果 =====")
        print("已知偏移 | 相关性方法 | 事件方法 | 视觉方法")
        print("---------|------------|----------|----------")
        
        for r in results:
            known = r['known_offset_ms']
            
            # 安全地获取各方法结果，可能某些方法失败了
            corr = r['methods'].get('correlation', {}).get('offset_ms', 0)
            event = r['methods'].get('event', {}).get('offset_ms', 0)
            visual = r['methods'].get('visual', {}).get('offset_ms', 0)
            
            corr_err = abs(corr - known)
            event_err = abs(event - known)
            visual_err = abs(visual - known)
            
            # 添加置信度和标准差信息
            corr_std = r['methods'].get('correlation', {}).get('std_offset', 0)
            event_std = r['methods'].get('event', {}).get('std_offset', 0)
            visual_conf = r['methods'].get('visual', {}).get('confidence', 0)
            
            print(f"{known:5d} ms | {corr:6.1f} ms (误差:{corr_err:5.1f}, 标准差:{corr_std:4.1f}) | "
                  f"{event:6.1f} ms (误差:{event_err:5.1f}, 标准差:{event_std:4.1f}) | "
                  f"{visual:6.1f} ms (误差:{visual_err:5.1f}, 置信度:{visual_conf:.2f})")
        
        # 计算方法有效性统计
        # 使用更严格的评估标准
        corr_success = sum(1 for r in results if abs(r['methods'].get('correlation', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
        event_success = sum(1 for r in results if abs(r['methods'].get('event', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
        visual_success = sum(1 for r in results if abs(r['methods'].get('visual', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
        
        # 计算平均误差
        corr_errors = [abs(r['methods'].get('correlation', {}).get('offset_ms', 0) - r['known_offset_ms']) for r in results]
        event_errors = [abs(r['methods'].get('event', {}).get('offset_ms', 0) - r['known_offset_ms']) for r in results]
        visual_errors = [abs(r['methods'].get('visual', {}).get('offset_ms', 0) - r['known_offset_ms']) for r in results]
        
        avg_corr_error = sum(corr_errors) / len(corr_errors) if corr_errors else float('inf')
        avg_event_error = sum(event_errors) / len(event_errors) if event_errors else float('inf')
        avg_visual_error = sum(visual_errors) / len(visual_errors) if visual_errors else float('inf')
        
        total_tests = len(results)
        
        # 总结发现
        print("\n===== 总结发现 =====")
        print(f"1. 相关性方法性能: 成功率 {corr_success}/{total_tests} ({corr_success/total_tests*100:.1f}%), 平均误差: {avg_corr_error:.1f}ms")
        print(f"2. 事件方法性能: 成功率 {event_success}/{total_tests} ({event_success/total_tests*100:.1f}%), 平均误差: {avg_event_error:.1f}ms")
        print(f"3. 视觉方法性能: 成功率 {visual_success}/{total_tests} ({visual_success/total_tests*100:.1f}%), 平均误差: {avg_visual_error:.1f}ms")
        
        # 建议最佳方法
        methods_performance = [
            ("相关性方法", corr_success/total_tests, avg_corr_error),
            ("事件方法", event_success/total_tests, avg_event_error),
            ("视觉方法", visual_success/total_tests, avg_visual_error)
        ]
        
        # 首先按成功率排序，然后按平均误差排序
        methods_performance.sort(key=lambda x: (-x[1], x[2]))
        
        best_method = methods_performance[0][0]
        print(f"4. 建议: 在当前测试条件下，{best_method}表现最佳，应优先采用")
        
        # 分析不同偏移范围的性能
        small_offsets = [r for r in results if r['known_offset_ms'] <= 50]
        medium_offsets = [r for r in results if 50 < r['known_offset_ms'] <= 200]
        large_offsets = [r for r in results if r['known_offset_ms'] > 200]
        
        if small_offsets:
            print("\n小偏移范围 (0-50ms) 性能分析:")
            corr_small = sum(1 for r in small_offsets if abs(r['methods'].get('correlation', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            event_small = sum(1 for r in small_offsets if abs(r['methods'].get('event', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            visual_small = sum(1 for r in small_offsets if abs(r['methods'].get('visual', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            
            print(f"  相关性方法: {corr_small}/{len(small_offsets)} ({corr_small/len(small_offsets)*100:.1f}%)")
            print(f"  事件方法: {event_small}/{len(small_offsets)} ({event_small/len(small_offsets)*100:.1f}%)")
            print(f"  视觉方法: {visual_small}/{len(small_offsets)} ({visual_small/len(small_offsets)*100:.1f}%)")
        
        if medium_offsets:
            print("\n中偏移范围 (51-200ms) 性能分析:")
            corr_medium = sum(1 for r in medium_offsets if abs(r['methods'].get('correlation', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            event_medium = sum(1 for r in medium_offsets if abs(r['methods'].get('event', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            visual_medium = sum(1 for r in medium_offsets if abs(r['methods'].get('visual', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            
            print(f"  相关性方法: {corr_medium}/{len(medium_offsets)} ({corr_medium/len(medium_offsets)*100:.1f}%)")
            print(f"  事件方法: {event_medium}/{len(medium_offsets)} ({event_medium/len(medium_offsets)*100:.1f}%)")
            print(f"  视觉方法: {visual_medium}/{len(medium_offsets)} ({visual_medium/len(medium_offsets)*100:.1f}%)")
        
        if large_offsets:
            print("\n大偏移范围 (>200ms) 性能分析:")
            corr_large = sum(1 for r in large_offsets if abs(r['methods'].get('correlation', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            event_large = sum(1 for r in large_offsets if abs(r['methods'].get('event', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            visual_large = sum(1 for r in large_offsets if abs(r['methods'].get('visual', {}).get('offset_ms', 0) - r['known_offset_ms']) <= 30)
            
            print(f"  相关性方法: {corr_large}/{len(large_offsets)} ({corr_large/len(large_offsets)*100:.1f}%)")
            print(f"  事件方法: {event_large}/{len(large_offsets)} ({event_large/len(large_offsets)*100:.1f}%)")
            print(f"  视觉方法: {visual_large}/{len(large_offsets)} ({visual_large/len(large_offsets)*100:.1f}%)")
    else:
        print("没有成功完成的测试，无法生成分析结果")

# 如果需要运行
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中发生错误: {e}")