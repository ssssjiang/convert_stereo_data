import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
import platform
import matplotlib.font_manager as fm
import matplotlib as mpl

# 设置全局字体为英文
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
mpl.rcParams['axes.unicode_minus'] = True  # 正确显示负号

# 中英文标签映射
label_map = {
    # 状态标签
    "静止": "Static",
    "匀速运动": "Uniform Motion",
    "动态运动": "Dynamic Motion",
    
    # 传感器和轴标签
    "加速度计": "Accelerometer",
    "陀螺仪": "Gyroscope",
    
    # 频段标签
    "very_low": "Very Low",
    "low": "Low",
    "mid": "Mid",
    "high": "High",
    
    # 噪声类型
    "白噪声": "White Noise",
    "粉红噪声(1/f)": "Pink Noise (1/f)",
    "褐噪声(1/f²)": "Brown Noise (1/f²)",
    "其他噪声": "Other Noise"
}

def translate(text):
    """将中文文本翻译为英文"""
    for cn, en in label_map.items():
        text = text.replace(cn, en)
    return text

def compare_motion_noise(file_path, save_dir="motion_noise_comparison"):
    """
    Compare IMU noise levels in different motion states
    
    Parameters:
        file_path (str): Path to IMU data file
        save_dir (str): Directory to save results
    
    Returns:
        dict: Noise metrics for different motion states
    """
    print(f"Analyzing file: {file_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load data
    imu_data = load_imu_data(file_path)
    print(f"Loaded {len(imu_data)} rows of data")
    
    # 2. Identify motion states
    motion_states = identify_motion_states(imu_data)
    print(f"Detected motion states: {list(motion_states.keys())}")
    
    # 3. Calculate noise metrics for each state
    noise_metrics = {}
    for state, indices in motion_states.items():
        if len(indices) > 100:  # Ensure enough data points
            print(f"Analyzing '{translate(state)}' state ({len(indices)} data points)")
            state_data = imu_data.iloc[indices]
            noise_metrics[state] = calculate_noise_metrics(state_data)
    
    # 4. Visualize comparison results
    visualize_noise_comparison(noise_metrics, save_dir)
    
    # 5. Generate detailed report
    generate_comparison_report(noise_metrics, save_dir)
    
    return noise_metrics

def load_imu_data(file_path):
    """加载IMU数据文件"""
    if file_path.endswith('.csv'):
        # 尝试加载CSV格式
        try:
            data = pd.read_csv(file_path, comment='#')
            if len(data.columns) >= 7:
                # 假设格式: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
                data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'] + \
                              [f'extra_{i}' for i in range(len(data.columns) - 7)]
            else:
                raise ValueError(f"CSV文件列数不足: {len(data.columns)}")
        except Exception as e:
            print(f"CSV加载错误: {e}")
            # 尝试无头CSV
            data = pd.read_csv(file_path, header=None)
            if len(data.columns) >= 7:
                data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'] + \
                              [f'extra_{i}' for i in range(len(data.columns) - 7)]
            else:
                raise ValueError(f"无头CSV文件列数不足: {len(data.columns)}")
    else:
        # 尝试加载LOG格式
        data = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
        if len(data.columns) >= 8:
            # 假设格式: timestamp, imu, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            data.columns = ['timestamp', 'imu', 'acc_x', 'acc_y', 'acc_z', 
                           'gyro_x', 'gyro_y', 'gyro_z'] + \
                           [f'extra_{i}' for i in range(len(data.columns) - 8)]
        else:
            raise ValueError(f"LOG文件列数不足: {len(data.columns)}")
    
    # 转换为数值类型
    for col in data.columns:
        if col != 'imu':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 计算合成加速度和角速度
    data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
    data['gyro_mag'] = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
    
    # 计算时间差
    data['dt'] = np.diff(data['timestamp'], prepend=data['timestamp'].iloc[0]) / 1000.0  # 秒
    
    return data

def identify_motion_states(imu_data, window_size=100):
    """识别不同的运动状态"""
    # 使用滑动窗口计算特征
    features = []
    indices = []
    
    for i in range(0, len(imu_data) - window_size, window_size // 2):
        window = imu_data.iloc[i:i+window_size]
        
        # 计算窗口内的特征
        acc_std = window['acc_mag'].std()
        gyro_std = window['gyro_mag'].std()
        acc_mean = window['acc_mag'].mean()
        
        features.append([acc_std, gyro_std, acc_mean])
        indices.append(i)
    
    # 使用K-means聚类
    features = np.array(features)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
    labels = kmeans.labels_
    
    # 分析聚类结果，确定状态类型
    centers = kmeans.cluster_centers_
    state_mapping = {}
    
    for i in range(len(centers)):
        acc_std, gyro_std, acc_mean = centers[i]
        
        if gyro_std < 0.1:  # 低角速度变化
            if abs(acc_mean - 9.8) < 0.5 and acc_std < 0.2:
                state_mapping[i] = "静止"
            else:
                state_mapping[i] = "匀速运动"
        else:
            state_mapping[i] = "动态运动"
    
    # 将数据点映射到状态
    motion_states = {state: [] for state in set(state_mapping.values())}
    
    for i, label in enumerate(labels):
        state = state_mapping[label]
        start_idx = indices[i]
        end_idx = min(start_idx + window_size, len(imu_data))
        motion_states[state].extend(list(range(start_idx, end_idx)))
    
    # 去除重复
    for state in motion_states:
        motion_states[state] = sorted(list(set(motion_states[state])))
    
    return motion_states

def calculate_noise_metrics(state_data):
    """计算噪声指标"""
    metrics = {
        'basic_stats': {},
        'spectral': {},
        'noise_types': {},
        'frequency_bands': {}
    }
    
    # 1. 基本统计量
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        data = state_data[axis].values
        metrics['basic_stats'][axis] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'p2p': np.max(data) - np.min(data),
            'rms': np.sqrt(np.mean(np.square(data - np.mean(data)))),
            'kurtosis': stats.kurtosis(data),
            'skewness': stats.skew(data)
        }
    
    # 2. 频谱分析
    sampling_rate = 1.0 / np.mean(state_data['dt'])
    
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        data = detrend(state_data[axis].values)
        f, psd = welch(data, fs=sampling_rate, nperseg=min(1024, len(data)//2))
        
        # 计算不同频段的能量和噪声水平
        freq_bands = {
            'very_low': (0.01, 0.1),
            'low': (0.1, 1.0),
            'mid': (1.0, 10.0),
            'high': (10.0, 50.0)
        }
        
        metrics['frequency_bands'][axis] = {}
        
        for band_name, (f_min, f_max) in freq_bands.items():
            band_mask = (f >= f_min) & (f < f_max)
            if np.any(band_mask):
                band_psd = psd[band_mask]
                band_f = f[band_mask]
                
                # 计算频段能量
                band_energy = np.sum(band_psd)
                
                # 计算频段噪声水平 (RMS)
                band_noise_level = np.sqrt(np.sum(band_psd) * (f_max - f_min) / len(band_psd))
                
                # 计算频段斜率
                if len(band_f) >= 5 and np.all(band_f > 0) and np.all(band_psd > 0):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        np.log10(band_f), np.log10(band_psd))
                else:
                    slope, r_value = 0, 0
                
                # 确定噪声类型
                noise_type = "未知"
                if -0.3 <= slope <= 0.3:
                    noise_type = "白噪声"
                elif -1.3 <= slope <= -0.7:
                    noise_type = "粉红噪声(1/f)"
                elif -2.3 <= slope <= -1.7:
                    noise_type = "褐噪声(1/f²)"
                
                metrics['frequency_bands'][axis][band_name] = {
                    'energy': band_energy,
                    'noise_level': band_noise_level,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'noise_type': noise_type
                }
    
    return metrics

def visualize_noise_comparison(noise_metrics, save_dir):
    """Visualize noise comparison between different motion states"""
    states = list(noise_metrics.keys())
    
    # 1. Compare basic noise levels (standard deviation)
    plt.figure(figsize=(15, 10))
    
    # Accelerometer noise levels
    plt.subplot(2, 1, 1)
    acc_axes = ['acc_x', 'acc_y', 'acc_z']
    acc_data = []
    
    for state in states:
        state_name = translate(state)
        for axis in acc_axes:
            acc_data.append({
                'State': state_name,
                'Axis': axis,
                'Std (m/s²)': noise_metrics[state]['basic_stats'][axis]['std']
            })
    
    acc_df = pd.DataFrame(acc_data)
    sns.barplot(x='Axis', y='Std (m/s²)', hue='State', data=acc_df)
    plt.title('Accelerometer Noise Level in Different Motion States (Std)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Motion State')
    
    # Gyroscope noise levels
    plt.subplot(2, 1, 2)
    gyro_axes = ['gyro_x', 'gyro_y', 'gyro_z']
    gyro_data = []
    
    for state in states:
        state_name = translate(state)
        for axis in gyro_axes:
            gyro_data.append({
                'State': state_name,
                'Axis': axis,
                'Std (rad/s)': noise_metrics[state]['basic_stats'][axis]['std']
            })
    
    gyro_df = pd.DataFrame(gyro_data)
    sns.barplot(x='Axis', y='Std (rad/s)', hue='State', data=gyro_df)
    plt.title('Gyroscope Noise Level in Different Motion States (Std)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Motion State')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_level_comparison.png'))
    plt.close()
    
    # 2. Compare noise levels in different frequency bands
    for sensor_type, axes in [('Accelerometer', acc_axes), ('Gyroscope', gyro_axes)]:
        plt.figure(figsize=(15, 12))
        
        for i, axis in enumerate(axes):
            plt.subplot(3, 1, i+1)
            
            band_data = []
            for state in states:
                state_name = translate(state)
                for band_name in ['very_low', 'low', 'mid', 'high']:
                    if band_name in noise_metrics[state]['frequency_bands'][axis]:
                        band_data.append({
                            'State': state_name,
                            'Frequency Band': translate(band_name),
                            'Noise Level': noise_metrics[state]['frequency_bands'][axis][band_name]['noise_level']
                        })
            
            band_df = pd.DataFrame(band_data)
            sns.barplot(x='Frequency Band', y='Noise Level', hue='State', data=band_df)
            plt.title(f'{axis} Noise Level in Different Frequency Bands')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Motion State')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sensor_type}_frequency_band_noise.png'))
        plt.close()
    
    # 3. Compare spectral slopes
    for sensor_type, axes in [('Accelerometer', acc_axes), ('Gyroscope', gyro_axes)]:
        plt.figure(figsize=(15, 12))
        
        for i, axis in enumerate(axes):
            plt.subplot(3, 1, i+1)
            
            slope_data = []
            for state in states:
                state_name = translate(state)
                for band_name in ['very_low', 'low', 'mid', 'high']:
                    if band_name in noise_metrics[state]['frequency_bands'][axis]:
                        slope_data.append({
                            'State': state_name,
                            'Frequency Band': translate(band_name),
                            'Slope': noise_metrics[state]['frequency_bands'][axis][band_name]['slope']
                        })
            
            slope_df = pd.DataFrame(slope_data)
            sns.barplot(x='Frequency Band', y='Slope', hue='State', data=slope_df)
            plt.title(f'{axis} Spectral Slope in Different Frequency Bands')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
            plt.axhline(y=-2, color='k', linestyle='--', alpha=0.3)
            plt.legend(title='Motion State')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sensor_type}_spectral_slope.png'))
        plt.close()
    
    # 4. Compare noise distributions
    for axis in acc_axes + gyro_axes:
        plt.figure(figsize=(15, 10))
        
        for i, state in enumerate(states):
            state_name = translate(state)
            data = noise_metrics[state]['basic_stats'][axis]
            
            plt.subplot(len(states), 1, i+1)
            
            # Create normal distribution
            x = np.linspace(data['mean'] - 4*data['std'], data['mean'] + 4*data['std'], 1000)
            y = stats.norm.pdf(x, data['mean'], data['std'])
            
            plt.plot(x, y, 'r-', lw=2)
            plt.fill_between(x, y, alpha=0.3)
            plt.title(f'{state_name} State {axis} Noise Distribution (μ={data["mean"]:.4f}, σ={data["std"]:.4f})')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            info_text = f"Kurtosis: {data['kurtosis']:.2f}\n"
            info_text += f"Skewness: {data['skewness']:.2f}\n"
            info_text += f"Peak-to-Peak: {data['p2p']:.4f}\n"
            info_text += f"RMS: {data['rms']:.4f}"
            
            plt.annotate(info_text, xy=(0.02, 0.7), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{axis}_distribution_comparison.png'))
        plt.close()

def generate_comparison_report(noise_metrics, save_dir):
    """Generate noise comparison report"""
    report_path = os.path.join(save_dir, 'noise_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== IMU Noise Level Comparison in Different Motion States ===\n\n")
        
        states = list(noise_metrics.keys())
        translated_states = [translate(state) for state in states]
        
        # 1. Basic noise level comparison
        f.write("## Basic Noise Level Comparison (Standard Deviation)\n\n")
        
        # Accelerometer
        f.write("### Accelerometer (m/s²)\n\n")
        f.write("| Axis | " + " | ".join(translated_states) + " |\n")
        f.write("|" + "-|"*(len(states)+1) + "\n")
        
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            f.write(f"| {axis} |")
            for state in states:
                std = noise_metrics[state]['basic_stats'][axis]['std']
                f.write(f" {std:.6f} |")
            f.write("\n")
        
        f.write("\n")
        
        # Gyroscope
        f.write("### Gyroscope (rad/s)\n\n")
        f.write("| Axis | " + " | ".join(translated_states) + " |\n")
        f.write("|" + "-|"*(len(states)+1) + "\n")
        
        for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
            f.write(f"| {axis} |")
            for state in states:
                std = noise_metrics[state]['basic_stats'][axis]['std']
                f.write(f" {std:.6f} |")
            f.write("\n")
        
        f.write("\n\n")
        
        # 2. Frequency band noise level comparison
        f.write("## Noise Level Comparison in Different Frequency Bands\n\n")
        
        for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            f.write(f"### {axis}\n\n")
            f.write("| Freq Band | Metric | " + " | ".join(translated_states) + " |\n")
            f.write("|" + "-|"*(len(states)+2) + "\n")
            
            for band in ['very_low', 'low', 'mid', 'high']:
                band_name = translate(band)
                # Noise level
                f.write(f"| {band_name} | Noise Level |")
                for state in states:
                    if band in noise_metrics[state]['frequency_bands'][axis]:
                        noise_level = noise_metrics[state]['frequency_bands'][axis][band]['noise_level']
                        f.write(f" {noise_level:.6e} |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
                
                # Slope
                f.write(f"| {band_name} | Slope |")
                for state in states:
                    if band in noise_metrics[state]['frequency_bands'][axis]:
                        slope = noise_metrics[state]['frequency_bands'][axis][band]['slope']
                        f.write(f" {slope:.2f} |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
                
                # Noise type
                f.write(f"| {band_name} | Noise Type |")
                for state in states:
                    if band in noise_metrics[state]['frequency_bands'][axis]:
                        noise_type = translate(noise_metrics[state]['frequency_bands'][axis][band]['noise_type'])
                        f.write(f" {noise_type} |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
            
            f.write("\n")
        
        # 3. Noise growth ratio
        if len(states) > 1 and "静止" in states:
            f.write("## Noise Growth Ratio Relative to Static State\n\n")
            
            # Accelerometer
            f.write("### Accelerometer\n\n")
            f.write("| Axis | State | Std Ratio | P2P Ratio | RMS Ratio |\n")
            f.write("|---|---|---|---|---|\n")
            
            for axis in ['acc_x', 'acc_y', 'acc_z']:
                static_std = noise_metrics["静止"]['basic_stats'][axis]['std']
                static_p2p = noise_metrics["静止"]['basic_stats'][axis]['p2p']
                static_rms = noise_metrics["静止"]['basic_stats'][axis]['rms']
                
                for state in states:
                    if state != "静止":
                        std = noise_metrics[state]['basic_stats'][axis]['std']
                        p2p = noise_metrics[state]['basic_stats'][axis]['p2p']
                        rms = noise_metrics[state]['basic_stats'][axis]['rms']
                        
                        std_ratio = std / static_std if static_std > 0 else float('inf')
                        p2p_ratio = p2p / static_p2p if static_p2p > 0 else float('inf')
                        rms_ratio = rms / static_rms if static_rms > 0 else float('inf')
                        
                        f.write(f"| {axis} | {translate(state)} | {std_ratio:.2f}x | {p2p_ratio:.2f}x | {rms_ratio:.2f}x |\n")
            
            f.write("\n")
            
            # Gyroscope
            f.write("### Gyroscope\n\n")
            f.write("| Axis | State | Std Ratio | P2P Ratio | RMS Ratio |\n")
            f.write("|---|---|---|---|---|\n")
            
            for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
                static_std = noise_metrics["静止"]['basic_stats'][axis]['std']
                static_p2p = noise_metrics["静止"]['basic_stats'][axis]['p2p']
                static_rms = noise_metrics["静止"]['basic_stats'][axis]['rms']
                
                for state in states:
                    if state != "静止":
                        std = noise_metrics[state]['basic_stats'][axis]['std']
                        p2p = noise_metrics[state]['basic_stats'][axis]['p2p']
                        rms = noise_metrics[state]['basic_stats'][axis]['rms']
                        
                        std_ratio = std / static_std if static_std > 0 else float('inf')
                        p2p_ratio = p2p / static_p2p if static_p2p > 0 else float('inf')
                        rms_ratio = rms / static_rms if static_rms > 0 else float('inf')
                        
                        f.write(f"| {axis} | {translate(state)} | {std_ratio:.2f}x | {p2p_ratio:.2f}x | {rms_ratio:.2f}x |\n")
    
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="比较不同运动状态下的IMU噪声水平")
    parser.add_argument('--file', type=str, required=True, help="IMU数据文件路径")
    parser.add_argument('--output', type=str, default="motion_noise_comparison", help="结果保存目录")
    
    args = parser.parse_args()
    
    compare_motion_noise(args.file, args.output) 