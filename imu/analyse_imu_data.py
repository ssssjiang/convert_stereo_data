import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq
from scipy import stats  # 用于线性回归
import matplotlib.font_manager as fm
import matplotlib as mpl
from sklearn.cluster import KMeans

# 配置matplotlib支持中文显示
try:
    # 尝试使用系统中常见的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
    
    # 检查系统中是否有这些字体
    available_fonts = [f for f in chinese_fonts if any(font.name == f for font in fm.fontManager.ttflist)]
    
    if available_fonts:
        # 使用找到的第一个中文字体
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        print(f"Using Chinese font: {available_fonts[0]}")
    else:
        # 如果没有找到中文字体，使用英文标签
        print("No Chinese fonts found, using English labels")
        # 这里可以设置一个标志，在后面的代码中使用英文标签
except Exception as e:
    print(f"Error configuring Chinese fonts: {e}")
    # 出错时使用英文标签

# 将calculate_slope函数移到最外层作用域
def calculate_slope(f, psd, f_min, f_max):
    """计算指定频率范围内的PSD斜率"""
    # 选择频率范围内的索引
    indices = np.where((f >= f_min) & (f <= f_max))[0]
    
    if len(indices) < 5:  # 需要足够的点进行拟合
        return None, None, None
    
    log_f = np.log10(f[indices])
    log_psd = np.log10(psd[indices])
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_psd)
    
    # 确定噪声类型
    noise_type = "未知"
    if -0.3 <= slope <= 0.3:
        noise_type = "白噪声"
    elif -1.3 <= slope <= -0.7:
        noise_type = "粉红噪声(1/f)"
    elif -2.3 <= slope <= -1.7:
        noise_type = "褐噪声(1/f²)"
    
    return slope, r_value**2, noise_type

def process_imu_data(file_path, cutoff_frequency=10.0, sampling_rate=50.0, order=2, start_time=0.0, save_dir=".", analysis_target="all", save_filtered=False):
    """
    Processes IMU data and generates analysis plots for each axis and magnitude, saved to specified directory.
    Supports both .log and .csv IMU data formats.

    Parameters:
        file_path (str): Path to the IMU data file.
        cutoff_frequency (float): Cutoff frequency for low-pass filter (Hz).
        sampling_rate (float): Sampling rate (Hz).
        order (int): Order of the low-pass filter.
        start_time (float): Start timestamp for analysis (in milliseconds).
        save_dir (str): Directory to save generated plots.
        analysis_target (str): Target data to analyze ("all", "accel", "gyro").
        save_filtered (bool): Whether to save filtered data plots.

    Returns:
        dict: Dictionary containing noise statistics for accelerometer and gyroscope
    """
    print("Loading data file:", file_path)
    
    # Determine file type by extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # CSV format: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
        try:
            # First try to load with header (for files with # header)
            imu_data = pd.read_csv(file_path, comment='#')
            if len(imu_data.columns) == 7:
                print("Detected CSV format with header")
                imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
            else:
                # Try without header
                imu_data = pd.read_csv(file_path, header=None)
                print("Detected CSV format without header")
                imu_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            raise
    else:
        # LOG format: timestamp, imu, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, quat...
        try:
            imu_data = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
            print(f"Detected LOG format: {len(imu_data.columns)} columns")
            
            # Rename columns based on LOG format
            if len(imu_data.columns) >= 12:
                imu_data.columns = ['timestamp', 'imu', 'acc_x', 'acc_y', 'acc_z', 
                                   'gyro_x', 'gyro_y', 'gyro_z',
                                   'quat_x', 'quat_y', 'quat_z', 'quat_w'] + \
                                   [f'extra_{i}' for i in range(len(imu_data.columns) - 12)]
            else:
                print("Warning: LOG file has fewer columns than expected")
                # Handle minimal case with just timestamp, acc, gyro
                cols = ['timestamp', 'imu']
                if len(imu_data.columns) > 2:
                    cols += ['acc_x', 'acc_y', 'acc_z']
                if len(imu_data.columns) > 5:
                    cols += ['gyro_x', 'gyro_y', 'gyro_z']
                cols += [f'extra_{i}' for i in range(len(imu_data.columns) - len(cols))]
                imu_data.columns = cols[:len(imu_data.columns)]
        except Exception as e:
            print(f"Error parsing LOG file: {e}")
            raise
    
    print(f"Loaded {len(imu_data)} rows with {len(imu_data.columns)} columns")
    print(f"Column names: {list(imu_data.columns)}")
    
    # Only convert numeric columns (skip 'imu' column if present)
    numeric_columns = [col for col in imu_data.columns if col != 'imu']
    
    # Convert only numeric columns
    for col in numeric_columns:
        imu_data[col] = pd.to_numeric(imu_data[col], errors='coerce')
    
    # Drop rows with NaN in critical columns
    critical_cols = ['timestamp']
    if 'acc_x' in imu_data.columns:
        critical_cols += ['acc_x', 'acc_y', 'acc_z']
    if 'gyro_x' in imu_data.columns:
        critical_cols += ['gyro_x', 'gyro_y', 'gyro_z']
        
    imu_data = imu_data.dropna(subset=critical_cols)
    print(f"After numeric conversion: {len(imu_data)} rows")
    
    # Filter data to include only rows after start_time
    print(f"Filtering data after timestamp {start_time}")
    filtered_data = imu_data[imu_data['timestamp'] >= start_time]
    
    if filtered_data.empty:
        print(f"WARNING: No data points exist after start_time {start_time}")
        print(f"Using full dataset instead.")
        filtered_data = imu_data
    
    imu_data = filtered_data
    
    # Check if we have enough data for filtering
    if len(imu_data) <= 10:
        raise ValueError(f"Not enough data points for analysis: {len(imu_data)} rows")

    # Define low-pass filter function
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Select target axes for analysis
    target_axes = []
    if analysis_target in ["all", "accel"] and 'acc_x' in imu_data.columns:
        target_axes.extend(['acc_x', 'acc_y', 'acc_z'])
    if analysis_target in ["all", "gyro"] and 'gyro_x' in imu_data.columns:
        target_axes.extend(['gyro_x', 'gyro_y', 'gyro_z'])

    # Apply low-pass filter to selected axes
    for axis in target_axes:
        imu_data[f'{axis}_filtered'] = butter_lowpass_filter(imu_data[axis], cutoff_frequency, sampling_rate, order)

    # Convert timestamp from milliseconds to seconds for plotting
    imu_data['time_seconds'] = imu_data['timestamp'] / 1000.0

    # Calculate noise statistics
    def calculate_noise_stats(data):
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'peak_to_peak': np.ptp(data)
        }

    noise_stats = {}
    if ("accel" in analysis_target or analysis_target == "all") and 'acc_x' in imu_data.columns:
        noise_stats['accelerometer'] = {
            'x': calculate_noise_stats(imu_data['acc_x']),
            'y': calculate_noise_stats(imu_data['acc_y']),
            'z': calculate_noise_stats(imu_data['acc_z'])
        }
        
    if ("gyro" in analysis_target or analysis_target == "all") and 'gyro_x' in imu_data.columns:
        noise_stats['gyroscope'] = {
            'x': calculate_noise_stats(imu_data['gyro_x']),
            'y': calculate_noise_stats(imu_data['gyro_y']),
            'z': calculate_noise_stats(imu_data['gyro_z'])
        }

    # Plot data based on analysis target
    if ("accel" in analysis_target or analysis_target == "all") and 'acc_x' in imu_data.columns:
        plt.figure(figsize=(20, 15))
        for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            plt.subplot(3, 1, i + 1)
            plt.plot(imu_data['time_seconds'], imu_data[axis], label=f'{axis} (raw)', linewidth=1.0)
            if save_filtered:
                plt.plot(imu_data['time_seconds'], imu_data[f'{axis}_filtered'], 
                        label=f'{axis} (filtered)', linewidth=1.0)
            plt.title(f'{axis} Data\nMean: {noise_stats["accelerometer"][axis[-1]]["mean"]:.6f}, '
                     f'Std: {noise_stats["accelerometer"][axis[-1]]["std"]:.6f}, '
                     f'RMS: {noise_stats["accelerometer"][axis[-1]]["rms"]:.6f}')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s²)')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        accel_plot_path = os.path.join(save_dir, 'acceleration_data.png')
        plt.savefig(accel_plot_path)
        plt.close()

    if ("gyro" in analysis_target or analysis_target == "all") and 'gyro_x' in imu_data.columns:
        plt.figure(figsize=(20, 15))
        for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            plt.subplot(3, 1, i + 1)
            plt.plot(imu_data['time_seconds'], imu_data[axis], label=f'{axis} (raw)', linewidth=1.0)
            if save_filtered:
                plt.plot(imu_data['time_seconds'], imu_data[f'{axis}_filtered'], 
                        label=f'{axis} (filtered)', linewidth=1.0)
            plt.title(f'{axis} Data\nMean: {noise_stats["gyroscope"][axis[-1]]["mean"]:.6f}, '
                     f'Std: {noise_stats["gyroscope"][axis[-1]]["std"]:.6f}, '
                     f'RMS: {noise_stats["gyroscope"][axis[-1]]["rms"]:.6f}')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular Velocity (rad/s)')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        gyro_plot_path = os.path.join(save_dir, 'gyroscope_data.png')
        plt.savefig(gyro_plot_path)
        plt.close()

    # 改进的频谱分析和噪声底计算函数
    def improved_frequency_analysis(signal, timestamps):
        dt = np.mean(np.diff(timestamps)) / 1000.0  # 转换为秒
        fs = 1 / dt
        
        # 检测静态区间 - 使用滑动窗口方差
        window_size = int(fs * 1.0)  # 1秒窗口
        if len(signal) <= window_size:
            window_size = max(10, len(signal) // 10)
        
        variances = []
        for i in range(0, len(signal) - window_size, window_size // 2):
            variances.append(np.var(signal[i:i+window_size]))
        
        # 找出最静态的区间（方差最小的几个窗口）
        static_threshold = np.percentile(variances, 10)  # 最静态的10%
        static_indices = [i for i, v in enumerate(variances) if v <= static_threshold]
        
        # 如果没有足够静态的区间，使用整个信号但打印警告
        if len(static_indices) < 3:
            print(f"WARNING: Could not find enough static data for noise floor estimation")
            # 使用welch方法计算功率谱密度，应用hann窗口
            f, psd = welch(signal, fs=fs, window='hann', nperseg=min(4096, len(signal)//4))
            # 使用低频部分中位数作为噪声底（除去DC）
            noise_floor = np.median(psd[1:int(len(psd)*0.2)])
            return f, psd, fs, noise_floor
        
        # 从静态区间创建蒙太奇数据
        static_data = []
        for idx in static_indices[:5]:  # 使用最多5个静态窗口
            start_idx = idx * (window_size // 2)
            static_data.extend(signal[start_idx:start_idx+window_size])
        
        static_data = np.array(static_data)
        
        # 使用welch方法计算静态数据的功率谱密度
        f, psd = welch(static_data, fs=fs, window='hann', nperseg=min(4096, len(static_data)//4))
        
        # 计算PSD中位数，忽略非常低频部分（通常包含DC和漂移）
        noise_floor = np.median(psd[int(len(psd)*0.05):int(len(psd)*0.5)])
        
        # 全信号的频谱分析用于绘图
        f_full, psd_full = welch(signal, fs=fs, window='hann', nperseg=min(4096, len(signal)//4))
        
        # 分析不同频率范围的PSD斜率
        # 1. 低频范围 (0.1Hz-1Hz) - 通常捕获褐噪声特性
        # 2. 中频范围 (1Hz-10Hz) - 通常捕获粉红噪声特性
        # 3. 高频范围 (10Hz-Nyquist) - 通常捕获白噪声特性
        
        # 定义频率范围
        f_low_min, f_low_max = 0.1, 1.0
        f_mid_min, f_mid_max = 1.0, 10.0
        f_high_min = 10.0
        f_high_max = fs/2 * 0.9  # 90% 奈奎斯特频率
        
        # 计算各频率范围的斜率
        slopes = {}
        noise_types = {}
        
        # 计算各频段斜率
        if f_full[0] < f_low_max and f_full[-1] > f_low_min:
            slopes['low'], r2_low, noise_types['low'] = calculate_slope(f_full, psd_full, f_low_min, f_low_max)
        
        if f_full[0] < f_mid_max and f_full[-1] > f_mid_min:
            slopes['mid'], r2_mid, noise_types['mid'] = calculate_slope(f_full, psd_full, f_mid_min, f_mid_max)
        
        if f_full[0] < f_high_max and f_full[-1] > f_high_min:
            slopes['high'], r2_high, noise_types['high'] = calculate_slope(f_full, psd_full, f_high_min, f_high_max)
        
        return f_full, psd_full, fs, noise_floor, slopes, noise_types

    # 更新频谱图绘制部分
    if ("accel" in analysis_target or analysis_target == "all") and 'acc_x' in imu_data.columns:
        plt.figure(figsize=(20, 15))
        for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            xf, ps, fs, noise_floor, slopes, noise_types = improved_frequency_analysis(imu_data[axis], imu_data['timestamp'])
            plt.subplot(3, 1, i + 1)
            plt.loglog(xf, ps, label=f'{axis} Spectrum')  # 使用对数-对数坐标
            plt.axhline(y=noise_floor, color='r', linestyle='--', label='Noise Floor')
            
            # 添加斜率拟合线和噪声类型标注
            title = f'{axis} Frequency Spectrum\nSampling Rate: {fs:.1f} Hz, Noise Floor: {noise_floor:.6f} m/s²\n'
            
            # 为每个频率范围添加斜率和噪声类型信息
            for range_name, range_label, color in [
                ('low', 'Low Freq(0.1-1Hz)', 'green'), 
                ('mid', 'Mid Freq(1-10Hz)', 'blue'), 
                ('high', 'High Freq(10Hz+)', 'purple')
            ]:
                if range_name in slopes and slopes[range_name] is not None:
                    slope = slopes[range_name]
                    noise_type = noise_types[range_name]
                    title += f"{range_label}: Slope={slope:.2f} ({noise_type}), "
                    
                    # 在图中绘制拟合线
                    if range_name == 'low':
                        x_fit = np.logspace(np.log10(0.1), np.log10(1.0), 100)
                    elif range_name == 'mid':
                        x_fit = np.logspace(np.log10(1.0), np.log10(10.0), 100)
                    else:  # high
                        x_fit = np.logspace(np.log10(10.0), np.log10(fs/2*0.9), 100)
                    
                    # 找到参考点
                    if range_name == 'low':
                        ref_idx = np.argmin(np.abs(xf - 0.5))
                    elif range_name == 'mid':
                        ref_idx = np.argmin(np.abs(xf - 3.0))
                    else:  # high
                        ref_idx = np.argmin(np.abs(xf - 20.0))
                    
                    if ref_idx < len(ps):
                        ref_point = ps[ref_idx]
                        ref_freq = xf[ref_idx]
                        y_fit = ref_point * (x_fit/ref_freq)**slope
                        plt.plot(x_fit, y_fit, color=color, linestyle='-', 
                                 label=f'{range_label} Slope: {slope:.2f}')
            
            plt.title(title)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (m²/s⁴/Hz)')
            plt.grid(True, which="both")
            plt.legend()

        plt.tight_layout()
        accel_spectrum_path = os.path.join(save_dir, 'acceleration_spectrum.png')
        plt.savefig(accel_spectrum_path)
        plt.close()

    if ("gyro" in analysis_target or analysis_target == "all") and 'gyro_x' in imu_data.columns:
        plt.figure(figsize=(20, 15))
        for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            xf, ps, fs, noise_floor, slopes, noise_types = improved_frequency_analysis(imu_data[axis], imu_data['timestamp'])
            plt.subplot(3, 1, i + 1)
            plt.loglog(xf, ps, label=f'{axis} Spectrum')  # 使用对数-对数坐标
            plt.axhline(y=noise_floor, color='r', linestyle='--', label='Noise Floor')
            
            # 添加斜率拟合线和噪声类型标注
            title = f'{axis} Frequency Spectrum\nSampling Rate: {fs:.1f} Hz, Noise Floor: {noise_floor:.6f} rad/s\n'
            
            # 为每个频率范围添加斜率和噪声类型信息
            for range_name, range_label, color in [
                ('low', 'Low Freq(0.1-1Hz)', 'green'), 
                ('mid', 'Mid Freq(1-10Hz)', 'blue'), 
                ('high', 'High Freq(10Hz+)', 'purple')
            ]:
                if range_name in slopes and slopes[range_name] is not None:
                    slope = slopes[range_name]
                    noise_type = noise_types[range_name]
                    title += f"{range_label}: Slope={slope:.2f} ({noise_type}), "
                    
                    # 在图中绘制拟合线
                    if range_name == 'low':
                        x_fit = np.logspace(np.log10(0.1), np.log10(1.0), 100)
                    elif range_name == 'mid':
                        x_fit = np.logspace(np.log10(1.0), np.log10(10.0), 100)
                    else:  # high
                        x_fit = np.logspace(np.log10(10.0), np.log10(fs/2*0.9), 100)
                    
                    # 找到参考点
                    if range_name == 'low':
                        ref_idx = np.argmin(np.abs(xf - 0.5))
                    elif range_name == 'mid':
                        ref_idx = np.argmin(np.abs(xf - 3.0))
                    else:  # high
                        ref_idx = np.argmin(np.abs(xf - 20.0))
                    
                    if ref_idx < len(ps):
                        ref_point = ps[ref_idx]
                        ref_freq = xf[ref_idx]
                        y_fit = ref_point * (x_fit/ref_freq)**slope
                        plt.plot(x_fit, y_fit, color=color, linestyle='-', 
                                 label=f'{range_label} Slope: {slope:.2f}')
            
            plt.title(title)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (rad²/s²/Hz)')
            plt.grid(True, which="both")
            plt.legend()
            
            # 将噪声底和斜率信息存储到噪声统计中
            noise_stats['gyroscope'][axis[-1]]['noise_floor'] = noise_floor
            for range_name in slopes:
                if slopes[range_name] is not None:
                    noise_stats['gyroscope'][axis[-1]][f'slope_{range_name}'] = slopes[range_name]
                    noise_stats['gyroscope'][axis[-1]][f'noise_type_{range_name}'] = noise_types[range_name]

        plt.tight_layout()
        gyro_spectrum_path = os.path.join(save_dir, 'gyroscope_spectrum.png')
        plt.savefig(gyro_spectrum_path)
        plt.close()

    # Print summary of noise statistics
    print("\n===== IMU Noise Analysis Summary =====")
    if 'accelerometer' in noise_stats:
        print("\nAccelerometer Noise Statistics:")
        for axis in ['x', 'y', 'z']:
            stats = noise_stats['accelerometer'][axis]
            print(f"  {axis}-axis: Mean={stats['mean']:.6f}, Std={stats['std']:.6f}, RMS={stats['rms']:.6f}, P2P={stats['peak_to_peak']:.6f}")
            if 'noise_floor' in stats:
                print(f"        Noise Floor={stats['noise_floor']:.6f} m/s²")
            for range_name, label in [('low','Low'), ('mid','Mid'), ('high','High')]:
                if f'slope_{range_name}' in stats:
                    print(f"       {label} Freq Slope={stats[f'slope_{range_name}']:.2f} ({stats[f'noise_type_{range_name}']})")
    
    if 'gyroscope' in noise_stats:
        print("\nGyroscope Noise Statistics:")
        for axis in ['x', 'y', 'z']:
            stats = noise_stats['gyroscope'][axis]
            print(f"  {axis}-axis: Mean={stats['mean']:.6f}, Std={stats['std']:.6f}, RMS={stats['rms']:.6f}, P2P={stats['peak_to_peak']:.6f}")
            if 'noise_floor' in stats:
                print(f"        Noise Floor={stats['noise_floor']:.6f} rad/s")
            for range_name, label in [('low','Low'), ('mid','Mid'), ('high','High')]:
                if f'slope_{range_name}' in stats:
                    print(f"       {label} Freq Slope={stats[f'slope_{range_name}']:.2f} ({stats[f'noise_type_{range_name}']})")
    
    return noise_stats

def analyze_imu_dynamic_noise(file_path, save_dir="dynamic_noise_results"):
    """分析IMU在不同运动状态下的噪声特性"""
    print(f"分析文件: {file_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载数据
    imu_data = load_imu_data(file_path)
    print(f"加载了 {len(imu_data)} 行数据")
    
    # 2. 识别运动状态
    motion_states = identify_motion_states(imu_data)
    print(f"识别到的运动状态: {list(motion_states.keys())}")
    
    # 3. 分析每种状态下的噪声
    results = {}
    for state, indices in motion_states.items():
        if len(indices) > 50:  # 确保有足够的数据点
            print(f"分析 '{state}' 状态 ({len(indices)} 个数据点)")
            state_data = imu_data.iloc[indices]
            results[state] = analyze_state_noise(state_data)
    
    # 4. 可视化结果
    visualize_results(imu_data, motion_states, results, save_dir)
    
    # 5. 生成报告
    generate_report(results, save_dir)
    
    return results

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

def analyze_state_noise(state_data):
    """分析特定状态下的噪声特性"""
    results = {}
    
    # 1. 基本统计量
    basic_stats = {}
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        basic_stats[axis] = {
            'mean': state_data[axis].mean(),
            'std': state_data[axis].std(),
            'min': state_data[axis].min(),
            'max': state_data[axis].max(),
            'p2p': state_data[axis].max() - state_data[axis].min(),
            'kurtosis': stats.kurtosis(state_data[axis].dropna()),
            'skewness': stats.skew(state_data[axis].dropna())
        }
    results['basic_stats'] = basic_stats
    
    # 2. 频谱分析
    spectral = {}
    fs = 1.0 / np.mean(state_data['dt'])  # 采样频率
    
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        f, psd = welch(state_data[axis].dropna(), fs=fs, nperseg=min(256, len(state_data)//4))
        
        # 计算不同频段的能量
        low_freq = np.sum(psd[(f >= 0.1) & (f < 1.0)])
        mid_freq = np.sum(psd[(f >= 1.0) & (f < 10.0)])
        high_freq = np.sum(psd[f >= 10.0])
        total = low_freq + mid_freq + high_freq
        
        # 计算频段能量比例
        spectral[axis] = {
            'low_freq_ratio': low_freq / total if total > 0 else 0,
            'mid_freq_ratio': mid_freq / total if total > 0 else 0,
            'high_freq_ratio': high_freq / total if total > 0 else 0
        }
        
        # 计算频谱斜率 (对数-对数尺度)
        if len(f) > 5 and len(psd) > 5:
            # 低频段斜率 (0.1-1Hz)
            low_indices = np.where((f >= 0.1) & (f <= 1.0))[0]
            if len(low_indices) >= 3:
                log_f_low = np.log10(f[low_indices])
                log_psd_low = np.log10(psd[low_indices])
                slope_low, _, _, _, _ = stats.linregress(log_f_low, log_psd_low)
                spectral[axis]['low_freq_slope'] = slope_low
            
            # 中频段斜率 (1-10Hz)
            mid_indices = np.where((f > 1.0) & (f <= 10.0))[0]
            if len(mid_indices) >= 3:
                log_f_mid = np.log10(f[mid_indices])
                log_psd_mid = np.log10(psd[mid_indices])
                slope_mid, _, _, _, _ = stats.linregress(log_f_mid, log_psd_mid)
                spectral[axis]['mid_freq_slope'] = slope_mid
            
            # 高频段斜率 (>10Hz)
            high_indices = np.where(f > 10.0)[0]
            if len(high_indices) >= 3:
                log_f_high = np.log10(f[high_indices])
                log_psd_high = np.log10(psd[high_indices])
                slope_high, _, _, _, _ = stats.linregress(log_f_high, log_psd_high)
                spectral[axis]['high_freq_slope'] = slope_high
    
    results['spectral'] = spectral
    
    # 3. 噪声类型分类
    noise_types = {}
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        noise_types[axis] = {}
        
        for freq_range in ['low_freq_slope', 'mid_freq_slope', 'high_freq_slope']:
            if freq_range in spectral[axis]:
                slope = spectral[axis][freq_range]
                
                if -0.3 <= slope <= 0.3:
                    noise_type = "白噪声"
                elif -1.3 <= slope <= -0.7:
                    noise_type = "粉红噪声(1/f)"
                elif -2.3 <= slope <= -1.7:
                    noise_type = "褐噪声(1/f²)"
                else:
                    noise_type = "其他噪声"
                
                noise_types[axis][freq_range.replace('_slope', '_type')] = noise_type
    
    results['noise_types'] = noise_types
    
    return results

def visualize_results(imu_data, motion_states, results, save_dir):
    """可视化分析结果"""
    # 1. 绘制运动状态分类
    plt.figure(figsize=(15, 10))
    
    # 转换为相对时间(秒)
    time_sec = (imu_data['timestamp'] - imu_data['timestamp'].iloc[0]) / 1000.0
    
    # 绘制加速度和角速度
    plt.subplot(2, 1, 1)
    plt.plot(time_sec, imu_data['acc_mag'], 'b-', alpha=0.5)
    
    # 标记不同状态
    colors = {'静止': 'green', '匀速运动': 'blue', '动态运动': 'red'}
    for state, indices in motion_states.items():
        if indices:
            plt.scatter(time_sec.iloc[indices], imu_data['acc_mag'].iloc[indices], 
                       c=colors.get(state, 'gray'), label=state, s=5, alpha=0.5)
    
    plt.title('加速度幅值与运动状态')
    plt.ylabel('加速度 (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_sec, imu_data['gyro_mag'], 'r-', alpha=0.5)
    
    for state, indices in motion_states.items():
        if indices:
            plt.scatter(time_sec.iloc[indices], imu_data['gyro_mag'].iloc[indices], 
                       c=colors.get(state, 'gray'), label=state, s=5, alpha=0.5)
    
    plt.title('角速度幅值与运动状态')
    plt.xlabel('时间 (秒)')
    plt.ylabel('角速度 (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'motion_states.png'))
    plt.close()
    
    # 2. 为每种状态绘制频谱斜率
    for state, state_results in results.items():
        if 'spectral' in state_results and 'noise_types' in state_results:
            plt.figure(figsize=(15, 10))
            
            # 陀螺仪频谱斜率
            plt.subplot(2, 1, 1)
            axes = ['gyro_x', 'gyro_y', 'gyro_z']
            freq_ranges = ['low_freq_slope', 'mid_freq_slope', 'high_freq_slope']
            
            data = []
            labels = []
            colors = []
            
            for axis in axes:
                for freq_range in freq_ranges:
                    if freq_range in state_results['spectral'][axis]:
                        slope = state_results['spectral'][axis][freq_range]
                        data.append(slope)
                        labels.append(f"{axis}_{freq_range.split('_')[0]}")
                        
                        # 根据噪声类型设置颜色
                        noise_type = state_results['noise_types'][axis].get(freq_range.replace('_slope', '_type'), '')
                        if '白噪声' in noise_type:
                            colors.append('blue')
                        elif '粉红噪声' in noise_type:
                            colors.append('red')
                        elif '褐噪声' in noise_type:
                            colors.append('brown')
                        else:
                            colors.append('gray')
            
            bars = plt.bar(range(len(data)), data, color=colors)
            plt.xticks(range(len(data)), labels, rotation=45)
            plt.title(f'{state}状态下陀螺仪频谱斜率')
            plt.ylabel('斜率')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
            plt.axhline(y=-2, color='k', linestyle='--', alpha=0.3)
            plt.grid(True)
            
            # 加速度计频谱斜率
            plt.subplot(2, 1, 2)
            axes = ['acc_x', 'acc_y', 'acc_z']
            
            data = []
            labels = []
            colors = []
            
            for axis in axes:
                for freq_range in freq_ranges:
                    if freq_range in state_results['spectral'][axis]:
                        slope = state_results['spectral'][axis][freq_range]
                        data.append(slope)
                        labels.append(f"{axis}_{freq_range.split('_')[0]}")
                        
                        noise_type = state_results['noise_types'][axis].get(freq_range.replace('_slope', '_type'), '')
                        if '白噪声' in noise_type:
                            colors.append('blue')
                        elif '粉红噪声' in noise_type:
                            colors.append('red')
                        elif '褐噪声' in noise_type:
                            colors.append('brown')
                        else:
                            colors.append('gray')
            
            bars = plt.bar(range(len(data)), data, color=colors)
            plt.xticks(range(len(data)), labels, rotation=45)
            plt.title(f'{state}状态下加速度计频谱斜率')
            plt.ylabel('斜率')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
            plt.axhline(y=-2, color='k', linestyle='--', alpha=0.3)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{state}_spectral_slopes.png'))
            plt.close()

def generate_report(results, save_dir):
    """生成分析报告"""
    report_path = os.path.join(save_dir, 'dynamic_noise_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== IMU动态噪声分析报告 ===\n\n")
        
        for state, state_results in results.items():
            f.write(f"== {state}状态噪声分析 ==\n\n")
            
            # 基本统计
            f.write("基本统计指标:\n")
            for sensor_type, axes in [('加速度计', ['acc_x', 'acc_y', 'acc_z']), 
                                     ('陀螺仪', ['gyro_x', 'gyro_y', 'gyro_z'])]:
                f.write(f"\n{sensor_type}:\n")
                for axis in axes:
                    stats = state_results['basic_stats'][axis]
                    f.write(f"  {axis}: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}, ")
                    f.write(f"峰峰值={stats['p2p']:.6f}\n")
            
            # 频谱特性
            if 'spectral' in state_results:
                f.write("\n频谱特性:\n")
                for sensor_type, axes in [('加速度计', ['acc_x', 'acc_y', 'acc_z']), 
                                         ('陀螺仪', ['gyro_x', 'gyro_y', 'gyro_z'])]:
                    f.write(f"\n{sensor_type}:\n")
                    for axis in axes:
                        spectral = state_results['spectral'][axis]
                        f.write(f"  {axis}:\n")
                        f.write(f"    能量分布: 低频={spectral.get('low_freq_ratio', 0):.2f}, ")
                        f.write(f"中频={spectral.get('mid_freq_ratio', 0):.2f}, ")
                        f.write(f"高频={spectral.get('high_freq_ratio', 0):.2f}\n")
                        
                        if 'low_freq_slope' in spectral:
                            f.write(f"    低频斜率: {spectral['low_freq_slope']:.2f}")
                            if 'noise_types' in state_results and 'low_freq_type' in state_results['noise_types'][axis]:
                                f.write(f" ({state_results['noise_types'][axis]['low_freq_type']})")
                            f.write("\n")
                        
                        if 'mid_freq_slope' in spectral:
                            f.write(f"    中频斜率: {spectral['mid_freq_slope']:.2f}")
                            if 'noise_types' in state_results and 'mid_freq_type' in state_results['noise_types'][axis]:
                                f.write(f" ({state_results['noise_types'][axis]['mid_freq_type']})")
                            f.write("\n")
                        
                        if 'high_freq_slope' in spectral:
                            f.write(f"    高频斜率: {spectral['high_freq_slope']:.2f}")
                            if 'noise_types' in state_results and 'high_freq_type' in state_results['noise_types'][axis]:
                                f.write(f" ({state_results['noise_types'][axis]['high_freq_type']})")
                            f.write("\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"报告已生成: {report_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析IMU在不同运动状态下的噪声特性")
    parser.add_argument('--file', type=str, required=True, help="IMU数据文件路径")
    parser.add_argument('--output', type=str, default="dynamic_noise_results", help="结果保存目录")
    
    args = parser.parse_args()
    
    analyze_imu_dynamic_noise(args.file, args.output)

