import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from scipy import stats
import allantools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_imu_noise_levels(file_path, save_dir="noise_analysis_results"):
    """
    对IMU数据进行全面的噪声水平分析
    
    Parameters:
        file_path (str): IMU数据文件路径
        save_dir (str): 结果保存目录
    """
    print(f"分析IMU噪声水平: {file_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载数据
    imu_data = load_imu_data(file_path)
    
    # 2. 识别运动状态
    motion_states = identify_motion_states(imu_data)
    
    # 3. 对每种状态进行噪声分析
    noise_results = {}
    for state, indices in motion_states.items():
        if len(indices) > 200:  # 确保有足够的数据点
            print(f"分析 '{state}' 状态噪声 ({len(indices)} 个数据点)")
            state_data = imu_data.iloc[indices]
            noise_results[state] = analyze_state_noise_levels(state_data, state, save_dir)
    
    # 4. 进行Allan方差分析
    print("执行Allan方差分析...")
    allan_results = perform_allan_variance_analysis(imu_data, motion_states, save_dir)
    
    # 5. 分析温度对噪声的影响 (如果有温度数据)
    if 'temperature' in imu_data.columns:
        print("分析温度对噪声的影响...")
        temp_noise_relation = analyze_temperature_effects(imu_data, save_dir)
    
    # 6. 生成详细报告
    generate_detailed_report(noise_results, allan_results, save_dir)
    
    return noise_results

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
    data['sampling_rate'] = 1.0 / data['dt']
    
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

def analyze_state_noise_levels(state_data, state_name, save_dir):
    """分析特定状态下的噪声水平"""
    results = {
        'basic_stats': {},
        'spectral': {},
        'noise_types': {},
        'noise_levels': {},
        'distribution': {}
    }
    
    # 1. 基本统计量
    for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        data = state_data[axis].values
        results['basic_stats'][axis] = {
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
        
        # 计算不同频段的能量比例
        low_freq_mask = (f >= 0.1) & (f < 1.0)
        mid_freq_mask = (f >= 1.0) & (f < 10.0)
        high_freq_mask = f >= 10.0
        
        total_energy = np.sum(psd)
        low_freq_energy = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
        mid_freq_energy = np.sum(psd[mid_freq_mask]) if np.any(mid_freq_mask) else 0
        high_freq_energy = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        # 计算频谱平坦度
        spectral_flatness = stats.gmean(psd[psd > 0]) / np.mean(psd[psd > 0]) if np.any(psd > 0) else 0
        
        # 计算各频段斜率
        results['spectral'][axis] = {
            'low_freq_ratio': low_freq_energy / total_energy if total_energy > 0 else 0,
            'mid_freq_ratio': mid_freq_energy / total_energy if total_energy > 0 else 0,
            'high_freq_ratio': high_freq_energy / total_energy if total_energy > 0 else 0,
            'spectral_flatness': spectral_flatness
        }
        
        results['noise_types'][axis] = {}
        
        # 低频斜率
        if np.any(low_freq_mask) and np.sum(low_freq_mask) >= 5:
            low_f = f[low_freq_mask]
            low_psd = psd[low_freq_mask]
            if np.all(low_psd > 0) and np.all(low_f > 0):
                slope, _, _, _, _ = stats.linregress(np.log10(low_f), np.log10(low_psd))
                results['spectral'][axis]['low_freq_slope'] = slope
                
                # 确定噪声类型
                noise_type = "其他噪声"
                if -0.3 <= slope <= 0.3:
                    noise_type = "白噪声"
                elif -1.3 <= slope <= -0.7:
                    noise_type = "粉红噪声(1/f)"
                elif -2.3 <= slope <= -1.7:
                    noise_type = "褐噪声(1/f²)"
                results['noise_types'][axis]['low_freq_type'] = noise_type
        
        # 中频斜率
        if np.any(mid_freq_mask) and np.sum(mid_freq_mask) >= 5:
            mid_f = f[mid_freq_mask]
            mid_psd = psd[mid_freq_mask]
            if np.all(mid_psd > 0) and np.all(mid_f > 0):
                slope, _, _, _, _ = stats.linregress(np.log10(mid_f), np.log10(mid_psd))
                results['spectral'][axis]['mid_freq_slope'] = slope
                
                noise_type = "其他噪声"
                if -0.3 <= slope <= 0.3:
                    noise_type = "白噪声"
                elif -1.3 <= slope <= -0.7:
                    noise_type = "粉红噪声(1/f)"
                elif -2.3 <= slope <= -1.7:
                    noise_type = "褐噪声(1/f²)"
                results['noise_types'][axis]['mid_freq_type'] = noise_type
        
        # 高频斜率
        if np.any(high_freq_mask) and np.sum(high_freq_mask) >= 5:
            high_f = f[high_freq_mask]
            high_psd = psd[high_freq_mask]
            if np.all(high_psd > 0) and np.all(high_f > 0):
                slope, _, _, _, _ = stats.linregress(np.log10(high_f), np.log10(high_psd))
                results['spectral'][axis]['high_freq_slope'] = slope
                
                noise_type = "其他噪声"
                if -0.3 <= slope <= 0.3:
                    noise_type = "白噪声"
                elif -1.3 <= slope <= -0.7:
                    noise_type = "粉红噪声(1/f)"
                elif -2.3 <= slope <= -1.7:
                    noise_type = "褐噪声(1/f²)"
                results['noise_types'][axis]['high_freq_type'] = noise_type
        
        # 绘制PSD图
        plt.figure(figsize=(12, 8))
        plt.loglog(f, psd, 'b-', alpha=0.7)
        
        # 标记不同频段
        if np.any(low_freq_mask):
            plt.loglog(f[low_freq_mask], psd[low_freq_mask], 'g-', linewidth=2, label='低频(0.1-1Hz)')
        if np.any(mid_freq_mask):
            plt.loglog(f[mid_freq_mask], psd[mid_freq_mask], 'r-', linewidth=2, label='中频(1-10Hz)')
        if np.any(high_freq_mask):
            plt.loglog(f[high_freq_mask], psd[high_freq_mask], 'y-', linewidth=2, label='高频(>10Hz)')
        
        # 添加斜率参考线
        if 'low_freq_slope' in results['spectral'][axis]:
            x_ref = np.logspace(np.log10(f[low_freq_mask][0]), np.log10(f[low_freq_mask][-1]), 100)
            y_ref = 10**(np.log10(psd[low_freq_mask][0]) + results['spectral'][axis]['low_freq_slope'] * 
                         (np.log10(x_ref) - np.log10(f[low_freq_mask][0])))
            plt.loglog(x_ref, y_ref, 'g--', alpha=0.7, 
                      label=f"低频斜率: {results['spectral'][axis]['low_freq_slope']:.2f}")
        
        plt.title(f'{state_name}状态下 {axis} 的功率谱密度')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('PSD')
        plt.grid(True, which="both", ls="-", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{state_name}_{axis}_psd.png'))
        plt.close()
        
        # 3. 计算噪声水平指标
        # 角度随机游走 (ARW) 或速度随机游走 (VRW)
        if axis.startswith('gyro'):
            # 对于陀螺仪，计算ARW (deg/√h)
            # ARW = σ * √(sampling_rate) * (180/π) * 60
            arw = results['basic_stats'][axis]['std'] * np.sqrt(sampling_rate) * (180/np.pi) * 60
            results['noise_levels'][axis] = {'ARW': arw}
        else:
            # 对于加速度计，计算VRW (m/s/√h)
            # VRW = σ * √(sampling_rate) * 60
            vrw = results['basic_stats'][axis]['std'] * np.sqrt(sampling_rate) * 60
            results['noise_levels'][axis] = {'VRW': vrw}
        
        # 4. 分析噪声分布
        data = detrend(state_data[axis].values)
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(data)
        pdf_fitted = stats.norm.pdf(bin_centers, mu, sigma)
        
        # 计算分布的峰度和偏度
        kurtosis = stats.kurtosis(data)
        skewness = stats.skew(data)
        
        # 进行Shapiro-Wilk正态性检验
        shapiro_test = stats.shapiro(data[:5000] if len(data) > 5000 else data)  # 限制样本大小
        
        results['distribution'][axis] = {
            'mu': mu,
            'sigma': sigma,
            'kurtosis': kurtosis,
            'skewness': skewness,
            'shapiro_test_p': shapiro_test.pvalue
        }
        
        # 绘制分布图
        plt.figure(figsize=(12, 8))
        plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
        plt.plot(bin_centers, pdf_fitted, 'r-', linewidth=2, 
                label=f'正态分布拟合 (μ={mu:.4f}, σ={sigma:.4f})')
        plt.title(f'{state_name}状态下 {axis} 的噪声分布\n'
                 f'峰度={kurtosis:.2f}, 偏度={skewness:.2f}, p值={shapiro_test.pvalue:.4f}')
        plt.xlabel('值')
        plt.ylabel('概率密度')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{state_name}_{axis}_distribution.png'))
        plt.close()
    
    return results

def perform_allan_variance_analysis(imu_data, motion_states, save_dir):
    """执行Allan方差分析"""
    results = {}
    
    # 对每种状态分别进行Allan方差分析
    for state, indices in motion_states.items():
        if len(indices) > 1000:  # 需要足够长的数据
            state_data = imu_data.iloc[indices]
            results[state] = {}
            
            # 计算平均采样率
            sampling_rate = 1.0 / np.mean(state_data['dt'])
            
            # 对每个轴进行分析
            for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                data = state_data[axis].values
                
                # 计算Allan方差
                t0 = 1.0 / sampling_rate  # 采样间隔
                (taus, adevs, _, _) = allantools.oadev(data, rate=sampling_rate, data_type="freq")
                
                # 存储结果
                results[state][axis] = {
                    'taus': taus,
                    'adevs': adevs
                }
                
                # 估计噪声参数
                # 1. 寻找最小点
                min_idx = np.argmin(adevs)
                min_tau = taus[min_idx]
                min_adev = adevs[min_idx]
                
                # 2. 拟合白噪声区域 (斜率 -0.5)
                short_term_idx = np.where(taus < min_tau/2)[0]
                if len(short_term_idx) > 5:
                    log_taus = np.log10(taus[short_term_idx])
                    log_adevs = np.log10(adevs[short_term_idx])
                    slope, intercept, _, _, _ = stats.linregress(log_taus, log_adevs)
                    
                    # 如果斜率接近 -0.5，则为白噪声区域
                    if -0.6 < slope < -0.4:
                        # 计算ARW或VRW
                        if axis.startswith('gyro'):
                            # 对于陀螺仪，计算ARW (deg/√h)
                            arw = 10**intercept * np.sqrt(1) * (180/np.pi) * 60
                            results[state][axis]['ARW'] = arw
                        else:
                            # 对于加速度计，计算VRW (m/s/√h)
                            vrw = 10**intercept * np.sqrt(1) * 60
                            results[state][axis]['VRW'] = vrw
                
                # 3. 拟合长期偏置不稳定性 (斜率 0.5)
                long_term_idx = np.where(taus > min_tau*2)[0]
                if len(long_term_idx) > 5:
                    log_taus = np.log10(taus[long_term_idx])
                    log_adevs = np.log10(adevs[long_term_idx])
                    slope, intercept, _, _, _ = stats.linregress(log_taus, log_adevs)
                    
                    # 如果斜率接近 0.5，则为随机游走区域
                    if 0.4 < slope < 0.6:
                        # 计算偏置不稳定性
                        if axis.startswith('gyro'):
                            # 对于陀螺仪，计算偏置不稳定性 (deg/h)
                            bi = 10**intercept / np.sqrt(1) * (180/np.pi) * 3600
                            results[state][axis]['BI'] = bi
                        else:
                            # 对于加速度计，计算偏置不稳定性 (m/s²)
                            bi = 10**intercept / np.sqrt(1)
                            results[state][axis]['BI'] = bi
                
                # 绘制Allan方差图
                plt.figure(figsize=(12, 8))
                plt.loglog(taus, adevs, 'b.-', alpha=0.7)
                
                # 标记最小点
                plt.plot(min_tau, min_adev, 'ro', markersize=8)
                plt.annotate(f'最小点: τ={min_tau:.2f}s', 
                            xy=(min_tau, min_adev), xytext=(min_tau*1.5, min_adev*1.5),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                
                # 添加参考斜率线
                tau_range = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 100)
                
                # 斜率 -0.5 (白噪声)
                ref_line = tau_range**(-0.5) * tau_range[0]**0.5 * adevs[0]
                plt.loglog(tau_range, ref_line, 'g--', alpha=0.7, label='斜率 -0.5 (白噪声/ARW/VRW)')
                
                # 斜率 0 (偏置不稳定性)
                ref_line = np.ones_like(tau_range) * min_adev
                plt.loglog(tau_range, ref_line, 'r--', alpha=0.7, label='斜率 0 (偏置不稳定性)')
                
                # 斜率 0.5 (随机游走)
                ref_line = tau_range**(0.5) * tau_range[0]**(-0.5) * adevs[-1]
                plt.loglog(tau_range, ref_line, 'y--', alpha=0.7, label='斜率 0.5 (随机游走)')
                
                # 添加噪声参数标注
                annotation_text = ""
                if axis.startswith('gyro'):
                    if 'ARW' in results[state][axis]:
                        annotation_text += f"ARW = {results[state][axis]['ARW']:.4f} deg/√h\n"
                    if 'BI' in results[state][axis]:
                        annotation_text += f"偏置不稳定性 = {results[state][axis]['BI']:.4f} deg/h"
                else:
                    if 'VRW' in results[state][axis]:
                        annotation_text += f"VRW = {results[state][axis]['VRW']:.4f} m/s/√h\n"
                    if 'BI' in results[state][axis]:
                        annotation_text += f"偏置不稳定性 = {results[state][axis]['BI']:.4f} m/s²"
                
                if annotation_text:
                    plt.annotate(annotation_text, xy=(0.05, 0.05), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
                
                plt.title(f'{state}状态下 {axis} 的Allan方差')
                plt.xlabel('积分时间 τ (秒)')
                plt.ylabel('Allan标准差')
                plt.grid(True, which="both", ls="-", alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{state}_{axis}_allan.png'))
                plt.close()
    
    return results

def analyze_temperature_effects(imu_data, save_dir):
    """分析温度对噪声的影响"""
    results = {}
    
    if 'temperature' not in imu_data.columns:
        print("数据中没有温度信息，跳过温度分析")
        return results
    
    # 计算温度范围
    temp_min = imu_data['temperature'].min()
    temp_max = imu_data['temperature'].max()
    
    # 如果温度变化不大，则跳过
    if temp_max - temp_min < 5:
        print(f"温度变化范围较小 ({temp_min:.1f}°C - {temp_max:.1f}°C)，跳过温度分析")
        return results
    
    # 将温度分成多个区间
    temp_bins = np.linspace(temp_min, temp_max, 6)
    bin_labels = [f"{temp_bins[i]:.1f}-{temp_bins[i+1]:.1f}°C" for i in range(len(temp_bins)-1)]
    
    # 对每个温度区间分析噪声
    for i in range(len(temp_bins)-1):
        temp_range = bin_labels[i]
        temp_indices = imu_data[(imu_data['temperature'] >= temp_bins[i]) & 
                               (imu_data['temperature'] < temp_bins[i+1])].index.tolist()
        
        if len(temp_indices) > 200:
            temp_data = imu_data.iloc[temp_indices]
            results[temp_range] = {}
            
            # 计算每个轴的噪声统计
            for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                data = temp_data[axis].values
                results[temp_range][axis] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'p2p': np.max(data) - np.min(data)
                }
    
    # 绘制温度与噪声关系图
    if results:
        for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro