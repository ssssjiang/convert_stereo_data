import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体为英文
def configure_english_display():
    """配置matplotlib使用英文显示"""
    # 使用Matplotlib内置字体
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
    mpl.rcParams['axes.unicode_minus'] = True  # 正确显示负号
    
    print("Using English display settings")
    
    # 中英文标签映射字典
    return {
        # 状态标签
        "静止": "Static",
        "匀速运动": "Uniform Motion",
        "动态运动": "Dynamic Motion",
        
        # 传感器和轴标签
        "加速度计": "Accelerometer",
        "陀螺仪": "Gyroscope",
        "acc_x": "Acc X",
        "acc_y": "Acc Y",
        "acc_z": "Acc Z",
        "gyro_x": "Gyro X",
        "gyro_y": "Gyro Y",
        "gyro_z": "Gyro Z",
        
        # 频段标签
        "very_low": "Very Low",
        "low": "Low",
        "mid": "Mid",
        "high": "High",
        "极低频": "Very Low Freq",
        "低频": "Low Freq",
        "中频": "Mid Freq",
        "高频": "High Freq",
        
        # 噪声类型
        "白噪声": "White Noise",
        "粉红噪声(1/f)": "Pink Noise (1/f)",
        "褐噪声(1/f²)": "Brown Noise (1/f²)",
        "其他噪声": "Other Noise",
        
        # 图表标题和标签
        "不同运动状态下加速度计噪声水平 (标准差)": "Accelerometer Noise Level in Different Motion States (Std)",
        "不同运动状态下陀螺仪噪声水平 (标准差)": "Gyroscope Noise Level in Different Motion States (Std)",
        "在不同频段的噪声水平": "Noise Level in Different Frequency Bands",
        "在不同频段的频谱斜率": "Spectral Slope in Different Frequency Bands",
        "的噪声分布": "Noise Distribution",
        "状态下": "State",
        "频率 (Hz)": "Frequency (Hz)",
        "标准差": "Std",
        "峰峰值": "Peak-to-Peak",
        "均值": "Mean",
        "概率密度": "Probability Density",
        "运动状态": "Motion State",
        "频段": "Frequency Band",
        "噪声水平": "Noise Level",
        "斜率": "Slope",
        "峰度": "Kurtosis",
        "偏度": "Skewness",
        "值": "Value",
        "分析文件": "Analyzing file",
        "加载了": "Loaded",
        "行数据": "rows of data",
        "识别到的运动状态": "Detected motion states",
        "分析": "Analyzing",
        "状态": "state",
        "个数据点": "data points",
        "报告已生成": "Report generated",
        
        # 报告标题和内容
        "=== IMU不同运动状态噪声水平比较 ===": "=== IMU Noise Level Comparison in Different Motion States ===",
        "## 基本噪声水平比较 (标准差)": "## Basic Noise Level Comparison (Standard Deviation)",
        "### 加速度计 (m/s²)": "### Accelerometer (m/s²)",
        "### 陀螺仪 (rad/s)": "### Gyroscope (rad/s)",
        "## 不同频段噪声水平比较": "## Noise Level Comparison in Different Frequency Bands",
        "## 相对于静止状态的噪声增长比例": "## Noise Growth Ratio Relative to Static State",
        "| 轴 | 状态 | 标准差比例 | 峰峰值比例 | RMS比例 |": "| Axis | State | Std Ratio | P2P Ratio | RMS Ratio |",
        "|---|---|---|---|---|": "|---|---|---|---|---|",
        "| 频段 | 指标 |": "| Freq Band | Metric |",
        "| 噪声水平 |": "| Noise Level |",
        "| 斜率 |": "| Slope |",
        "| 噪声类型 |": "| Noise Type |"
    }

# 翻译函数
def translate_text(text, label_map):
    """将中文文本翻译为英文"""
    for cn, en in label_map.items():
        text = text.replace(cn, en)
    return text 