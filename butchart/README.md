# 传感器数据帧率分析工具

这个工具用于分析IMU、rawgyroodo和图像数据的帧率和丢帧情况。

## 功能

- 分析IMU和rawgyroodo数据的帧率
- 检测IMU和rawgyroodo数据的丢帧情况
- 分析图像时间戳的帧率和丢帧情况
- 比较不同传感器数据的时间同步情况
- 生成帧率和丢帧分析图表（仅`analyze_frame_rate.py`）

## 依赖

```bash
pip install numpy matplotlib
```

## 使用方法

### 简单分析（无图表）

```bash
python check_frame_rate.py /path/to/RRLDR_fprintf.log [--image_dir /path/to/image_directory]
```

### 详细分析（带图表）

```bash
python analyze_frame_rate.py /path/to/RRLDR_fprintf.log [--image_dir /path/to/image_directory]
```

## 参数说明

- `log_file`: 包含IMU和rawgyroodo数据的日志文件路径
- `--image_dir`: 可选参数，图像目录路径，该目录下应包含camera0和camera1子目录

## 图像目录结构

图像目录应具有以下结构：

```
image_dir/
  ├── camera0/
  │   ├── 123456.jpg  # 文件名为时间戳
  │   ├── 123476.jpg
  │   └── ...
  └── camera1/
      ├── 123456.jpg
      ├── 123476.jpg
      └── ...
```

## 输出说明

脚本会输出以下信息：

1. IMU和rawgyroodo数据的帧率统计
2. 可能的丢帧情况
3. 图像数据的帧率统计（如果提供了图像目录）
4. 不同传感器之间的时间同步分析

`analyze_frame_rate.py`还会生成以下图表：

1. 帧间隔分布直方图
2. 帧间隔随时间变化图
3. 不同传感器时间戳对比图
4. 时间戳差异分析图

## 示例

```bash
# 只分析IMU和rawgyroodo数据
python check_frame_rate.py /home/roborock/datasets/roborock/stereo/0306_mower/RRLDR_fprintf.log

# 同时分析IMU、rawgyroodo和图像数据
python analyze_frame_rate.py /home/roborock/datasets/roborock/stereo/0306_mower/RRLDR_fprintf.log --image_dir /home/roborock/datasets/roborock/stereo/0306_mower/images
```

## 注意事项

1. 图像文件名必须是数字时间戳（可以有扩展名）
2. 时间戳单位假设为毫秒，帧率计算基于此假设
3. 丢帧检测基于帧间隔中位数的1.5倍作为阈值 