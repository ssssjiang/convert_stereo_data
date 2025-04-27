# Rosbag图像尺寸调整工具

这个工具用于读取rosbag文件中的图像数据，将图像尺寸调整为原始尺寸的一半（或指定的缩放比例），然后保存为新的rosbag文件。

## 功能特点

- 支持处理rosbag中的所有图像话题（sensor_msgs/Image类型）
- 默认将图像尺寸调整为原始尺寸的一半
- 可以自定义缩放比例
- 保持图像的原始编码格式和元数据
- 实时显示处理进度

## 依赖项

在使用此工具之前，请确保安装了以下依赖：

```bash
pip install -r requirements.txt
```

同时，您需要有一个正常工作的ROS环境，包括`cv_bridge`。

## 使用方法

### 基本用法

```bash
python resize_rosbag_images.py <输入rosbag路径> <输出rosbag路径>
```

这将读取输入rosbag文件，将其中所有图像尺寸调整为原始尺寸的一半，并将结果保存到新的rosbag文件中。

### 自定义缩放比例

```bash
python resize_rosbag_images.py <输入rosbag路径> <输出rosbag路径> <缩放比例>
```

例如，将图像缩放为原始尺寸的75%：

```bash
python resize_rosbag_images.py input.bag output.bag 0.75
```

## 示例

假设有一个名为`mk1_4_657.bag`的rosbag文件，包含`/image`和`/image1`两个图像话题：

```bash
python resize_rosbag_images.py mk1_4_657.bag mk1_4_657_resized.bag
```

这将创建一个新的rosbag文件`mk1_4_657_resized.bag`，其中的图像尺寸为原始尺寸的一半。

## 注意事项

- 处理大型rosbag文件可能需要较长时间，请耐心等待。
- 确保有足够的磁盘空间存储输出的rosbag文件。
- 如果处理过程中出现错误，程序会保留原始消息，确保输出的rosbag文件结构完整。 