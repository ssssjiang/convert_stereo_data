#!/bin/bash

# 定义根目录
root_dir=$1
sensor_yaml=$2

# # 如果命名中有空格，先去掉空格，重命名
# root_dir_new=$(echo "$root_dir" | tr -d ' ')
# mv "$root_dir" "$root_dir_new"
# root_dir=$root_dir_new

# 调用 parse_image.py 脚本
python ../vision/parse_image.py --yuv_folder "$root_dir" --rotate_90 --resize_factor 0.5

root_dir=$root_dir"_rgb"

# 重构目录结构
mkdir -p "$root_dir/camera/camera0"
mkdir -p "$root_dir/camera/camera1"

# 查找所有图像文件
for file in "$root_dir"/*.png; do
  if [ -f "$file" ]; then
    # 获取文件名
    filename=$(basename "$file")
    
    # 提取时间戳字段（假设格式为 AT_544x640_4998_386_58435_left_0.png）
    # 格式也可能为 L_22001_640X544.png
    if [[ "$filename" == *"left"* || "$filename" == *"right"* ]]; then
      timestamp=$(echo "$filename" | awk -F'_' '{print $5}')
    elif [[ "$filename" == *"L_"* || "$filename" == *"R_"* ]]; then
      timestamp=$(echo "$filename" | awk -F'_' '{print $2}')
    fi
    
    # 判断是左目还是右目
    if [[ "$filename" == *"left"*  || "$filename" == *"L_"* ]]; then
      # 左目图像移动到camera0
      mv "$file" "$root_dir/camera/camera0/${timestamp}.png"
      echo "Left image: $file -> $root_dir/camera/camera0/${timestamp}.png"
    elif [[ "$filename" == *"right"* || "$filename" == *"R_"* ]]; then
      # 右目图像移动到camera1
      mv "$file" "$root_dir/camera/camera1/${timestamp}.png"
      echo "Right image: $file -> $root_dir/camera/camera1/${timestamp}.png"
    else
      echo "跳过不符合命名规则的文件: $file"
    fi
  fi
done

echo "目录重构和重命名完成！"

# convert sensor_yaml to vslam format
sensor_yaml_new=$sensor_yaml"_new"
python ../stereo_analyzer/convert_stereo_yaml.py --input "$sensor_yaml"  --output  "$sensor_yaml_new"  --template /home/roborock/下载/sy_calibr/sensor_sy_25.yaml

slam-toolkit --input "$root_dir" --steps convert
# slam-toolkit --input "$root_dir" --steps analyze --analyzers stereo --params /home/roborock/下载/sensor.yaml --stereo-sampling-rate 100
slam-toolkit --input "$root_dir" --steps analyze --analyzers frame_rate,stereo --params "$sensor_yaml_new" --stereo-sampling-rate 100