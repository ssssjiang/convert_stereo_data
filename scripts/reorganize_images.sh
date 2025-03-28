#!/bin/bash

# 定义根目录
root_dir=$1

# 如果命名中有空格，先去掉空格，重命名
root_dir_new=$(echo "$root_dir" | tr -d ' ')
mv "$root_dir" "$root_dir_new"
root_dir=$root_dir_new

# 调用 parse_image.py 脚本
python vision/parse_image.py --yuv_folder "$root_dir" --rotate_90

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
    timestamp=$(echo "$filename" | awk -F'_' '{print $5}')
    
    # 判断是左目还是右目
    if [[ "$filename" == *"left"* ]]; then
      # 左目图像移动到camera1
      mv "$file" "$root_dir/camera/camera1/${timestamp}.png"
      echo "Left image: $file -> $root_dir/camera/camera1/${timestamp}.png"
    elif [[ "$filename" == *"right"* ]]; then
      # 右目图像移动到camera0
      mv "$file" "$root_dir/camera/camera0/${timestamp}.png"
      echo "Right image: $file -> $root_dir/camera/camera0/${timestamp}.png"
    else
      echo "跳过不符合命名规则的文件: $file"
    fi
  fi
done

echo "目录重构和重命名完成！"

slam-toolkit --input "$root_dir" --steps convert
slam-toolkit --input "$root_dir" --steps analyze --analyzers stereo --params /home/roborock/下载/sensor.yaml --stereo-sampling-rate 100
