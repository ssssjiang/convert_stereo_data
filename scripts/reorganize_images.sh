#!/bin/bash

# 定义根目录
root_dir=$1

# 重构目录结构
mkdir -p "$root_dir/camera"
mv "$root_dir/fisheye1" "$root_dir/camera/camera0"
mv "$root_dir/fisheye2" "$root_dir/camera/camera1"

# 遍历 camera/camera0 和 camera/camera1 目录
for camera_dir in "$root_dir/camera/camera0" "$root_dir/camera/camera1"; do
  # 遍历目录中的所有 .png 文件
  for file in "$camera_dir"/*.png; do
    # 提取文件名（不包括路径和扩展名）
    base_name=$(basename "$file" .png)

    # 转换时间戳（秒 -> 毫秒）
    timestamp_ms=$(echo "$base_name * 1000" | bc | awk '{printf "%.0f", $0}')

    # 构造新的文件名
    new_file="$camera_dir/${timestamp_ms}.png"

    # 重命名文件
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
  done
done

echo "目录重构和重命名完成！"
