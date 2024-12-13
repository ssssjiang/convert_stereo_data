#!/bin/bash

# 原始图像文件夹
SOURCE_DIR="camera0"
# 新的目标文件夹
TARGET_DIR="selected_images"
# 起始时间和结束时间（格式：整数时间戳，例如：1680000000）
START_TIME=1680000000
END_TIME=1680003600

# 创建目标文件夹，如果不存在
mkdir -p "$TARGET_DIR"

# 遍历文件夹中的图像文件
for image in "$SOURCE_DIR"/*; do
    if [[ -f "$image" ]]; then
        # 提取文件名中的时间戳
        filename=$(basename -- "$image")
        timestamp="${filename%.*}"

        # 确保时间戳是整数，避免非时间戳文件报错
        if [[ "$timestamp" =~ ^[0-9]+$ ]]; then
            # 检查时间戳是否在范围内
            if (( timestamp >= START_TIME && timestamp <= END_TIME )); then
                echo "Copying $image to $TARGET_DIR"
                cp "$image" "$TARGET_DIR/"
            fi
        fi
    fi
done

echo "Images within the time range [$START_TIME, $END_TIME] have been successfully copied to $TARGET_DIR."
