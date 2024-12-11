#!/bin/bash

# 原始图像文件夹
SOURCE_DIR="camera0"
# 新的目标文件夹
TARGET_DIR="selected_images"

# 创建目标文件夹，如果不存在
mkdir -p "$TARGET_DIR"

# 计数器初始化
counter=0

# 遍历文件夹中的图像文件
for image in "$SOURCE_DIR"/*; do
    if [[ -f "$image" ]]; then
        # 如果计数器能被10整除，复制图像到目标文件夹
        if (( counter % 3 == 0 )); then
            echo "Copying $image to $TARGET_DIR"
            cp "$image" "$TARGET_DIR/"
        fi
        # 增加计数器
        ((counter++))
    fi
done

echo "Images have been successfully selected and copied to $TARGET_DIR."
