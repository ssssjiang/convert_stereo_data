#!/bin/bash

# 检查是否提供了目标目录参数
if [ $# -eq 0 ]; then
    echo "请提供目标目录路径"
    echo "使用方法: ./copy_vslam_results.sh <目标目录>"
    exit 1
fi

# 目标目录
TARGET_DIR="$1"

# 创建嵌套目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 复制文件
cp log_stereo.txt pose.txt statistics.yaml timing.yaml "$TARGET_DIR/"

echo "文件已复制到: $TARGET_DIR"