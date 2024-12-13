#!/bin/bash

# 输入参数
IMAGE_DIR=$1
OVERWRITE=$2  # 覆写选项：可选值为 "yes" 或 "no"

# 检查文件夹是否存在
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: Directory '$IMAGE_DIR' not found!"
  exit 1
fi

# 默认值：如果未提供 overwrite 参数，默认跳过
if [[ -z "$OVERWRITE" ]]; then
  OVERWRITE="no"
fi

# YAML 路径
YAML_PATH_FISHEYE="/home/roborock/下载/mower/#1Z.png_stereo_fisheye.yml"
YAML_PATH_STANDARD8="/home/roborock/下载/mower/P1_2_#1_116_Z.raw_stereo_nofix8.yml"
YAML_PATH_STANDARD5="/home/roborock/下载/mower/#1Z.png_stereo_nofix5.yml"

# 遍历目录中的所有 PNG 图片
for IMAGE_PATH in "$IMAGE_DIR"/*.png; do
  # 检查文件是否存在
  if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "No PNG files found in directory: $IMAGE_DIR"
    exit 1
  fi

  # 跳过已生成的 undistorted 文件
  if [[ "$IMAGE_PATH" == *_undistorted_fisheye_grid.png || "$IMAGE_PATH" == *_undistorted_standard_grid.png ]]; then
    echo "Skipping already processed file: $IMAGE_PATH"
    continue
  fi

  # 获取文件名和目录
  IMAGE_BASENAME=$(basename "$IMAGE_PATH")
  IMAGE_FILENAME="${IMAGE_BASENAME%.*}"  # 去掉扩展名

  # 解析文件名并设置相机矩阵和畸变系数节点
  if [[ "$IMAGE_PATH" == *IR0* ]]; then
    CAMERA_MATRIX_NODE="M1"
    DIST_COEFFS_NODE="D1"
  elif [[ "$IMAGE_PATH" == *IR1* ]]; then
    CAMERA_MATRIX_NODE="M2"
    DIST_COEFFS_NODE="D2"
  else
    echo "Skipping file without 'IR0' or 'IR1' in name: $IMAGE_PATH"
    continue
  fi

  # 输出文件路径
  OUTPUT_FISHEYE="$IMAGE_DIR/${IMAGE_FILENAME}_undistorted_fisheye_grid.png"
  OUTPUT_STANDARD8="$IMAGE_DIR/${IMAGE_FILENAME}_undistorted_standard8_grid.png"
  OUTPUT_STANDARD5="$IMAGE_DIR/${IMAGE_FILENAME}_undistorted_standard5_grid.png"

  # 检查是否需要运行 fisheye 模式
  if [[ -f "$OUTPUT_FISHEYE" && "$OVERWRITE" == "no" ]]; then
    echo "Fisheye undistorted image already exists: $OUTPUT_FISHEYE. Skipping..."
  else
    echo "Processing fisheye undistortion for $IMAGE_PATH..."
    python distort_images_yaml.py \
      --image_path "$IMAGE_PATH" \
      --yaml_path "$YAML_PATH_FISHEYE" \
      -m fisheye \
      -cm "$CAMERA_MATRIX_NODE" \
      -dc "$DIST_COEFFS_NODE"
  fi

  # 检查是否需要运行 standard8 模式
  if [[ -f "$OUTPUT_STANDARD8" && "$OVERWRITE" == "no" ]]; then
    echo "Standard8 undistorted image already exists: $OUTPUT_STANDARD8. Skipping..."
  else
    echo "Processing standard8 undistortion for $IMAGE_PATH..."
    python distort_images_yaml.py \
      --image_path "$IMAGE_PATH" \
      --yaml_path "$YAML_PATH_STANDARD8" \
      -m standard8 \
      -cm "$CAMERA_MATRIX_NODE" \
      -dc "$DIST_COEFFS_NODE"
  fi

  # 检查是否需要运行 standard5 模式
  if [[ -f "$OUTPUT_STANDARD5" && "$OVERWRITE" == "no" ]]; then
    echo "Standard5 undistorted image already exists: $OUTPUT_STANDARD5. Skipping..."
  else
    echo "Processing standard5 undistortion for $IMAGE_PATH..."
    python distort_images_yaml.py \
      --image_path "$IMAGE_PATH" \
      --yaml_path "$YAML_PATH_STANDARD5" \
      -m standard5 \
      -cm "$CAMERA_MATRIX_NODE" \
      -dc "$DIST_COEFFS_NODE"
  fi

  echo "Processing completed for $IMAGE_PATH"
done
