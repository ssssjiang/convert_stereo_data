#!/bin/bash

# 显示使用方法
function show_usage {
    echo "用法: $0 <数据目录> <sensor_yaml文件> [--method <1|2>]"
    echo "参数说明:"
    echo "  <数据目录>          输入数据的根目录"
    echo "  <sensor_yaml文件>   传感器YAML配置文件路径"
    echo "  --method <1|2>      选择数据转换方式: 1=使用parse_image.py, 2=使用slam-toolkit (默认: 1)"
    exit 1
}

# 参数解析
if [ $# -lt 2 ]; then
    show_usage
fi

root_dir=$1
sensor_yaml=$2
method=1  # 默认使用方式一

# 解析额外参数
shift 2
while [ "$#" -gt 0 ]; do
    case "$1" in
        --method)
            if [ "$2" -eq 1 ] || [ "$2" -eq 2 ]; then
                method=$2
                shift 2
            else
                echo "错误: 方法必须是1或2"
                show_usage
            fi
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            ;;
    esac
done

# 创建输出目录名
output_dir="${root_dir}_rgb"

# 根据选择的方法进行数据转换
if [ $method -eq 1 ]; then
    echo "使用方式一: parse_image.py进行数据转换..."
    
    # 转换数据方式一: 使用parse_image.py
    # --resize_factor 0.5
    python ../vision/parse_image.py --yuv_folder "$root_dir" --rotate_90 --image_width 544 --image_height 640
    
    # 重构目录结构
    mkdir -p "$output_dir/camera/camera0"
    mkdir -p "$output_dir/camera/camera1"
    
    # 查找所有图像文件
    for file in "$output_dir"/*.png; do
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
          mv "$file" "$output_dir/camera/camera0/${timestamp}.png"
          echo "Left image: $file -> $output_dir/camera/camera0/${timestamp}.png"
        elif [[ "$filename" == *"right"* || "$filename" == *"R_"* ]]; then
          # 右目图像移动到camera1
          mv "$file" "$output_dir/camera/camera1/${timestamp}.png"
          echo "Right image: $file -> $output_dir/camera/camera1/${timestamp}.png"
        else
          echo "跳过不符合命名规则的文件: $file"
        fi
      fi
    done
    
    echo "目录重构和重命名完成！"
    
elif [ $method -eq 2 ]; then
    echo "使用方式二: slam-toolkit进行数据转换..."
    
    # 转换数据方式二: 使用slam-toolkit
    slam-toolkit --input "$root_dir" --cameras-dir "$root_dir" --output "$output_dir" --steps organize
    slam-toolkit --input "$output_dir" --steps convert
fi

# 公共部分: 转换sensor_yaml和分析
echo "转换sensor_yaml文件并进行数据分析..."

# convert sensor_yaml to vslam format
# --no_divide_intrinsics
sensor_yaml_new="${sensor_yaml}_new"
python ../stereo_analyzer/convert_stereo_yaml.py --input "$sensor_yaml" --output "$sensor_yaml_new" --template /home/roborock/下载/sy_calibr/sensor_sy_25.yaml

slam-toolkit --input "$output_dir" --steps convert
slam-toolkit --input "$output_dir" --steps analyze --analyzers frame_rate,stereo --params "$sensor_yaml_new" --stereo-sampling-rate 100

echo "处理完成!"