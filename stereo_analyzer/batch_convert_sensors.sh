#!/bin/bash

# 批量转换sensor.yaml脚本
# 用法: ./batch_convert_sensors.sh <tartan_calibr目录路径> [选项]

# 错误处理
set -e

# 函数：显示使用帮助
function show_help {
    echo "用法: $0 <tartan_calibr目录路径> [选项]"
    echo "选项:"
    echo "  --no-swap              不交换左右相机"
    echo "  --no-divide-intrinsics 不将内参除以2"
    echo "  --template <文件路径>  使用特定的模板sensor文件"
    echo "  --pattern <模式>       查找特定模式的camchain文件（默认: *-camchain-imucam.yaml）"
    echo "  --dry-run              仅显示会执行的操作，不实际执行"
    echo "例如:"
    echo "  $0 /path/to/tartan_calibr"
    echo "  $0 /path/to/tartan_calibr --no-swap --template /path/to/template.yaml"
}

# 检查参数
if [ "$#" -lt 1 ]; then
    show_help
    exit 1
fi

# 设置基本参数
BASE_DIR="$1"
TEMPLATE_OPTION=""
SWAP_OPTION="--swap"
DIVIDE_OPTION=""
FILE_PATTERN="*-camchain-imucam.yaml"
DRY_RUN=0

# 处理选项参数
shift 1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --template)
            if [ -f "$2" ]; then
                TEMPLATE_OPTION="--template \"$2\""
            else
                echo "警告: 模板文件 '$2' 不存在!"
            fi
            shift
            ;;
        --no-swap)
            SWAP_OPTION=""
            ;;
        --no-divide-intrinsics)
            DIVIDE_OPTION="--no_divide_intrinsics"
            ;;
        --pattern)
            FILE_PATTERN="$2"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 检查目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录 '$BASE_DIR' 不存在!"
    exit 1
fi

# 生成文件名函数
function generate_sensor_filename {
    local camchain_name="$1"
    
    # 移除 -camchain-imucam.yaml 后缀
    local base_name=$(basename "$camchain_name" -camchain-imucam.yaml)
    
    # 替换 - 为 _
    local sensor_base_name=${base_name//-/_}
    
    # 生成最终的sensor文件名
    echo "sensor_${sensor_base_name}.yaml"
}

# 开始批量处理
echo "===== 开始批量转换相机参数 ====="
echo "基础目录: $BASE_DIR"
echo "查找模式: $FILE_PATTERN"

# 初始化计数器
total_files=0
converted_files=0
failed_files=0

# 使用进程替换而不是管道，这样变量修改会在当前shell中保留
while read camchain_file; do
    total_files=$((total_files + 1))
    
    # 获取文件所在目录
    dir_path=$(dirname "$camchain_file")
    
    # 生成输出文件名
    file_name=$(basename "$camchain_file")
    sensor_file_name=$(generate_sensor_filename "$file_name")
    sensor_output="$dir_path/$sensor_file_name"
    
    echo -e "\n处理 ($total_files): $camchain_file"
    echo "  ==> $sensor_output"
    
    # 构建直接调用Python脚本的命令
    cmd_options=""
    if [ ! -z "$SWAP_OPTION" ]; then
        cmd_options="$cmd_options $SWAP_OPTION"
    fi
    if [ ! -z "$DIVIDE_OPTION" ]; then
        cmd_options="$cmd_options $DIVIDE_OPTION"
    fi
    if [ ! -z "$TEMPLATE_OPTION" ]; then
        cmd_options="$cmd_options $TEMPLATE_OPTION"
    fi
    
    CONVERT_CMD="python3 $(dirname "$0")/convert_stereo_yaml.py --input \"$camchain_file\" --output \"$sensor_output\" --camchain $cmd_options"
    
    # 创建输出目录（如果不存在）
    mkdir -p "$dir_path"
    
    # 是否实际执行
    if [ $DRY_RUN -eq 1 ]; then
        echo "  [DRY RUN] 将执行: $CONVERT_CMD"
    else
        echo "  执行: $CONVERT_CMD"
        if eval $CONVERT_CMD; then
            converted_files=$((converted_files + 1))
            echo "  转换成功: $sensor_output"
        else
            failed_files=$((failed_files + 1))
            echo "  转换失败: $camchain_file"
        fi
    fi
done < <(find "$BASE_DIR" -name "$FILE_PATTERN")

# 打印统计信息
echo -e "\n===== 批量转换完成 ====="
if [ $DRY_RUN -eq 1 ]; then
    echo "DRY RUN模式，没有实际执行转换"
else
    echo "总文件数: $total_files"
    echo "成功转换: $converted_files"
    echo "失败转换: $failed_files"
fi 