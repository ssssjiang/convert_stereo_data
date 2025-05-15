#!/bin/bash

# SLAM数据处理流程自动化脚本
# 用法：./process_slam_data.sh <数据根目录> <sensor目录> [选项]

# 错误处理
set -e

# 函数：显示使用帮助
function show_help {
    echo "用法: $0 <数据根目录> <sensor目录> [选项] [<数据标识>]"
    echo ""
    echo "参数:"
    echo "  <数据根目录>       - 包含所有数据集的根目录（例如包含多个MK*_*数据集的目录）"
    echo "  <sensor目录>       - 包含所有sensor*.yaml文件的根目录"
    echo "  <数据标识>         - 可选。具体要处理的单个数据集标识（如MK1-5_78_normalz_sunlight）"
    echo "                       如果不指定，则默认处理根目录下的所有数据集"
    echo "选项:"
    echo "  --force            - 强制覆盖已存在的分析结果，不询问确认"
    echo "  --skip-organize    - 跳过数据组织步骤"
    echo "  --skip-convert     - 跳过数据转换步骤"
    echo "  --skip-prep        - 同时跳过数据组织和转换步骤"
    echo "  --no-auto-skip     - 禁用自动跳过（默认已启用：对已有输出目录的数据自动跳过转换步骤）"
    echo "  --no-auto-match    - 禁用自动匹配sensor文件（默认会根据数据集前缀自动匹配对应的sensor文件）"
    echo "  --debug            - 显示详细的调试信息"
    echo ""
    echo "例如:"
    echo "  $0 /path/to/data /path/to/sensors"
    echo "  $0 /path/to/data /path/to/sensors MK1-5_78_normalz_sunlight"
    echo "  $0 /path/to/data /path/to/sensors --skip-prep"
}

# 函数：从完整数据标识中提取前缀（MK*-*部分）
function extract_prefix {
    local DATA_ID="$1"
    if [[ $DATA_ID =~ ^(MK[0-9]+-[0-9]+)_ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# 函数：在sensor目录中查找匹配指定前缀的sensor.yaml文件
function find_matching_sensors {
    local PREFIX="$1"
    local SENSOR_DIR="$2"
    
    if [ -z "$PREFIX" ]; then
        echo "错误: 无法从数据标识中提取前缀"
        return 1
    fi
    
    # 查找所有匹配前缀的sensor*.yaml文件
    local SENSORS=($(find "$SENSOR_DIR" -type f -name "${PREFIX}*sensor*.yaml" 2>/dev/null))
    
    if [ ${#SENSORS[@]} -eq 0 ]; then
        echo "错误: 未找到匹配前缀 ${PREFIX} 的sensor文件"
        return 1
    fi
    
    # 返回找到的sensor文件列表
    for SENSOR in "${SENSORS[@]}"; do
        echo "$SENSOR"
    done
    
    return 0
}

# 函数：处理单个sensor文件
function process_single_sensor {
    local SENSOR_FILE="$1"
    local ROOT_DIR="$2"
    local DIST_DIR="$3"
    local CAMERAS_DIR="$4"
    
    # 检查文件是否存在
    if [ ! -f "$SENSOR_FILE" ]; then
        echo "错误: sensor.yaml文件不存在: $SENSOR_FILE"
        return 1
    fi
    
    # 提取sensor.yaml的数据标识
    local SENSOR_FILENAME=$(basename "$SENSOR_FILE")
    if [[ "$SENSOR_FILENAME" =~ ^(MK[0-9]+-[0-9]+)_sensor(.*)\.yaml$ ]]; then
        local SENSOR_PREFIX="${BASH_REMATCH[1]}"
        local SENSOR_SUFFIX="${BASH_REMATCH[2]}"
        
        # 修改: 不在SENSOR_ID中包含重复的"_sensor"字符串
        if [ -n "$SENSOR_SUFFIX" ] && [ "$SENSOR_SUFFIX" != "_sensor" ]; then
            # 如果后缀不为空且不等于"_sensor"，则只使用有意义的部分
            if [[ "$SENSOR_SUFFIX" == _* ]]; then
                # 如果后缀以_开头，直接使用
                local SENSOR_ID="${SENSOR_PREFIX}${SENSOR_SUFFIX}"
            else
                # 如果后缀不以_开头，添加_
                local SENSOR_ID="${SENSOR_PREFIX}_${SENSOR_SUFFIX}"
            fi
        else
            # 如果后缀为空或者就是"_sensor"，只使用前缀
            local SENSOR_ID="${SENSOR_PREFIX}"
        fi
        
        echo "提取到sensor数据标识: $SENSOR_ID"
    else
        # 如果提取失败，使用文件名作为标识，但去掉sensor和yaml部分
        local SENSOR_ID=$(echo "${SENSOR_FILENAME}" | sed -e 's/_sensor//g' -e 's/\.yaml$//')
        echo "警告: 无法从sensor文件名提取数据标识格式，将使用: $SENSOR_ID"
    fi
    
    # 构建分析目录路径
    local ANALYSIS_DIR="${DIST_DIR}/analysis"
    local RENAMED_ANALYSIS_DIR="${DIST_DIR}/analysis_${SENSOR_ID}"
    
    # 调试信息：打印提取到的各部分
    if [ "$DEBUG_MODE" = "true" ]; then
        echo "调试信息: 文件名=${SENSOR_FILENAME}"
        echo "调试信息: 前缀=${SENSOR_PREFIX}, 后缀=${SENSOR_SUFFIX}"
        echo "调试信息: 最终ID=${SENSOR_ID}"
        echo "调试信息: 原分析目录=${ANALYSIS_DIR}"
        echo "调试信息: 新分析目录=${RENAMED_ANALYSIS_DIR}"
    fi
    
    # 检查目标分析目录是否已存在
    if [ -d "${RENAMED_ANALYSIS_DIR}" ]; then
        echo "警告: 目标分析目录已存在: ${RENAMED_ANALYSIS_DIR}"
        if [ "$FORCE_OVERWRITE" == "true" ]; then
            echo "将覆盖已有分析结果。"
            rm -rf "${RENAMED_ANALYSIS_DIR}"
        else
            read -p "是否覆盖已有分析结果? (y/n): " OVERWRITE
            if [[ "$OVERWRITE" != "y" && "$OVERWRITE" != "Y" ]]; then
                echo "跳过处理: ${SENSOR_FILE}"
                return 0
            fi
            echo "将覆盖已有分析结果。"
            rm -rf "${RENAMED_ANALYSIS_DIR}"
        fi
    fi
    
    # 输出信息
    echo -e "\n===== 开始处理 Sensor文件: $SENSOR_FILENAME ====="
    echo "Sensor标识: ${SENSOR_ID}"
    echo "分析结果将保存为: ${RENAMED_ANALYSIS_DIR}"
    
    # 使用sensor.yaml执行SLAM分析
    echo -e "\n----- 执行SLAM分析 -----"
    echo "执行: slam-toolkit --input \"${DIST_DIR}\" --steps analyze --analyzers frame_rate,stereo,synchronization,anomaly --params \"${SENSOR_FILE}\""
    slam-toolkit --input "${DIST_DIR}" --steps analyze --analyzers frame_rate,stereo,synchronization,anomaly --params "${SENSOR_FILE}"
    
    # 重命名分析结果目录
    echo -e "\n----- 重命名分析结果目录 -----"
    if [ -d "${ANALYSIS_DIR}" ]; then
        echo "重命名: ${ANALYSIS_DIR} -> ${RENAMED_ANALYSIS_DIR}"
        # 如果目标目录已存在（可能由于之前未完全清理），先删除
        [ -d "${RENAMED_ANALYSIS_DIR}" ] && rm -rf "${RENAMED_ANALYSIS_DIR}"
        mv "${ANALYSIS_DIR}" "${RENAMED_ANALYSIS_DIR}"
        echo "分析结果已重命名。"
    else
        echo "警告: 分析目录不存在: ${ANALYSIS_DIR}"
        return 1
    fi
    
    echo -e "\n===== 处理完成: $SENSOR_FILENAME ====="
    echo "结果已输出到: ${RENAMED_ANALYSIS_DIR}"
    
    return 0
}

# 函数：处理单个数据集
function process_single_dataset {
    local DATA_ROOT="$1"     # 数据根目录
    local DATA_ID="$2"       # 数据标识（可能带_raw后缀）
    local SENSOR_DIR="$3"    # sensor目录
    local SKIP_ORGANIZE="$4"
    local SKIP_CONVERT="$5"
    local AUTO_MATCH="$6"
    
    # 构建路径 - 支持在指定目录中精确查找
    echo "在目录 ${DATA_ROOT} 中查找数据标识 ${DATA_ID}"
    
    # 查找完整的数据路径
    local DATA_PATH=$(find "${DATA_ROOT}" -maxdepth 1 -type d -name "${DATA_ID}" | head -n 1)
    
    if [ -z "${DATA_PATH}" ]; then
        echo "错误: 未找到数据标识 ${DATA_ID} 的数据目录"
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试在 ${DATA_ROOT} 中搜索..."
            find "${DATA_ROOT}" -maxdepth 1 -type d | sort
        fi
        return 1
    fi
    
    echo "找到数据目录: ${DATA_PATH}"
    
    # 检查是否是原始数据目录（以_raw结尾）
    if [[ "${DATA_PATH}" == *"_raw" ]]; then
        # 这是原始数据目录，我们将处理它并输出到不带_raw的目录
        local RAW_PATH="${DATA_PATH}"
        # 创建对应的输出目录名（去掉_raw后缀）
        local OUTPUT_NAME=$(basename "${DATA_PATH}")
        local OUTPUT_NAME_NO_RAW="${OUTPUT_NAME%_raw}"
        local DIST_DIR="${DATA_ROOT}/${OUTPUT_NAME_NO_RAW}"
    else
        # 这不是原始数据目录，可能是输出目录
        echo "警告: 指定的目录不是带_raw后缀的原始数据目录"
        echo "将假设这是处理后的输出目录，寻找对应的原始数据目录"
        local DIST_DIR="${DATA_PATH}"
        local RAW_PATH="${DATA_PATH}_raw"
        
        if [ ! -d "${RAW_PATH}" ]; then
            echo "错误: 未找到对应的原始数据目录: ${RAW_PATH}"
            return 1
        fi
    fi
    
    echo "使用原始数据目录: ${RAW_PATH}"
    echo "使用输出目录: ${DIST_DIR}"
    
    # 查找相机目录
    local CAMERAS_DIR="${RAW_PATH}/image"
    if [ ! -d "${CAMERAS_DIR}" ]; then
        echo "错误: 相机图像目录不存在: ${CAMERAS_DIR}"
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试查找可能的相机目录:"
            find "${RAW_PATH}" -maxdepth 3 -type d -name "*image*" 2>/dev/null || echo "未找到相机相关目录"
        fi
        return 1
    fi
    
    # 查找输入数据文件夹
    local ROOT_DIR=$(find "${RAW_PATH}" -maxdepth 2 -type d -name "*DEV" | head -n 1)
    
    if [ -z "${ROOT_DIR}" ]; then
        echo "错误: 无法找到原始数据目录。请检查 ${RAW_PATH} 路径。"
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试查找目录内容:"
            ls -la "${RAW_PATH}" 2>/dev/null || echo "无法访问目录"
            echo "调试信息: 查找任何可能的子目录:"
            find "${RAW_PATH}" -maxdepth 2 -type d 2>/dev/null || echo "无法执行find命令"
        fi
        return 1
    fi
    
    # 输出公共信息
    local DISPLAY_ID=$(basename "${DIST_DIR}")  # 用于显示的数据标识（不带_raw后缀）
    echo -e "\n\n===== 处理数据集: ${DISPLAY_ID} ====="
    echo "数据根目录: ${DATA_ROOT}"
    echo "原始数据目录: ${RAW_PATH}"
    echo "输入目录: ${ROOT_DIR}"
    echo "相机目录: ${CAMERAS_DIR}"
    echo "输出目录: ${DIST_DIR}"
    
    # 如果启用了自动跳过，并且输出目录已存在，则跳过转换步骤
    if [ "$AUTO_SKIP" = "true" ] && [ -d "${DIST_DIR}" ]; then
        echo "检测到输出目录已存在: ${DIST_DIR}"
        echo "启用自动跳过：将跳过数据整理和转换步骤"
        SKIP_ORGANIZE="true"
        SKIP_CONVERT="true"
    fi
    
    # 创建输出目录（如果不存在）
    mkdir -p "${DIST_DIR}"
    
    # 数据整理和转换步骤（可选）
    if [ "$SKIP_ORGANIZE" = "false" ]; then
        echo -e "\n===== 步骤1.1：数据整理 ====="
        echo "执行: slam-toolkit --input \"${ROOT_DIR}\" --output \"${DIST_DIR}\" --steps organize --cameras-dir \"${CAMERAS_DIR}\""
        slam-toolkit --input "${ROOT_DIR}" --output "${DIST_DIR}" --steps organize --cameras-dir "${CAMERAS_DIR}"
    else
        echo -e "\n===== 步骤1.1：数据整理 [已跳过] ====="
    fi
    
    if [ "$SKIP_CONVERT" = "false" ]; then
        echo -e "\n===== 步骤1.2：数据转换 ====="
        echo "执行: slam-toolkit --input \"${DIST_DIR}\" --steps convert"
        slam-toolkit --input "${DIST_DIR}" --steps convert
    else
        echo -e "\n===== 步骤1.2：数据转换 [已跳过] ====="
    fi
    
    # 准备sensor文件处理
    echo -e "\n===== 步骤2：处理sensor文件 ====="
    
    # 自动匹配对应的sensor文件（如果启用）
    if [ "$AUTO_MATCH" = "true" ]; then
        echo "启用自动匹配sensor文件"
        
        # 从数据标识中提取前缀（MK*-*部分）
        DATA_PREFIX=$(extract_prefix "${DISPLAY_ID}")
        echo "从数据标识中提取前缀: ${DATA_PREFIX}"
        
        if [ -n "${DATA_PREFIX}" ]; then
            # 查找匹配的sensor文件
            MATCHED_SENSORS=($(find_matching_sensors "${DATA_PREFIX}" "${SENSOR_DIR}"))
            MATCH_RESULT=$?
            
            if [ $MATCH_RESULT -eq 0 ] && [ ${#MATCHED_SENSORS[@]} -gt 0 ]; then
                echo "找到 ${#MATCHED_SENSORS[@]} 个匹配的sensor文件:"
                for sensor in "${MATCHED_SENSORS[@]}"; do
                    echo "  - $(basename "$sensor")"
                done
                
                # 询问是否处理所有匹配的sensor文件
                if [ ${#MATCHED_SENSORS[@]} -gt 1 ] && [ "$FORCE_OVERWRITE" != "true" ]; then
                    read -p "处理所有匹配的sensor文件? (y/n): " PROCESS_ALL
                    if [[ "$PROCESS_ALL" != "y" && "$PROCESS_ALL" != "Y" ]]; then
                        echo "请手动指定要处理的sensor文件"
                        return 0
                    fi
                fi
                
                # 处理所有匹配的sensor文件
                for sensor in "${MATCHED_SENSORS[@]}"; do
                    process_single_sensor "$sensor" "${ROOT_DIR}" "${DIST_DIR}" "${CAMERAS_DIR}"
                done
                
                return 0
            else
                echo "警告: 未找到匹配前缀 ${DATA_PREFIX} 的sensor文件"
                echo "将退回到手动指定sensor文件"
            fi
        else
            echo "警告: 无法从数据标识 ${DISPLAY_ID} 中提取前缀"
            echo "将退回到手动指定sensor文件"
        fi
    fi
    
    # 如果自动匹配失败或禁用，则提示用户手动指定sensor文件
    echo "请指定要处理的sensor文件:"
    echo "例如: process_single_sensor /path/to/sensor.yaml \"${ROOT_DIR}\" \"${DIST_DIR}\" \"${CAMERAS_DIR}\""
    return 0
}

# 检查必须参数
if [ "$#" -lt 2 ]; then
    show_help
    exit 1
fi

# 设置基本参数
DATA_ROOT="$1"
SENSOR_DIR="$2"
shift 2

# 检查是否有在前两个参数后的单个数据标识（不是以--开头）
SINGLE_DATASET=""
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    SINGLE_DATASET="$1"
    shift 1
fi

# 设置默认选项
SKIP_ORGANIZE=false
SKIP_CONVERT=false
FORCE_OVERWRITE=false
AUTO_SKIP=true   # 默认启用自动跳过
AUTO_MATCH=true  # 默认启用自动匹配sensor文件
DEBUG_MODE=false # 调试模式默认关闭

# 处理选项参数
while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)
            FORCE_OVERWRITE=true
            echo "强制覆盖模式已启用"
            ;;
        --skip-organize)
            SKIP_ORGANIZE=true
            echo "跳过数据组织步骤"
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            echo "跳过数据转换步骤"
            ;;
        --skip-prep)
            SKIP_ORGANIZE=true
            SKIP_CONVERT=true
            echo "跳过数据准备步骤（组织和转换）"
            ;;
        --no-auto-skip)
            AUTO_SKIP=false
            echo "自动跳过模式已禁用"
            ;;
        --no-auto-match)
            AUTO_MATCH=false
            echo "自动匹配sensor文件已禁用"
            ;;
        --debug)
            DEBUG_MODE=true
            echo "调试模式已启用，将显示详细信息"
            set -x  # 开启调试输出
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "警告: 未知参数: $1"
            ;;
    esac
    shift
done

# 检查关键参数是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 数据根目录不存在: $DATA_ROOT"
    exit 1
fi

if [ ! -d "$SENSOR_DIR" ]; then
    echo "错误: sensor目录不存在: $SENSOR_DIR"
    exit 1
fi

echo "数据根目录: $DATA_ROOT"
echo "Sensor目录: $SENSOR_DIR"

# 主程序开始
echo "===== SLAM数据处理流程开始 ====="

# 保存原始错误处理设置
OLD_SET_E_VALUE=$-
if [[ $OLD_SET_E_VALUE == *e* ]]; then
    echo "临时禁用错误退出(set -e)，确保所有数据集都被处理"
    set +e
fi

if [ -n "$SINGLE_DATASET" ]; then
    # 处理单个指定的数据集
    echo "处理单个指定的数据集: $SINGLE_DATASET"
    process_single_dataset "$DATA_ROOT" "$SINGLE_DATASET" "$SENSOR_DIR" "$SKIP_ORGANIZE" "$SKIP_CONVERT" "$AUTO_MATCH"
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "数据集 $SINGLE_DATASET 处理成功"
    else
        echo "数据集 $SINGLE_DATASET 处理失败 (错误码: $RESULT)"
    fi
else
    # 批量处理所有数据集
    # 查找所有数据集目录（包含_raw后缀的目录）
    echo "在 $DATA_ROOT 中查找所有数据集目录..."
    DATASETS=($(find "$DATA_ROOT" -maxdepth 1 -type d -name "MK*_*_raw" | sort))
    
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "错误: 未找到任何带_raw后缀的数据集目录"
        exit 1
    fi
    
    echo "找到 ${#DATASETS[@]} 个数据集:"
    DATASET_IDS=()
    for dir in "${DATASETS[@]}"; do
        # 提取数据标识
        dataset_id=$(basename "$dir")
        DATASET_IDS+=("$dataset_id")
        echo "  - $dataset_id"
    done
    
    # 询问是否继续
    if [ "$FORCE_OVERWRITE" != "true" ]; then
        read -p "是否处理以上所有数据集? (y/n): " CONTINUE
        if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
            echo "操作取消。"
            exit 0
        fi
    fi
    
    # 批量处理所有数据集
    SUCCESSFUL=0
    FAILED=0
    SKIPPED=0
    
    echo -e "\n===== 开始批量处理数据集 ====="
    
    for ((i=0; i<${#DATASET_IDS[@]}; i++)); do
        current_dataset="${DATASET_IDS[$i]}"
        echo -e "\n[$((i+1))/${#DATASET_IDS[@]}] 准备处理数据集: ${current_dataset}"
        
        # 处理数据集
        process_single_dataset "$DATA_ROOT" "${current_dataset}" "$SENSOR_DIR" "$SKIP_ORGANIZE" "$SKIP_CONVERT" "$AUTO_MATCH"
        PROCESS_RESULT=$?
        
        if [ $PROCESS_RESULT -eq 0 ]; then
            echo "数据集 ${current_dataset} 处理成功"
            SUCCESSFUL=$((SUCCESSFUL+1))
        else
            echo "数据集 ${current_dataset} 处理失败 (错误码: $PROCESS_RESULT)"
            FAILED=$((FAILED+1))
        fi
    done
    
    echo -e "\n===== 批量处理结果 ====="
    echo "成功: $SUCCESSFUL"
    echo "失败: $FAILED"
    echo "跳过: $SKIPPED"
    echo "总计: ${#DATASET_IDS[@]} 个数据集"
fi

# 恢复原来的错误处理设置
if [[ $OLD_SET_E_VALUE == *e* ]]; then
    echo "恢复错误退出设置(set -e)"
    set -e
fi

# 如果启用了调试模式，关闭它
if [ "$DEBUG_MODE" = "true" ]; then
    set +x  # 关闭调试输出
fi

echo -e "\n===== SLAM数据处理流程完成 =====" 