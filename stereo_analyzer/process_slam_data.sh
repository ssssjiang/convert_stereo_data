#!/bin/bash

# SLAM数据处理流程自动化脚本
# 用法：./process_slam_data.sh <MK系列目录> <数据标识> <sensor.yaml文件路径或包含sensor.yaml的目录> [选项]

# 错误处理
set -e

# 函数：显示使用帮助
function show_help {
    echo "用法: $0 <MK系列目录> <数据标识> <sensor.yaml文件路径或包含sensor.yaml的目录> [选项]"
    echo "参数:"
    echo "  <MK系列目录>       - MK系列根目录路径"
    echo "  <数据标识>         - 单个数据标识(如78_normalz_sunlight)，不需要包含完整路径"
    echo "                       如启用批量模式(--batch)，则此参数可省略"
    echo "  <sensor路径>       - 单个sensor.yaml文件路径或包含sensor*.yaml文件的目录"
    echo "选项:"
    echo "  --force            - 强制覆盖已存在的分析结果，不询问确认"
    echo "  --skip-organize    - 跳过数据组织步骤"
    echo "  --skip-convert     - 跳过数据转换步骤"
    echo "  --skip-prep        - 同时跳过数据组织和转换步骤"
    echo "  --batch            - 批量模式，仅需传入MK系列目录"
    echo "  --no-auto-skip     - 禁用自动跳过（默认已启用：对已有输出目录的数据自动跳过转换步骤）"
    echo "  --debug            - 显示详细的调试信息"
    echo "例如:"
    echo "  $0 /home/roborock/下载/slam_data/MK1-5 78_normalz_sunlight /path/to/sensor.yaml"
    echo "  $0 /home/roborock/下载/slam_data/MK1-5 78_normalz_sunlight /path/to/sensors/directory/ --skip-prep"
    echo "  $0 /home/roborock/下载/slam_data/MK1-5 --batch /path/to/sensor.yaml"
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
    if [[ "$SENSOR_FILENAME" =~ ^sensor_(.+)\.yaml$ ]]; then
        local SENSOR_ID="${BASH_REMATCH[1]}"
        echo "提取到sensor数据标识: $SENSOR_ID"
    else
        # 如果提取失败，使用默认值
        local SENSOR_ID="default"
        echo "警告: 无法从sensor文件名提取数据标识，将使用默认值: $SENSOR_ID"
    fi
    
    # 构建分析目录路径
    local ANALYSIS_DIR="${DIST_DIR}/analysis"
    local RENAMED_ANALYSIS_DIR="${DIST_DIR}/analysis_${SENSOR_ID}"
    
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

# 函数：处理单个数据标识
function process_single_data_id {
    local MK_DIR="$1"     # MK系列目录
    local DATA_ID="$2"    # 数据标识
    local SENSOR_PATH="$3"
    local LOCAL_SKIP_ORGANIZE="$4"
    local LOCAL_SKIP_CONVERT="$5"
    
    # 构建路径 - 支持在多层子目录中查找
    echo "在目录 ${MK_DIR} 中查找数据标识 ${DATA_ID}"
    
    # 查找数据标识的原始数据目录 (*_raw)
    local RAW_DIR_FOUND=false
    local RAW_DIR=""
    
    # 递归查找最多3层子目录中的数据标识目录
    RAW_DIR=$(find "${MK_DIR}" -maxdepth 3 -type d -name "${DATA_ID}_raw" | head -n 1)
    
    if [ -n "${RAW_DIR}" ]; then
        RAW_DIR_FOUND=true
        echo "找到原始数据目录: ${RAW_DIR}"
    else
        echo "错误: 无法找到数据标识 ${DATA_ID} 的原始数据目录"
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试在 ${MK_DIR} 中搜索..."
            find "${MK_DIR}" -maxdepth 3 -type d -name "*_raw" | sort
        fi
        return 1
    fi
    
    # 构建输出目录路径（将输出到与原始数据同级的目录）
    local DIST_DIR=$(dirname "${RAW_DIR}")/${DATA_ID}
    local CAMERAS_DIR="${RAW_DIR}/image"
    
    # 如果启用了自动跳过，并且输出目录已存在，则跳过转换步骤
    if [ "$AUTO_SKIP" = "true" ] && [ -d "${DIST_DIR}" ]; then
        echo "检测到输出目录已存在: ${DIST_DIR}"
        echo "启用自动跳过：将跳过数据整理和转换步骤"
        LOCAL_SKIP_ORGANIZE="true"
        LOCAL_SKIP_CONVERT="true"
    fi
    
    # 查找输入数据文件夹
    local ROOT_DIR_FOUND=true
    local ROOT_DIR=$(find "${RAW_DIR}" -maxdepth 2 -type d -name "*DEV" | head -n 1)
    
    if [ -z "${ROOT_DIR}" ]; then
        echo "错误: 无法找到原始数据目录。请检查 ${RAW_DIR} 路径。"
        ROOT_DIR_FOUND=false
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试查找目录内容:"
            ls -la "${RAW_DIR}" 2>/dev/null || echo "无法访问目录"
            echo "调试信息: 查找任何可能的子目录:"
            find "${RAW_DIR}" -maxdepth 2 -type d 2>/dev/null || echo "无法执行find命令"
        fi
        return 1
    fi
    
    # 检查相机目录
    local CAMERAS_DIR_FOUND=true
    if [ ! -d "${CAMERAS_DIR}" ]; then
        echo "错误: 相机图像目录不存在: ${CAMERAS_DIR}"
        CAMERAS_DIR_FOUND=false
        if [ "$DEBUG_MODE" = "true" ]; then
            echo "调试信息: 尝试查找可能的相机目录:"
            find "${RAW_DIR}" -maxdepth 3 -type d -name "*image*" 2>/dev/null || echo "未找到相机相关目录"
        fi
        return 1
    fi
    
    # 如果关键目录不存在，提前返回
    if [ "$ROOT_DIR_FOUND" = "false" ] || [ "$CAMERAS_DIR_FOUND" = "false" ]; then
        echo "错误: 缺少关键目录，无法处理数据标识 ${DATA_ID}"
        return 1
    fi
    
    # 输出公共信息
    echo -e "\n\n===== 处理数据标识: ${DATA_ID} ====="
    echo "MK目录: ${MK_DIR}"
    echo "数据标识: ${DATA_ID}"
    echo "输入目录: ${ROOT_DIR}"
    echo "相机目录: ${CAMERAS_DIR}"
    echo "输出目录: ${DIST_DIR}"
    
    # 创建输出目录（如果不存在）
    mkdir -p "${DIST_DIR}"
    
    # 数据整理和转换步骤（可选）
    if [ "$LOCAL_SKIP_ORGANIZE" = "false" ]; then
        echo -e "\n===== 步骤1.1：数据整理 ====="
        echo "执行: slam-toolkit --input \"${ROOT_DIR}\" --output \"${DIST_DIR}\" --steps organize --cameras-dir \"${CAMERAS_DIR}\""
        slam-toolkit --input "${ROOT_DIR}" --output "${DIST_DIR}" --steps organize --cameras-dir "${CAMERAS_DIR}"
    else
        echo -e "\n===== 步骤1.1：数据整理 [已跳过] ====="
    fi
    
    if [ "$LOCAL_SKIP_CONVERT" = "false" ]; then
        echo -e "\n===== 步骤1.2：数据转换 ====="
        echo "执行: slam-toolkit --input \"${DIST_DIR}\" --steps convert"
        slam-toolkit --input "${DIST_DIR}" --steps convert
    else
        echo -e "\n===== 步骤1.2：数据转换 [已跳过] ====="
    fi
    
    # 处理sensor.yaml（单个文件或批量）
    echo -e "\n===== 步骤2：处理sensor.yaml ====="
    
    # 判断是文件还是目录
    if [ -f "${SENSOR_PATH}" ]; then
        # 处理单个文件
        echo "检测到单个sensor文件: ${SENSOR_PATH}"
        process_single_sensor "${SENSOR_PATH}" "${ROOT_DIR}" "${DIST_DIR}" "${CAMERAS_DIR}"
        local SENSOR_RESULT=$?
        
    elif [ -d "${SENSOR_PATH}" ]; then
        # 处理目录中的所有sensor*.yaml文件
        echo "检测到sensor文件目录: ${SENSOR_PATH}"
        local SENSOR_FILES=($(find "${SENSOR_PATH}" -maxdepth 1 -name "sensor*.yaml" | sort))
        
        if [ ${#SENSOR_FILES[@]} -eq 0 ]; then
            echo "错误: 未找到任何sensor*.yaml文件在目录: ${SENSOR_PATH}"
            return 1
        fi
        
        echo "找到 ${#SENSOR_FILES[@]} 个sensor文件:"
        for file in "${SENSOR_FILES[@]}"; do
            echo "  - $(basename "$file")"
        done
        
        # 询问是否继续
        if [ "$FORCE_OVERWRITE" != "true" ]; then
            read -p "是否处理以上所有sensor文件? (y/n): " CONTINUE
            if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
                echo "跳过此数据标识的处理"
                return 0
            fi
        fi
        
        # 批量处理文件
        local SUCCESSFUL=0
        local FAILED=0
        for ((sensor_idx=0; sensor_idx<${#SENSOR_FILES[@]}; sensor_idx++)); do
            echo -e "\n[$((sensor_idx+1))/${#SENSOR_FILES[@]}] 处理: ${SENSOR_FILES[$sensor_idx]}"
            if process_single_sensor "${SENSOR_FILES[$sensor_idx]}" "${ROOT_DIR}" "${DIST_DIR}" "${CAMERAS_DIR}"; then
                SUCCESSFUL=$((SUCCESSFUL+1))
            else
                FAILED=$((FAILED+1))
                echo "处理失败: ${SENSOR_FILES[$sensor_idx]}"
            fi
        done
        
        echo -e "\n===== 批量处理结果 ====="
        echo "成功: $SUCCESSFUL"
        echo "失败: $FAILED"
        echo "总计: ${#SENSOR_FILES[@]}"
        
        if [ $SUCCESSFUL -gt 0 ]; then
            SENSOR_RESULT=0  # 只要有一个传感器处理成功，我们就认为整体成功
        else
            SENSOR_RESULT=1  # 所有传感器都失败，则认为整体失败
        fi
    else
        echo "错误: 指定的sensor路径既不是文件也不是目录: ${SENSOR_PATH}"
        return 1
    fi
    
    echo -e "\n===== 数据标识 ${DATA_ID} 处理完成 ====="
    return $SENSOR_RESULT
}

# 检查必须参数
if [ "$#" -lt 1 ]; then
    show_help
    exit 1
fi

# 设置基本参数
MK_DIR="$1"
BATCH_MODE=false
SKIP_ORGANIZE=false
SKIP_CONVERT=false
FORCE_OVERWRITE=false
AUTO_SKIP=true  # 默认启用自动跳过
DEBUG_MODE=false # 调试模式默认关闭

# 先检查是否有 --batch 选项
for arg in "$@"; do
    if [ "$arg" = "--batch" ]; then
        BATCH_MODE=true
        break
    fi
done

# 根据是否为批量模式，处理参数
if [ "$BATCH_MODE" = "true" ]; then
    # 批量模式
    if [ "$#" -lt 2 ]; then
        echo "错误: 批量模式下至少需要提供MK目录和sensor路径"
        show_help
        exit 1
    fi
    
    # 在批量模式下，第二个参数是sensor路径
    shift 1
    # 检查下一个参数是否是选项
    if [[ "$1" == --* ]]; then
        # 如果是选项，说明没有提供sensor路径
        echo "错误: 缺少sensor路径参数"
        show_help
        exit 1
    fi
    SENSOR_PATH="$1"
    DATA_ID_OR_PATH="" # 在批量模式下不使用
    shift 1
else
    # 单个数据标识模式
    if [ "$#" -lt 3 ]; then
        echo "错误: 单个处理模式下需要提供MK目录、数据标识和sensor路径"
        show_help
        exit 1
    fi
    
    DATA_ID_OR_PATH="$2"
    SENSOR_PATH="$3"
    shift 3
fi

# 处理剩余的可选参数
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
        --batch)
            # 已经在前面处理过
            echo "批量模式已启用"
            ;;
        --auto-skip)
            # 保留此选项但不做任何事情，自动跳过现在是默认行为
            echo "自动跳过模式已启用（默认）"
            ;;
        --no-auto-skip)
            AUTO_SKIP=false
            echo "自动跳过模式已禁用"
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
if [ ! -d "$MK_DIR" ]; then
    echo "错误: MK目录不存在: $MK_DIR"
    exit 1
fi

echo "MK目录: $MK_DIR"
if [ "$BATCH_MODE" != "true" ]; then
    echo "数据标识: $DATA_ID_OR_PATH"
fi
echo "Sensor路径: $SENSOR_PATH"

# 主程序开始
echo "===== SLAM数据处理流程开始 ====="

# 判断是单个处理还是批量处理
if [ "$BATCH_MODE" = "true" ]; then
    # 批量处理模式 - 直接使用 MK_DIR 作为搜索根目录
    
    # 临时禁用错误退出，以确保所有数据都被处理
    OLD_SET_E_VALUE=$-
    if [[ $OLD_SET_E_VALUE == *e* ]]; then
        echo "临时禁用错误退出(set -e)，确保所有数据标识都被处理"
        set +e
    fi
    
    # 查找所有匹配的*_raw目录
    echo "正在搜索目录: $MK_DIR"
    RAW_DIRS=($(find "$MK_DIR" -maxdepth 3 -type d -name "*_raw" | sort))
    
    if [ ${#RAW_DIRS[@]} -eq 0 ]; then
        echo "错误: 未找到任何*_raw目录"
        exit 1
    fi
    
    echo "找到 ${#RAW_DIRS[@]} 个数据标识:"
    DATA_IDS=()
    for dir in "${RAW_DIRS[@]}"; do
        # 提取数据标识
        dir_name=$(basename "$dir")
        data_id=${dir_name%_raw}
        DATA_IDS+=("$data_id")
        echo "  - $data_id (来自: $dir)"
    done
    
    # 询问是否继续
    if [ "$FORCE_OVERWRITE" != "true" ]; then
        read -p "是否处理以上所有数据标识? (y/n): " CONTINUE
        if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
            echo "操作取消。"
            exit 0
        fi
    fi
    
    # 批量处理所有数据标识
    SUCCESSFUL=0
    FAILED=0
    SKIPPED=0
    TOTAL_PROCESSED=0
    echo -e "\n===== 开始批量处理数据标识 ====="
    
    # 打印所有将要处理的数据标识，用于调试
    echo "准备处理以下数据标识:"
    for ((j=0; j<${#DATA_IDS[@]}; j++)); do
        echo "  $((j+1))/${#DATA_IDS[@]}: ${DATA_IDS[$j]}"
    done
    
    # 确保处理所有数据标识
    for ((i=0; i<${#DATA_IDS[@]}; i++)); do
        echo "[DEBUG] Loop iteration start: i=$i"
        current_data_id="${DATA_IDS[$i]}"
        echo -e "\n[$((i+1))/${#DATA_IDS[@]}] 准备处理数据标识: ${current_data_id}"
        
        # 处理数据标识 - 由于已经确认了*_raw目录存在，这里直接传递目录和数据标识
        echo "[DEBUG] Before calling process_single_data_id for ${current_data_id}"
        echo "开始处理数据标识: ${current_data_id}"
        set +e  # 临时禁用错误退出
        process_single_data_id "$MK_DIR" "${current_data_id}" "$SENSOR_PATH" "$SKIP_ORGANIZE" "$SKIP_CONVERT"
        PROCESS_RESULT=$?
        # set -e  # 重新启用错误退出 <-- 移除此行
        echo "[DEBUG] After calling process_single_data_id for ${current_data_id}, result: ${PROCESS_RESULT}"
        
        if [ $PROCESS_RESULT -eq 0 ]; then
            echo "数据标识 ${current_data_id} 处理成功"
            SUCCESSFUL=$((SUCCESSFUL+1))
        else
            echo "数据标识 ${current_data_id} 处理失败 (错误码: $PROCESS_RESULT)"
            FAILED=$((FAILED+1))
        fi
        
        TOTAL_PROCESSED=$((TOTAL_PROCESSED+1))
        echo "当前处理进度: $TOTAL_PROCESSED/${#DATA_IDS[@]} (成功: $SUCCESSFUL, 失败: $FAILED, 跳过: $SKIPPED)"
        echo "[DEBUG] Loop iteration end: i=$i"
    done
    
    echo "[DEBUG] Loop finished."
    
    echo -e "\n===== 批量数据标识处理结果 ====="
    echo "成功: $SUCCESSFUL"
    echo "失败: $FAILED"
    echo "跳过: $SKIPPED"
    echo "总计处理: $TOTAL_PROCESSED 个数据标识"
    echo "总计发现: ${#DATA_IDS[@]} 个数据标识"
    
    # 验证所有数据标识是否都被处理
    if [ $((SUCCESSFUL + FAILED + SKIPPED)) -ne ${#DATA_IDS[@]} ]; then
        echo "警告: 处理统计数据不一致，可能有数据未被正确处理"
        echo "  - 预期处理: ${#DATA_IDS[@]} 个数据标识"
        echo "  - 实际统计: $((SUCCESSFUL + FAILED + SKIPPED)) 个数据标识"
    fi
    
    # 恢复原来的错误处理设置
    if [[ $OLD_SET_E_VALUE == *e* ]]; then
        echo "恢复错误退出设置(set -e)"
        set -e
    fi
else
    # 单个数据标识处理
    process_single_data_id "$MK_DIR" "$DATA_ID_OR_PATH" "$SENSOR_PATH" "$SKIP_ORGANIZE" "$SKIP_CONVERT"
fi

# 如果启用了调试模式，关闭它
if [ "$DEBUG_MODE" = "true" ]; then
    set +x  # 关闭调试输出
fi

echo -e "\n===== SLAM数据处理流程完成 =====" 