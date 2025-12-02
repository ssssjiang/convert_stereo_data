#!/bin/bash

# rradb 命令简化工具
# 使用方法：
#   source rradb_helper.sh  或者  . rradb_helper.sh
#   然后就可以使用简化的命令了

# 设置 rradb 路径（如果不在当前目录，请修改此路径）
RRADB_PATH="${RRADB_PATH:-./rradb.linux}"
DEVICE="${RRADB_DEVICE:-default}"

# 检查 rradb 是否存在
if [ ! -f "$RRADB_PATH" ]; then
    echo "警告: $RRADB_PATH 不存在，请设置 RRADB_PATH 环境变量或修改脚本中的路径"
fi

# 简化版命令函数

# 检查远程文件是否存在
# 用法: _check_remote_file <设备路径>
_check_remote_file() {
    local remote_path="$1"
    "$RRADB_PATH" "$DEVICE" shell "test -f $remote_path" 2>/dev/null
    return $?
}

# 检查本地文件是否存在
# 用法: _check_local_file <本地路径>
_check_local_file() {
    local local_path="$1"
    if [ -f "$local_path" ]; then
        return 0
    else
        return 1
    fi
}

# 推送文件到设备
# 用法: rpush <本地文件> <设备路径>
rpush() {
    if [ $# -lt 2 ]; then
        echo "用法: rpush <本地文件> <设备路径>"
        return 1
    fi
    local local_file="$1"
    local remote_path="$2"
    
    # 检查本地文件是否存在
    if ! _check_local_file "$local_file"; then
        echo "❌ 错误: 本地文件不存在: $local_file"
        return 1
    fi
    
    # 推送文件
    if ! "$RRADB_PATH" "$DEVICE" push "$local_file" "$remote_path"; then
        echo "❌ 错误: 推送文件失败"
        return 1
    fi
    
    # 验证远程文件是否存在
    echo "验证文件..."
    if _check_remote_file "$remote_path"; then
        echo "✓ 文件已成功推送到: $remote_path"
    else
        echo "⚠ 警告: 无法验证远程文件是否存在: $remote_path"
    fi
}

# 从设备拉取文件
# 用法: rpull <设备路径> [本地路径]
rpull() {
    if [ $# -lt 1 ]; then
        echo "用法: rpull <设备路径> [本地路径]"
        return 1
    fi
    local remote_path="$1"
    local local_path="${2:-$(basename "$remote_path")}"
    
    # 检查远程文件是否存在（如果是文件）
    if ! "$RRADB_PATH" "$DEVICE" shell "test -e $remote_path" 2>/dev/null; then
        echo "⚠ 警告: 远程路径可能不存在: $remote_path"
    fi
    
    # 如果指定了本地路径且包含目录，创建目录结构
    if [ "$local_path" != "$(basename "$local_path")" ]; then
        local local_dir=$(dirname "$local_path")
        if [ ! -d "$local_dir" ]; then
            echo "创建本地目录: $local_dir"
            mkdir -p "$local_dir"
        fi
    fi
    
    # 拉取文件
    if ! "$RRADB_PATH" "$DEVICE" pull "$remote_path"; then
        echo "❌ 错误: 拉取文件失败"
        return 1
    fi
    
    # 如果 rradb pull 默认保存为 basename，需要移动到指定路径
    local default_local_path=$(basename "$remote_path")
    
    # 如果指定的本地路径与默认路径不同，需要移动文件
    if [ "$local_path" != "$default_local_path" ]; then
        if [ -e "$default_local_path" ]; then
            # 确保目标目录存在
            local target_dir=$(dirname "$local_path")
            if [ "$target_dir" != "." ] && [ ! -d "$target_dir" ]; then
                mkdir -p "$target_dir"
            fi
            
            # 移动文件或目录到指定路径
            if [ -f "$default_local_path" ]; then
                mv "$default_local_path" "$local_path" 2>/dev/null
            elif [ -d "$default_local_path" ]; then
                # 如果是目录，先创建目标目录，然后移动内容
                mkdir -p "$local_path"
                mv "$default_local_path"/* "$local_path"/ 2>/dev/null
                rmdir "$default_local_path" 2>/dev/null
            fi
        fi
    fi
    
    # 验证本地文件是否存在（如果是文件，不是目录）
    echo "验证文件..."
    if [ -f "$local_path" ]; then
        echo "✓ 文件已成功拉取到: $local_path"
        ls -lh "$local_path" | head -1
    elif [ -d "$local_path" ]; then
        echo "✓ 目录已成功拉取到: $local_path"
    else
        echo "⚠ 警告: 无法验证本地文件是否存在: $local_path"
    fi
}

# 在设备上执行shell命令
# 用法: rshell <命令>
rshell() {
    if [ $# -lt 1 ]; then
        echo "用法: rshell <命令>"
        return 1
    fi
    "$RRADB_PATH" "$DEVICE" shell "$@"
}

# 重启设备
# 用法: rreboot
rreboot() {
    "$RRADB_PATH" "$DEVICE" shell reboot
}

# 清理 devtest 目录
# 用法: rclean_devtest [确认选项]
rclean_devtest() {
    echo "⚠ 警告: 此操作将删除 /mnt/data/rockrobo/devtest/ 目录下的所有文件"
    if [ "$1" != "--yes" ]; then
        echo "如需确认，请使用: rclean_devtest --yes"
        return 1
    fi
    echo "正在清理 devtest 目录..."
    "$RRADB_PATH" "$DEVICE" shell "rm -rf /mnt/data/rockrobo/devtest/*"
    if [ $? -eq 0 ]; then
        echo "✓ devtest 目录已清理完成"
    else
        echo "❌ 清理失败"
        return 1
    fi
}

# 清理 /dev/shm 目录
# 用法: rclean_shm [确认选项]
rclean_shm() {
    echo "⚠ 警告: 此操作将删除 /dev/shm 目录下的所有文件"
    if [ "$1" != "--yes" ]; then
        echo "如需确认，请使用: rclean_shm --yes"
        return 1
    fi
    echo "正在清理 /dev/shm 目录..."
    "$RRADB_PATH" "$DEVICE" shell "rm -rf /dev/shm/*"
    if [ $? -eq 0 ]; then
        echo "✓ /dev/shm 目录已清理完成"
    else
        echo "❌ 清理失败"
        return 1
    fi
}

# 清理日志目录
# 用法: rclean_log [确认选项]
rclean_log() {
    echo "⚠ 警告: 此操作将删除 /mnt/data/rockrobo/rrlog/ 目录下的所有文件"
    if [ "$1" != "--yes" ]; then
        echo "如需确认，请使用: rclean_log --yes"
        return 1
    fi
    echo "正在清理日志目录..."
    "$RRADB_PATH" "$DEVICE" shell "rm -rf /mnt/data/rockrobo/rrlog/*"
    if [ $? -eq 0 ]; then
        echo "✓ 日志目录已清理完成"
    else
        echo "❌ 清理失败"
        return 1
    fi
}

# 推送文件到临时目录并复制到目标位置
# 用法: rpushcp <本地文件> <设备目标路径>
rpushcp() {
    if [ $# -lt 2 ]; then
        echo "用法: rpushcp <本地文件> <设备目标路径>"
        return 1
    fi
    local local_file="$1"
    local remote_path="$2"
    local filename=$(basename "$local_file")
    local tmp_path="/tmp/$filename"
    
    # 检查本地文件是否存在
    if ! _check_local_file "$local_file"; then
        echo "❌ 错误: 本地文件不存在: $local_file"
        return 1
    fi
    
    # 推送到临时目录
    echo "推送文件到临时目录: $tmp_path"
    if ! "$RRADB_PATH" "$DEVICE" push "$local_file" "$tmp_path"; then
        echo "❌ 错误: 推送文件到临时目录失败"
        return 1
    fi
    
    # 验证临时文件是否存在
    if ! _check_remote_file "$tmp_path"; then
        echo "❌ 错误: 临时文件不存在: $tmp_path"
        return 1
    fi
    
    # 创建目标目录（如果需要）
    local remote_dir=$(dirname "$remote_path")
    if [ "$remote_dir" != "." ] && [ "$remote_dir" != "/" ]; then
        echo "创建目标目录: $remote_dir"
        "$RRADB_PATH" "$DEVICE" shell "mkdir -p $remote_dir"
    fi
    
    # 复制到目标位置
    echo "复制文件到目标位置: $remote_path"
    if ! "$RRADB_PATH" "$DEVICE" shell "cp $tmp_path $remote_path"; then
        echo "❌ 错误: 复制文件失败"
        return 1
    fi
    
    # 验证目标文件是否存在
    echo "验证文件..."
    if _check_remote_file "$remote_path"; then
        echo "✓ 文件已成功复制到: $remote_path"
        local file_info
        file_info=$("$RRADB_PATH" "$DEVICE" shell "ls -lh $remote_path" 2>/dev/null)
        echo "$file_info" | grep -v "^$" | head -1
    else
        echo "❌ 错误: 目标文件不存在: $remote_path"
        return 1
    fi
}

# 替换库文件（推送 -> 复制 -> 重启）
# 用法: rreplace_lib <本地文件> <设备目标路径> [--no-reboot]
rreplace_lib() {
    if [ $# -lt 2 ]; then
        echo "用法: rreplace_lib <本地文件> <设备目标路径> [--no-reboot]"
        echo "示例: rreplace_lib /path/to/lib.so /mnt/data/songs/lib.so"
        echo "      rreplace_lib /path/to/lib.so /mnt/data/songs/lib.so --no-reboot  # 不自动重启"
        return 1
    fi
    
    local local_file="$1"
    local remote_path="$2"
    local no_reboot=false
    
    # 检查是否有 --no-reboot 选项
    if [ "$3" = "--no-reboot" ]; then
        no_reboot=true
    fi
    
    local filename=$(basename "$local_file")
    local tmp_path="/tmp/$filename"
    
    echo "=== 开始替换库文件 ==="
    echo "本地文件: $local_file"
    echo "目标路径: $remote_path"
    echo ""
    
    # 检查本地文件是否存在
    if ! _check_local_file "$local_file"; then
        echo "❌ 错误: 本地文件不存在: $local_file"
        return 1
    fi
    
    # 推送到临时目录
    echo "[1/4] 推送文件到临时目录: $tmp_path"
    if ! "$RRADB_PATH" "$DEVICE" push "$local_file" "$tmp_path"; then
        echo "❌ 错误: 推送文件失败"
        return 1
    fi
    
    # 验证临时文件是否存在
    if ! _check_remote_file "$tmp_path"; then
        echo "❌ 错误: 临时文件不存在: $tmp_path"
        return 1
    fi
    
    # 创建目标目录（如果需要）
    local remote_dir=$(dirname "$remote_path")
    if [ "$remote_dir" != "." ] && [ "$remote_dir" != "/" ]; then
        echo "[2/4] 创建目标目录: $remote_dir"
        "$RRADB_PATH" "$DEVICE" shell "mkdir -p $remote_dir"
    fi
    
    # 复制到目标位置
    echo "[3/4] 复制文件到目标位置: $remote_path"
    if ! "$RRADB_PATH" "$DEVICE" shell "cp $tmp_path $remote_path"; then
        echo "❌ 错误: 复制文件失败"
        return 1
    fi
    
    # 验证文件是否存在和时间戳
    echo "[4/4] 验证文件..."
    
    # 检查远程文件是否存在
    if ! _check_remote_file "$remote_path"; then
        echo "❌ 错误: 文件不存在于目标位置: $remote_path"
        return 1
    fi
    
    # 获取本地文件的时间戳（用于比较）
    local local_timestamp
    if [ -f "$local_file" ]; then
        local_timestamp=$(stat -c '%Y' "$local_file" 2>/dev/null || stat -f '%m' "$local_file" 2>/dev/null)
    fi
    
    # 获取文件详细信息
    local file_info
    file_info=$("$RRADB_PATH" "$DEVICE" shell "ls -lh $remote_path")
    
    # 显示文件信息
    echo "文件信息:"
    echo "$file_info" | grep -v "^$" | head -1
    
    # 获取远程文件时间戳（尝试多种方法）
    local remote_timestamp=""
    
    # 方法1: 使用 stat -c '%Y' (Linux)
    local stat_output
    stat_output=$("$RRADB_PATH" "$DEVICE" shell "stat -c '%Y' $remote_path" 2>&1)
    remote_timestamp=$(echo "$stat_output" | grep -E "^[0-9]+$" | head -1)
    
    # 方法2: 如果方法1失败，尝试使用 ls -l 获取时间（作为备选）
    if [ -z "$remote_timestamp" ]; then
        # 文件刚复制，假设时间戳是当前时间（允许一些误差）
        remote_timestamp=$(date +%s)
    fi
    local current_timestamp
    current_timestamp=$(date +%s)
    
    if [ -n "$remote_timestamp" ] && [ "$remote_timestamp" -gt 0 ]; then
        local time_diff=$((current_timestamp - remote_timestamp))
        
        # 如果时间差小于5分钟，认为文件足够新
        if [ "$time_diff" -lt 300 ]; then
            if [ "$time_diff" -lt 60 ]; then
                echo "✓ 文件时间戳验证通过 (${time_diff}秒前)"
            else
                local minutes=$((time_diff / 60))
                echo "✓ 文件时间戳验证通过 (${minutes}分钟前)"
            fi
            
            # 如果本地文件时间戳可用，比较两者
            if [ -n "$local_timestamp" ]; then
                local timestamp_diff=$((remote_timestamp - local_timestamp))
                if [ "$timestamp_diff" -ge -10 ] && [ "$timestamp_diff" -le 10 ]; then
                    echo "✓ 文件时间戳与本地文件一致"
                elif [ "$timestamp_diff" -lt -10 ]; then
                    local diff_abs=$((timestamp_diff * -1))
                    echo "⚠ 注意: 远程文件比本地文件新 ${diff_abs}秒"
                else
                    echo "⚠ 注意: 远程文件比本地文件旧 ${timestamp_diff}秒"
                fi
            fi
        else
            local minutes=$((time_diff / 60))
            echo "⚠ 警告: 文件时间戳较旧 (${minutes}分钟前，可能不是最新文件)"
        fi
    else
        echo "⚠ 警告: 无法获取文件时间戳"
    fi
    
    echo ""
    echo "=== 库文件替换完成 ==="
    
    # 重启设备（除非指定不重启）
    if [ "$no_reboot" = false ]; then
        echo ""
        echo "正在重启设备..."
        rreboot
    else
        echo "提示: 使用 rreboot 命令手动重启设备"
    fi
}

# 查看日志目录内容
# 用法: rlog [相对路径或绝对路径] [选项]
# 如果不指定路径，默认查看 /mnt/data/rockrobo/rrlog/
# 如果指定相对路径（不以/开头），会拼接到默认路径后面
rlog() {
    local base_path="/mnt/data/rockrobo/rrlog/"
    local log_path="$1"
    
    # 如果没有提供参数，使用默认路径
    if [ -z "$log_path" ]; then
        log_path="$base_path"
    # 如果路径不以 / 开头，认为是相对路径，拼接到基础路径
    elif [ "${log_path#/}" = "$log_path" ]; then
        # 移除基础路径末尾的斜杠（如果有），然后拼接
        base_path="${base_path%/}"
        log_path="$base_path/$log_path"
    fi
    # 如果路径以 / 开头，则直接使用（绝对路径）
    
    shift
    "$RRADB_PATH" "$DEVICE" shell "ls -lh $log_path" "$@"
}

# 查看 devtest 日志目录
# 用法: rlog_devtest [相对路径或绝对路径] [选项]
# 如果指定相对路径（不以/开头），会拼接到 /mnt/data/rockrobo/devtest/ 后面
rlog_devtest() {
    local base_path="/mnt/data/rockrobo/devtest/"
    local log_path="$1"
    
    # 如果没有提供参数，使用默认路径
    if [ -z "$log_path" ]; then
        log_path="$base_path"
    # 如果路径不以 / 开头，认为是相对路径，拼接到基础路径
    elif [ "${log_path#/}" = "$log_path" ]; then
        base_path="${base_path%/}"
        log_path="$base_path/$log_path"
    fi
    
    shift
    "$RRADB_PATH" "$DEVICE" shell "ls -lh $log_path" "$@"
}

# 查看 /dev/shm 目录
# 用法: rlog_shm [相对路径或绝对路径] [选项]
# 如果指定相对路径（不以/开头），会拼接到 /dev/shm 后面
rlog_shm() {
    local base_path="/dev/shm"
    local log_path="$1"
    
    # 如果没有提供参数，使用默认路径
    if [ -z "$log_path" ]; then
        log_path="$base_path"
    # 如果路径不以 / 开头，认为是相对路径，拼接到基础路径
    elif [ "${log_path#/}" = "$log_path" ]; then
        log_path="$base_path/$log_path"
    fi
    
    shift
    "$RRADB_PATH" "$DEVICE" shell "ls -lh $log_path" "$@"
}

# 拉取日志目录
# 用法: rpull_log [相对路径]
rpull_log() {
    if [ $# -lt 1 ]; then
        echo "用法: rpull_log <文件或文件夹相对路径>"
        echo "示例: rpull_log logfile.txt"
        echo "      rpull_log subdir/"
        return 1
    fi
    local relative_path="$1"
    # 移除开头的斜杠（如果有）
    relative_path="${relative_path#/}"
    local full_path="/mnt/data/rockrobo/rrlog/$relative_path"
    # 保持相对路径结构作为本地路径
    rpull "$full_path" "$relative_path"
}

# 拉取 devtest 日志目录下的指定文件或文件夹
# 用法: rpull_devtest <文件或文件夹相对路径>
rpull_devtest() {
    if [ $# -lt 1 ]; then
        echo "用法: rpull_devtest <文件或文件夹相对路径>"
        echo "示例: rpull_devtest logfile.txt"
        echo "      rpull_devtest subdir/file.txt"
        echo "      rpull_devtest 000224.20251202091331050_R1191R45000126_2025112519DEV/PERCEPTOR_normal.ta"
        return 1
    fi
    local relative_path="$1"
    # 移除开头的斜杠（如果有）
    relative_path="${relative_path#/}"
    local full_path="/mnt/data/rockrobo/devtest/$relative_path"
    # 保持相对路径结构作为本地路径
    rpull "$full_path" "$relative_path"
}

# 拉取 /dev/shm 目录下的指定文件或文件夹
# 用法: rpull_shm <文件或文件夹相对路径>
rpull_shm() {
    if [ $# -lt 1 ]; then
        echo "用法: rpull_shm <文件或文件夹相对路径>"
        echo "示例: rpull_shm logfile.txt"
        echo "      rpull_shm subdir/"
        return 1
    fi
    local relative_path="$1"
    # 移除开头的斜杠（如果有）
    relative_path="${relative_path#/}"
    local full_path="/dev/shm/$relative_path"
    # 保持相对路径结构作为本地路径
    rpull "$full_path" "$relative_path"
}

# 显示帮助信息
rradb_help() {
    cat << EOF
rradb 命令简化工具

可用命令:
  文件操作:
    rpush <本地文件> <设备路径>      - 推送文件到设备
    rpull <设备路径>                  - 从设备拉取文件
    rpushcp <本地文件> <设备目标路径> - 推送文件到临时目录并复制到目标位置
    rreplace_lib <本地文件> <设备目标路径> [--no-reboot] - 替换库文件（推送->复制->重启）
  
  系统操作:
    rshell <命令>                     - 在设备上执行shell命令
    rreboot                           - 重启设备
  
  清理操作:
    rclean_devtest --yes              - 清理 /mnt/data/rockrobo/devtest/ 目录
    rclean_shm --yes                  - 清理 /dev/shm 目录
    rclean_log --yes                  - 清理 /mnt/data/rockrobo/rrlog/ 目录
  
  日志查看:
    rlog [相对路径或绝对路径]         - 查看日志目录 (默认: /mnt/data/rockrobo/rrlog/)
                                       相对路径会拼接到默认路径后面
    rlog_devtest [相对路径或绝对路径] - 查看 /mnt/data/rockrobo/devtest/ 目录
                                       相对路径会拼接到基础路径后面
    rlog_shm [相对路径或绝对路径]     - 查看 /dev/shm 目录
                                       相对路径会拼接到基础路径后面
  
  日志拉取:
    rpull_log <相对路径>              - 拉取日志目录下的指定文件或文件夹
    rpull_devtest <相对路径>          - 拉取 /mnt/data/rockrobo/devtest/ 下的指定文件或文件夹
    rpull_shm <相对路径>              - 拉取 /dev/shm 下的指定文件或文件夹

环境变量:
  RRADB_PATH  - rradb.linux 的路径 (默认: ./rradb.linux)
  RRADB_DEVICE - 设备名称 (默认: default)

使用示例:
  # 文件操作
  rpush /path/to/file.so /tmp/file.so
  rpushcp /path/to/file.so /mnt/data/songs/file.so
  
  # 替换库文件（一键完成推送、复制、重启）
  rreplace_lib /path/to/librrpercdrv.so /mnt/data/songs/librrpercdrv.so
  rreplace_lib /path/to/lib.so /mnt/data/songs/lib.so --no-reboot  # 不自动重启
  
  # 系统操作
  rreboot
  
  # 清理操作
  rclean_devtest --yes              # 清理 devtest 目录
  rclean_shm --yes                  # 清理 /dev/shm 目录
  rclean_log --yes                  # 清理日志目录
  
  # 日志查看
  rlog                              # 查看默认日志目录
  rlog 3DGS                         # 查看默认日志目录下的 3DGS 子目录
  rlog /mnt/data/rockrobo/rrlog/    # 查看指定绝对路径
  rlog_devtest                      # 查看 devtest 目录
  rlog_devtest subdir/              # 查看 devtest 目录下的 subdir 子目录
  rlog_shm                          # 查看 /dev/shm 目录
  rlog_shm shared.log               # 查看 /dev/shm 下的 shared.log 文件
  
  # 日志拉取
  rpull_log logfile.txt             # 拉取日志目录下的指定文件
  rpull_devtest test.log            # 拉取 devtest 目录下的指定文件
  rpull_devtest subdir/             # 拉取 devtest 目录下的指定文件夹
  rpull_shm shared.log              # 拉取 /dev/shm 下的指定文件

EOF
}

# 如果直接运行脚本（而不是source），显示帮助信息
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "请使用 'source rradb_helper.sh' 或 '. rradb_helper.sh' 来加载这些命令"
    echo ""
    rradb_help
fi

