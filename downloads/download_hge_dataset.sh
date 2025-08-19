#!/bin/bash
set -e

BASE_URL="https://cvg-data.inf.ethz.ch/lamar/raw/HGE"
OUTPUT_DIR="HGE"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# URL解码函数
url_decode() {
    local url_encoded="${1//+/ }"
    printf '%b' "${url_encoded//%/\\x}"
}

# 递归下载函数
download_directory() {
    local url="$1"
    local local_dir="$2"
    
    echo "🔍 正在扫描目录: $url"
    echo "📁 本地路径: $local_dir"
    
    # 创建本地目录
    mkdir -p "$local_dir"
    
    # 获取目录内容
    local content
    content=$(wget -q -O - "$url/" || {
        echo "❌ 无法访问 $url"
        return 1
    })
    
    # 解析Apache目录列表，提取href链接
    local links
    links=$(echo "$content" | grep -oP 'href="[^"]*"' | sed 's/href="//g' | sed 's/"//g' | grep -vE '^(\.\.?/?|\?|/)' || true)
    
    if [ -z "$links" ]; then
        echo "⚠️  目录 $url 为空或无法解析"
        return 0
    fi
    
    echo "📋 找到 $(echo "$links" | wc -l) 个项目"
    
    for link in $links; do
        # URL解码文件名
        local decoded_name
        decoded_name=$(url_decode "$link")
        
        local full_url="${url}/${link}"
        local local_path="${local_dir}/${decoded_name}"
        
        # 检查是否是目录（以/结尾）
        if [[ "$decoded_name" == */ ]]; then
            echo "📂 发现子目录: $decoded_name"
            # 递归下载子目录
            download_directory "${url}/${link%/}" "$local_path"
        else
            echo "📥 准备下载文件: $decoded_name"
            
            # 检查文件是否已存在且大小正确
            if [ -f "$local_path" ]; then
                echo "ℹ️  文件已存在，检查是否需要续传: $decoded_name"
            fi
            
            # 使用aria2c下载文件，支持断点续传
            if aria2c -c -x 16 -s 16 -j 1 --max-tries=3 --retry-wait=5 \
                      -d "$(dirname "$local_path")" \
                      -o "$(basename "$local_path")" \
                      "$full_url"; then
                echo "✅ 下载成功: $decoded_name"
            else
                echo "❌ 下载失败: $decoded_name (URL: $full_url)"
                # 继续下载其他文件
                continue
            fi
        fi
    done
}

# 检查必要的工具
check_dependencies() {
    if ! command -v wget &> /dev/null; then
        echo "❌ 错误：需要安装 wget"
        exit 1
    fi
    
    if ! command -v aria2c &> /dev/null; then
        echo "❌ 错误：需要安装 aria2"
        echo "💡 安装命令: sudo apt-get install aria2  # Ubuntu/Debian"
        echo "💡 安装命令: sudo yum install aria2     # CentOS/RHEL"
        exit 1
    fi
}

echo "🚀 开始下载 HGE 数据集..."
echo "源地址: $BASE_URL"
echo "目标目录: $OUTPUT_DIR"
echo ""

# 检查依赖
check_dependencies

# 首先下载根目录的metadata文件
echo "📋 下载根目录文件..."
root_content=$(wget -q -O - "$BASE_URL/" || {
    echo "❌ 无法访问根目录"
    exit 1
})

# 提取根目录的文件（不是目录）
root_files=$(echo "$root_content" | grep -oP 'href="[^"]*"' | sed 's/href="//g' | sed 's/"//g' | grep -vE '^(\.\.?/?|\?|/|.*/$)' || true)

for file in $root_files; do
    decoded_name=$(url_decode "$file")
    echo "📥 下载根目录文件: $decoded_name"
    if aria2c -c -x 16 -s 16 -j 1 --max-tries=3 --retry-wait=5 \
              -d "$OUTPUT_DIR" \
              -o "$decoded_name" \
              "${BASE_URL}/${file}"; then
        echo "✅ 根目录文件下载成功: $decoded_name"
    else
        echo "❌ 根目录文件下载失败: $decoded_name"
    fi
done

# 下载sessions目录
echo ""
echo "📂 开始下载 sessions 目录..."
download_directory "${BASE_URL}/sessions" "${OUTPUT_DIR}/sessions"

echo ""
echo "✅ HGE 数据集下载完成！"
echo "所有文件已保存到: $OUTPUT_DIR"

# 显示下载统计
echo ""
echo "📊 下载统计:"
if [ -d "$OUTPUT_DIR" ]; then
    file_count=$(find "$OUTPUT_DIR" -type f | wc -l)
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "未知")
    echo "总文件数: $file_count"
    echo "总大小: $total_size"
    
    # 显示文件类型统计
    echo ""
    echo "📁 文件类型分布:"
    find "$OUTPUT_DIR" -type f -name "*.zip" | wc -l | xargs -I {} echo "ZIP文件: {} 个"
    find "$OUTPUT_DIR" -type f -name "*.json" | wc -l | xargs -I {} echo "JSON文件: {} 个"
else
    echo "下载目录不存在"
fi 