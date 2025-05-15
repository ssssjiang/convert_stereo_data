#!/bin/bash

# 检查是否已安装依赖
if [ ! -f "requirements.txt" ]; then
    echo "错误: requirements.txt 文件不存在"
    exit 1
fi

# 显示当前Python环境信息
echo "=== Python环境信息 ==="
which python3
python3 --version
echo "========================"

# 检测是否在Conda环境中
if command -v conda &> /dev/null; then
    echo "检测到Conda环境"
    CONDA_PREFIX=$(conda info --base)
    echo "Conda安装路径: $CONDA_PREFIX"
    
    # 检查是否已有streamlit
    if python3 -c "import streamlit" &> /dev/null; then
        echo "Streamlit已安装，直接启动应用..."
    else
        echo "在Conda环境中未检测到streamlit"
        echo "请先运行Conda专用安装脚本: ./conda_install.sh"
        exit 1
    fi
else
    # 非Conda环境，使用pip安装
    echo "使用pip安装/更新依赖..."
    python3 -m pip install --user -r requirements.txt
    
    # 检查安装结果
    if [ $? -ne 0 ]; then
        echo "依赖安装失败，请尝试手动安装："
        echo "python3 -m pip install --user streamlit pandas Pillow blinker"
        exit 1
    fi
    
    # 验证关键包是否已安装
    echo "验证关键包安装..."
    python3 -c "import streamlit; print('Streamlit版本:', streamlit.__version__)" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Streamlit安装验证失败，请尝试手动安装："
        echo "python3 -m pip install --user streamlit"
        exit 1
    fi
    
    python3 -c "import blinker; print('Blinker已成功安装')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Blinker安装验证失败，请尝试手动安装："
        echo "python3 -m pip install --user blinker"
        exit 1
    fi
fi

# 运行应用
echo "启动同步可视化结果查看器..."
if command -v streamlit &> /dev/null; then
    # 如果streamlit命令可用，直接使用
    streamlit run app.py
else
    # 否则使用python模块方式启动
    python3 -m streamlit run app.py
fi 