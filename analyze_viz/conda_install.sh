#!/bin/bash

# Conda环境专用安装脚本

echo "===== Conda环境中安装依赖 ====="

# 显示当前Conda信息
echo "当前Conda环境:"
conda info --envs
echo ""

# 添加conda-forge通道
echo "添加conda-forge通道..."
conda config --add channels conda-forge
conda config --set channel_priority flexible

# 安装依赖
echo "使用conda安装主要依赖..."
conda install -y -c conda-forge streamlit pandas pillow blinker

# 如果conda安装失败，尝试在conda环境中使用pip
if [ $? -ne 0 ]; then
    echo "Conda安装失败，尝试使用pip在conda环境中安装..."
    pip install streamlit pandas Pillow blinker
fi

# 验证安装
echo ""
echo "===== 验证安装 ====="
python -c "
try:
    import streamlit
    print(f'✓ Streamlit已安装: {streamlit.__version__}')
except ImportError:
    print('✗ Streamlit未安装')

try:
    import blinker
    print(f'✓ Blinker已安装')
except ImportError:
    print('✗ Blinker未安装')

try:
    import pandas
    print(f'✓ Pandas已安装: {pandas.__version__}')
except ImportError:
    print('✗ Pandas未安装')

try:
    from PIL import Image
    import PIL
    print(f'✓ Pillow已安装: {PIL.__version__}')
except ImportError:
    print('✗ Pillow未安装')
"

echo ""
echo "安装完成后，请运行:"
echo "cd analyze_viz"
echo "streamlit run app.py" 