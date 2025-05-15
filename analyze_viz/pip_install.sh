#!/bin/bash

# 最简单的安装方式 - 直接使用pip安装
# 对于任何环境都有效（包括Conda）

echo "===== 使用pip安装依赖 ====="

# 使用pip直接安装全部依赖
echo "安装streamlit..."
pip install streamlit

echo "安装blinker..."
pip install blinker

echo "安装其他依赖..."
pip install pandas Pillow

# 验证安装
echo ""
echo "验证安装..."
python -c "
import sys
print(f'Python版本: {sys.version}')

try:
    import streamlit
    print(f'✓ Streamlit已安装')
except ImportError:
    print('✗ Streamlit未安装')

try:
    import blinker
    print(f'✓ Blinker已安装')
except ImportError:
    print('✗ Blinker未安装')
"

echo ""
echo "安装完成，请尝试运行应用:"
echo "cd analyze_viz"
echo "streamlit run app.py" 