#!/bin/bash

echo "===== 检查所有静态库的C++标准版本 ====="
find /home/roborock/repos/okvis_depend/prebuild/x86_64-linux-gnu -name "*.a" | while read lib; do
  echo -e "\n=== 检查库: $lib ==="
  # 提取完整的编译器版本信息
  objdump -s --section=.comment "$lib" 2>/dev/null | head -20 | grep -v "^$" | xxd -r -p | strings 
  
  # 搜索C++标准相关标志
  echo -e "\nC++标准相关标志:"
  strings "$lib" | grep -E "std=c\+\+|std=gnu\+\+|c\+\+1[1-9]|__cplusplus" | sort -u
done

echo -e "\n===== 检查所有动态库的C++标准版本 ====="
find /home/roborock/repos/okvis_depend/prebuild/x86_64-linux-gnu -name "*.so" | while read lib; do
  echo -e "\n=== 检查库: $lib ==="
  # 提取完整的编译器版本信息
  readelf -p .comment "$lib" 2>/dev/null | grep -v "String dump"
  
  # 搜索C++标准相关标志
  echo -e "\nC++标准相关标志:"
  strings "$lib" | grep -E "std=c\+\+|std=gnu\+\+|c\+\+1[1-9]|__cplusplus" | sort -u
done

#!/bin/bash

echo "===== 检查OKVIS2库文件的C++标准版本 ====="

find_cpp_std() {
    local file="$1"
    echo -e "\n=== 检查文件: $file ==="
    
    # 搜索编译器版本
    echo "编译器信息:"
    if [[ "$file" == *.so ]]; then
        readelf -p .comment "$file" 2>/dev/null | grep -v "String dump" | head -3
    else
        strings "$file" | grep -E "GCC|clang" | grep -E "version" | head -3
    fi
    
    # 搜索C++标准标志
    echo -e "\nC++标准标志:"
    strings "$file" | grep -E "std=c\+\+|std=gnu\+\+" | sort -u
    
    # 搜索特定的C++标准库特性
    echo -e "\nC++标准库特性:"
    nm -C "$file" 2>/dev/null | grep -E "std::filesystem|std::optional|std::variant|std::string_view" | head -3 > /tmp/cpp17.txt
    if [ -s /tmp/cpp17.txt ]; then
        echo "发现C++17特性:"
        cat /tmp/cpp17.txt
    fi
    
    nm -C "$file" 2>/dev/null | grep -E "std::make_unique|std::integer_sequence" | head -3 > /tmp/cpp14.txt
    if [ -s /tmp/cpp14.txt ]; then
        echo "发现C++14特性:"
        cat /tmp/cpp14.txt
    fi
    
    # 检查Eigen相关定义
    echo -e "\nEigen相关定义:"
    strings "$file" | grep -E "EIGEN_MAX_ALIGN_BYTES|EIGEN_MAX_STATIC_ALIGN_BYTES" | sort -u
}

# 检查库文件
LIBS=$(find /home/roborock/repos/okvis_depend/prebuild/x86_64-linux-gnu -name "*.a" -o -name "*.so" | sort)
for lib in $LIBS; do
    find_cpp_std "$lib"
done

echo -e "\n===== 检查完成 ====="
