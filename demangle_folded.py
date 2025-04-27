#!/usr/bin/env python3
"""
解码perf.folded文件中的C++名称修饰符号

用法:
    ./demangle_folded.py input.folded output.folded

此脚本读取perf.folded格式文件，利用c++filt解码所有的C++名称修饰符号，
然后输出解码后的文件，便于后续生成火焰图。
"""

import sys
import subprocess
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def demangle_cpp_symbol(symbol):
    """使用c++filt解码单个C++名称修饰符号"""
    # 清理符号中的空格和特殊字符
    symbol = symbol.strip()
    
    if symbol.startswith('_Z'):
        try:
            # 调用c++filt命令解码符号
            result = subprocess.run(['c++filt', symbol], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   timeout=1)
            if result.returncode == 0:
                # 移除解码结果中可能出现的不兼容字符
                demangled = result.stdout.strip()
                # 移除换行符，确保格式一致
                demangled = demangled.replace('\n', ' ')
                # 替换可能导致解析问题的字符
                demangled = demangled.replace(';', ':')
                return demangled
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"解码符号 '{symbol}' 时出错: {e}")
            # 如果命令失败或找不到c++filt，返回原始符号
            pass
    return symbol

def process_line(line):
    """处理单行，解码所有C++符号"""
    # 保留原始的换行符
    original_ending = ''
    if line.endswith('\n'):
        original_ending = '\n'
        line = line[:-1]
    
    if not line.strip():
        return line + original_ending
    
    # 分割堆栈和计数
    parts = line.strip().rsplit(' ', 1)
    if len(parts) != 2:
        return line + original_ending  # 不符合格式，返回原始行
    
    stack, count = parts
    
    # 分割堆栈中的各个符号
    symbols = stack.split(';')
    
    # 解码每个符号
    demangled_symbols = []
    for symbol in symbols:
        demangled_symbols.append(demangle_cpp_symbol(symbol))
    
    # 重新组合成一行，并确保保留原始的换行符
    return ';'.join(demangled_symbols) + ' ' + count + original_ending

def process_folded_file(input_file, output_file, max_workers=None):
    """处理整个folded文件，解码所有C++符号"""
    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return False
    
    # 尝试不同的编码方式读取文件
    encodings = ['utf-8', 'latin-1', 'ascii']
    lines = None
    
    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            print(f"使用 {encoding} 编码读取文件失败，尝试下一种编码...")
    
    if lines is None:
        print("错误: 无法读取输入文件，请检查文件编码")
        return False
    
    total_lines = len(lines)
    print(f"开始处理 {total_lines} 行数据...")
    
    # 增加调试信息
    print(f"前3行样本:")
    for i in range(min(3, len(lines))):
        print(f"  行 {i+1}: {lines[i].strip()}")
    
    # 使用线程池并行处理
    processed_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_line = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        
        # 处理结果
        for future in tqdm(as_completed(future_to_line), total=total_lines, desc="解码符号"):
            line_index = future_to_line[future]
            try:
                processed_line = future.result()
                processed_lines.append((line_index, processed_line))
            except Exception as e:
                print(f"处理第 {line_index+1} 行时出错: {e}")
                processed_lines.append((line_index, lines[line_index]))  # 使用原始行
    
    # 按原始顺序排序结果
    processed_lines.sort(key=lambda x: x[0])
    
    # 检查处理后的行是否有换行符
    has_newlines = any(line.endswith('\n') for _, line in processed_lines)
    if not has_newlines:
        print("警告: 处理后的行没有换行符，添加换行符...")
    
    # 写入输出文件，确保每行都有换行符
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, line in processed_lines:
                # 确保每行都有换行符
                if not line.endswith('\n'):
                    line += '\n'
                f.write(line)
                
        print(f"处理完成! 解码后的数据已保存到 {output_file}")
        
        # 验证输出文件是否可读
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                check_lines = f.readlines()
                print(f"输出文件包含 {len(check_lines)} 行")
                print(f"前3行样本:")
                for i in range(min(3, len(check_lines))):
                    print(f"  行 {i+1}: {check_lines[i].strip()}")
            return True
        except Exception as e:
            print(f"警告: 无法验证输出文件: {e}")
            return True  # 仍然返回成功，因为写入成功了
        
    except Exception as e:
        print(f"错误: 写入输出文件时出错: {e}")
        return False

def main():
    # 解析命令行参数
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} input.folded output.folded")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 显示版本信息
    print(f"C++符号解码工具 v1.1")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 处理文件
    success = process_folded_file(input_file, output_file)
    
    if success:
        print("成功完成符号解码")
    else:
        print("符号解码失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 