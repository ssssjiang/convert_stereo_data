#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import os
import glob
from PIL import Image
import re
import pandas as pd
from pathlib import Path
import time

def find_analysis_folders(root_dir):
    """
    扫描根目录，找到所有符合格式的数据集及其所有分析项
    返回: {数据集名称: {分析项名称: [图片路径列表]}}
    """
    dataset_analysis = {}
    
    # 查找所有analysis_*和analysis目录
    pattern1 = os.path.join(root_dir, "*", "analysis_*")
    pattern2 = os.path.join(root_dir, "*", "analysis")
    
    analysis_dirs = glob.glob(pattern1) + glob.glob(pattern2)
    
    for analysis_dir in analysis_dirs:
        # 提取数据集信息
        parts = analysis_dir.split(os.sep)
        dataset_name = parts[-2]  # 数据集名称
        analysis_id = parts[-1]   # 分析ID
        
        # 格式化数据集显示名称
        if analysis_id == "analysis":
            key = f"{dataset_name} (分析)"
        else:
            key = f"{dataset_name} ({analysis_id})"
            
        # 查找所有子文件夹（分析项）
        analysis_items = [d for d in os.listdir(analysis_dir) 
                         if os.path.isdir(os.path.join(analysis_dir, d))]
        
        if analysis_items:
            dataset_analysis[key] = {}
            
            # 遍历每个分析项文件夹
            for item in analysis_items:
                item_path = os.path.join(analysis_dir, item)
                png_files = glob.glob(os.path.join(item_path, "*.png"))
                
                if png_files:
                    dataset_analysis[key][item] = sorted(png_files)
    
    return dataset_analysis

def group_framerate_images(images):
    """
    将framerate分析项的图像按名称分组
    分组: vs, hist, intervals, 其他
    """
    groups = {
        "vs": [],
        "hist": [],
        "intervals": [],
        "其他": []
    }
    
    for img in images:
        filename = os.path.basename(img).lower()
        if "vs" in filename:
            groups["vs"].append(img)
        elif "hist" in filename:
            groups["hist"].append(img)
        elif "interval" in filename:
            groups["intervals"].append(img)
        else:
            groups["其他"].append(img)
            
    return groups

def display_gallery(images, cols=3, filter_text=""):
    """以网格形式显示图片列表，支持文本过滤"""
    if filter_text:
        # 根据文件名过滤图片
        filtered_images = [img for img in images if filter_text.lower() in os.path.basename(img).lower()]
    else:
        filtered_images = images
    
    if not filtered_images:
        st.warning("没有符合过滤条件的图片")
        return
        
    # 将图片列表分成多行，每行包含cols个图片
    rows = [filtered_images[i:i+cols] for i in range(0, len(filtered_images), cols)]
    
    for row in rows:
        cols_list = st.columns(len(row))
        for i, img_path in enumerate(row):
            with cols_list[i]:
                # 添加图片点击查看大图功能
                img = Image.open(img_path)
                st.image(img_path, use_container_width=True)
                st.caption(os.path.basename(img_path))

def extract_metadata(dataset_name):
    """从数据集名称提取元数据"""
    metadata = {}
    
    # 简化提取逻辑，只分为整机信息和场景信息两部分
    # 例如 MK2-23_105_slope1_10x8_sunlight
    # 整机信息: MK2-23
    # 场景信息: 105_slope1_10x8_sunlight
    
    parts = dataset_name.split('_', 1)  # 只在第一个下划线处分割
    
    if len(parts) >= 1:
        metadata['整机信息'] = parts[0]  # 第一部分为整机信息，如 MK2-23
    
    if len(parts) >= 2:
        metadata['场景信息'] = parts[1]  # 第二部分及以后都作为场景信息
    
    return metadata

def initialize_session_state():
    """初始化会话状态变量"""
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'selected_items' not in st.session_state:
        st.session_state.selected_items = []
    if 'all_selected' not in st.session_state:
        st.session_state.all_selected = True
    if 'show_hist' not in st.session_state:
        st.session_state.show_hist = False

def main():
    st.set_page_config(
        page_title="分析结果可视化浏览器",
        page_icon="🔍",
        layout="wide",
    )
    
    initialize_session_state()
    
    st.title("分析结果可视化查看器")
    
    # 输入根目录路径
    default_path = os.getcwd()  # 默认使用当前工作目录
    root_dir = st.text_input("输入数据根目录路径", value=default_path)
    
    if not os.path.isdir(root_dir):
        st.error(f"目录不存在: {root_dir}")
        return
    
    # 扫描数据目录
    dataset_analysis = find_analysis_folders(root_dir)
    
    if not dataset_analysis:
        st.warning(f"未在 {root_dir} 找到符合格式的数据集。请确保目录结构为: /root/数据名称/analysis_分析ID/ 或 /root/数据名称/analysis/")
        st.info("示例: /root/MK2-23_105_slope1_10x8_sunlight/analysis_MK2-23-49/ 或 /root/MK2-23_105_slope1_10x8_sunlight/analysis/")
        return
    
    # 显示找到的数据集数量
    st.success(f"找到 {len(dataset_analysis)} 个数据集")
    
    # 创建侧边栏导航
    st.sidebar.header("数据集导航")
    
    # 提供搜索过滤功能
    search_term = st.sidebar.text_input("搜索数据集", "")
    filtered_datasets = {k: v for k, v in dataset_analysis.items() if search_term.lower() in k.lower()}
    
    # 通过侧边栏选择数据集
    selected_dataset = st.sidebar.selectbox(
        "选择数据集", 
        list(filtered_datasets.keys()),
        index=0 if list(filtered_datasets.keys()) else None,
        key="dataset_selector"
    )
    
    if selected_dataset:
        st.session_state.selected_dataset = selected_dataset
        
        # 分析项筛选（多选）
        st.sidebar.subheader("分析项筛选")
        
        analysis_items = list(filtered_datasets[selected_dataset].keys())
        
        # 全选/取消全选按钮
        all_selected = st.sidebar.checkbox("全选/取消全选", value=st.session_state.all_selected, key="all_select")
        st.session_state.all_selected = all_selected
        
        # 如果点击全选，则默认选中所有分析项
        if all_selected and not st.session_state.selected_items:
            default_selections = analysis_items
        elif not all_selected and st.session_state.selected_items == analysis_items:
            default_selections = []
        else:
            default_selections = st.session_state.selected_items
        
        # 多选框，选择要显示的分析项
        selected_items = st.sidebar.multiselect(
            "选择要显示的分析项", 
            analysis_items,
            default=default_selections if selected_dataset == st.session_state.selected_dataset else analysis_items
        )
        st.session_state.selected_items = selected_items
        
        # 常用分析组合
        st.sidebar.subheader("常用分析组合")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("质量检查组合", help="framerate + synchronization + anomaly"):
                quality_items = [item for item in ["framerate", "synchronization", "anomaly"] if item in analysis_items]
                st.session_state.selected_items = quality_items
                st.rerun()
                
        with col2:
            if st.button("双目视觉组合", help="stereo"):
                stereo_items = [item for item in ["stereo"] if item in analysis_items]
                st.session_state.selected_items = stereo_items
                st.rerun()
        
        # framerate特殊选项
        if "framerate" in analysis_items:
            st.sidebar.subheader("Framerate选项")
            show_hist = st.sidebar.checkbox("显示直方图(hist)图像", value=st.session_state.show_hist)
            st.session_state.show_hist = show_hist
        
        # 显示选项
        st.sidebar.header("显示选项")
        cols_count = st.sidebar.slider("每行图片数量", 1, 5, 3)
        
        # 主内容区域：使用标签页
        if selected_items:
            # 显示选中数据集的元数据
            st.header(selected_dataset)
            
            # 提取并显示元数据
            dataset_name = selected_dataset.split(" (")[0]  # 移除括号中的分析ID
            metadata = extract_metadata(dataset_name)
            
            if metadata:
                meta_cols = st.columns(len(metadata))
                for i, (key, value) in enumerate(metadata.items()):
                    with meta_cols[i]:
                        st.metric(key, value)
            
            # 创建标签列表：添加"全部"标签和各个分析项标签
            tabs = ["全部"] + selected_items
            selected_tab = st.tabs(tabs)
            
            # "全部"标签显示所有选中的分析项
            with selected_tab[0]:
                for item in selected_items:
                    images = filtered_datasets[selected_dataset][item]
                    
                    # 为framerate分析项使用特殊处理
                    if item == "framerate":
                        with st.expander(f"{item} ({len(images)} 张图片)", expanded=True):
                            # 分组显示
                            grouped_images = group_framerate_images(images)
                            
                            # VS 组
                            if grouped_images["vs"]:
                                st.subheader("VS图像")
                                filter_text = st.text_input(f"筛选VS图像", "", key=f"filter_vs")
                                display_gallery(grouped_images["vs"], cols=cols_count, filter_text=filter_text)
                            
                            # Intervals 组
                            if grouped_images["intervals"]:
                                st.subheader("Intervals图像")
                                filter_text = st.text_input(f"筛选Intervals图像", "", key=f"filter_intervals")
                                display_gallery(grouped_images["intervals"], cols=cols_count, filter_text=filter_text)
                            
                            # Hist 组 (可选显示)
                            if grouped_images["hist"] and st.session_state.show_hist:
                                st.subheader("Hist图像")
                                filter_text = st.text_input(f"筛选Hist图像", "", key=f"filter_hist")
                                display_gallery(grouped_images["hist"], cols=cols_count, filter_text=filter_text)
                            
                            # 其他图像
                            if grouped_images["其他"]:
                                st.subheader("其他图像")
                                filter_text = st.text_input(f"筛选其他图像", "", key=f"filter_other")
                                display_gallery(grouped_images["其他"], cols=cols_count, filter_text=filter_text)
                    else:
                        # 使用常规方式显示其他分析项
                        with st.expander(f"{item} ({len(images)} 张图片)", expanded=True):
                            # 添加图片过滤功能
                            filter_text = st.text_input(f"筛选 {item} 图片", "", key=f"filter_{item}")
                            
                            # 显示图片
                            display_gallery(images, cols=cols_count, filter_text=filter_text)
            
            # 各个分析项标签
            for i, item in enumerate(selected_items, 1):
                with selected_tab[i]:
                    st.subheader(f"{item}")
                    
                    # 为framerate分析项使用特殊处理
                    if item == "framerate":
                        images = filtered_datasets[selected_dataset][item]
                        # 分组显示
                        grouped_images = group_framerate_images(images)
                        
                        # VS 组
                        if grouped_images["vs"]:
                            st.subheader("VS图像")
                            filter_text = st.text_input(f"筛选VS图像", "", key=f"filter_tab_vs")
                            display_gallery(grouped_images["vs"], cols=cols_count, filter_text=filter_text)
                        
                        # Intervals 组
                        if grouped_images["intervals"]:
                            st.subheader("Intervals图像")
                            filter_text = st.text_input(f"筛选Intervals图像", "", key=f"filter_tab_intervals")
                            display_gallery(grouped_images["intervals"], cols=cols_count, filter_text=filter_text)
                        
                        # Hist 组 (可选显示)
                        if grouped_images["hist"] and st.session_state.show_hist:
                            st.subheader("Hist图像")
                            filter_text = st.text_input(f"筛选Hist图像", "", key=f"filter_tab_hist")
                            display_gallery(grouped_images["hist"], cols=cols_count, filter_text=filter_text)
                        
                        # 其他图像
                        if grouped_images["其他"]:
                            st.subheader("其他图像")
                            filter_text = st.text_input(f"筛选其他图像", "", key=f"filter_tab_other")
                            display_gallery(grouped_images["其他"], cols=cols_count, filter_text=filter_text)
                    else:
                        # 图片过滤
                        filter_text = st.text_input(f"筛选图片", "", key=f"filter_tab_{item}")
                        
                        # 排序选项
                        sort_options = ["按名称升序", "按名称降序", "按时间最新", "按时间最旧"]
                        sort_method = st.selectbox("排序方式", sort_options, key=f"sort_{item}")
                        
                        images = filtered_datasets[selected_dataset][item]
                        
                        # 应用排序
                        if sort_method == "按名称降序":
                            images = sorted(images, reverse=True)
                        elif sort_method == "按时间最新":
                            images = sorted(images, key=os.path.getmtime, reverse=True)
                        elif sort_method == "按时间最旧":
                            images = sorted(images, key=os.path.getmtime)
                        
                        # 显示当前分析项的图片
                        st.info(f"共 {len(images)} 张图片")
                        display_gallery(images, cols=cols_count, filter_text=filter_text)
        else:
            st.warning("请在侧边栏选择至少一个分析项")

if __name__ == "__main__":
    main() 