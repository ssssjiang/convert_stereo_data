#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像顶部裁剪工具
功能：裁剪指定目录下图像的顶部N行像素，保存到输出目录
同时输出修改后的相机内参（无畸变情况下）
"""

import cv2
import numpy as np
import argparse
import os
import glob
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="裁剪指定目录下图像的顶部N行像素",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python crop_top_images.py --input_dir ./images --output_dir ./cropped_images --crop_rows 20
  python crop_top_images.py --input_dir ./images --output_dir ./cropped_images --crop_rows 20 --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0

相机内参修改说明（无畸变情况）:
  裁剪顶部N行后，内参修改如下：
  - fx（焦距x）: 保持不变
  - fy（焦距y）: 保持不变
  - cx（主点x坐标）: 保持不变
  - cy（主点y坐标）: cy_new = cy_old - crop_rows
  - 图像宽度: 保持不变
  - 图像高度: height_new = height_old - crop_rows
        """
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help="输入图像目录路径")
    parser.add_argument('--output_dir', type=str, required=True,
                       help="输出图像目录路径")
    parser.add_argument('--crop_rows', type=int, default=100,
                       help="裁剪顶部的行数（像素） (默认: 20)")
    parser.add_argument('--supported_formats', type=str, nargs='+',
                       default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
                       help="支持的图像格式 (默认: jpg jpeg png bmp tiff tif)")
    parser.add_argument('--verbose', action='store_true',
                       help="显示详细处理信息")

    # 可选的相机内参，用于计算修改后的内参
    parser.add_argument('--fx', type=float, default=None,
                       help="原始相机焦距 fx（可选，用于显示修改后的内参）")
    parser.add_argument('--fy', type=float, default=None,
                       help="原始相机焦距 fy（可选，用于显示修改后的内参）")
    parser.add_argument('--cx', type=float, default=None,
                       help="原始相机主点 cx（可选，用于显示修改后的内参）")
    parser.add_argument('--cy', type=float, default=None,
                       help="原始相机主点 cy（可选，用于显示修改后的内参）")

    return parser.parse_args()


def crop_top_rows(image, crop_rows):
    """
    裁剪图像顶部的指定行数

    Args:
        image: 输入图像 (numpy数组)
        crop_rows: 要裁剪的顶部行数

    Returns:
        cropped_image: 裁剪后的图像
    """
    height, width = image.shape[:2]

    if crop_rows >= height:
        raise ValueError(f"裁剪行数 ({crop_rows}) 必须小于图像高度 ({height})")

    # 裁剪顶部crop_rows行，保留剩余部分
    cropped_image = image[crop_rows:, :].copy()

    return cropped_image


def get_image_files(input_dir, supported_formats):
    """
    获取指定目录下所有支持格式的图像文件

    Args:
        input_dir: 输入目录路径
        supported_formats: 支持的图像格式列表

    Returns:
        image_files: 图像文件路径列表（排序后）
    """
    image_files = []
    for fmt in supported_formats:
        # 同时支持大小写
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{fmt}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{fmt.upper()}")))

    # 去重并排序
    image_files = sorted(list(set(image_files)))

    return image_files


def calculate_new_intrinsics(fx, fy, cx, cy, crop_rows, orig_width, orig_height):
    """
    计算裁剪后的相机内参

    Args:
        fx, fy: 原始焦距
        cx, cy: 原始主点坐标
        crop_rows: 裁剪的顶部行数
        orig_width: 原始图像宽度
        orig_height: 原始图像高度

    Returns:
        dict: 包含新旧内参对比的字典
    """
    new_fx = fx
    new_fy = fy
    new_cx = cx
    new_cy = cy - crop_rows
    new_width = orig_width
    new_height = orig_height - crop_rows

    return {
        'original': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width': orig_width,
            'height': orig_height
        },
        'new': {
            'fx': new_fx,
            'fy': new_fy,
            'cx': new_cx,
            'cy': new_cy,
            'width': new_width,
            'height': new_height
        }
    }


def print_intrinsics_info(intrinsics_dict):
    """
    打印相机内参修改信息

    Args:
        intrinsics_dict: 内参字典
    """
    print("\n" + "="*60)
    print("相机内参修改说明（无畸变情况）")
    print("="*60)

    orig = intrinsics_dict['original']
    new = intrinsics_dict['new']

    print("\n原始内参:")
    print(f"  焦距:      fx = {orig['fx']:.6f},  fy = {orig['fy']:.6f}")
    print(f"  主点:      cx = {orig['cx']:.6f},  cy = {orig['cy']:.6f}")
    print(f"  图像尺寸:  width = {orig['width']},  height = {orig['height']}")

    print("\n裁剪后内参:")
    print(f"  焦距:      fx = {new['fx']:.6f},  fy = {new['fy']:.6f}  (保持不变)")
    print(f"  主点:      cx = {new['cx']:.6f},  cy = {new['cy']:.6f}  (cy减少)")
    print(f"  图像尺寸:  width = {new['width']},  height = {new['height']}  (高度减少)")

    print("\n关键修改:")
    print(f"  cy: {orig['cy']:.6f} → {new['cy']:.6f}  (减少 {orig['cy'] - new['cy']:.1f} 像素)")
    print(f"  height: {orig['height']} → {new['height']}  (减少 {orig['height'] - new['height']} 像素)")

    print("\n内参矩阵 K:")
    print("  原始:")
    print(f"    [{orig['fx']:.6f},  0.0,          {orig['cx']:.6f}]")
    print(f"    [0.0,          {orig['fy']:.6f},  {orig['cy']:.6f}]")
    print(f"    [0.0,          0.0,          1.0]")

    print("\n  裁剪后:")
    print(f"    [{new['fx']:.6f},  0.0,          {new['cx']:.6f}]")
    print(f"    [0.0,          {new['fy']:.6f},  {new['cy']:.6f}]")
    print(f"    [0.0,          0.0,          1.0]")

    print("\n" + "="*60)
    print("\n")


def print_general_intrinsics_rule(crop_rows):
    """
    打印通用的内参修改规则

    Args:
        crop_rows: 裁剪的行数
    """
    print("\n" + "="*60)
    print("相机内参修改规则（无畸变情况）")
    print("="*60)
    print(f"\n裁剪顶部 {crop_rows} 行后，内参修改规则如下：")
    print("\n保持不变的参数:")
    print("  • fx (焦距x): 保持不变")
    print("  • fy (焦距y): 保持不变")
    print("  • cx (主点x坐标): 保持不变")
    print("  • 图像宽度 (width): 保持不变")
    print("  • 畸变系数 (k1, k2, p1, p2, k3等): 保持不变（无畸变时均为0）")

    print("\n需要修改的参数:")
    print(f"  • cy (主点y坐标): cy_new = cy_old - {crop_rows}")
    print(f"  • 图像高度 (height): height_new = height_old - {crop_rows}")

    print("\n原理说明:")
    print("  裁剪顶部像素相当于改变了图像坐标系的原点位置。")
    print(f"  原来在 (u, v) 位置的像素，裁剪后在 (u, v-{crop_rows}) 位置。")
    print("  因此主点的 y 坐标需要相应减少裁剪的行数。")

    print("\n注意事项:")
    print("  • 如果原始图像有畸变，应先去畸变再裁剪，或重新标定")
    print("  • 如果是双目相机，两个相机的内参都需要相应修改")
    print("  • 如果有外参（如双目相机的基线），外参保持不变")

    print("\n" + "="*60)
    print("\n")


def main():
    """主函数"""
    args = parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {args.output_dir}")

    # 获取所有图像文件
    image_files = get_image_files(args.input_dir, args.supported_formats)

    if len(image_files) == 0:
        print(f"警告：在 {args.input_dir} 中未找到支持格式的图像文件")
        print(f"支持的格式: {', '.join(args.supported_formats)}")
        return

    print(f"找到 {len(image_files)} 个图像文件")
    print(f"将裁剪顶部 {args.crop_rows} 行像素\n")

    # 处理第一张图像以获取尺寸信息
    first_image = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        print(f"错误：无法读取图像 {image_files[0]}")
        return

    orig_height, orig_width = first_image.shape[:2]
    print(f"原始图像尺寸: {orig_width} x {orig_height}")
    print(f"裁剪后尺寸: {orig_width} x {orig_height - args.crop_rows}\n")

    # 如果提供了内参，计算并显示新内参
    if all([args.fx is not None, args.fy is not None,
            args.cx is not None, args.cy is not None]):
        intrinsics = calculate_new_intrinsics(
            args.fx, args.fy, args.cx, args.cy,
            args.crop_rows, orig_width, orig_height
        )
        print_intrinsics_info(intrinsics)
    else:
        # 显示通用的内参修改规则
        print_general_intrinsics_rule(args.crop_rows)

    # 处理所有图像
    success_count = 0
    fail_count = 0

    for idx, img_path in enumerate(image_files, 1):
        try:
            # 读取图像
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"警告：无法读取图像 {img_path}")
                fail_count += 1
                continue

            # 裁剪顶部
            cropped_image = crop_top_rows(image, args.crop_rows)

            # 生成输出文件路径
            filename = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, filename)

            # 保存裁剪后的图像
            cv2.imwrite(output_path, cropped_image)

            success_count += 1

            if args.verbose:
                print(f"[{idx}/{len(image_files)}] 处理完成: {filename}")
            elif idx % 10 == 0 or idx == len(image_files):
                print(f"进度: {idx}/{len(image_files)} ({100*idx/len(image_files):.1f}%)")

        except Exception as e:
            print(f"错误：处理图像 {img_path} 时出错: {str(e)}")
            fail_count += 1

    # 打印处理结果摘要
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个文件")
    if fail_count > 0:
        print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

