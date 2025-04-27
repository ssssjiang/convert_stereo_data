#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse


def yuv_to_rgb(yuv_data, width, height, yuv_format='NV12'):
    """
    Convert YUV data to RGB.
    
    Args:
        yuv_data: YUV data as bytes or numpy array
        width: Image width
        height: Image height
        yuv_format: YUV format ('NV12', 'NV21', 'I420', etc.)
    
    Returns:
        RGB image as numpy array
    """
    # Convert bytes to numpy array if needed
    if isinstance(yuv_data, bytes):
        yuv_data = np.frombuffer(yuv_data, dtype=np.uint8)
    
    # Create a YUV image
    if yuv_format == 'NV12':
        # For NV12: Y plane followed by interleaved UV plane
        yuv_img = cv2.cvtColor(
            yuv_data.reshape(height + height // 2, width), 
            cv2.COLOR_YUV2BGR_NV12
        )
    elif yuv_format == 'NV21':
        # For NV21: Y plane followed by interleaved VU plane
        yuv_img = cv2.cvtColor(
            yuv_data.reshape(height + height // 2, width), 
            cv2.COLOR_YUV2BGR_NV21
        )
    elif yuv_format == 'I420' or yuv_format == 'YUV420P':
        # For I420/YUV420P: Y plane followed by U plane then V plane
        yuv_img = cv2.cvtColor(
            yuv_data.reshape(height * 3 // 2, width), 
            cv2.COLOR_YUV2BGR_I420
        )
    else:
        raise ValueError(f"Unsupported YUV format: {yuv_format}")
    
    return yuv_img


def compare_conversion_paths_from_yuv(yuv_file, width, height, yuv_format='NV12'):
    """
    Compare two different conversion paths from YUV to grayscale.
    
    Path 1: YUV->RGB->PNG->BGR->Gray
    Path 2: YUV->RGB->Gray->PNG->Gray
    
    Args:
        yuv_file: Path to YUV file
        width: Image width
        height: Image height
        yuv_format: YUV format
    
    Returns:
        True if both paths produce identical results, False otherwise
    """
    # Create output directory if it doesn't exist
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read YUV data
    with open(yuv_file, 'rb') as f:
        yuv_data = f.read()
    
    # Convert YUV to RGB
    rgb_img = yuv_to_rgb(yuv_data, width, height, yuv_format)
    
    return compare_conversion_paths_from_rgb(rgb_img, output_dir)


def compare_conversion_paths_from_rgb(rgb_img, output_dir="comparison_results"):
    """
    Compare two different conversion paths from RGB to grayscale.
    
    Path 1: RGB->PNG->BGR->Gray
    Path 2: RGB->Gray->PNG->Gray
    
    Args:
        rgb_img: RGB image as numpy array or path to RGB image file
        output_dir: Directory to save comparison results
    
    Returns:
        True if both paths produce identical results, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load RGB image if path is provided
    if isinstance(rgb_img, str):
        rgb_img = cv2.imread(rgb_img)
        if rgb_img is None:
            raise ValueError(f"Could not read image file: {rgb_img}")
    
    height, width = rgb_img.shape[:2]
    
    # Path 1: RGB->PNG->BGR->Gray
    path1_rgb_png = os.path.join(output_dir, "path1_rgb.png")
    cv2.imwrite(path1_rgb_png, rgb_img)
    path1_bgr = cv2.imread(path1_rgb_png)
    path1_gray = cv2.cvtColor(path1_bgr, cv2.COLOR_BGR2GRAY)
    path1_gray_png = os.path.join(output_dir, "path1_gray.png")
    cv2.imwrite(path1_gray_png, path1_gray)
    
    # Path 2: RGB->Gray->PNG->Gray
    path2_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    path2_gray_png = os.path.join(output_dir, "path2_gray.png")
    cv2.imwrite(path2_gray_png, path2_gray)
    path2_gray_loaded = cv2.imread(path2_gray_png, cv2.IMREAD_GRAYSCALE)
    
    # Save direct conversion for reference
    direct_gray_png = os.path.join(output_dir, "direct_gray.png")
    cv2.imwrite(direct_gray_png, path2_gray)
    
    # Compare results
    are_equal = np.array_equal(path1_gray, path2_gray_loaded)
    
    # Calculate differences
    if not are_equal:
        diff = cv2.absdiff(path1_gray, path2_gray_loaded)
        diff_scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_png = os.path.join(output_dir, "difference.png")
        cv2.imwrite(diff_png, diff_scaled)
        
        # Calculate statistics
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        num_diff_pixels = np.count_nonzero(diff)
        percent_diff = (num_diff_pixels / (height * width)) * 100
        
        return False, {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "num_diff_pixels": num_diff_pixels,
            "percent_diff": percent_diff
        }
    
    return True, None


def main():
    parser = argparse.ArgumentParser(description='Compare YUV to grayscale conversion paths')
    
    # Create a mutually exclusive group for input type
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--yuv_file', help='Path to YUV file')
    input_group.add_argument('--rgb_file', help='Path to RGB/BGR image file (PNG, JPG, etc.)')
    
    # YUV specific arguments
    parser.add_argument('--width', type=int, help='Image width (required for YUV)')
    parser.add_argument('--height', type=int, help='Image height (required for YUV)')
    parser.add_argument('--yuv_format', default='NV12', 
                        choices=['NV12', 'NV21', 'I420', 'YUV420P'],
                        help='YUV format (only needed for YUV input)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.yuv_file and (not args.width or not args.height):
        parser.error("--width and --height are required when using --yuv_file")
    
    output_dir = "comparison_results"
    
    if args.yuv_file:
        print(f"Comparing YUV to grayscale conversion paths for {args.yuv_file}")
        print(f"Image dimensions: {args.width}x{args.height}, Format: {args.yuv_format}")
        
        are_equal, diff_stats = compare_conversion_paths_from_yuv(
            args.yuv_file, args.width, args.height, args.yuv_format
        )
    else:
        print(f"Comparing RGB to grayscale conversion paths for {args.rgb_file}")
        
        are_equal, diff_stats = compare_conversion_paths_from_rgb(args.rgb_file, output_dir)
    
    if are_equal:
        print("✅ RESULT: Both conversion paths produce identical grayscale images.")
    else:
        print("❌ RESULT: The conversion paths produce different grayscale images.")
        print("\nDifference Statistics:")
        print(f"  Maximum difference: {diff_stats['max_diff']}")
        print(f"  Mean difference: {diff_stats['mean_diff']:.4f}")
        print(f"  Standard deviation: {diff_stats['std_diff']:.4f}")
        print(f"  Number of different pixels: {diff_stats['num_diff_pixels']} " 
              f"({diff_stats['percent_diff']:.2f}% of image)")
        print(f"\nCheck the '{output_dir}' directory for visual comparison.")


if __name__ == "__main__":
    main() 
