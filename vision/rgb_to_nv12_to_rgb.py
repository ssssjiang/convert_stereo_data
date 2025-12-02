#!/usr/bin/env python3
"""
Simple script to convert RGB image to YUV NV12 format and back to RGB.
"""

import cv2
import numpy as np
import argparse
import os


def rgb_to_nv12(rgb_image):
    """
    Convert RGB image to YUV NV12 format.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
    
    Returns:
        nv12_data: NV12 format data as 1D numpy array
    """
    height, width = rgb_image.shape[:2]
    
    # Convert RGB to YUV (I420 format first)
    yuv_i420 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV_I420)
    
    # I420 has Y plane (H*W), U plane (H/2*W/2), V plane (H/2*W/2)
    # NV12 has Y plane (H*W), interleaved UV plane (H/2*W/2*2)
    y_size = height * width
    uv_size = (height // 2) * (width // 2)
    
    # Extract Y, U, V planes from I420
    y_plane = yuv_i420[:height, :]
    u_plane = yuv_i420[height:height + height // 4, :].reshape(height // 2, width // 2)
    v_plane = yuv_i420[height + height // 4:, :].reshape(height // 2, width // 2)
    
    # Create NV12 format: Y plane followed by interleaved UV
    nv12_data = np.zeros(y_size + uv_size * 2, dtype=np.uint8)
    nv12_data[:y_size] = y_plane.flatten()
    
    # Interleave U and V for UV plane
    uv_interleaved = np.zeros((height // 2, width), dtype=np.uint8)
    uv_interleaved[:, 0::2] = u_plane
    uv_interleaved[:, 1::2] = v_plane
    nv12_data[y_size:] = uv_interleaved.flatten()
    
    return nv12_data


def nv12_to_rgb(nv12_data, width, height):
    """
    Convert YUV NV12 format to RGB image.
    
    Args:
        nv12_data: NV12 format data as 1D numpy array
        width: Image width
        height: Image height
    
    Returns:
        rgb_image: RGB image as numpy array (H, W, 3)
    """
    y_size = height * width
    
    # Extract Y plane
    y_plane = nv12_data[:y_size].reshape(height, width)
    
    # Extract interleaved UV plane
    uv_plane = nv12_data[y_size:].reshape(height // 2, width)
    u_plane = uv_plane[:, 0::2]
    v_plane = uv_plane[:, 1::2]
    
    # Construct YUV image in I420 format for OpenCV conversion
    yuv_i420 = np.zeros((height + height // 2, width), dtype=np.uint8)
    yuv_i420[:height, :] = y_plane
    yuv_i420[height:height + height // 4, :] = u_plane.flatten().reshape(height // 4, width)
    yuv_i420[height + height // 4:, :] = v_plane.flatten().reshape(height // 4, width)
    
    # Convert YUV I420 to RGB
    rgb_image = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2RGB_I420)
    
    return rgb_image


def main():
    parser = argparse.ArgumentParser(description='Convert RGB image to NV12 and back to RGB')
    parser.add_argument('input_image', type=str, help='Input RGB image path')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output image path (default: input_name_nv12_converted.png)')
    parser.add_argument('--save-yuv', '-y', type=str, default=None,
                        help='Save YUV NV12 raw data to file (e.g., output.yuv)')

    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' does not exist!")
        return
    
    # Read RGB image
    print(f"Reading image: {args.input_image}")
    bgr_image = cv2.imread(args.input_image)
    if bgr_image is None:
        print(f"Error: Failed to read image '{args.input_image}'!")
        return
    
    # Convert BGR to RGB (OpenCV reads as BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Ensure dimensions are even (required for NV12)
    if height % 2 != 0 or width % 2 != 0:
        print(f"Warning: Image dimensions must be even for NV12 format. Cropping...")
        height = (height // 2) * 2
        width = (width // 2) * 2
        rgb_image = rgb_image[:height, :width, :]
        print(f"New size: {width}x{height}")
    
    # Convert RGB to NV12
    print("Converting RGB to NV12...")
    nv12_data = rgb_to_nv12(rgb_image)
    print(f"NV12 data size: {len(nv12_data)} bytes")
    
    # Save YUV NV12 file if requested
    if args.save_yuv:
        yuv_path = args.save_yuv
        print(f"Saving YUV NV12 data to: {yuv_path}")
        with open(yuv_path, 'wb') as f:
            f.write(nv12_data.tobytes())
        print(f"YUV file saved successfully (size: {len(nv12_data)} bytes, {width}x{height})")

    # Convert NV12 back to RGB
    print("Converting NV12 back to RGB...")
    rgb_converted = nv12_to_rgb(nv12_data, width, height)
    
    # Convert RGB to BGR for saving with OpenCV
    bgr_converted = cv2.cvtColor(rgb_converted, cv2.COLOR_RGB2BGR)
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.input_image)[0]
        output_path = f"{base_name}_nv12_converted.png"
    else:
        output_path = args.output
    
    # Save the converted image
    print(f"Saving converted image to: {output_path}")
    cv2.imwrite(output_path, bgr_converted)
    print("Done!")
    
    # Calculate and print PSNR to show conversion quality
    mse = np.mean((bgr_image[:height, :width, :].astype(float) - bgr_converted.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255 ** 2 / mse)
        print(f"PSNR (quality metric): {psnr:.2f} dB")


if __name__ == '__main__':
    main()

