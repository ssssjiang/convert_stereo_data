#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

def create_test_image(width=640, height=480):
    """
    Create a test image with various elements to help visualize color conversion effects.
    """
    # Create a base image with gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # Create RGB channels with different patterns
    r_channel = np.uint8((np.sin(xv * 10) + 1) * 127.5)
    g_channel = np.uint8((np.cos(yv * 10) + 1) * 127.5)
    b_channel = np.uint8(xv * yv * 255)
    
    # Combine channels
    image = np.stack([b_channel, g_channel, r_channel], axis=2)  # OpenCV uses BGR
    
    # Add some shapes
    cv2.circle(image, (width // 4, height // 4), 50, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(image, (width // 2, height // 2), (width // 2 + 100, height // 2 + 100), (0, 255, 0), -1)  # Green rectangle
    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5)  # Red line
    
    return image

def bgr_to_yuv(bgr_img, yuv_format='NV12'):
    """
    Convert BGR image to YUV format.
    """
    height, width = bgr_img.shape[:2]
    
    if yuv_format == 'NV12':
        # Convert to YUV
        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
        return yuv_img
    elif yuv_format == 'NV21':
        # Convert to YUV
        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
        # Rearrange UV planes for NV21
        y_size = width * height
        uv_size = y_size // 4
        y_plane = yuv_img[:y_size]
        u_plane = yuv_img[y_size:y_size + uv_size]
        v_plane = yuv_img[y_size + uv_size:y_size + 2 * uv_size]
        
        # Interleave V and U planes for NV21
        vu_interleaved = np.zeros(uv_size * 2, dtype=np.uint8)
        vu_interleaved[0::2] = v_plane
        vu_interleaved[1::2] = u_plane
        
        return np.concatenate([y_plane, vu_interleaved])
    else:
        raise ValueError(f"Unsupported YUV format: {yuv_format}")

def main():
    # Create output directory
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a test image
    width, height = 640, 480
    test_image = create_test_image(width, height)
    
    # Save as PNG
    png_path = os.path.join(output_dir, "test_image.png")
    cv2.imwrite(png_path, test_image)
    print(f"Saved test image as PNG: {png_path}")
    
    # Convert to YUV and save
    yuv_format = 'NV12'
    yuv_data = bgr_to_yuv(test_image, yuv_format)
    
    yuv_path = os.path.join(output_dir, f"test_image_{yuv_format}.yuv")
    with open(yuv_path, 'wb') as f:
        f.write(yuv_data.tobytes())
    print(f"Saved test image as YUV ({yuv_format}): {yuv_path}")
    
    # Also save grayscale version for reference
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, "test_image_gray.png")
    cv2.imwrite(gray_path, gray_image)
    print(f"Saved grayscale reference image: {gray_path}")
    
    print(f"\nImage dimensions: {width}x{height}")
    print(f"YUV format: {yuv_format}")
    print("\nYou can now run the comparison script with:")
    print(f"python yuv_to_gray_comparison.py --yuv_file {yuv_path} --width {width} --height {height} --yuv_format {yuv_format}")
    print("or")
    print(f"python yuv_to_gray_comparison.py --rgb_file {png_path}")

if __name__ == "__main__":
    main() 