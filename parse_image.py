import os
import shutil
import subprocess
import cv2
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Process log files and images.'
    )
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder')
    parser.add_argument('--image_width', type=int, default=800, help='Image width')
    parser.add_argument('--image_height', type=int, default=600, help='Image height')
    parser.add_argument('--NID', type=str, default='941bd09bda0d6d57', help='NID value to use in commands')
    return parser.parse_args()

def parse_yuv_image(yuv_folder_path, output_folder_path, width, height):
    if not os.path.exists(yuv_folder_path):
        raise FileNotFoundError(f"YUV folder not found: {yuv_folder_path}")

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Output folder: {output_folder_path}")

    for root, dirs, files in os.walk(yuv_folder_path):
        for filename in files:
            if filename.endswith('.yuv'):
                yuv_file_path = os.path.join(root, filename)

                relative_path = os.path.relpath(root, yuv_folder_path)
                output_subfolder = os.path.join(output_folder_path, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                output_image_path = os.path.join(output_subfolder, filename.replace('.yuv', '.png'))

                if os.path.exists(output_image_path):
                    print(f"Image already converted: {output_image_path}. Skipping.")
                    continue

                try:
                    with open(yuv_file_path, 'rb') as yuv_file:
                        yuv_data = yuv_file.read()

                    yuv_image = np.frombuffer(yuv_data, dtype=np.uint8)
                    yuv_image = yuv_image.reshape((height * 3 // 2, width))

                    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
                    cv2.imwrite(output_image_path, bgr_image)
                    print(f"Converted: {yuv_file_path} -> {output_image_path}")
                except Exception as e:
                    print(f"Error processing file {yuv_file_path}: {e}")

def main():
    args = parse_args()
    root_folder = args.root_folder
    image_width = args.image_width
    image_height = args.image_height

    # Parse YUV images
    yuv_folder_path = next(
        (os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if folder.endswith('DEV')), None)

    if not yuv_folder_path or not os.path.isdir(yuv_folder_path):
        raise FileNotFoundError(f"No DEV folder found in {root_folder}")

    rgb_folder_path = yuv_folder_path + '_rgb'
    parse_yuv_image(yuv_folder_path, rgb_folder_path, image_width, image_height)

if __name__ == '__main__':
    main()
