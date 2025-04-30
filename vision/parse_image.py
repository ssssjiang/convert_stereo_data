import os
import cv2
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Process log files and images.'
    )
    parser.add_argument('--root_folder', type=str, help='Path to the root folder')
    parser.add_argument('--yuv_folder', type=str, help='Path to the YUV folder')
    parser.add_argument('--yuv_file', type=str, help='Path to a single YUV file')
    parser.add_argument('--image_width', type=int, default=1088, help='Image width')
    parser.add_argument('--image_height', type=int, default=1280, help='Image height')
    parser.add_argument('--rotate_90', action='store_true', help='Rotate image 90 degrees clockwise')
    return parser.parse_args()

def parse_yuv_image(yuv_file_path, output_image_path, width, height, rotate_90=False):
    try:
        with open(yuv_file_path, 'rb') as yuv_file:
            yuv_data = yuv_file.read()

        yuv_image = np.frombuffer(yuv_data, dtype=np.uint8)
        target_size = height * 3 // 2 * width
        if len(yuv_image) > target_size:
            print(f"Warning: Data size {len(yuv_image)} exceeds target size {target_size}. Truncating data.")
            yuv_image = yuv_image[:target_size]
        elif len(yuv_image) < target_size:
            raise ValueError(f"Data size {len(yuv_image)} is smaller than target size {target_size}.")

        yuv_image = yuv_image.reshape((height * 3 // 2, width))
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
        
        # 如果需要旋转图像，顺时针旋转90度
        if rotate_90:
            # 临时需求：left 图像逆时针旋转90度，right 图像顺时针旋转90度
            if 'L' in yuv_file_path or 'left' in yuv_file_path:
                # 左相机图像逆时针旋转90度
                bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                print(f"左相机图像逆时针旋转90度: {yuv_file_path}")
            elif 'R' in yuv_file_path or 'right' in yuv_file_path:
                # 右相机图像顺时针旋转90度
                bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)
                print(f"右相机图像顺时针旋转90度: {yuv_file_path}")
            
        cv2.imwrite(output_image_path, bgr_image)
        print(f"Converted: {yuv_file_path} -> {output_image_path}")
    except Exception as e:
        print(f"Error processing file {yuv_file_path}: {e}")

def process_folder(yuv_folder_path, output_folder_path, width, height, rotate_90=False):
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

                parse_yuv_image(yuv_file_path, output_image_path, width, height, rotate_90)

def main():
    args = parse_args()
    image_width = args.image_width
    image_height = args.image_height
    rotate_90 = args.rotate_90

    if args.yuv_file:
        # Process single YUV file
        output_image_path = args.yuv_file.replace('.yuv', '.png')
        parse_yuv_image(args.yuv_file, output_image_path, image_width, image_height, rotate_90)

    elif args.yuv_folder:
        # Process YUV folder
        rgb_folder_path = args.yuv_folder + '_rgb'
        process_folder(args.yuv_folder, rgb_folder_path, image_width, image_height, rotate_90)

    elif args.root_folder:
        # Search and process all DEV folders under root folder
        dev_folders = [
            os.path.join(args.root_folder, folder)
            for folder in os.listdir(args.root_folder)
            if folder.endswith('DEV') and os.path.isdir(os.path.join(args.root_folder, folder))
        ]

        if not dev_folders:
            raise FileNotFoundError(f"No DEV folders found in {args.root_folder}")

        for dev_folder in dev_folders:
            print(f"Processing folder: {dev_folder}")
            rgb_folder_path = dev_folder + '_rgb'
            process_folder(dev_folder, rgb_folder_path, image_width, image_height, rotate_90)

    else:
        raise ValueError("You must specify --yuv_file, --yuv_folder, or --root_folder")

if __name__ == '__main__':
    main()
