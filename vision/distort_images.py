import cv2
import numpy as np
import argparse
import os
import yaml

# Rk1=7.94875317
# Rk2=4.62162266
# Rp1=0.00004919
# Rp2=0.00006952
# Rk3=0.21034777
# Rk4=8.25617308
# Rk5=7.06237124
# Rk6=1.02222154
def parse_args():
    parser = argparse.ArgumentParser(description="Undistort an image and calculate FOV after de-distortion.")
    parser.add_argument('--image_path', type=str, default='', help="Path to the input image.")
    parser.add_argument('--image_folder', type=str, default='', help="Path to the folder containing images for batch processing.")
    parser.add_argument('--output_folder', type=str, default='', help="Path to save undistorted images (for batch processing).")
    parser.add_argument('--camchain_yaml', type=str, default='', help="Path to camchain YAML file (e.g., log1-camchain.yaml). If provided, intrinsics and distortion coefficients will be loaded from this file.")
    parser.add_argument('--cam_name', type=str, default='cam0', help="Camera name in the camchain YAML file (default: cam0).")
    parser.add_argument('--fx', type=float, default=None, help="Focal length fx of the camera.")
    parser.add_argument('--fy', type=float, default=None, help="Focal length fy of the camera.")
    parser.add_argument('--cx', type=float, default=None, help="Principal point cx of the camera.")
    parser.add_argument('--cy', type=float, default=None, help="Principal point cy of the camera.")
    parser.add_argument('--k1', type=float, default=None, help="Distortion coefficient k1.")
    parser.add_argument('--k2', type=float, default=None, help="Distortion coefficient k2.")
    parser.add_argument('--k3', type=float, default=None, help="Distortion coefficient k3.")
    parser.add_argument('--k4', type=float, default=None, help="Distortion coefficient k4.")
    parser.add_argument('--k5', type=float, default=None, help="Distortion coefficient k5.")
    parser.add_argument('--k6', type=float, default=None, help="Distortion coefficient k6.")
    parser.add_argument('--p1', type=float, default=None, help="Distortion coefficient p1.")
    parser.add_argument('--p2', type=float, default=None, help="Distortion coefficient p2.")
    parser.add_argument('--zoom_factor', type=float, default=0.95, help="Zoom factor for de-distortion (default: 1.0).")
    parser.add_argument('--save_output', type=str, default='', help="Path to save the undistorted image (for single image processing).")
    parser.add_argument('--no_display', action='store_true', help="Do not display images, just save if --save_output is specified.")
    parser.add_argument('--image_ext', type=str, default='.jpg,.jpeg,.png,.bmp', help="Image file extensions to process (comma-separated, default: .jpg,.jpeg,.png,.bmp).")
    parser.add_argument('--save_camchain_yaml', action='store_true', help="Save new camchain YAML file with undistorted camera parameters. File will be saved as <original_name>_undistort.yaml")
    return parser.parse_args()


def load_camchain_params(yaml_path, cam_name='cam0'):
    """
    Load camera parameters from camchain YAML file.

    Args:
        yaml_path: Path to the camchain YAML file
        cam_name: Camera name in the YAML file (default: 'cam0')

    Returns:
        Dictionary with camera parameters: fx, fy, cx, cy, distortion_coeffs, distortion_model
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Camchain YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        camchain = yaml.safe_load(f)

    if cam_name not in camchain:
        available_cams = list(camchain.keys())
        raise ValueError(f"Camera '{cam_name}' not found in YAML file. Available cameras: {available_cams}")

    cam_data = camchain[cam_name]

    # Extract intrinsics: [fx, fy, cx, cy]
    intrinsics = cam_data['intrinsics']
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Extract distortion coefficients
    distortion_coeffs = cam_data['distortion_coeffs']
    distortion_model = cam_data.get('distortion_model', 'radtan')

    print(f"从 {yaml_path} 加载相机 '{cam_name}' 的参数:")
    print(f"  相机模型: {cam_data.get('camera_model', 'unknown')}")
    print(f"  畸变模型: {distortion_model}")
    print(f"  内参: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"  畸变系数: {distortion_coeffs}")
    if 'resolution' in cam_data:
        print(f"  分辨率: {cam_data['resolution']}")

    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'distortion_coeffs': distortion_coeffs,
        'distortion_model': distortion_model
    }


def undistort_image(bgr_image, camera_matrix, dist_coeffs, zoom_factor=1.0):
    """
    Undistort a single image.

    Args:
        bgr_image: Input BGR image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        zoom_factor: Zoom factor for undistortion

    Returns:
        Undistorted image, new camera matrix, horizontal FOV, vertical FOV
    """
    h, w = bgr_image.shape[:2]

    # Create a new camera matrix adjusted for zoom factor
    new_camera_matrix = camera_matrix.copy()
    new_camera_matrix[0, 0] *= zoom_factor  # Adjust fx
    new_camera_matrix[1, 1] *= zoom_factor  # Adjust fy

    # Compute undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

    # Remap the original image to get the undistorted image
    undistorted_img = cv2.remap(
        bgr_image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )

    # Calculate FOV after de-distortion
    fovx_after = 2 * np.arctan((w / 2) / new_camera_matrix[0, 0]) * (180 / np.pi)
    fovy_after = 2 * np.arctan((h / 2) / new_camera_matrix[1, 1]) * (180 / np.pi)

    return undistorted_img, new_camera_matrix, fovx_after, fovy_after


def save_camchain_yaml(output_yaml_path, cam_name, new_camera_matrix, resolution, original_camchain_path=None):
    """
    Save new camchain YAML file with undistorted camera parameters.

    Args:
        output_yaml_path: Path to save the new YAML file
        cam_name: Camera name
        new_camera_matrix: New camera intrinsic matrix after undistortion
        resolution: Image resolution [width, height]
        original_camchain_path: Path to original camchain file (to copy rostopic if exists)
    """
    fx = new_camera_matrix[0, 0]
    fy = new_camera_matrix[1, 1]
    cx = new_camera_matrix[0, 2]
    cy = new_camera_matrix[1, 2]

    camchain_data = {
        cam_name: {
            'cam_overlaps': [],
            'camera_model': 'pinhole',
            'intrinsics': [float(fx), float(fy), float(cx), float(cy)],
            'distortion_model': 'radtan',
            'distortion_coeffs': [0.0, 0.0, 0.0, 0.0],  # No distortion after undistortion
            'resolution': resolution
        }
    }

    # Try to copy rostopic from original camchain if available
    if original_camchain_path and os.path.exists(original_camchain_path):
        try:
            with open(original_camchain_path, 'r') as f:
                original_camchain = yaml.safe_load(f)
                if cam_name in original_camchain and 'rostopic' in original_camchain[cam_name]:
                    camchain_data[cam_name]['rostopic'] = original_camchain[cam_name]['rostopic']
        except:
            pass

    # Ensure output directory exists
    output_dir = os.path.dirname(output_yaml_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to YAML file
    with open(output_yaml_path, 'w') as f:
        yaml.dump(camchain_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n新的 camchain YAML 文件已保存至: {output_yaml_path}")
    print(f"去畸变后的相机参数:")
    print(f"  相机模型: pinhole")
    print(f"  畸变模型: radtan (无畸变)")
    print(f"  内参: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"  畸变系数: [0.0, 0.0, 0.0, 0.0]")
    print(f"  分辨率: {resolution}")


def process_single_image(image_path, camera_matrix, dist_coeffs, zoom_factor, save_output, no_display):
    """
    Process a single image.
    """
    # Load image
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"警告: 无法加载图像 '{image_path}'")
        return False

    # Undistort the image
    undistorted_img, new_camera_matrix, fovx_after, fovy_after = undistort_image(
        bgr_image, camera_matrix, dist_coeffs, zoom_factor
    )

    print(f"处理图像: {os.path.basename(image_path)}")
    print(f"  校正后的FOV: 水平={fovx_after:.2f}度, 垂直={fovy_after:.2f}度")

    # Save the undistorted image if requested
    if save_output:
        output_dir = os.path.dirname(save_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(save_output, undistorted_img)
        print(f"  已保存至: {save_output}")

    # Display images if not disabled
    if not no_display:
        # 添加提示信息到图像上
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bgr_image, "按 'q' 键退出", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(undistorted_img, "按 'q' 键退出", (10, 30), font, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("原始图像", bgr_image)
        cv2.imshow("校正后图像", undistorted_img)
        
        print("  按 'q' 键退出程序...")

        # 等待按键，每100ms检查一次，如果是q或Q则退出
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            # 检查窗口是否已关闭
            try:
                if cv2.getWindowProperty("原始图像", cv2.WND_PROP_VISIBLE) < 1 or \
                   cv2.getWindowProperty("校正后图像", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break

        cv2.destroyAllWindows()
        print("  程序已正常退出")

    return True


def process_image_folder(image_folder, output_folder, camera_matrix, dist_coeffs, zoom_factor, image_extensions):
    """
    Process all images in a folder.

    Returns:
        Tuple of (new_camera_matrix, resolution) or (None, None) if no images processed
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图像文件夹不存在: {image_folder}")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])

    # Sort by filename
    image_files.sort()

    if len(image_files) == 0:
        print(f"警告: 在文件夹 '{image_folder}' 中未找到图像文件")
        print(f"支持的扩展名: {image_extensions}")
        return None, None

    print(f"找到 {len(image_files)} 张图像，开始批量处理...")
    print(f"输入文件夹: {image_folder}")
    print(f"输出文件夹: {output_folder}")
    print("")

    success_count = 0
    new_camera_matrix = None
    resolution = None

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        print(f"[{idx}/{len(image_files)}] 处理: {image_file}")

        # Load image
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            print(f"  警告: 无法加载图像，跳过")
            continue

        # Get resolution from first image
        if resolution is None:
            h, w = bgr_image.shape[:2]
            resolution = [w, h]

        # Undistort the image
        undistorted_img, new_camera_matrix, fovx_after, fovy_after = undistort_image(
            bgr_image, camera_matrix, dist_coeffs, zoom_factor
        )

        # Save the undistorted image
        cv2.imwrite(output_path, undistorted_img)
        success_count += 1

        # Print progress every 10 images or last image
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  进度: {idx}/{len(image_files)} ({100*idx//len(image_files)}%)")

    print("")
    print(f"批量处理完成！成功处理 {success_count}/{len(image_files)} 张图像")
    if new_camera_matrix is not None:
        print(f"校正后的FOV: 水平={fovx_after:.2f}度, 垂直={fovy_after:.2f}度")

    return new_camera_matrix, resolution


def main():
    args = parse_args()

    # Validate input arguments
    if not args.image_path and not args.image_folder:
        print("错误: 必须指定 --image_path 或 --image_folder 中的一个")
        return

    if args.image_path and args.image_folder:
        print("错误: --image_path 和 --image_folder 不能同时指定")
        return

    if args.image_folder and not args.output_folder:
        print("错误: 使用 --image_folder 时必须指定 --output_folder")
        return

    # Load camera parameters from YAML if provided
    if args.camchain_yaml:
        cam_params = load_camchain_params(args.camchain_yaml, args.cam_name)
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        distortion_coeffs = cam_params['distortion_coeffs']
        distortion_model = cam_params['distortion_model']

        # Convert distortion coefficients to numpy array
        dist_coeffs = np.array(distortion_coeffs)

        print(f"使用畸变模型: {distortion_model}")
    else:
        # Use command line arguments
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            print("错误: 未指定 --camchain_yaml 时，必须提供所有相机内参 (fx, fy, cx, cy)")
            return

        fx = args.fx
        fy = args.fy
        cx = args.cx
        cy = args.cy

        # Build distortion coefficients
        dist_list = []
        if args.k1 is not None:
            dist_list.append(args.k1)
        if args.k2 is not None:
            dist_list.append(args.k2)
        if args.p1 is not None:
            dist_list.append(args.p1)
        if args.p2 is not None:
            dist_list.append(args.p2)
        if args.k3 is not None:
            dist_list.append(args.k3)
        if args.k4 is not None:
            dist_list.append(args.k4)
        if args.k5 is not None:
            dist_list.append(args.k5)
        if args.k6 is not None:
            dist_list.append(args.k6)

        dist_coeffs = np.array(dist_list)
        print("使用命令行参数指定的相机参数")

    # Camera matrix
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]])

    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs}")
    print(f"缩放因子: {args.zoom_factor}")
    print("")

    # Process based on mode
    if args.image_folder:
        # Batch processing mode
        image_extensions = [ext.strip() for ext in args.image_ext.split(',')]
        new_camera_matrix, resolution = process_image_folder(
            args.image_folder,
            args.output_folder,
            camera_matrix,
            dist_coeffs,
            args.zoom_factor,
            image_extensions
        )

        # Save new camchain YAML if requested and processing was successful
        if args.save_camchain_yaml and new_camera_matrix is not None and args.camchain_yaml:
            # Generate output YAML path: <original_name>_undistort.yaml
            yaml_dir = os.path.dirname(args.camchain_yaml)
            yaml_basename = os.path.basename(args.camchain_yaml)
            yaml_name_without_ext = os.path.splitext(yaml_basename)[0]
            output_yaml_path = os.path.join(yaml_dir, f"{yaml_name_without_ext}_undistort.yaml")

            save_camchain_yaml(
                output_yaml_path,
                args.cam_name,
                new_camera_matrix,
                resolution,
                args.camchain_yaml
            )
    else:
        # Single image mode
        process_single_image(
            args.image_path,
            camera_matrix,
            dist_coeffs,
            args.zoom_factor,
            args.save_output,
            args.no_display
        )


if __name__ == "__main__":
    main()
