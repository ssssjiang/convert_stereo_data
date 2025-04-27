import cv2
import numpy as np
import argparse
import os

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
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--fx', type=float, default=624.393420 / 2, help="Focal length fx of the camera.")
    parser.add_argument('--fy', type=float, default=624.403147 / 2, help="Focal length fy of the camera.")
    parser.add_argument('--cx', type=float, default=636.068917 / 2, help="Principal point cx of the camera.")
    parser.add_argument('--cy', type=float, default=587.965537 / 2, help="Principal point cy of the camera.")
    parser.add_argument('--k1', type=float, default=7.948753, help="Distortion coefficient k1.")
    parser.add_argument('--k2', type=float, default=4.621623, help="Distortion coefficient k2.")
    parser.add_argument('--k3', type=float, default=0.210347, help="Distortion coefficient k3.")
    parser.add_argument('--k4', type=float, default=8.256173, help="Distortion coefficient k4.")
    parser.add_argument('--k5', type=float, default=7.062371, help="Distortion coefficient k5.")
    parser.add_argument('--k6', type=float, default=1.022221, help="Distortion coefficient k6.")
    parser.add_argument('--p1', type=float, default=0.000049, help="Distortion coefficient p1.")
    parser.add_argument('--p2', type=float, default=0.000069, help="Distortion coefficient p2.")
    parser.add_argument('--zoom_factor', type=float, default=1, help="Zoom factor for de-distortion (default: 0.5).")
    parser.add_argument('--save_output', type=str, default='', help="Path to save the undistorted image (optional).")
    parser.add_argument('--no_display', action='store_true', help="Do not display images, just save if --save_output is specified.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image
    bgr_image = cv2.imread(args.image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"图像路径 '{args.image_path}' 无法加载。")

    # Camera matrix
    camera_matrix = np.array([[args.fx, 0, args.cx],
                              [0, args.fy, args.cy],
                              [0,  0,  1]])

    # Distortion coefficients
    dist_coeffs = np.array([args.k1, args.k2, args.p1, args.p2, args.k3, args.k4, args.k5, args.k6]) # 这个顺序太冷门了
    # dist_coeffs = np.array([args.k1, args.k2, args.k3, args.k4])
    # Get image dimensions
    h, w = bgr_image.shape[:2]

    # Create a new camera matrix adjusted for zoom factor
    new_camera_matrix = camera_matrix.copy()
    new_camera_matrix[0, 0] *= args.zoom_factor  # Adjust fx
    new_camera_matrix[1, 1] *= args.zoom_factor  # Adjust fy

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
    print("校正后的FOV:")
    print("水平FOV: {:.2f} 度".format(fovx_after))
    print("垂直FOV: {:.2f} 度".format(fovy_after))

    # Save the undistorted image if requested
    if args.save_output:
        output_dir = os.path.dirname(args.save_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(args.save_output, undistorted_img)
        print(f"校正后的图像已保存至: {args.save_output}")

    # Display images if not disabled
    if not args.no_display:
        # 添加提示信息到图像上
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bgr_image, "按 'q' 键退出", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(undistorted_img, "按 'q' 键退出", (10, 30), font, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("原始图像", bgr_image)
        cv2.imshow("校正后图像", undistorted_img)
        
        print("按 'q' 键退出程序...")
        
        # 等待按键，每100ms检查一次，如果是q或Q则退出
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            # 检查窗口是否已关闭
            if cv2.getWindowProperty("原始图像", cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty("校正后图像", cv2.WND_PROP_VISIBLE) < 1:
                break
                
        cv2.destroyAllWindows()
        print("程序已正常退出")
    else:
        print("已禁用图像显示")


if __name__ == "__main__":
    main()
