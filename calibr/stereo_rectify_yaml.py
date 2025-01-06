import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Stereo Rectification for images.")
    parser.add_argument('-l', '--left_image', type=str, required=True, help="Path to the left input image.")
    parser.add_argument('-r', '--right_image', type=str, required=True, help="Path to the right input image.")
    parser.add_argument('-y', '--yaml_path', type=str, required=True, help="Path to the calibration YAML file.")
    parser.add_argument('-m', '--model', type=str, choices=['fisheye', 'standard'], required=True, help="Camera model: 'fisheye' or 'standard'.")
    parser.add_argument('-z', '--zoom_factor', type=float, default=1, help="Zoom factor for de-distortion (default: 1.0).")
    parser.add_argument('-g', '--grid_spacing', type=int, default=50, help="Spacing between grid lines in pixels.")
    return parser.parse_args()


def load_calibration(yaml_path):
    """从YAML文件加载相机内参、畸变系数、旋转矩阵、平移向量和投影矩阵"""
    print(f"Loading calibration parameters from: {yaml_path}")

    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"YAML file at path '{yaml_path}' could not be loaded.")

    camera_matrix_l = fs.getNode("M1").mat()
    dist_coeffs_l = fs.getNode("D1").mat().flatten()
    camera_matrix_r = fs.getNode("M2").mat()
    dist_coeffs_r = fs.getNode("D2").mat().flatten()
    R = fs.getNode("R").mat()  # 旋转矩阵
    T = fs.getNode("T").mat().flatten()  # 平移向量
    # PL = fs.getNode("PL").mat()  # 左相机投影矩阵
    # PR = fs.getNode("PR").mat()  # 右相机投影矩阵

    # 旋转矩阵 R
    # R = R.T
    # T = -R @ T

    fs.release()

    if camera_matrix_l is None or dist_coeffs_l is None or camera_matrix_r is None or dist_coeffs_r is None or R is None or T is None:
        raise ValueError("Failed to load calibration parameters from the YAML file.")

    print("Loaded Left Camera Matrix:")
    print(camera_matrix_l)
    print("\nLoaded Left Distortion Coefficients:")
    print(dist_coeffs_l)
    print("\nLoaded Right Camera Matrix:")
    print(camera_matrix_r)
    print("\nLoaded Right Distortion Coefficients:")
    print(dist_coeffs_r)
    print("\nLoaded Rotation Matrix:")
    print(R)
    print("\nLoaded Translation Vector:")
    print(T)

    return camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T


def apply_zoom_to_projection_matrix(P, zoom_factor, center_x, center_y):
    """
    调整投影矩阵的焦距和主点坐标以应用缩放因子。
    """
    P_new = P.copy()
    P_new[0, 0] *= zoom_factor  # 调整 fx
    P_new[1, 1] *= zoom_factor  # 调整 fy
    P_new[0, 2] = center_x  # 调整 cx
    P_new[1, 2] = center_y  # 调整 cy
    return P_new


# R is R_right_left, T is T_right_left
def rectify_stereo_images(left_img, right_img, K1, D1, K2, D2, R, T, image_size, model, zoom_factor=1.0):
    """进行双目极线校正，并调整图像的缩放比例"""
    # 获取图像中心
    center_x = image_size[0] / 2
    center_y = image_size[1] / 2

    if model == 'fisheye':
        # 使用 fisheye 模型进行极线校正
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, balance=0, fov_scale=1
        )
    else:
        # 使用标准模型进行极线校正
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )

    # 调整投影矩阵以应用缩放因子
    P1_zoomed = apply_zoom_to_projection_matrix(P1, zoom_factor, center_x, center_y)
    P2_zoomed = apply_zoom_to_projection_matrix(P2, zoom_factor, center_x, center_y)

    # 生成去畸变校正映射
    if model == 'fisheye':
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1_zoomed, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2_zoomed, image_size, cv2.CV_32FC1)
    else:
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1_zoomed, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2_zoomed, image_size, cv2.CV_32FC1)

    # 使用映射对左右图像进行校正
    rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

    return rectified_left, rectified_right, P1, P2


def draw_grid(img, step=50, color=(255, 0, 0)):
    """在图像上绘制网格线，便于验证校正效果"""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y in range(0, img.shape[0], step):
        cv2.line(img_color, (0, y), (img.shape[1], y), color, 1)
    for x in range(0, img.shape[1], step):
        cv2.line(img_color, (x, 0), (x, img.shape[0]), color, 1)
    return img_color


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载相机校准参数
    K1, D1, K2, D2, R, T = load_calibration(args.yaml_path)

    # 读取左、右图像
    img1 = cv2.imread(args.left_image, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.right_image, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Failed to load images at '{args.left_image}' or '{args.right_image}'.")

    # 检查图像尺寸是否匹配
    if img1.shape != img2.shape:
        raise ValueError("Left and right images must have the same dimensions.")

    # 获取图像尺寸
    image_size = (img1.shape[1], img1.shape[0])

    # 执行双目极线校正并获取投影矩阵
    rectified_left, rectified_right, P1, P2 = rectify_stereo_images(
        img1, img2, K1, D1, K2, D2, R, T, image_size, args.model, args.zoom_factor
    )

    # 在校正后的图像上绘制网格
    grid_left = draw_grid(rectified_left.copy())
    grid_right = draw_grid(rectified_right.copy())

    # 比较投影矩阵（PL, PR）与校正后的投影矩阵（P1, P2）
    print("\nLeft Projection Matrix (P1) from StereoRectify:")
    print(P1)
    print("\nRight Projection Matrix (P2) from StereoRectify:")
    print(P2)

    # 拼接左右校正后的图像
    combined_image = np.hstack((grid_left, grid_right))

    # 保存拼接后的图像
    output_dir = os.path.dirname(args.left_image)  # 输出路径与输入图像目录相同
    os.makedirs(output_dir, exist_ok=True)

    output_path_combined = os.path.join(output_dir, "rectified_combined.png")
    cv2.imwrite(output_path_combined, combined_image)

    # 显示拼接后的图像
    plt.figure(figsize=(12, 6))
    plt.title("Rectified Combined Image")
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
