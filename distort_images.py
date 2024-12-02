import cv2
import numpy as np

# right
# fx = 259.7568821036086
# fy = 260.10819099573615
# cx = 394.9905360190833
# cy = 294.44467823631834
#
# k1 = 0.0008605939481375175
# k2 = 0.015921588486384006
# p1 = 0
# p2 = 0
# k3 = -0.012233412348966891
# k4 = 0.0012503893360738545

# left
fx = 260.063551592498
fy = 259.9904115230021
cx = 400.7237754048461
cy = 300.40231457638737

k1 = -0.0025081048464266195
k2 = 0.022744694807417455
p1 = 0
p2 = 0
k3 = -0.018000412523496625
k4 = 0.0026870339959659795

brg_image_path = '/home/roborock/下载/20241121095001014_R1098D43000318_2024112120DEV_rgb_images/encpic_000041/SL_R_71291_IR_0_NoPose_800X900.png'
bgr_image = cv2.imread(brg_image_path)

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])  # 相机矩阵
dist_coeffs = np.array([k1, k2, k3, k4])  # 畸变系数

h, w = bgr_image.shape[:2]

# 复制原始相机矩阵
new_camera_matrix = camera_matrix.copy()

# 设置缩放因子（小于1表示缩小，保留更多黑边；大于1表示放大，裁剪更多图像）
zoom_factor = 0.5  # 您可以根据需要调整这个值

# 调整焦距以改变缩放
new_camera_matrix[0, 0] *= zoom_factor  # 调整 fx
new_camera_matrix[1, 1] *= zoom_factor  # 调整 fy

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)
undistorted_img = cv2.remap(
    bgr_image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
)

# ==============================
# Calculate FOV After De-distortion
# ==============================

# Compute horizontal and vertical FOV after de-distortion using pinhole model
fovx_after = 2 * np.arctan((w / 2) / new_camera_matrix[0, 0]) * (180 / np.pi)
fovy_after = 2 * np.arctan((h / 2) / new_camera_matrix[1, 1]) * (180 / np.pi)
print("FOV after de-distortion:")
print("Horizontal FOV: {:.2f} degrees".format(fovx_after))
print("Vertical FOV: {:.2f} degrees".format(fovy_after))

# 显示结果
cv2.imshow("Original Image", bgr_image)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去畸变后的图片
# cv2.imwrite("/mnt/data/undistorted_with_black_borders.png", undistorted_img)
