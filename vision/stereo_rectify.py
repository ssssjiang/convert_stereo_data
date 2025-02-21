import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --------------------------------------------
# 1. Define Camera Parameters
# --------------------------------------------

# Left camera intrinsic parameters
K1 = np.array([
    [338.6803142637706, 0.0, 258.0690560860475],
    [0.0, 338.75365280379106, 320.413904138195],
    [0.0, 0.0, 1.0]
])

D1 = np.array([
    -0.05000465875052348,
    0.013638910705411799,
    -0.004489042932918501,
    -0.0022342707993817414
])

# Right camera intrinsic parameters
K2 = np.array([
    [338.2025698693698, 0.0, 261.4748043067035],
    [0.0, 338.3759984276547, 323.62424124741983],
    [0.0, 0.0, 1.0]
])

D2 = np.array([
    -0.053200225373542735,
    0.03182203889525646,
    -0.03173614929236543,
    0.009767890209496633
])

# # Rotation and translation from left to right camera
R = np.array([
    [9.99964716e-01, 5.22597035e-03, -6.66162030e-03],
    [-5.24784793e-03, 9.99980944e-01, -3.28138833e-03],
    [6.64477369e-03, 3.31630614e-03, 9.99972337e-01]
])

T = np.array([-5.11477817e-04, -6.50395836e-02, -7.49796887e-05])

# --------------------------------------------

# R = np.array([
#     [0.9999641553975523, 0.0052259809020549775, -0.00666160969022284],
#     [-0.005247928242327688, 0.999980845685386, -0.00328138865380209],
#     [0.006644333617218029, 0.0033162306833630123, 0.9999724273422901]
# ])
#
# T = np.array([-0.0005115137246098722, -0.06503955280279526, -7.49217089659946e-05])

# --------------------------------------------
# 2. Load Images and Determine Image Size
# --------------------------------------------

# Paths to the left and right images
left_image_path = "/home/roborock/下载/000385_converted_log/camera/camera0/156039.png"
right_image_path = "/home/roborock/下载/000385_converted_log/camera/camera1/156039.png"

# Read images in grayscale
img1 = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

if img1.shape != img2.shape:
    raise ValueError("Left and right images must have the same dimensions.")

# Set image size based on the loaded images
image_size = (img1.shape[1], img1.shape[0])  # (width, height)

# --------------------------------------------
# 3. Perform Stereo Rectification
# --------------------------------------------
zoom_factor = 0.7  # 0 < zoom_factor <= 1.0，根据需要调整

# Perform stereo rectification using fisheye module
R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    balance=0,
    fov_scale=1
)

print("R1:\n", R1)
print("R2:\n", R2)
print("P1:\n", P1)
print("P2:\n", P2)
print("Q:\n", Q)

# --------------------------------------------
# 4. Calculate the Rectification Maps
# --------------------------------------------

def apply_zoom_to_projection_matrix(P, zoom, center_x, center_y):
    """
    调整投影矩阵的焦距和主点坐标以应用缩放因子。

    参数:
        P (np.ndarray): 3x4 投影矩阵。
        zoom (float): 缩放因子。
        center_x (float): 图像中心的 x 坐标。
        center_y (float): 图像中心的 y 坐标。

    返回:
        np.ndarray: 调整后的投影矩阵。
    """
    P_new = P.copy()
    P_new[0, 0] *= zoom  # 调整 fx
    P_new[1, 1] *= zoom  # 调整 fy
    P_new[0, 2] = center_x  # 调整 cx
    P_new[1, 2] = center_y  # 调整 cy
    return P_new

# compute the center of the image
center_x = image_size[0] / 2
center_y = image_size[1] / 2

# adjust the projection matrices for zoom
P1_zoomed = apply_zoom_to_projection_matrix(P1, zoom_factor, center_x, center_y)
P2_zoomed = apply_zoom_to_projection_matrix(P2, zoom_factor, center_x, center_y)

print("P1_zoomed:\n", P1_zoomed)
print("P2_zoomed:\n", P2_zoomed)

# Generate undistort rectify map for the left camera
map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
    K1, D1, R1, P1_zoomed, image_size, cv2.CV_32FC1
)

# Generate undistort rectify map for the right camera
map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
    K2, D2, R2, P2_zoomed, image_size, cv2.CV_32FC1
)

# --------------------------------------------
# 5. Remap the Images to Rectify
# --------------------------------------------

# Apply the remapping to rectify the left image
rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)

# Apply the remapping to rectify the right image
rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# --------------------------------------------
# 7. Visualize Rectified Images
# --------------------------------------------

def draw_grid(img, step=50, color=(255, 0, 0)):
    """
    Draws a grid on the image for visual verification.
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y in range(0, img.shape[0], step):
        cv2.line(img_color, (0, y), (img.shape[1], y), color, 1)
    for x in range(0, img.shape[1], step):
        cv2.line(img_color, (x, 0), (x, img.shape[0]), color, 1)
    return img_color

# Draw grid on rectified images for verification
grid_img1 = draw_grid(rectified_img1.copy())
grid_img2 = draw_grid(rectified_img2.copy())

# 拼接左右校正后的图像
combined_image = np.hstack((grid_img1, grid_img2))

# 保存拼接后的图像
output_dir = os.path.dirname("/home/roborock/")  # 输出路径与输入图像目录相同
os.makedirs(output_dir, exist_ok=True)

output_path_combined = os.path.join(output_dir, "rectified_combined.png")
cv2.imwrite(output_path_combined, combined_image)

# 显示拼接后的图像
plt.figure(figsize=(12, 6))
plt.title("Rectified Combined Image")
plt.imshow(combined_image)
plt.axis('off')
plt.show()

