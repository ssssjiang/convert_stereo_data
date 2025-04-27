import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --------------------------------------------
# 1. Define Camera Parameters
# --------------------------------------------
# cam0:
#   cam_overlaps: [1]
#   camera_model: pinhole
#   distortion_coeffs: [1.9264416073422455, 0.3352504441349091, -8.260465789774901e-05, -0.0002827431714135497, -0.01261188664138058, 2.2847458015574476, 0.9000561794297213, 0.0029298021182558994]
#   distortion_model: radtan8
#   intrinsics: [327.97005666465026, 328.10382442993483, 262.04137719767436, 325.5019849239798]
#   resolution: [544, 640]
#   rostopic: /image
# Right camera intrinsic parameters
K1 = np.array([
    [327.97005666465026, 0.0, 262.04137719767436],
    [0.0, 328.10382442993483, 325.5019849239798],
    [0.0, 0.0, 1.0]
])

# For standard camera model, using 8 coefficients format (k1, k2, p1, p2, k3, k4, k5, k6)
D1 = np.array([
    1.9264416073422455,  # k1
    0.3352504441349091,  # k2
    -8.260465789774901e-05,  # p1
    -0.0002827431714135497,  # p2
    -0.01261188664138058,  # k3
    2.2847458015574476,  # k4
    0.9000561794297213,  # k5
    0.0029298021182558994  # k6
])

#   camera_model: pinhole
#   distortion_coeffs: [0.5729525396536029, 0.0020189103859255446, -0.00011978871703312007, 4.151954995754269e-05, -0.0012806796277598455, 0.9254672528339936, 0.10857346697421204, -0.006363662121772228]
#   distortion_model: radtan8
#   intrinsics: [327.57536130200486, 327.66355688970094, 266.1246896917541, 326.05116417420976]
#   resolution: [544, 640]
#   rostopic: /image1
K2 = np.array([
    [327.57536130200486, 0.0, 266.1246896917541],
    [0.0, 327.66355688970094, 326.05116417420976],
    [0.0, 0.0, 1.0]
])

# For standard camera model, using 8 coefficients format (k1, k2, p1, p2, k3, k4, k5, k6)
D2 = np.array([
    0.5729525396536029,  # k1
    0.0020189103859255446,  # k2
    -0.00011978871703312007,  # p1
    4.151954995754269e-05,  # p2
    -0.0012806796277598455,  # k3
    0.9254672528339936,  # k4
    0.10857346697421204,  # k5
    -0.006363662121772228  # k6
])

#   - [0.9999871153098752, 0.001003502077869496, -0.004976162960663991, -0.0004100737193986931]
#   - [-0.0009912983287079897, 0.9999964968587095, 0.0024543019230421985, -0.0649217727292534]
#   - [0.004978608425541555, -0.002449337438096166, 0.9999846069836558, -3.724150566495725e-05]
#   - [0.0, 0.0, 0.0, 1.0]
# Rotation and translation from left to right camera
R = np.array([
    [0.9999871153098752, 0.001003502077869496, -0.004976162960663991],
    [-0.0009912983287079897, 0.9999964968587095, 0.0024543019230421985],
    [0.004978608425541555, -0.002449337438096166, 0.9999846069836558]
])

T = np.array([-0.0004100737193986931, -0.0649217727292534, -3.724150566495725e-05])


# R = R.T
# T = -R @ T
# --------------------------------------------
# 2. Load Images and Determine Image Size
# --------------------------------------------

# Paths to the left and right images
image1_path = "/home/roborock/datasets/roborock/stereo/mower/78_normal_z_10X8_cloudy/camera/camera0/487690.png"
image0_path = "/home/roborock/datasets/roborock/stereo/mower/78_normal_z_10X8_cloudy/camera/camera1/487690.png"

# Read images in grayscale
img1 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)

if img1.shape != img2.shape:
    raise ValueError("Left and right images must have the same dimensions.")

# Set image size based on the loaded images
image_size = (img1.shape[1], img1.shape[0])  # (width, height)

# --------------------------------------------
# 3. Perform Stereo Rectification
# --------------------------------------------
zoom_factor = 0.55  # 0 < zoom_factor <= 1.0，根据需要调整

# Perform stereo rectification using fisheye module
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY
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
map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1_zoomed, image_size, cv2.CV_32FC1
)

# Generate undistort rectify map for the right camera
map2x, map2y = cv2.initUndistortRectifyMap(
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

def draw_grid(img, step=30, color=(225, 0, 0)):
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
combined_image = np.vstack((grid_img1, grid_img2))

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

