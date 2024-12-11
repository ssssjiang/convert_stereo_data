import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --------------------------------------------
# 1. Define Camera Parameters
# --------------------------------------------

# Left camera intrinsic parameters
K1 = np.array([
    [260.063551592498, 0.0, 400.7237754048461],
    [0.0, 259.9904115230021, 300.40231457638737],
    [0.0, 0.0, 1.0]
])

D1 = np.array([
    -0.0025081048464266195,
    0.022744694807417455,
    -0.018000412523496625,
    0.0026870339959659795
])

# Right camera intrinsic parameters
K2 = np.array([
    [259.7568821036086, 0.0, 394.9905360190833],
    [0.0, 260.10819099573615, 294.44467823631834],
    [0.0, 0.0, 1.0]
])

D2 = np.array([
    0.0008605939481375175,
    0.015921588486384006,
    -0.012233412348966891,
    0.0012503893360738545
])

# Rotation and translation from left to right camera
R = np.array([
    [9.99098793e-01, -2.05292927e-02, 3.71502418e-02],
    [1.79610022e-02, 9.97510473e-01, 6.81926137e-02],
    [-3.84577621e-02, -6.74638908e-02, 9.96980307e-01]
])

T = np.array([-5.83901193e-02, 5.96484850e-04, 1.92816866e-04])

# --------------------------------------------
# 2. Load Images and Determine Image Size
# --------------------------------------------

# Paths to the left and right images
left_image_path = "/home/roborock/下载/images/encpic_000054/SL_R_18840977_IR_0_NoPose_800X900.png"
right_image_path = "/home/roborock/下载/images/encpic_000054/SL_L_18840977_IR_0_NoPose_800X900.png"

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
zoom_factor = 0.3   # 0 < zoom_factor <= 1.0，根据需要调整

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
# 6. Save Rectified Images
# --------------------------------------------

# Define output paths
output_dir = "/home/roborock/"
output_path_left = os.path.join(output_dir, "24854721_left.png")
output_path_right = os.path.join(output_dir, "24854721_right.png")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save rectified images
cv2.imwrite(output_path_left, rectified_img1)
cv2.imwrite(output_path_right, rectified_img2)

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

# Display rectified images with grid
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Rectified Left with Grid")
plt.imshow(grid_img1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Rectified Right with Grid")
plt.imshow(grid_img2)
plt.axis('off')

plt.show()
