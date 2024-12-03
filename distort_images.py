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
# fx = 260.063551592498
# fy = 259.9904115230021
# cx = 400.7237754048461
# cy = 300.40231457638737
#
# k1 = -0.0025081048464266195
# k2 = 0.022744694807417455
# p1 = 0
# p2 = 0
# k3 = -0.018000412523496625
# k4 = 0.0026870339959659795

import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Undistort an image and calculate FOV after de-distortion.")
    parser.add_argument('--image_path', type=str, required=True help="Path to the input image.")
    parser.add_argument('--fx', type=float, default=260.063551592498, help="Focal length fx of the camera.")
    parser.add_argument('--fy', type=float, default=259.9904115230021, help="Focal length fy of the camera.")
    parser.add_argument('--cx', type=float, default=400.7237754048461, help="Principal point cx of the camera.")
    parser.add_argument('--cy', type=float, default=300.40231457638737, help="Principal point cy of the camera.")
    parser.add_argument('--k1', type=float, default=-0.0025081048464266195, help="Distortion coefficient k1.")
    parser.add_argument('--k2', type=float, default=0.022744694807417455, help="Distortion coefficient k2.")
    parser.add_argument('--k3', type=float, default=-0.018000412523496625, help="Distortion coefficient k3.")
    parser.add_argument('--k4', type=float, default=0.0026870339959659795, help="Distortion coefficient k4.")
    parser.add_argument('--zoom_factor', type=float, default=0.5, help="Zoom factor for de-distortion (default: 0.5).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image
    bgr_image = cv2.imread(args.image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Image at path '{args.image_path}' could not be loaded.")

    # Camera matrix
    camera_matrix = np.array([[args.fx, 0, args.cx],
                              [0, args.fy, args.cy],
                              [0,  0,  1]])

    # Distortion coefficients
    dist_coeffs = np.array([args.k1, args.k2, args.k3, args.k4])

    # Get image dimensions
    h, w = bgr_image.shape[:2]

    # Create a new camera matrix adjusted for zoom factor
    new_camera_matrix = camera_matrix.copy()
    new_camera_matrix[0, 0] *= args.zoom_factor  # Adjust fx
    new_camera_matrix[1, 1] *= args.zoom_factor  # Adjust fy

    # Compute undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

    # Remap the original image to get the undistorted image
    undistorted_img = cv2.remap(
        bgr_image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )

    # Calculate FOV after de-distortion
    fovx_after = 2 * np.arctan((w / 2) / new_camera_matrix[0, 0]) * (180 / np.pi)
    fovy_after = 2 * np.arctan((h / 2) / new_camera_matrix[1, 1]) * (180 / np.pi)
    print("FOV after de-distortion:")
    print("Horizontal FOV: {:.2f} degrees".format(fovx_after))
    print("Vertical FOV: {:.2f} degrees".format(fovy_after))

    # Display results
    cv2.imshow("Original Image", bgr_image)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the undistorted image if needed
    # cv2.imwrite("/mnt/data/undistorted_with_black_borders.png", undistorted_img)


if __name__ == "__main__":
    main()
