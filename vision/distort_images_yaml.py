import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Undistort an image with fisheye or standard model.")
    parser.add_argument('-i', '--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('-y', '--yaml_path', type=str, required=True, help="Path to the calibration YAML file.")
    parser.add_argument('-cm', '--camera_matrix_node', type=str, default="M1", help="Node name for camera matrix.")
    parser.add_argument('-dc', '--dist_coeffs_node', type=str, default="D1", help="Node name for distortion coefficients.")
    parser.add_argument('-m', '--model', type=str, choices=["standard", "fisheye"], required=True, help="Distortion model to use: 'standard' or 'fisheye'.")
    parser.add_argument('-z', '--zoom_factor', type=float, default=0.5, help="Zoom factor for de-distortion (default: 1.0).")
    parser.add_argument('-g', '--grid_spacing', type=int, default=50, help="Spacing between grid lines in pixels.")
    return parser.parse_args()


def load_calibration(yaml_path, camera_matrix_node, dist_coeffs_node):
    """Load calibration parameters from a YAML file."""
    print(f"Loading calibration parameters from: {yaml_path}")

    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"YAML file at path '{yaml_path}' could not be loaded.")

    camera_matrix = fs.getNode(camera_matrix_node).mat()
    dist_coeffs = fs.getNode(dist_coeffs_node).mat().flatten()

    fs.release()

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("Failed to load calibration parameters from the YAML file.")

    print("Loaded Camera Matrix:")
    print(camera_matrix)
    print("\nLoaded Distortion Coefficients:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs


def undistort_image(image, camera_matrix, dist_coeffs, model, zoom_factor):
    """Undistort an image using the selected model."""
    h, w = image.shape[:2]

    if model == "fisheye":
        print("Using fisheye distortion model.")
        new_camera_matrix = camera_matrix.copy()
        new_camera_matrix[0, 0] *= zoom_factor
        new_camera_matrix[1, 1] *= zoom_factor

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )
    else:  # standard model
        print("Using standard distortion model.")
        new_camera_matrix = camera_matrix.copy()
        new_camera_matrix[0, 0] *= zoom_factor
        new_camera_matrix[1, 1] *= zoom_factor
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
        )

    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def draw_grid(image, spacing):
    """Draw a grid on the image with the specified spacing."""
    h, w = image.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(image, (x, 0), (x, h), color=(0, 255, 0), thickness=1)
    for y in range(0, h, spacing):
        cv2.line(image, (0, y), (w, y), color=(0, 255, 0), thickness=1)
    return image


def main():
    args = parse_args()

    # Load calibration parameters
    camera_matrix, dist_coeffs = load_calibration(args.yaml_path, args.camera_matrix_node, args.dist_coeffs_node)

    # Load image
    bgr_image = cv2.imread(args.image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Image at path '{args.image_path}' could not be loaded.")

    # Undistort image
    undistorted_img = undistort_image(bgr_image, camera_matrix, dist_coeffs, args.model, args.zoom_factor)

    # Draw grid on the undistorted image
    grid_image = draw_grid(undistorted_img.copy(), args.grid_spacing)

    # Save the undistorted image
    if args.model == "fisheye":
        output_path = args.image_path.replace(".png", "_undistorted_fisheye_grid.png")
    else:
        output_path = args.image_path.replace(".png", "_undistorted_standard_grid.png")
    cv2.imwrite(output_path, grid_image)

    print(f"Undistorted image with grid saved to: {output_path}")


if __name__ == "__main__":
    main()
