import argparse
import numpy as np
import yaml
import transformations as tf


def modify_euler_angles(file_path, axis, angle):
    # Load YAML file
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    # Read T_B_C matrices
    cam0_T_B_C_data = data["sensors"][1]["cameras"][0]["T_B_C"]["data"]

    # Convert to matrices
    T_B_C0 = np.array(cam0_T_B_C_data).reshape(4, 4)

    # 输出矩阵
    print("T_B_C0:\n", T_B_C0)

    euler_B_C0 = tf.euler_from_matrix(T_B_C0, axes='sxyz')
    print("\nEuler B_C0:")
    print(euler_B_C0)

    T = np.array([
        [0.0, -1.0, 0.0, 0],
        [0, 0.0, -1, 0],
        [1, 0.0, 0, 0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    T_B_C0 = np.dot(T, T_B_C0)

    euler_B_C0 = tf.euler_from_matrix(T_B_C0, axes='sxyz')
    print("\nEuler B_C0 after:")
    print(euler_B_C0)

    # ================== Add noise ==================
    euler_B_C0 = np.array(euler_B_C0)
    euler_B_C0["xyz".index(axis)] += np.deg2rad(angle)

    # Recalculate the transformation matrix
    T_B_C0_r = tf.euler_matrix(euler_B_C0[0], euler_B_C0[1], euler_B_C0[2], axes='sxyz')
    T_B_C0_r[:3, 3] = T_B_C0[:3, 3]  # Preserve translation
    print("\nT_B_C0 with noise, axis:", axis, "angle:", angle)
    print(T_B_C0_r)

    # recover the original matrix
    T_B_C0_r = np.dot(np.linalg.inv(T), T_B_C0_r)
    print("\nT_B_C0_r recover:")
    print(T_B_C0_r)

    # ================== Save to YAML ==================
    # Save the modified T_B_C1 back to the data structure
    data["sensors"][1]["cameras"][0]["T_B_C"]["data"] = [float(value) for value in T_B_C0_r.flatten()]

    suffix = f"_{axis}_{int(angle)}deg"
    new_file_path = file_path.replace(".yaml", f"{suffix}.yaml")
    with open(new_file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, width=120, indent=4)

    print(f"Modified YAML saved to: {new_file_path}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Modify YAML file with rotation noise.")
    parser.add_argument("--file_path", type=str, help="Path to the input YAML file.")
    parser.add_argument("--axis", type=str, choices=['x', 'y', 'z'], help="Rotation axis: roll (x), pitch (y), yaw (z).")
    parser.add_argument("--angle", type=float, help="Angle to add to the specified axis (in degrees).")

    args = parser.parse_args()

    # Call the function with parsed arguments
    modify_euler_angles(args.file_path, args.axis, args.angle)