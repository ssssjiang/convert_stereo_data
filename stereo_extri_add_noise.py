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
    cam1_T_B_C_data = data["sensors"][1]["cameras"][1]["T_B_C"]["data"]

    # Convert to matrices
    T_B_C0 = np.array(cam0_T_B_C_data).reshape(4, 4)
    T_B_C1 = np.array(cam1_T_B_C_data).reshape(4, 4)

    # 输出矩阵
    print("T_B_C0:\n", T_B_C0)
    print("T_B_C1:\n", T_B_C1)

    euler_B_C0 = tf.euler_from_matrix(T_B_C0, axes='sxyz')
    print("\nEuler B_C0:")
    print(euler_B_C0)

    euler_B_C1 = tf.euler_from_matrix(T_B_C1, axes='sxyz')
    print("\nEuler B_C1:")
    print(euler_B_C1)

    T_C1_C0 = np.dot(np.linalg.inv(T_B_C1), T_B_C0)
    print("\nT_C1_C0:")
    print(T_C1_C0)

    euler_C1_C0 = tf.euler_from_matrix(T_C1_C0, axes='sxyz')
    print("\nEuler C1_C0:")
    print(euler_C1_C0)

    T_C0_C1 = np.linalg.inv(T_C1_C0)
    print("\nT_C0_C1:")
    print(T_C0_C1)

    euler_C0_C1 = tf.euler_from_matrix(T_C0_C1, axes='sxyz')
    print("\nEuler C0_C1:")
    print(euler_C0_C1)

    # ================== Add noise ==================
    euler_C0_C1 = np.array(euler_C0_C1)
    euler_C0_C1["xyz".index(axis)] += np.deg2rad(angle)

    # Recalculate the transformation matrix
    T_C0_C1_r = tf.euler_matrix(euler_C0_C1[0], euler_C0_C1[1], euler_C0_C1[2], axes='sxyz')
    T_C0_C1_r[:3, 3] = T_C0_C1[:3, 3]  # Preserve translation
    print("\nT_C0_C1 with noise, axis:", axis, "angle:", angle)
    print(T_C0_C1_r)

    # Update T_B_C1 with the new transformation
    T_B_C1_r = np.dot(T_B_C0, T_C0_C1_r)
    print("\nT_B_C1 with noise, axis:", axis, "angle:", angle)
    print(T_B_C1_r)

    euler_B_C1_r = tf.euler_from_matrix(T_B_C1_r, axes='sxyz')
    print("\nEuler B_C1 with noise, axis:", axis, "angle:", angle)
    print(euler_B_C1_r)

    # ================== Save to YAML ==================
    # Save the modified T_B_C1 back to the data structure
    data["sensors"][1]["cameras"][1]["T_B_C"]["data"] = [float(value) for value in T_B_C1_r.flatten()]

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
