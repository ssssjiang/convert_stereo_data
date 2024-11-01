import numpy as np
import pandas as pd
import transformations as tf
from convert_stereo_data.convert_extrinsic import T_body_imu

# 定义文件路径
slampose_file = '/home/roborock/datasets/roborock/stereo/2020-08-14/slampose/data.csv'
output_file = '/home/roborock/datasets/roborock/stereo/2020-08-14/slampose/imu_data.csv'

# 定义 slam pose 数据的列名
slampose_columns = [
    'timestamp',
    'pose_x',
    'pose_y',
    'pose_yaw',
]

# 读取 slam pose 数据
slampose_df = pd.read_csv(
    slampose_file,
    header=None,
    names=slampose_columns,
    dtype={
        'timestamp': str,
        'pose_x': float,
        'pose_y': float,
        'pose_yaw': float,
    }
)

# 查看数据是否成功读取
print("slam pose 数据预览:")
print(slampose_df.tail(20))

# 将 slam pose 数据转换为 6DOF 数据
# 6DOF 数据的列名
output_columns = [
    'timestamp',
    'pose_x',
    'pose_y',
    'pose_z',
    'quat_w',
    'quat_x',
    'quat_y',
    'quat_z',
]

# 初始化输出数据
output_data = []

# convert frame from body to imu
# T_imu_body:
# [[-0.     -1.     -0.     -0.    ]
#  [ 1.      0.      0.     -0.104 ]
# [ 0.      0.      1.     -0.0162]
# [ 0.      0.      0.      1.    ]]
T_imu_body = np.array([
    [0.0, -1.0, 0.0, 0],
    [1.0, 0.0, 0.0, -0.104],
    [0.0, 0.0, 1.0, -0.0162],
    [0.0, 0.0, 0.0, 1.0]])
T_body_imu = np.linalg.inv(T_imu_body)
print(T_body_imu)

# 逐行处理 slam pose 数据
for index, row in slampose_df.iterrows():
    # 读取每一行的数据
    timestamp = row['timestamp']
    pose_x = row['pose_x']
    pose_y = row['pose_y']
    pose_yaw = row['pose_yaw']

    # 计算 6DOF 数据
    # 3x3 rotation matrix
    R = tf.euler_matrix(0, 0, pose_yaw)[:3, :3]
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [pose_x, pose_y, 0]
    M = np.dot(T_imu_body, M)
    M = np.dot(M, T_body_imu)
    # convert R to quaternion
    quat = tf.quaternion_from_matrix(M)

    # 将数据添加到输出数据中
    output_data.append([
        timestamp,
        M[0, 3],
        M[1, 3],
        M[2, 3],
        quat[3],
        quat[0],
        quat[1],
        quat[2],
    ])

# 将输出数据转换为 DataFrame
output_df = pd.DataFrame(output_data, columns=output_columns)

# 查看最终输出的数据(最后10行)
print("\n最终输出的数据预览:")
print(output_df.tail(20))

# 将整合后的数据输出到 CSV 文件，以空格分隔, header以#开头
output_df.to_csv(output_file, index=False, sep=',', header=True, float_format='%.6f')

print("\n数据已保存到:", output_file)
