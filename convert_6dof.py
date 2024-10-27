import numpy as np
import pandas as pd
import transformations as tf

# 定义文件路径
slampose_file = '/home/songshu/datasets/2020-08-14/slampose/data.csv'
output_file = '/home/songshu/datasets/2020-08-14/slampose/6dof_data.csv'

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
print(slampose_df.tail(10))

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

# 逐行处理 slam pose 数据
for index, row in slampose_df.iterrows():
    # 读取每一行的数据
    timestamp = row['timestamp']
    pose_x = row['pose_x']
    pose_y = row['pose_y']
    pose_yaw = row['pose_yaw']

    # 计算 6DOF 数据
    # 3x3 rotation matrix
    R = tf.rotation_matrix(pose_yaw, [0, 0, 1])[:3, :3]
    M = np.eye(4)
    M[:3, :3] = R
    # convert R to quaternion
    quat = tf.quaternion_from_matrix(M)

    # 将数据添加到输出数据中
    output_data.append([
        timestamp,
        pose_x,
        pose_y,
        0.0,
        quat[3],
        quat[0],
        quat[1],
        quat[2],
    ])

# 将输出数据转换为 DataFrame
output_df = pd.DataFrame(output_data, columns=output_columns)

# 查看最终输出的数据(最后10行)
print("\n最终输出的数据预览:")
print(output_df.tail(10))

# 将整合后的数据输出到 CSV 文件，以空格分隔, header以#开头
output_df.to_csv(output_file, index=False, sep=',', header=True, float_format='%.6f')

print("\n数据已保存到:", output_file)
