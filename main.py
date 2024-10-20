import numpy as np
import pandas as pd

# 定义文件路径
imu_file = '/home/roborock/datasets/roborock/stereo/2020-08-14/imu0/data.csv'
wheel_file = '/home/roborock/datasets/roborock/stereo/2020-08-14/odo/odo.csv'
output_file = 'merged_data.csv'

# 定义 IMU 数据的列名
imu_columns = [
    'timestamp',
    'gyro_x',      # 陀螺仪X轴角速度
    'gyro_y',      # 陀螺仪Y轴角速度
    'gyro_z',      # 陀螺仪Z轴角速度
    'accel_x',     # 加速度计X轴加速度
    'accel_y',     # 加速度计Y轴加速度
    'accel_z'      # 加速度计Z轴加速度
]

# 定义轮速计数据的列名
wheel_columns = [
    'timestamp',
    'left_count',
    'right_count',
    'gyro_roll',
    'gyro_pitch',
    'gyro_yaw',
    'speed_v', # actual speed v
    'speed_w', # actual speed w
]

# 读取 IMU 数据
imu_df = pd.read_csv(
    imu_file,
    header=None,
    names=imu_columns,
    dtype={
        'timestamp': str,
        'gyro_x': float,
        'gyro_y': float,
        'gyro_z': float,
        'accel_x': float,
        'accel_y': float,
        'accel_z': float
    }
)

# 读取轮速计数据
wheel_df = pd.read_csv(
    wheel_file,
    header=None,
    names=wheel_columns,
    dtype={
        'timestamp': str,
        'left_count': int,
        'right_count': int,
        'gyro_roll': float,
        'gyro_pitch': float,
        'gyro_yaw': float,
        'speed_v': float,
        'speed_w': float,
    }
)

T_odo_imu = np.array([
    [0.0, 1.0, 0.0, 0.104],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.003309],
    [0.0, 0.0, 0.0, 1.0]
])

# gyro roll / pitch / yaw data in imu frame
# convert gyro roll / pitch / yaw data to odo frame
R_odo_imu = T_odo_imu[:3, :3]
gyro_roll = imu_df['gyro_x'].values
gyro_pitch = imu_df['gyro_y'].values
gyro_yaw = imu_df['gyro_z'].values

gyro_odo_roll = []
gyro_odo_pitch = []
gyro_odo_yaw = []

for i in range(len(gyro_roll)):
    gyro_odo = np.dot(R_odo_imu, np.array([gyro_roll[i], gyro_pitch[i], gyro_yaw[i]]))
    gyro_odo_roll.append(gyro_odo[0])
    gyro_odo_pitch.append(gyro_odo[1])
    gyro_odo_yaw.append(gyro_odo[2])

imu_df['gyro_odo_roll'] = gyro_odo_roll
imu_df['gyro_odo_pitch'] = gyro_odo_pitch
imu_df['gyro_odo_yaw'] = gyro_odo_yaw


# 查看数据是否成功读取
print("IMU 数据预览:")
print(imu_df.head())

print("\n轮速计数据预览:")
print(wheel_df.head())

# 合并数据集，基于时间戳
merged_df = pd.merge(
    imu_df[['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_odo_roll', 'gyro_odo_pitch', 'gyro_odo_yaw']],
    wheel_df[['timestamp', 'left_count', 'right_count', 'speed_v', 'speed_w']],
    on='timestamp',
    how='inner'  # 仅保留两个数据集中都存在的时间戳
)

# 查看合并后的数据
print("\n合并后的数据预览:")
print(merged_df.head())

# 选择并重命名所需的列
output_df = merged_df[[
    'timestamp',
    'accel_x',
    'accel_y',
    'accel_z',
    'gyro_odo_roll',      # roll
    'gyro_odo_pitch',      # pitch
    'gyro_odo_yaw',      # yaw
    'left_count',
    'right_count',
    'speed_v',
    'speed_w',
]].copy()

# 添加 data_type 列，并设置为 'gyroOdo'
output_df.insert(1, 'data_type', 'gyroOdo')

# add 0 to 2 - 10 columns
output_df.insert(2, 'pose_x', 0)
output_df.insert(3, 'pose_y', 0)
output_df.insert(4, 'pose_theta', 0)
output_df.insert(5, 'odo_gyro_pose_x', 0)
output_df.insert(6, 'odo_gyro_pose_y', 0)
output_df.insert(7, 'odo_gyro_pose_theta', 0)
output_df.insert(8, 'revised_pose_x', 0)
output_df.insert(9, 'revised_pose_y', 0)
output_df.insert(10, 'revised_pose_theta', 0)

# add 0 to 14 - 16 columns
output_df.insert(14, 'euler_roll', 0)
output_df.insert(15, 'euler_pitch', 0)
output_df.insert(16, 'euler_yaw', 0)

# add 0 to 20 - 23 columns
output_df.insert(22, 'target_v', 0)
output_df.insert(23, 'target_w', 0)

# timestamp from ns to ms (int)
output_df['timestamp'] = output_df['timestamp'].astype(int) // 1_000_000

# # remove left_count and right_count < 0 rows
# output_df = output_df[output_df['left_count'] >= 0]

# 查看最终输出的数据
print("\n最终输出的数据预览:")
print(output_df.head())

# 将整合后的数据输出到 CSV 文件
output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n数据已成功整合并输出到 {output_file}")
