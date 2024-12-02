import numpy as np
import transformations as tf


# T_body_cam0: 4x4 matrix
# data: [2.67949e-08,   0.0871558,    0.996195,   0.155,
#        -1.0, 2.33533e-09, 2.66929e-08, -0.0175,
#        5.55112e-17,   -0.996195,   0.0871558,  0.0065,
#        0.0,         0.0,         0.0,         1.0]

# T_body_cam1: 4x4 matrix
# data: [-0.00329331,   0.0852561,    0.996354,    0.154995,
#        -0.999924,    0.011519, -0.00429097,    0.017353,
#        -0.0118431,   -0.996293,   0.0852123,  0.0065518,
#        0.0,         0.0,         0.0,         1.0]

# T_body_odo: 4x4 matrix
# data: [1.0, 0.0, 0.0,  0.0,
#        0.0, 1.0, 0.0,  0.0,
#        0.0, 0.0, 1.0, -0.012891,
#        0.0, 0.0, 0.0, 1.0]

# T_body_imu: 4x4 matrix
# data: [ 0.0,  1.0,  0.0,  0.104,
#         -1.0,  0.0,  0.0,    0.0,
#         0.0,  0.0,  1.0, 0.0162,
#         0.0,  0.0,  0.0,  1.0]


# convert extrinsic matrix, odo as the body frame
T_body_cam0 = np.array([
    [2.67949e-08, 0.0871558, 0.996195, 0.155],
    [-1.0, 2.33533e-09, 2.66929e-08, -0.0175],
    [5.55112e-17, -0.996195, 0.0871558, 0.0065],
    [0.0, 0.0, 0.0, 1.0]
])

T_body_cam1 = np.array([
    [-0.00329331, 0.0852561, 0.996354, 0.154995],
    [-0.999924, 0.011519, -0.00429097, 0.017353],
    [-0.0118431, -0.996293, 0.0852123, 0.0065518],
    [0.0, 0.0, 0.0, 1.0]
])

T_body_odo = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.012891],
    [0.0, 0.0, 0.0, 1.0]
])

T_body_imu = np.array([
    [0.0, 1.0, 0.0, 0.104],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0162],
    [0.0, 0.0, 0.0, 1.0]
])

# T_odo_cam0 = T_odo_body * T_body_cam0
T_odo_cam0 = np.dot(np.linalg.inv(T_body_odo), T_body_cam0)
T_cam0_odo = np.linalg.inv(T_odo_cam0)
# convert T_cam0_odo[:3][:3] to quaternion
q_cam0_odo = tf.quaternion_from_matrix(T_cam0_odo)

# T_odo_cam1 = T_odo_body * T_body_cam1
T_odo_cam1 = np.dot(np.linalg.inv(T_body_odo), T_body_cam1)
T_cam1_odo = np.linalg.inv(T_odo_cam1)
# convert T_cam1_odo[:3][:3] to quaternion
q_cam1_odo = tf.quaternion_from_matrix(T_cam1_odo)

T_cam0_cam1 = np.dot(np.linalg.inv(T_body_cam0), T_body_cam1)
q_cam0_cam1 = tf.quaternion_from_matrix(T_cam0_cam1)

T_cam1_cam0 = np.dot(np.linalg.inv(T_body_cam1), T_body_cam0)
q_cam1_cam0 = tf.quaternion_from_matrix(T_cam1_cam0)

# T_odo_imu = T_odo_body * T_body_imu
T_odo_imu = np.dot(np.linalg.inv(T_body_odo), T_body_imu)

T_imu_cam0 = np.dot(np.linalg.inv(T_body_imu), T_body_cam0)
q_imu_cam0 = tf.quaternion_from_matrix(T_imu_cam0)

T_imu_cam1 = np.dot(np.linalg.inv(T_body_imu), T_body_cam1)
q_imu_cam1 = tf.quaternion_from_matrix(T_imu_cam1)

T_imu_body = np.linalg.inv(T_body_imu)
T_odo_body = np.linalg.inv(T_body_odo)

print("T_odo_cam0:")
print(T_odo_cam0)
print("\nT_cam0_odo:")
print(T_cam0_odo)
print("\nq_cam0_odo:")
print(q_cam0_odo)

print("\nT_odo_cam1:")
print(T_odo_cam1)
print("\nT_cam1_odo:")
print(T_cam1_odo)
print("\nq_cam1_odo:")
print(q_cam1_odo)

print("\nT_cam0_cam1:")
print(T_cam0_cam1)

print("\nT_cam1_cam0:")
print(T_cam1_cam0)

print("\nT_odo_imu:")
print(T_odo_imu)

print("\nT_imu_cam0:")
print(T_imu_cam0)

print("\nT_imu_cam1:")
print(T_imu_cam1)

print("\nT_imu_body:")
print(T_imu_body)

print("\nT_odo_body:")
print(T_odo_body)

# T_C_B extrinsic:
# 0.0205522,    -0.9996291,    0.0178662,    -0.0297000
# 0.0898282,    -0.0159515,    -0.9958295,    0.0612000
# 0.9957452,    0.0220714,    0.0894670,    -0.1670000

T_CR_B = np.array([
    [0.0205522, -0.9996291, 0.0178662, -0.0297000],
    [0.0898282, -0.0159515, -0.9958295, 0.0612000],
    [0.9957452, 0.0220714, 0.0894670, -0.1670000],
    [0.0, 0.0, 0.0, 1.0]
])
# convert to


T_B_CR = np.linalg.inv(T_CR_B)
print("\nT_B_CR:")
print(T_B_CR)

# T = np.array([
#     [0.0, -1.0, 0.0, 0],
#     [0, 0.0, -1, 0],
#     [1, 0.0, 0, 0],
#     [0.0, 0.0, 0.0, 1.0]
# ])
#
# T_B_CR = np.dot(T, T_B_CR)

eular_B_CR = tf.euler_from_matrix(T_B_CR, axes='sxyz')
print(eular_B_CR)

# extrinsic:
# 0.9990988,    0.0179610,    -0.0384577,    0.0583342
# -0.0205293,    0.9975105,    -0.0674639,    -0.0017807
# 0.0371503,    0.0681926,    0.9969802,    0.0019363

T_CL_CR = np.array([
    [0.9990988, 0.0179610, -0.0384577, 0.0583342],
    [-0.0205293, 0.9975105, -0.0674639, -0.0017807],
    [0.0371503, 0.0681926, 0.9969802, 0.0019363],
    [0.0, 0.0, 0.0, 1.0]
])

eular_B_CL = tf.euler_from_matrix(T_B_CR *  np.linalg.inv(T_CL_CR), axes='sxyz')
print(eular_B_CL)

eular_CL_CR = tf.euler_from_matrix(T_CL_CR, axes='sxyz')
print(eular_CL_CR)

T_B_CL = np.dot(T_B_CR, np.linalg.inv(T_CL_CR))
print("\nT_B_CL:")
print(T_B_CL)