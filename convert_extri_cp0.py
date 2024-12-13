import numpy as np
import transformations as tf


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

T_CR_CL = np.linalg.inv(T_CL_CR)
print("\nT_CR_CL:")
print(T_CR_CL)

eular_B_CL = tf.euler_from_matrix(T_B_CR *  np.linalg.inv(T_CL_CR), axes='sxyz')
print(eular_B_CL)

eular_CL_CR = tf.euler_from_matrix(T_CL_CR, axes='sxyz')
print(eular_CL_CR)

T_B_CL = np.dot(T_B_CR, np.linalg.inv(T_CL_CR))
print("\nT_B_CL:")
print(T_B_CL)

T_CL_B =np.dot(T_CL_CR, np.linalg.inv(T_B_CR))
print("\nT_CL_B:")
print(T_CL_B)