# ('base_link', 'd400_color', [0.2264836849091656, -0.05114194035652147, 0.916, -0.49676229968284147, 0.4998795887129772, -0.49510681269354095, 0.5081504289345848])
# ('base_link', 't265_fisheye1', [0.23757012582874237, -0.038724749600647444, 0.8950755692783204, -0.4959736284625998, 0.4983307159156683, -0.49589211227589786, 0.5096740825539087])
# ('base_link', 'marker', [0.17731624201267646, -0.08315129523911616, 0.9199814590558101, 0.017978434511175256, 0.011876376471955656, -0.7075455983002495, 0.7063391210320762])
# ('base_link', 'laser', [0.14350911104973352, -0.09642488461422752, 0.9980486189885366, 0.007946763733172666, 0.01829294482641466, 0.0008308669479766101, 0.9998007435363616])
# ('d400_color', 'd400_depth', [0.014881294220686, -2.32995425903937e-05, 0.000588475959375501, 0.00174536, 0.00293073, -0.00191947, 0.99999237])
# ('d400_color', 'd400_imu', [0.0203127935528755, -0.0051032523624599, -0.0112013882026076, 0.00174536, 0.00293073, -0.00191947, 0.99999237])
# ('t265_fisheye1', 't265_fisheye2', [0.0639765113592148, 0.000148267135955393, -0.000398468371713534, 0.00188497, 0.00347595, 0.00154952, 0.999991])
# ('t265_fisheye1', 't265_imu', [0.0106999985873699, 7.27595761418343e-12, -2.91038304567337e-11, 0.00554841, 0.00136098, 0.99998062, -0.00245956])


import transformations as tf
import numpy as np
# 计算两两之间的外参，得到外参矩阵，print出来
# 例如：
T_base_link_t265_fisheye1 = np.eye(4)
t_base_link_t265_fisheye1 = np.array([0.23757012582874237, -0.038724749600647444, 0.8950755692783204])
q_base_link_t265_fisheye1 = np.array([-0.4959736284625998, 0.4983307159156683, -0.49589211227589786, 0.5096740825539087])
R_base_link_t265_fisheye1 = tf.quaternion_matrix(q_base_link_t265_fisheye1)[:3, :3]
T_base_link_t265_fisheye1[:3, :3] = R_base_link_t265_fisheye1
T_base_link_t265_fisheye1[:3, 3] = t_base_link_t265_fisheye1
print("T_base_link_t265_fisheye1:")
print(T_base_link_t265_fisheye1)

T_t265_fisheye1_t265_fisheye2 = np.eye(4)
t_t265_fisheye1_t265_fisheye2 = np.array([0.0639765113592148, 0.000148267135955393, -0.000398468371713534])
q_t265_fisheye1_t265_fisheye2 = np.array([0.00188497, 0.00347595, 0.00154952, 0.999991])
R_t265_fisheye1_t265_fisheye2 = tf.quaternion_matrix(q_t265_fisheye1_t265_fisheye2)[:3, :3]
T_t265_fisheye1_t265_fisheye2[:3, :3] = R_t265_fisheye1_t265_fisheye2
T_t265_fisheye1_t265_fisheye2[:3, 3] = t_t265_fisheye1_t265_fisheye2
print("T_t265_fisheye1_t265_fisheye2:")
print(T_t265_fisheye1_t265_fisheye2)

T_base_link_t265_fisheye2 = np.dot(T_base_link_t265_fisheye1, T_t265_fisheye1_t265_fisheye2)
print("T_base_link_t265_fisheye2:")
print(T_base_link_t265_fisheye2)

T_t265_fisheye1_t265_imu = np.eye(4)
t_t265_fisheye1_t265_imu = np.array([0.0106999985873699, 7.27595761418343e-12, -2.91038304567337e-11])
q_t265_fisheye1_t265_imu = np.array([0.00554841, 0.00136098, 0.99998062, -0.00245956])
R_t265_fisheye1_t265_imu = tf.quaternion_matrix(q_t265_fisheye1_t265_imu)[:3, :3]
T_t265_fisheye1_t265_imu[:3, :3] = R_t265_fisheye1_t265_imu
T_t265_fisheye1_t265_imu[:3, 3] = t_t265_fisheye1_t265_imu
print("T_t265_fisheye1_t265_imu:")
print(T_t265_fisheye1_t265_imu)

T_base_link_t265_imu = np.dot(T_base_link_t265_fisheye1, T_t265_fisheye1_t265_imu)
print("T_base_link_t265_imu:")
print(T_base_link_t265_imu)