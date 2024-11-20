import cv2
import numpy as np

yuv_file_path = '/home/roborock/下载/SL_L_3423105_IR_0_NoPose_800X900.yuv'
with open(yuv_file_path, 'rb') as yuv_file:
    yuv_data = yuv_file.read()

width = 800
height = 600

yuv_image = np.frombuffer(yuv_data, dtype=np.uint8)
yuv_image = yuv_image.reshape((height * 3 // 2, width))
bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

cv2.imshow('BGR Image', bgr_image)
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()
