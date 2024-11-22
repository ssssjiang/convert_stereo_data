import cv2
import numpy as np

import os
import cv2
import numpy as np

# 文件夹路径
yuv_folder_path = '/home/roborock/下载/IR'
output_folder_path = '/home/roborock/下载/IR_rgb'

# 创建保存输出图像的文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 图像的宽度和高度
width = 800
height = 600

# 使用 os.walk 递归遍历文件夹中的所有子文件夹和文件
for root, dirs, files in os.walk(yuv_folder_path):
    for filename in files:
        if filename.endswith('.yuv'):
            yuv_file_path = os.path.join(root, filename)

            # 读取 .yuv 文件数据
            with open(yuv_file_path, 'rb') as yuv_file:
                yuv_data = yuv_file.read()

            # 将 YUV 数据转换为 NumPy 数组并重塑为 NV12 格式
            yuv_image = np.frombuffer(yuv_data, dtype=np.uint8)
            yuv_image = yuv_image.reshape((height * 3 // 2, width))

            # 转换为 BGR 图像
            bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

            # 生成输出文件路径
            relative_path = os.path.relpath(root, yuv_folder_path)
            output_subfolder = os.path.join(output_folder_path, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            output_image_path = os.path.join(output_subfolder, filename.replace('.yuv', '.png'))

            # 保存 BGR 图像
            cv2.imwrite(output_image_path, bgr_image)

            # 显示 BGR 图像（可选）
            cv2.imshow('BGR Image', bgr_image)
            cv2.waitKey(500)  # 显示 500 毫秒
            cv2.destroyAllWindows()

print("YUV 文件已全部转换并保存为 RGB 图像。")