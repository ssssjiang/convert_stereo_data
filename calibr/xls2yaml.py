import cv2
import pandas as pd
import yaml
import os

# 创建输出目录
root_dir = '/home/roborock/下载/oms_data/'

# 读取 .xls 文件
xls_data = pd.read_excel(root_dir + '/DualCal-2025年1月3日离线更新数据.xls', sheet_name=None)

# 提取并保存每组数据
for i, record in enumerate(xls_data.get("DualCal2024_12_25", []).to_dict(orient='records')):
    # 检查数据完整性
    required_keys = [
        "Left Fx", "Left Fy", "Left Cx", "Left Cy",
        "Right Fx", "Right Fy", "Right Cx", "Right Cy",
        "Left K1", "Left K2", "Left K3", "Left K4",
        "Right K1", "Right K2", "Right K3", "Right K4",
        "ROTATE_TRANS R00", "ROTATE_TRANS R01", "ROTATE_TRANS R02",
        "ROTATE_TRANS R10", "ROTATE_TRANS R11", "ROTATE_TRANS R12",
        "ROTATE_TRANS R20", "ROTATE_TRANS R21", "ROTATE_TRANS R22",
        "ROTATE_TRANS TX", "ROTATE_TRANS TY", "ROTATE_TRANS TZ"
    ]

    if all(key in record for key in required_keys):
        # 组织数据
        K1 = [[record["Left Fx"], 0, record["Left Cx"]],
              [0, record["Left Fy"], record["Left Cy"]],
              [0, 0, 1]]
        D1 = [record["Left K1"], record["Left K2"], record["Left K3"], record["Left K4"]]
        K2 = [[record["Right Fx"], 0, record["Right Cx"]],
              [0, record["Right Fy"], record["Right Cy"]],
              [0, 0, 1]]
        D2 = [record["Right K1"], record["Right K2"], record["Right K3"], record["Right K4"]]
        R = [[record["ROTATE_TRANS R00"], record["ROTATE_TRANS R01"], record["ROTATE_TRANS R02"]],
             [record["ROTATE_TRANS R10"], record["ROTATE_TRANS R11"], record["ROTATE_TRANS R12"]],
             [record["ROTATE_TRANS R20"], record["ROTATE_TRANS R21"], record["ROTATE_TRANS R22"]]]
        T = [[record["ROTATE_TRANS TX"]],
             [record["ROTATE_TRANS TY"]],
             [record["ROTATE_TRANS TZ"]]]

        # 转换为 OpenCV 可读格式
        opencv_data = {
            "M1": {"rows": 3, "cols": 3, "dt": "d", "data": [item for sublist in K1 for item in sublist]},
            "D1": {"rows": 1, "cols": 4, "dt": "d", "data": D1},
            "M2": {"rows": 3, "cols": 3, "dt": "d", "data": [item for sublist in K2 for item in sublist]},
            "D2": {"rows": 1, "cols": 4, "dt": "d", "data": D2},
            "R": {"rows": 3, "cols": 3, "dt": "d", "data": [item for sublist in R for item in sublist]},
            "T": {"rows": 3, "cols": 1, "dt": "d", "data": [item for sublist in T for item in sublist]},
            "lens": str(record.get("lens", "nan")),
            "SN": str(record.get("SN", "unknown_SN")),
            "TestResult": str(record.get("TestResult", "UNKNOWN")),
            "DateTime": str(record.get("DateTime", "unknown_date"))
        }

        # 确保SN和DateTime是字符串
        file_sn = str(record.get("SN", "unknown_SN")).replace("/", "_")
        file_datetime = str(record.get("DateTime", "unknown_date")).replace(" ", "_").replace(":", "_")
        output_file = os.path.join(root_dir, f'{file_sn}_{file_datetime}.yaml')

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("%YAML:1.0\n")
            file.write("---\n")
            for key, value in opencv_data.items():
                if isinstance(value, dict) and "data" in value:
                    rows = value["rows"]
                    cols = value["cols"]
                    dt   = value["dt"]
                    data_str = ', '.join(map(str, value["data"]))

                    file.write(f"{key}: !!opencv-matrix\n")
                    file.write(f"   rows: {rows}\n")              # 3 spaces
                    file.write(f"   cols: {cols}\n")              # 3 spaces
                    file.write(f"   dt: {dt}\n")                  # 3 spaces
                    file.write(f"   data: [ {data_str} ]\n")      # 3 spaces
                    file.write("\n")  # optional extra newline
                else:
                    # 如果是字符串，就用引号包裹起来
                    if isinstance(value, str):
                        file.write(f"{key}: \"{value}\"\n")
                    else:
                        file.write(f"{key}: {value}\n")

        print(f"Saved record {i+1} to {output_file}")

        fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_READ)

        if fs.isOpened():
            print(f"Loading calibration parameters from: {output_file}")
        else:
            print(f"Failed to load calibration parameters from: {output_file}")
        fs.release()
    else:
        print(f"Skipping record {i+1}: Missing required data.")


