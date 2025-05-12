# MK2样机双目相机数据处理工具

这个工具用于处理MK2样机的双目相机标定数据，包括重命名文件夹和批量转换标定文件。

## 功能

1. 从CSV文件中读取整机编号和双目编号的对应关系
2. 将子文件夹重命名为`MK2-整机编号-双目编号`格式
3. 为每个子文件夹中的dualcamera_calibration.json文件生成对应的sensor.yaml和okvis.yaml文件
4. 生成的yaml文件命名为`MK2-整机编号-双目编号_sensor.yaml`和`MK2-整机编号-双目编号_okvis.yaml`
5. 支持生成不同格式的输出文件：仅sensor格式、仅okvis格式或同时生成两种格式
6. 生成处理汇总报告，记录所有处理的文件夹和生成的文件

## 使用方法

1. 确保您有整机编号和双目编号的对应关系CSV文件（例如`BUTCHART样机清单 - MK2样机.csv`）
2. 确保您的双目标定数据存放在一个根目录下，每个子文件夹名称为双目编号，文件夹内包含`dualcamera_calibration.json`文件
3. 如果需要生成sensor或okvis格式的YAML文件，请准备相应的模板文件

### 运行脚本

```bash
python3 process_mk2_stereo_data.py --csv "BUTCHART样机清单 - MK2样机.csv" --root "/path/to/stereo_data" [--format {sensor,okvis,all}] [--sensor_template "/path/to/sensor_template.yaml"] [--okvis_template "/path/to/okvis_template.yaml"] [--keep-original] [--report "/path/to/report.md"]
```

### 参数说明

- `--csv`：必需，MK2样机清单CSV文件路径
- `--root`：必需，包含双目编号子文件夹的根目录路径
- `--format`：可选，指定输出格式，可选值为'sensor'（默认）、'okvis'或'all'（同时输出两种格式）
- `--sensor_template`：可选，用于生成sensor.yaml的模板文件路径
- `--okvis_template`：可选，用于生成okvis.yaml的模板文件路径。如果format为'okvis'则必须提供
- `--keep-original`：可选，保留原始文件夹（默认会重命名）
- `--report`：可选，指定生成处理报告的文件路径。如果不提供，将自动生成一个带时间戳的默认报告

### 示例

```bash
# 仅生成sensor.yaml文件（默认模式）
python3 process_mk2_stereo_data.py --csv "下载/BUTCHART样机清单 - MK2样机.csv" --root "datasets/roborock/stereo/mower_mk2/new_mk2" --sensor_template "templates/sensor_template.yaml"

# 仅生成okvis.yaml文件
python3 process_mk2_stereo_data.py --csv "下载/BUTCHART样机清单 - MK2样机.csv" --root "datasets/roborock/stereo/mower_mk2/new_mk2" --format okvis --okvis_template "templates/okvis_template.yaml"

# 同时生成sensor.yaml和okvis.yaml文件
python3 process_mk2_stereo_data.py --csv "下载/BUTCHART样机清单 - MK2样机.csv" --root "datasets/roborock/stereo/mower_mk2/new_mk2" --format all --sensor_template "templates/sensor_template.yaml" --okvis_template "templates/okvis_template.yaml"

# 保留原始文件夹，同时生成两种格式文件，并指定生成报告
python3 process_mk2_stereo_data.py --csv "下载/BUTCHART样机清单 - MK2样机.csv" --root "datasets/roborock/stereo/mower_mk2/new_mk2" --format all --sensor_template "templates/sensor_template.yaml" --okvis_template "templates/okvis_template.yaml" --keep-original --report "MK2处理报告.md"
```

## 报告说明

处理完成后，脚本会生成一个Markdown格式的处理报告，包含以下内容：

1. **处理统计**：已处理文件夹数量和未匹配文件夹数量
2. **已处理文件夹**：表格形式展示每个处理的文件夹，包含原文件夹名、新文件夹名和生成的文件列表
3. **未匹配文件夹**：列出所有在CSV文件中未找到对应整机编号的文件夹名称

如果未指定报告文件路径，将自动生成一个带时间戳的默认报告文件。

## 注意事项

1. CSV文件格式中，整机编号应该在第1列（列索引0），双目编号应该在第5列（列索引4）
2. 脚本会自动跳过CSV文件的标题行
3. 如果某个子文件夹的双目编号在CSV文件中没有找到对应的整机编号，该文件夹将被跳过处理，并在最后列出
4. 输出文件的命名格式为：`MK2-整机编号-双目编号_sensor.yaml`和`MK2-整机编号-双目编号_okvis.yaml`
5. 如果选择`--format okvis`，则必须提供`--okvis_template`参数
6. 如果选择`--format all`，推荐同时提供`--sensor_template`和`--okvis_template`参数
7. 脚本能够智能处理多种格式的双目编号（如完整编号和短编号）
8. 如果文件夹已经是正确格式（`MK2-整机编号-双目编号`），将不会被重命名，但仍会生成yaml文件 