# 立体相机标定数据转换工具指南

## 1. 概述

此文档介绍了两个立体相机标定数据转换工具：`convert_camchain_to_sensor.py` 和 `convert_stereo_yaml.py`。这些工具用于将不同格式的相机标定数据转换为统一的 `sensor.yaml` 格式或 OKVIS 格式，便于后续的立体视觉或 SLAM 应用。

## 2. 坐标系统和转换概念

### 2.1 坐标系统定义

- **相机坐标系（C）**：以相机光心为原点，z轴指向相机前方的坐标系
- **机体坐标系（B）**：以机器人/设备主体为参考的坐标系
- **IMU坐标系（I）**：以IMU为中心的坐标系

### 2.2 重要的变换矩阵

- **T_B_C**：从相机坐标系到机体坐标系的变换
- **T_cn_cnm1**：从第n-1个相机到第n个相机的变换（例如从右相机到左相机）
- **T_cam_imu**：从IMU坐标系到相机坐标系的变换
- **T_B_I**：从IMU坐标系到机体坐标系的变换

## 3. 转换公式和逻辑

### 3.1 相机内参转换

相机内参通常包含以下参数：
- **fx, fy**：焦距（x和y方向）
- **cx, cy**：主点坐标（光轴与图像平面的交点）

如果启用 `divide_intrinsics` 选项，则所有内参都会除以2：
```
fx_new = fx / 2
fy_new = fy / 2
cx_new = cx / 2
cy_new = cy / 2
```

这通常用于处理降采样后的图像。

### 3.2 畸变模型转换

畸变模型在不同框架下有不同命名，工具会进行映射：

| 源模型        | 目标模型（sensor.yaml） | 目标模型（OKVIS）   |
|--------------|------------------------|-------------------|
| radtan       | radial-tangential      | radialtangential  |
| radtan8      | radial-tangential8     | radialtangential8 |
| equidistant  | kannala-brandt         | equidistant       |
| none         | double-sphere          | -                 |

畸变参数数量根据模型不同而有所差异，通常会限制在8个参数以内。

### 3.3 坐标变换计算

#### 从Tbc文件和camchain计算变换矩阵

当提供Tbc文件时（基于 `use_Tbc1` 参数）：

- **如果使用Tbc0**（默认）：
  ```
  T_B_C0 = Tbc文件中读取的矩阵
  T_B_C1 = T_B_C0 @ inv(T_c1_c0)
  T_B_I = T_B_C0 @ T_cam0_imu
  ```

- **如果使用Tbc1**：
  ```
  T_B_C1 = Tbc文件中读取的矩阵
  T_B_C0 = T_B_C1 @ T_c1_c0
  T_B_I = T_B_C1 @ T_cam1_imu
  ```

#### 当未提供Tbc文件时

- **如果要交换相机**（--swap）：
  ```
  T_B_C0 = 单位矩阵
  T_B_C1 = T_c1_c0
  ```

- **如果不交换相机**：
  ```
  T_B_C0 = T_c1_c0
  T_B_C1 = 单位矩阵
  ```

#### 对于OpenCV的立体标定结果

从R（旋转矩阵）和T（平移向量）构建变换矩阵：
```
T_B_C = [ R   T ]
        [ 0   1 ]
```

## 4. convert_camchain_to_sensor.py 工具

### 4.1 功能

此工具将Kalibr工具生成的camchain-imucam.yaml文件和可选的Tbc文件转换为sensor.yaml或OKVIS格式。

### 4.2 命令行参数

```
--camchain      输入的camchain-imucam.yaml文件路径
--Tbc           输入的Tbc0或Tbc1文本文件路径
--use_Tbc1      如果设置，则将Tbc视为Tbc1而非Tbc0
--output        输出的sensor.yaml文件路径
--template      模板sensor.yaml文件路径
--divide_intrinsics 是否将内参除以2（用于半分辨率图像）
--swap          是否交换左右相机（camera0和camera1）
--format        输出格式：'sensor'或'okvis'
--okvis_template OKVIS模板YAML文件路径
```

### 4.3 主要流程

1. 读取camchain-imucam.yaml文件
2. 如果提供了Tbc文件，解析它并计算相应的变换矩阵
3. 读取模板文件（如果提供）
4. 处理相机参数（内参、畸变系数、变换矩阵）
5. 根据指定格式（sensor或okvis）生成输出文件

### 4.4 输出模式详解

工具支持两种不同的输出格式：`sensor`模式（默认）和`okvis`模式。这两种模式有着不同的参数计算和数据组织方式。

#### 4.4.1 sensor模式

sensor模式生成的是用于RoboRock内部SLAM系统的标准sensor.yaml文件。

**数据结构示例：**

```yaml
sensor:
  cameras:
    - camera:
        intrinsics:
          cols: 1
          rows: 4
          data: [fx, fy, cx, cy]
        distortion:
          cols: 1
          rows: 8
          data: [k1, k2, p1, p2, k3, k4, k5, k6]
        distortion_type: radial-tangential8
        image_width: 1280
        image_height: 720
      T_B_C:
        cols: 4
        rows: 4
        data: [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3, 0, 0, 0, 1]
```

**参数计算规则：**

1. **内参处理**:
   - 直接从camchain文件中提取内参参数[fx, fy, cx, cy]
   - 如果`divide_intrinsics=True`，则所有内参会除以2

2. **畸变模型转换**:
   - 使用`map_distortion_model`函数将Kalibr的畸变模型映射到sensor.yaml格式
   - 最多限制在8个参数以内，使用`limit_distortion_params`函数处理

3. **变换矩阵处理**:
   - T_B_C根据前面列出的规则生成
   - 如果有T_B_I数据（通过Tbc文件计算得到），则也会包含在输出文件中

4. **分辨率处理**:
   - 如果camchain中包含分辨率信息，则使用该信息
   - 如果启用`divide_intrinsics`，分辨率也会相应地除以2
   - 如果没有指定分辨率，则使用模板中的值

#### 4.4.2 okvis模式

okvis模式生成的是OKVIS（Open Keyframe-Based Visual-Inertial SLAM）系统使用的配置文件格式。

**数据结构示例：**

```yaml
%YAML:1.0
cameras:
     - {T_SC:
        [r11, r12, r13, t1,
        r21, r22, r23, t2,
        r31, r32, r33, t3,
        0, 0, 0, 1],
        image_dimension: [width, height],
        distortion_coefficients: [k1, k2, p1, p2, ...],
        distortion_type: radialtangential8,
        focal_length: [fx, fy],
        principal_point: [cx, cy],
        camera_type: gray,
        slam_use: okvis}
```

**参数计算规则：**

1. **坐标变换处理**:
   - OKVIS使用T_SC（从传感器/相机到IMU的变换）而不是T_B_C
   - 计算方式：`T_SC = inv(T_cam_imu)`

2. **内参表示方式**:
   - 与sensor模式不同，OKVIS使用`focal_length`和`principal_point`两个单独的数组
   - 计算公式相同，但组织形式不同

3. **畸变模型映射**:
   - 使用`map_okvis_distortion_model`函数进行映射，与sensor模式略有不同
   - OKVIS支持`radialtangential`、`radialtangential8`和`equidistant`三种模型

4. **畸变参数处理**:
   - 根据不同模型有不同处理方式：
     * equidistant模型：通常使用4个参数
     * radialtangential模型：通常使用4个参数
     * radialtangential8模型：最多使用14个参数，不足部分用0填充

5. **T_BS处理**:
   - 对于轮式里程计：`T_BS = inv(T_B_I)`（车体到IMU的逆变换）
   - 对于IMU：提供特定的T_BS矩阵

#### 4.4.3 两种模式的区别总结

| 特性 | sensor模式 | okvis模式 |
|-----|------------|-----------|
| 内参表示 | 单一[fx, fy, cx, cy]数组 | 分为focal_length和principal_point |
| 坐标变换 | 使用T_B_C（相机到车体） | 使用T_SC（传感器/相机到IMU） |
| 畸变模型名称 | radial-tangential, kannala-brandt等 | radialtangential, equidistant等 |
| 畸变参数数量 | 最多8个 | 根据模型不同，equidistant 4个，radialtangential8最多14个 |
| 配置文件格式 | YAML标准格式 | 以%YAML:1.0开头的特殊格式 |
| IMU表示 | 使用T_B_I（IMU 到 车体） | 使用T_BS（IMU就是body，其他传感器到body） |

#### 4.4.4 使用建议

1. **针对RoboRock SLAM系统**：
   - 使用sensor模式（默认）
   - 根据实际图像分辨率决定是否使用`--divide_intrinsics`

2. **针对OKVIS系统**：
   - 使用`--format okvis`和`--okvis_template`参数
   - 确保提供正确的Tbc文件用于计算完整的变换链
   - 谨慎选择合适的畸变模型

3. **参数测试建议**：
   - 首次使用时建议输出详细日志查看变换矩阵和参数计算结果
   - 分别测试有无`--swap`选项的效果，选择合适的相机顺序

## 5. convert_stereo_yaml.py 工具

### 5.1 功能

此工具将OpenCV格式的立体标定结果或camchain-imucam.yaml文件转换为sensor.yaml格式。

### 5.2 命令行参数

```
--input         输入的标定文件路径
--output        输出的sensor.yaml文件路径
--template      模板sensor.yaml文件路径
--camchain      如果设置，则使用camchain-imucam.yaml格式
--swap          是否交换左右相机（camera0和camera1）
--no_divide_intrinsics 如果设置，则不将内参除以2
```

### 5.3 主要流程

1. 根据 `--camchain` 标志选择加载方式
2. 如果是OpenCV格式：
   - 读取相机矩阵（M1, M2）和畸变系数（D1, D2）
   - 读取旋转矩阵（R）和平移向量（T）
   - 生成变换矩阵
3. 处理相机参数（内参、畸变系数、变换矩阵）
4. 生成sensor.yaml输出文件

## 6. 常见配置示例

### 6.1 普通立体相机配置

```bash
python convert_stereo_yaml.py --input stereo_calibration.yml --output sensor.yaml --template sensor_template.yaml
```

### 6.2 将camchain转换为sensor.yaml，并应用Tbc0

```bash
python convert_camchain_to_sensor.py --camchain camchain-imucam.yaml --Tbc Tbc0.txt --output sensor.yaml --template sensor_template.yaml
```

### 6.3 将camchain转换为OKVIS格式，并交换相机

```bash
python convert_camchain_to_sensor.py --camchain camchain-imucam.yaml --Tbc Tbc1.txt --use_Tbc1 --output mower_stereo_light.yaml --format okvis --okvis_template okvis_template.yaml --swap
```

## 7. 故障排除

### 7.1 常见问题

1. **畸变模型不匹配**：如果源畸变模型与目标系统不兼容，检查日志中的模型映射信息。

2. **分辨率问题**：使用 `--divide_intrinsics` 或 `--no_divide_intrinsics` 调整相机内参和分辨率。

3. **变换矩阵问题**：
   - 检查Tbc文件的格式是否正确（3x4或4x4矩阵）
   - 确认是否使用了正确的 `--use_Tbc1` 标志

### 7.2 如何验证转换结果

可以通过以下方法验证结果：
1. 使用可视化工具查看变换矩阵表示的相机位置关系
2. 用转换后的参数进行立体匹配，检查视差图
3. 在SLAM系统中验证相机轨迹是否合理 