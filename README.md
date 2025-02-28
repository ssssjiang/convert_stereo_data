# IMU Data Visualization Tool

This tool is used to parse IMU sensor data and generate visualizations of 3-axis acceleration and angular velocity.

## Features

- Extract IMU data from log files
- Plot 3-axis acceleration charts (with inverted values)
- Plot 3-axis angular velocity charts
- Save charts as PNG images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your IMU data file is named `Sensor_fprintf.log` and placed in the same directory as the script
2. Run the script:

```bash
python parse_imu_data.py
```

3. The script will generate an image file named `imu_data_plot.png` and display the charts on screen

## Data Format

The script assumes IMU data is formatted as follows:
```
timestamp imu accel_x accel_y accel_z gyro_x gyro_y gyro_z quat_w quat_x quat_y quat_z
```

Example:
```
143468 imu -0.195726 2.970007 -9.200917 -0.365386 0.067112 -0.151268 0.509130 0.498711 -0.486398 -0.505462
```

## Data Processing

- All acceleration values are multiplied by -1 to invert their direction
- This inversion can help correct sensor orientation or coordinate system differences 