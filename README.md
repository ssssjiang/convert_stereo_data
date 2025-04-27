# Stereo Camera Parameter Conversion Tool

This tool converts stereo camera calibration parameters from OpenCV YAML format to the sensor.yaml format used by the robot.

## Features

- Reads camera matrices (M1, D1, M2, D2) from OpenCV YAML format
- Divides intrinsic matrix values by 2 as required
- Uses R and T from source file to create transformation matrix for camera0
- Uses identity matrix for camera1's transformation
- Preserves other settings in the target YAML file

## Requirements

- Python 3.6+
- OpenCV (cv2)
- PyYAML

## Usage

### Basic Conversion

To convert camera parameters from an OpenCV YAML file to a sensor YAML file:

```bash
./convert_stereo_yaml.py --input /path/to/opencv_stereo.yml --output /path/to/sensor.yaml
```

Default paths are:
- Input: `/home/roborock/下载/37_stereo.yml`
- Output: `/home/roborock/下载/sensor_sy_19.yaml`

### Verifying Conversion

To verify that the conversion has been done correctly:

```bash
./verify_conversion.py
```

This script will:
1. Create a backup of the original sensor YAML file
2. Load and display the original parameters
3. Run the conversion script
4. Load and display the updated parameters
5. Verify that the changes match the expected values

## File Format Details

### Input OpenCV YAML Format

The input file should be in OpenCV YAML format with these nodes:
- `M1`: 3x3 camera matrix for the first camera
- `D1`: Distortion coefficients for the first camera
- `M2`: 3x3 camera matrix for the second camera
- `D2`: Distortion coefficients for the second camera
- `R`: 3x3 rotation matrix between the two cameras
- `T`: 3x1 translation vector between the two cameras

### Output Sensor YAML Format

The output file will update these parameters in the sensor YAML:
- `camera0.camera.intrinsics.data`: Camera matrix parameters for camera0 (divided by 2)
- `camera0.camera.distortion.data`: First 8 distortion coefficients for camera0
- `camera0.T_B_C.data`: 4x4 transformation matrix from R and T
- `camera1.camera.intrinsics.data`: Camera matrix parameters for camera1 (divided by 2)
- `camera1.camera.distortion.data`: First 8 distortion coefficients for camera1
- `camera1.T_B_C.data`: Identity matrix (4x4)

## Notes

- The script preserves all other settings in the target sensor YAML file
- A backup of the original sensor YAML file is created with `.bak` extension
- Only the first 8 distortion coefficients are used, as the target format supports 8 parameters 