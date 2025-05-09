#!/usr/bin/env python3
import numpy as np
import yaml
import re

def read_yaml_safely(file_path):
    """Read YAML file safely, handling syntax issues."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # 处理OKVIS风格的YAML文件，它们通常以"%YAML:1.0"开头
    # 将其替换为标准YAML解析器可以理解的格式
    if content.strip().startswith("%YAML:"):
        # 去除YAML版本标记行
        content = re.sub(r'^%YAML:[0-9.]+\s*$', '', content, flags=re.MULTILINE)
    
    # Fix problematic data arrays with leading commas
    content = re.sub(r'data: \[\s*,', 'data: [', content)
    
    # Parse the fixed content
    return yaml.safe_load(content)

def map_distortion_model(distortion_model):
    """Map distortion model names from camchain format to sensor.yaml format."""
    mapping = {
        'radtan': 'radial-tangential',
        'radtan8': 'radial-tangential8',
        'equidistant': 'kannala-brandt',
        'none': 'double-sphere',
    }
    return mapping.get(distortion_model, distortion_model)

def limit_distortion_params(distortion_params, max_dim=8):
    """Limit distortion parameters to a maximum dimension.
    
    Args:
        distortion_params: List or numpy array of distortion parameters
        max_dim: Maximum number of dimensions allowed (default: 8)
        
    Returns:
        List of distortion parameters limited to max_dim
    """
    # Convert to list if it's not already
    if isinstance(distortion_params, np.ndarray):
        distortion_list = distortion_params.flatten().tolist()
    else:
        distortion_list = list(distortion_params)
    
    # Limit to max_dim parameters
    if len(distortion_list) > max_dim:
        limited_params = distortion_list[:max_dim]
        print(f"Warning: Truncated distortion parameters from {len(distortion_list)} to {max_dim} dimensions")
        return limited_params
    
    return distortion_list

def create_T_B_C(R, T):
    """Create a 4x4 transformation matrix from R and T."""
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = R
    T_B_C[:3, 3] = T.reshape(3)
    return T_B_C

def matrix_to_yaml_list(matrix):
    """Convert a 4x4 transformation matrix to a flattened list for YAML."""
    return [
        float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3]),
        float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3]),
        float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3]),
        float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3])
    ]

def process_camera_resolution(camera, resolution, divide_intrinsics=False):
    """Process and set camera resolution, with optional downscaling.
    
    Args:
        camera: Camera dict from sensor.yaml
        resolution: Original resolution [width, height]
        divide_intrinsics: Whether to divide resolution by 2
    """
    if divide_intrinsics:
        camera['camera']['image_width'] = resolution[0] // 2
        camera['camera']['image_height'] = resolution[1] // 2
        print(f"Camera resolution set to {camera['camera']['image_width']}x{camera['camera']['image_height']} (divided by 2)")
    else:
        camera['camera']['image_width'] = resolution[0]
        camera['camera']['image_height'] = resolution[1]
        print(f"Camera resolution set to {camera['camera']['image_width']}x{camera['camera']['image_height']}")
    
    return camera

def scale_template_resolution(camera_dict, divide_intrinsics=False):
    """Scale resolution from template if divide_intrinsics is True.
    
    Args:
        camera_dict: Camera dictionary from template
        divide_intrinsics: Whether to divide resolution by 2
    """
    if divide_intrinsics and 'camera' in camera_dict:
        if 'image_width' in camera_dict['camera'] and 'image_height' in camera_dict['camera']:
            width = camera_dict['camera']['image_width']
            height = camera_dict['camera']['image_height']
            camera_dict['camera']['image_width'] = width // 2
            camera_dict['camera']['image_height'] = height // 2
            print(f"Camera resolution scaled down to {width//2}x{height//2}")
    
    return camera_dict

def load_camchain_data(camchain_path):
    """Load and validate camchain data from a YAML file.
    
    Args:
        camchain_path: Path to the camchain YAML file
        
    Returns:
        A dictionary containing the camchain data
        
    Raises:
        ValueError: If the file does not contain expected cam0 and cam1 entries
    """
    with open(camchain_path, 'r') as file:
        camchain_data = yaml.safe_load(file)
    
    # Validate the data structure
    if 'cam0' not in camchain_data or 'cam1' not in camchain_data:
        raise ValueError("Input YAML file does not contain expected cam0 and cam1 entries.")
    
    return camchain_data

def extract_camera_params(cam_data, divide_intrinsics=False):
    """Extract and process camera parameters from camchain data.
    
    Args:
        cam_data: Camera data dictionary from camchain
        divide_intrinsics: Whether to divide intrinsics by 2
        
    Returns:
        Tuple of (processed_intrinsics, distortion_model_type, distortion_params)
    """
    # Extract intrinsics
    intrinsics = np.array(cam_data['intrinsics'])
    
    # Process intrinsics based on divide_intrinsics flag
    if divide_intrinsics:
        intrinsics_processed = intrinsics / 2
    else:
        intrinsics_processed = intrinsics.copy()
    
    # Get distortion model and coefficients
    distortion_model = cam_data.get('distortion_model', 'none')
    distortion_type = map_distortion_model(distortion_model)
    
    # Get distortion coefficients and limit to 8 dimensions
    distortion_coeffs = limit_distortion_params(np.array(cam_data.get('distortion_coeffs', [])))
    
    return intrinsics_processed, distortion_type, distortion_coeffs, distortion_model

def update_camera_params(camera, intrinsics, distortion_type, distortion_params):
    """Update camera parameters in sensor.yaml.
    
    Args:
        camera: Camera dictionary from sensor.yaml
        intrinsics: Processed intrinsics array
        distortion_type: Mapped distortion model type
        distortion_params: Processed distortion parameters
        
    Returns:
        Updated camera dictionary
    """
    # Update distortion parameters
    camera['camera']['distortion']['data'] = distortion_params
    camera['camera']['distortion']['cols'] = 1
    camera['camera']['distortion']['rows'] = len(distortion_params)
    camera['camera']['distortion_type'] = distortion_type
    
    # Update intrinsics
    camera['camera']['intrinsics']['data'] = [
        float(intrinsics[0]),  # fx
        float(intrinsics[1]),  # fy
        float(intrinsics[2]),  # cx
        float(intrinsics[3])   # cy
    ]
    
    return camera

def process_sensor_cameras(sensor_data, cam0_data, cam1_data, swap_cameras, divide_intrinsics, 
                           T_B_C0=None, T_B_C1=None, identity_matrix=None, update_resolution=True):
    """Process and update camera parameters in sensor.yaml.
    
    Args:
        sensor_data: Sensor YAML data dictionary
        cam0_data: Cam0 data from camchain
        cam1_data: Cam1 data from camchain
        swap_cameras: Whether to swap cameras
        divide_intrinsics: Whether to divide intrinsics
        T_B_C0: Transformation matrix for camera0 (optional)
        T_B_C1: Transformation matrix for camera1 (optional)
        identity_matrix: Identity matrix (optional)
        update_resolution: Whether to update resolution (default: True)
        
    Returns:
        Updated sensor_data dictionary
    """
    if 'cameras' not in sensor_data['sensor'] or len(sensor_data['sensor']['cameras']) < 1:
        return sensor_data
    
    # Get camera objects
    camera0 = sensor_data['sensor']['cameras'][0]
    camera1 = sensor_data['sensor']['cameras'][1] if len(sensor_data['sensor']['cameras']) > 1 else None
    
    if camera1 is None and swap_cameras:
        print("Warning: Cannot swap cameras as there is only one camera in the sensor.yaml file.")
        swap_cameras = False
    
    # Extract camera parameters
    cam0_intrinsics, cam0_distortion_type, cam0_distortion, cam0_distortion_model = extract_camera_params(cam0_data, divide_intrinsics)
    cam1_intrinsics, cam1_distortion_type, cam1_distortion, cam1_distortion_model = extract_camera_params(cam1_data, divide_intrinsics)
    
    # Determine whether we have resolutions from camchain
    has_cam0_resolution = 'resolution' in cam0_data
    has_cam1_resolution = 'resolution' in cam1_data
    
    # Transformation matrices to use (if provided)
    T_B_C_for_camera0 = None
    T_B_C_for_camera1 = None
    
    if swap_cameras and camera1 is not None:
        # Update camera parameters with swapped data
        camera0 = update_camera_params(camera0, cam1_intrinsics, cam1_distortion_type, cam1_distortion)
        camera1 = update_camera_params(camera1, cam0_intrinsics, cam0_distortion_type, cam0_distortion)
        
        # 根据 update_resolution 参数决定是否更新分辨率
        if update_resolution:
            # Set resolution if available from camchain
            if has_cam1_resolution:
                camera0 = process_camera_resolution(camera0, cam1_data['resolution'], divide_intrinsics)
            elif divide_intrinsics and 'camera' in camera0 and 'image_width' in camera0['camera'] and 'image_height' in camera0['camera']:
                # Only scale if no camchain resolution is available
                width = camera0['camera']['image_width']
                height = camera0['camera']['image_height']
                camera0['camera']['image_width'] = width // 2
                camera0['camera']['image_height'] = height // 2
                print(f"Camera0 resolution scaled from template to {width//2}x{height//2}")
            
            if has_cam0_resolution:
                camera1 = process_camera_resolution(camera1, cam0_data['resolution'], divide_intrinsics)
            elif divide_intrinsics and camera1 is not None and 'camera' in camera1 and 'image_width' in camera1['camera'] and 'image_height' in camera1['camera']:
                # Only scale if no camchain resolution is available
                width = camera1['camera']['image_width']
                height = camera1['camera']['image_height']
                camera1['camera']['image_width'] = width // 2
                camera1['camera']['image_height'] = height // 2
                print(f"Camera1 resolution scaled from template to {width//2}x{height//2}")
        else:
            print("保持模板中的图像分辨率不变")
        
        # Set transformation matrices if provided
        if T_B_C0 is not None and T_B_C1 is not None:
            T_B_C_for_camera0 = T_B_C1
            T_B_C_for_camera1 = T_B_C0
        elif identity_matrix is not None:
            # Special case for camchain format
            T_B_C_for_camera0 = identity_matrix
            
        print("Swapping cameras: camera0 gets cam1 data, camera1 gets cam0 data")
    else:
        # Update camera parameters without swapping
        camera0 = update_camera_params(camera0, cam0_intrinsics, cam0_distortion_type, cam0_distortion)
        if camera1 is not None:
            camera1 = update_camera_params(camera1, cam1_intrinsics, cam1_distortion_type, cam1_distortion)
        
        # 根据 update_resolution 参数决定是否更新分辨率
        if update_resolution:
            # Set resolution if available from camchain
            if has_cam0_resolution:
                camera0 = process_camera_resolution(camera0, cam0_data['resolution'], divide_intrinsics)
            elif divide_intrinsics and 'camera' in camera0 and 'image_width' in camera0['camera'] and 'image_height' in camera0['camera']:
                # Only scale if no camchain resolution is available
                width = camera0['camera']['image_width']
                height = camera0['camera']['image_height']
                camera0['camera']['image_width'] = width // 2
                camera0['camera']['image_height'] = height // 2
                print(f"Camera0 resolution scaled from template to {width//2}x{height//2}")
            
            if has_cam1_resolution and camera1 is not None:
                camera1 = process_camera_resolution(camera1, cam1_data['resolution'], divide_intrinsics)
            elif divide_intrinsics and camera1 is not None and 'camera' in camera1 and 'image_width' in camera1['camera'] and 'image_height' in camera1['camera']:
                # Only scale if no camchain resolution is available
                width = camera1['camera']['image_width']
                height = camera1['camera']['image_height']
                camera1['camera']['image_width'] = width // 2
                camera1['camera']['image_height'] = height // 2
                print(f"Camera1 resolution scaled from template to {width//2}x{height//2}")
        else:
            print("保持模板中的图像分辨率不变")
        
        # Set transformation matrices if provided
        if T_B_C0 is not None and T_B_C1 is not None:
            T_B_C_for_camera0 = T_B_C0
            T_B_C_for_camera1 = T_B_C1
        
        print("Not swapping cameras: camera0 gets cam0 data, camera1 gets cam1 data")
    
    # Update transformation matrices if provided
    if T_B_C_for_camera0 is not None:
        camera0['T_B_C']['data'] = matrix_to_yaml_list(T_B_C_for_camera0)
    
    if T_B_C_for_camera1 is not None and camera1 is not None:
        camera1['T_B_C']['data'] = matrix_to_yaml_list(T_B_C_for_camera1)
    
    # Return updated distortion model information for logging
    model_info = {
        'cam0_model': cam0_distortion_model,
        'cam0_type': cam0_distortion_type,
        'cam1_model': cam1_distortion_model,
        'cam1_type': cam1_distortion_type
    }
    
    return sensor_data, model_info 

def process_okvis_resolution(resolution, divide_intrinsics=False):
    """Process camera resolution for OKVIS format.
    
    Args:
        resolution: Original resolution [width, height]
        divide_intrinsics: Whether to divide resolution by 2
        
    Returns:
        Processed resolution as a list [height, width]
    """
    if divide_intrinsics:
        # OKVIS格式的image_dimension是[height, width]
        processed_resolution = [resolution[1] // 2, resolution[0] // 2]
        print(f"OKVIS camera resolution set to {processed_resolution[0]}x{processed_resolution[1]} (divided by 2)")
    else:
        # OKVIS格式的image_dimension是[height, width]
        processed_resolution = [resolution[1], resolution[0]]
        print(f"OKVIS camera resolution set to {processed_resolution[0]}x{processed_resolution[1]}")
    
    return processed_resolution 