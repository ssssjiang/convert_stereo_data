import os


def is_image_file(filename):
    """
    判断文件是否为图像文件（根据后缀名）
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    ext = os.path.splitext(filename)[-1].lower()
    return ext in image_extensions


def get_all_images(folder_path):
    """
    递归获取文件夹下所有图像的绝对路径
    """
    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if is_image_file(file):
                images.append(os.path.abspath(os.path.join(root, file)))
    return images


def get_camera_parameters(path):
    """
    根据路径返回相应的相机参数
    """
    if "camera0" in path:
        return "OPENCV_FISHEYE 259.7568821036086 260.10819099573615 394.9905360190833 294.44467823631834 0.0008605939481375175 0.015921588486384006 -0.012233412348966891 0.0012503893360738545"
    elif "camera1" in path:
        return "OPENCV_FISHEYE 260.063551592498 259.9904115230021 400.7237754048461 300.40231457638737 -0.0025081048464266195 0.022744694807417455 -0.018000412523496625 0.0026870339959659795"
    return ""


def save_images_to_file_with_params(image_paths, output_file, folder_path, save_absolute=True):
    """
    将图像路径及相机参数保存到文件中
    :param image_paths: 图像路径列表
    :param output_file: 输出文件路径
    :param folder_path: 基准文件夹路径
    :param save_absolute: 是否保存绝对路径（True: 绝对路径, False: 相对路径）
    """
    with open(output_file, 'w') as f:
        # resort the image paths by timestamp (camerax/timestamp.png)
        image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        for path in image_paths:
            if save_absolute:
                file_path = path  # 绝对路径
            else:
                file_path = os.path.relpath(path, folder_path)  # 相对路径

            # 获取相机参数
            # camera_params = get_camera_parameters(file_path)
            camera_params = ""
            if camera_params:
                f.write(f"{file_path} {camera_params}\n")  # 保存路径和参数
            else:
                f.write(f"{file_path}\n")  # 只保存路径


if __name__ == "__main__":
    folder_path = input("请输入要搜索的文件夹路径: ").strip()
    save_type = input("请选择保存路径类型（1: 绝对路径, 2: 相对路径）: ").strip()

    # 定义输出文件路径：与 folder_path 同级目录
    parent_folder = os.path.dirname(folder_path)  # 获取 folder_path 的上级目录
    output_file = os.path.join(parent_folder, "images_list.txt")

    # 获取所有图像路径
    images = get_all_images(folder_path)

    # 判断保存路径类型
    save_absolute = save_type == "1"

    # 保存到文件
    save_images_to_file_with_params(images, output_file, folder_path, save_absolute=save_absolute)

    print(f"共找到 {len(images)} 张图像，路径及相机参数已保存到 {output_file} 中。")
