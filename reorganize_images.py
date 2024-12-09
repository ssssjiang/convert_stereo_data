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


def reorganize_images(folder_path):
    # 获取所有图像路径
    images = get_all_images(folder_path)
    parent_folder = os.path.dirname(folder_path)  # 获取 folder_path 的上级目录

    # Reorganize images (copy camera/camera0/xxx.png -> camera_map/xxx_0.png, camera/camera1/xxx.png -> camera_map/xxx_1.png)
    camera_map_folder = os.path.join(parent_folder, "camera_map")
    if not os.path.exists(camera_map_folder):
        os.makedirs(camera_map_folder)
    for image in images:
        camera = "0" if "camera0" in image else "1"
        image_name = os.path.basename(image)
        new_image_name = image_name.split('.')[0] + f"_{camera}.png"
        new_image_path = os.path.join(camera_map_folder, new_image_name)
        os.system(f"cp {image} {new_image_path}")

    print(f"图像已整理到 {camera_map_folder} 文件夹中")


if __name__ == "__main__":
    folder_path = input("请输入要搜索的文件夹路径: ").strip()

    reorganize_images(folder_path)
