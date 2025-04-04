import fnmatch
import os
import shutil
import subprocess
import cv2
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process log files and images.')
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder')
    parser.add_argument('--image_width', type=int, default=800, help='Image width')
    parser.add_argument('--image_height', type=int, default=600, help='Image height')
    parser.add_argument('--NID', type=str, default='941bd09bda0d6d57', help='NID')
    parser.add_argument("--L", type=str, default='camera0', help="Left camera directory name")
    parser.add_argument('--R', type=str, default='camera1', help='Right camera directory name')
    parser.add_argument(
        '--steps',
        type=str,
        nargs='*',
        default=[],
        help=(
            'Steps to execute:\n'
            '  step1: Process process_mt_log.\n'
            '  step2: Handle Bin47ToText and log file processing.\n'
            '  step3: Perform cleanup of intermediate files.\n'
            '  step4: Parse YUV images to PNG format.\n'
            '  step5: Reorganize images into structured directories.\n'
            '  step6: Check timestamp gaps between images.'
        )
    )
    return parser.parse_args()



def copy_files_from_mnt(mnt_folder, dest_folder):
    """
    Copy all files from the mounted folder to the destination folder, skipping existing files.
    """
    for root, dirs, files in os.walk(mnt_folder):
        for filename in files:
            src_file = os.path.join(root, filename)
            dest_file = os.path.join(dest_folder, filename)
            copy_file_with_check(src_file, dest_file)


def copy_file_with_check(src_file, dest_file):
    """
    Check if the destination file exists, and if not, copy the file.
    """
    if not os.path.exists(dest_file):
        shutil.copy(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")
    else:
        print(f"File already exists: {dest_file}. Skipping.")


def execute_command(command, cwd, done_file):
    if not os.path.exists(done_file):
        try:
            result = subprocess.run(command, shell=True, cwd=cwd, check=True)
            print(f"Command executed successfully: {command}")
            with open(done_file, 'w') as f:
                f.write("Process completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}. {e}")
            raise
    else:
        print(f"Command already executed: {command}. Skipping.")


def copy_log_files(dest_folder):
    """
    Copy log files matching specific patterns from the source folder to the destination folder.
    """
    log_patterns = [
        "SLAM_fprintf.*", "SLAM_normal.*", "RRLDR_binId4.*", "RRLDR_binId7.*",
        "slamloadmap.*", "user.*", "relocatesucc.*", "relocatemap.*"
    ]

    log_file_path = None
    for root, dirs, files in os.walk(dest_folder):
        for filename in files:
            for pattern in log_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    log_file_path = os.path.join(root, filename)
                    dest_log_file_path = os.path.join(dest_folder, filename)
                    copy_file_with_check(log_file_path, dest_log_file_path)

    if not log_file_path:
        raise FileNotFoundError(f"No log file matching the specified patterns was found in {dest_folder}")


def resort_log_file(dst, filename):
    """
    Resort the log file by the first column (assumed to be an integer).
    """
    output_file = os.path.join(dst, filename)
    output_file_sorted = os.path.join(dst, filename.replace('.log', '_sort.log'))

    with open(output_file, 'r') as f:
        sorted_lines = sorted(f, key=lambda line: int(line.split(' ')[0]))

    with open(output_file_sorted, 'w') as f:
        f.writelines(sorted_lines)

    os.remove(output_file)
    os.rename(output_file_sorted, output_file)
    print(f"Resorted {filename} and replaced the original.")


def parse_yuv_image(yuv_folder_path, output_folder_path, width, height):
    if not os.path.exists(yuv_folder_path):
        raise FileNotFoundError(f"YUV folder not found: {yuv_folder_path}")

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Output folder: {output_folder_path}")

    for root, dirs, files in os.walk(yuv_folder_path):
        for filename in files:
            if filename.endswith('.yuv'):
                yuv_file_path = os.path.join(root, filename)

                relative_path = os.path.relpath(root, yuv_folder_path)
                output_subfolder = os.path.join(output_folder_path, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                output_image_path = os.path.join(output_subfolder, filename.replace('.yuv', '.png'))

                if os.path.exists(output_image_path):
                    print(f"Image already converted: {output_image_path}. Skipping.")
                    continue

                try:
                    with open(yuv_file_path, 'rb') as yuv_file:
                        yuv_data = yuv_file.read()

                    yuv_image = np.frombuffer(yuv_data, dtype=np.uint8)
                    yuv_image = yuv_image.reshape((height * 3 // 2, width))

                    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
                    cv2.imwrite(output_image_path, bgr_image)
                    print(f"Converted: {yuv_file_path} -> {output_image_path}")
                except Exception as e:
                    print(f"Error processing file {yuv_file_path}: {e}")


def reorganize_images(input_folder, output_folder, L_folder, R_folder):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) >= 4:
                    camera_folder = L_folder if parts[1] == 'L' else R_folder
                    timestamp = parts[2]

                    dest_folder = os.path.join(output_folder, 'camera', camera_folder)
                    os.makedirs(dest_folder, exist_ok=True)

                    dest_file = os.path.join(dest_folder, f"{timestamp}.png")
                    src_file = os.path.join(root, filename)

                    shutil.move(src_file, dest_file)
                    print(f"Moved {src_file} to {dest_file}")


def save_logs_to_file(logs, file_path):
    """
    Save logs to a specified file.
    :param logs: List of log messages to save
    :param file_path: Path to the log file
    """
    with open(file_path, 'w') as file:
        for log in logs:
            file.write(log + '\n')


def check_timestamp_gaps(timestamps, log_file):
    """
    Check gaps between image timestamps.
    Ensure that timestamps for stereo data are correctly paired and continuous at 15Hz.
    :param timestamps: List of image timestamps
    :param log_file: Path to save the log for abnormal gaps
    """
    if os.path.exists(log_file):
        os.remove(log_file)

    timestamps.sort()
    gaps = np.diff(timestamps)
    valid_gaps = 1000 // 15  # Expected gap in milliseconds for 15Hz

    abnormal_gaps = []
    logs = []
    for i, gap in enumerate(gaps):
        if gap > valid_gaps + 34:  # Allow slight deviation for larger gaps
            message = f"Warning: Gap too large: {gap}ms between timestamps {timestamps[i]} and {timestamps[i+1]}."
            print(message)
            logs.append(message)
            abnormal_gaps.append((timestamps[i], timestamps[i + 1], gap))
        elif gap < valid_gaps - 16 and gap > 0:  # Allow slight deviation for smaller gaps
            message = f"Warning: Gap too small: {gap}ms between timestamps {timestamps[i]} and {timestamps[i+1]}."
            print(message)
            logs.append(message)
            abnormal_gaps.append((timestamps[i], timestamps[i + 1], gap))

    save_logs_to_file(logs, log_file)
    return abnormal_gaps


def validate_stereo_timestamps(timestamps, log_file):
    """
    Validate that stereo data has paired timestamps.
    :param timestamps: List of image timestamps
    :param log_file: Path to save the log for unpaired timestamps
    """
    if os.path.exists(log_file):
        os.remove(log_file)

    timestamp_counts = {}
    logs = []
    for ts in timestamps:
        if ts in timestamp_counts:
            timestamp_counts[ts] += 1
        else:
            timestamp_counts[ts] = 1

    unpaired_timestamps = [ts for ts, count in timestamp_counts.items() if count != 2]

    if unpaired_timestamps:
        message = "Warning: Unpaired timestamps detected:"
        print(message)
        logs.append(message)
        for ts in unpaired_timestamps:
            log_message = f"Timestamp {ts} has {timestamp_counts[ts]} occurrences (expected 2)."
            print(log_message)
            logs.append(log_message)
    else:
        message = "All timestamps are properly paired."
        print(message)
        logs.append(message)

    save_logs_to_file(logs, log_file)


def parse_timestamps_from_files(root_folder):
    """
    Parse timestamps from PNG files first; if none found, fallback to YUV files.
    :param root_folder: Root folder containing image data
    :return: List of parsed timestamps
    """
    timestamps = []
    png_files_found = False

    # First, look for PNG files
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith('.png'):
                png_files_found = True
                try:
                    timestamp = int(filename.split('.')[0])  # Expect format: timestamp.png
                    timestamps.append(timestamp)
                except ValueError:
                    print(f"Invalid timestamp in file name: {filename}")

    # If no PNG files are found, fallback to YUV files
    if not png_files_found:
        for root, dirs, files in os.walk(root_folder):
            for filename in files:
                if filename.endswith('.yuv'):
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        try:
                            timestamp = int(parts[2])
                            timestamps.append(timestamp)
                        except ValueError:
                            print(f"Invalid timestamp in file name: {filename}")

    return timestamps


def check_timestamps(root_folder):
    """
    Check timestamp gaps and validate stereo pairing.
    :param root_folder: Root folder containing image data
      """
    timestamps = parse_timestamps_from_files(root_folder)

    if not timestamps:
        print("No image timestamps found.")
        return

    print(f"Total images: {len(timestamps)}")

    unpaired_log_file = os.path.join(root_folder, "unpaired_timestamps.log")
    gap_log_file = os.path.join(root_folder, "timestamp_gaps.log")

    # Validate stereo timestamps
    validate_stereo_timestamps(timestamps, unpaired_log_file)

    # Check for abnormal gaps
    abnormal_gaps = check_timestamp_gaps(timestamps, gap_log_file)
    if abnormal_gaps:
        print("Abnormal timestamp gaps detected. Check the log file for details:", gap_log_file)
    else:
        print("All timestamp gaps are within the acceptable range.")


def has_yuv_files(folder_path):
    """递归检查 folder_path 及其子目录中是否存在 .yuv 文件"""
    for _, _, files in os.walk(folder_path):
        if any(file.endswith('.yuv') for file in files):
            return True
    return False


def main():
    args = parse_args()
    root_folder = args.root_folder
    image_width = args.image_width
    image_height = args.image_height
    L_foler = args.L
    R_foler = args.R
    NID = args.NID
    steps_to_execute = set(args.steps)

    mnt_folder = os.path.join(root_folder, 'mnt')
    process_mt_log_dest = os.path.join(root_folder, 'process_mt_log')
    bin47totext_dest = os.path.join(root_folder, 'Bin47ToText')

    if not steps_to_execute or 'step1' in steps_to_execute:
        # Step 1.1: Copy process_mt_log to root folder
        process_mt_log_path = os.path.join(os.path.dirname(__file__), 'process_mt_log')
        copy_file_with_check(process_mt_log_path, process_mt_log_dest)

        # Step 1.2: Copy all log from sub-dir to root folder
        copy_log_files(root_folder)

        # Step 1.3: Execute process_mt_log
        execute_command('NID=' + NID + ' ./process_mt_log', root_folder, os.path.join(root_folder, 'process_mt_log.done'))

    if not steps_to_execute or 'step2' in steps_to_execute:
        # Step 2.1: Copy Bin47ToText to root folder
        bin47totext_path = os.path.join(os.path.dirname(__file__), 'Bin47ToText')
        copy_file_with_check(bin47totext_path, bin47totext_dest)

        # Step 2.2: Copy all log from mnt to root folder
        copy_files_from_mnt(mnt_folder, root_folder)

        # Step 2.3: Execute Bin47ToText
        execute_command('./Bin47ToText RRLDR_binId4.log ./', root_folder, os.path.join(root_folder, 'Bin47ToText.done'))

        # Step 2.4: resort the RRLDR_fprintf.log
        resort_log_file(root_folder, 'RRLDR_fprintf.log')

    if not steps_to_execute or 'step3' in steps_to_execute:
        # Step 3: Cleanup
        for file_path in [process_mt_log_dest, bin47totext_dest, os.path.join(root_folder, 'RRLDR_binId4.log')]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")
        if os.path.exists(mnt_folder):
            shutil.rmtree(mnt_folder)
            print(f"Removed folder {mnt_folder}")

    if not steps_to_execute or 'step4' in steps_to_execute:
        # Step 7: Parse YUV images
        dev_folders = [
            os.path.join(root_folder, folder)
            for folder in os.listdir(root_folder)
            if folder.endswith('DEV') and os.path.isdir(os.path.join(root_folder, folder))
        ]

        # find the first folder containing .yuv files
        yuv_folder_path = None
        for folder in dev_folders:
            if has_yuv_files(folder):
                yuv_folder_path = folder
                break

        if not yuv_folder_path or not os.path.isdir(yuv_folder_path):
            raise FileNotFoundError(f"No DEV folder (has *.yuv) found in {root_folder}")

        rgb_folder_path = yuv_folder_path + '_rgb'
        parse_yuv_image(yuv_folder_path, rgb_folder_path, image_width, image_height)

    if not steps_to_execute or 'step5' in steps_to_execute:
        rgb_folder_path = next(
            (os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if folder.endswith('DEV_rgb')), None)

        if not rgb_folder_path or not os.path.isdir(rgb_folder_path):
            raise FileNotFoundError(f"No DEV_rgb folder found in {root_folder}")

        reorganize_images(rgb_folder_path, root_folder, L_foler, R_foler)

    # step 6: check timestamp gap between images
    # normal hz is 15hz, so the gap should be 66ms
    # if the gap is larger than 100ms, we should check the reason
    # if the gap is smaller than 50ms, we should check the reason
    if not steps_to_execute or 'step6' in steps_to_execute:
        check_timestamps(root_folder)


if __name__ == '__main__':
    main()
