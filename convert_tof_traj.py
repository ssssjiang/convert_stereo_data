import argparse
import math

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert VSLAM log data to TUM trajectory format.")
    parser.add_argument('input_file', type=str, help="Path to the input VSLAM log file.")
    parser.add_argument('output_file', type=str, help="Path to save the converted TUM trajectory file.")
    parser.add_argument('--keyword', type=str, default="estimate", help="Keyword to filter relevant data lines.")
    return parser.parse_args()


def convert_vslam_to_tum(input_file_path, output_file_path, keyword="estimate"):
    """
    Converts VSLAM log data to TUM trajectory format.

    Parameters:
        input_file_path (str): Path to the input VSLAM log file.
        output_file_path (str): Path to save the converted TUM trajectory file.
        keyword (str): Keyword to filter lines containing relevant data (default is "estimate").

    Returns:
        None
    """
    # Read the input file
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract data based on the keyword
    vslam_data = []
    for line in lines:
        if keyword in line:
            parts = line.split()
            try:
                timestamp = float(parts[0].strip('#'))
                x = float(parts[2])
                y = float(parts[3])
                theta = float(parts[4])
                vslam_data.append((timestamp, x, y, theta))
            except (ValueError, IndexError):
                print(f"Error parsing line: {line.strip()}")

    # Convert to TUM format
    tum_data = []
    for timestamp, x, y, theta in vslam_data:
        qw = math.cos(theta / 2)
        qz = math.sin(theta / 2)
        tum_data.append(f"{timestamp} {x} {y} 0 0 0 {qz} {qw}\n")

    # Write the TUM-formatted data to the output file
    try:
        with open(output_file_path, 'w') as file:
            file.writelines(tum_data)
        print(f"TUM data successfully written to {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def main():
    args = parse_args()
    convert_vslam_to_tum(args.input_file, args.output_file, args.keyword)
