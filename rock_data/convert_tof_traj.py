import argparse
import math
import subprocess

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


def is_valid_value(x, min_value=-1e6, max_value=1e6):
    """
    Check if a value is within a specified range.

    Parameters:
        x (float): The value to check.
        min_value (float): Minimum allowed value.
        max_value (float): Maximum allowed value.

    Returns:
        bool: True if the value is valid, False otherwise.
    """
    return min_value <= x <= max_value


def convert_vslam_to_tum(input_file_path, output_file_path, keyword="estimate"):
    """
    Converts VSLAM log data to TUM trajectory format with error handling and anomaly filtering.

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
                # Parse timestamp, x, y, theta
                timestamp = float(parts[0].strip('#'))
                x = float(parts[2])
                y = float(parts[3])
                theta = float(parts[4])

                # Validate values
                if is_valid_value(x) and is_valid_value(y) and is_valid_value(theta, -math.pi, math.pi):
                    vslam_data.append((timestamp, x, y, theta))
                else:
                    print(f"Skipping line with out-of-range values: {line.strip()}")
            except (ValueError, IndexError):
                print(f"Error parsing line: {line.strip()}")

    # Sort data by timestamp
    vslam_data.sort(key=lambda x: x[0])

    # Convert to TUM format
    tum_data = []
    for timestamp, x, y, theta in vslam_data:
        try:
            qw = math.cos(theta / 2)
            qz = math.sin(theta / 2)
            tum_data.append(f"{timestamp} {x} {y} 0 0 0 {qz} {qw}\n")
        except Exception as e:
            print(f"Error converting data to TUM format: {e}")

    # Write the TUM-formatted data to the output file
    try:
        with open(output_file_path, 'w') as file:
            file.writelines(tum_data)
        print(f"TUM data successfully written to {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def plot_tum_trajectory(tum_file_path, output_image_path):
    """
    Use evo to plot the XY trajectory from a TUM file and save the plot as an image.

    Parameters:
        tum_file_path (str): Path to the TUM trajectory file.
        output_image_path (str): Path to save the trajectory image.

    Returns:
        None
    """
    try:
        # Construct the evo command
        command = [
            "evo_traj",
            "tum",
            tum_file_path,
            "--plot_mode", "xy",
            "--save_plot", output_image_path
        ]

        # Run the command
        subprocess.run(command, check=True)
        print(f"Trajectory plot saved to {output_image_path}")
    except FileNotFoundError:
        print("Error: evo command not found. Ensure evo is installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing evo command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    args = parse_args()
    convert_vslam_to_tum(args.input_file, args.output_file, args.keyword)

    # Plot the trajectory
    output_image_file = args.output_file.replace(".txt", ".png")
    plot_tum_trajectory(args.output_file, output_image_file)


if __name__ == "__main__":
    main()
