import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_rtk_and_wheel_data(rtk_file, wheel_file, imu_file):
    """
    Reads RTK, wheel encoder, and IMU data, aligns timestamps, and creates an interactive plot.

    Args:
        rtk_file (str): Path to the RTK pose data file.
        wheel_file (str): Path to the wheel encoder data file.
        imu_file (str): Path to the IMU data file.
    """
    # Read RTK data
    rtk_df = pd.read_csv(
        rtk_file,
        sep='\\s+',
        skiprows=1,
        header=None,
        names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'num_sats', 'sol_status', 'vel_status']
    )

    # Read wheel encoder data
    wheel_df = pd.read_csv(
        wheel_file,
        skiprows=1,
        header=None,
        names=['timestamp', 'left_ticks', 'right_ticks']
    )

    # Convert wheel encoder timestamp from milliseconds to seconds
    wheel_df['timestamp'] = wheel_df['timestamp'] / 1000.0

    # Read IMU data
    imu_df = pd.read_csv(
        imu_file,
        skiprows=1,
        header=None,
        names=['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    )

    # Convert IMU timestamp from milliseconds to seconds
    imu_df['timestamp'] = imu_df['timestamp'] / 1000.0

    # Create figure with multiple y-axes
    fig = go.Figure()

    # Add RTK position traces to the first y-axis
    fig.add_trace(go.Scatter(x=rtk_df['timestamp'], y=rtk_df['tx'], name='RTK tx', yaxis='y1'))
    fig.add_trace(go.Scatter(x=rtk_df['timestamp'], y=rtk_df['ty'], name='RTK ty', yaxis='y1'))
    fig.add_trace(go.Scatter(x=rtk_df['timestamp'], y=rtk_df['tz'], name='RTK tz', yaxis='y1'))

    # Add wheel encoder traces to the second y-axis
    fig.add_trace(go.Scatter(x=wheel_df['timestamp'], y=wheel_df['left_ticks'], name='Left Ticks', yaxis='y2'))
    fig.add_trace(go.Scatter(x=wheel_df['timestamp'], y=wheel_df['right_ticks'], name='Right Ticks', yaxis='y2'))

    # Add IMU Z-axis acceleration trace to the third y-axis
    fig.add_trace(go.Scatter(x=imu_df['timestamp'], y=imu_df['accel_z'], name='IMU Accel Z', yaxis='y3'))

    # Update layout for multiple y-axes
    fig.update_layout(
        title_text="RTK, Wheel, and IMU Data vs. Time",
        hovermode='x unified',

        xaxis=dict(
            title="Time (s)",
            domain=[0, 0.88],  # Make space for y-axes on the right
            spikesnap="cursor",
            spikemode="across",
            spikedash="dot"
        ),

        # Primary Y-axis for RTK
        yaxis=dict(
            title="Position (m)"
        ),

        # Secondary Y-axis for Wheel Encoder
        yaxis2=dict(
            title="Ticks",
            overlaying="y",
            side="right"
        ),

        # Tertiary Y-axis for IMU
        yaxis3=dict(
            title="Accel Z (m/s^2)",
            overlaying="y",
            side="right",
            position=0.95,
            anchor="x"
        )
    )

    # fig.show()
    output_filename = "rtk_and_wheel_plot.html"
    fig.write_html(output_filename)
    print(f"Plot saved to {output_filename}. Please open this file in a web browser to see the interactive plot.")

if __name__ == '__main__':
    # NOTE: Please update these paths to your local file paths
    rtk_data_file = '/home/roborock/下载/20hz_RTK采集数据/022c103c0090308121a2_022c103c0090308121a2-2025.6.17.10.48.0/000364.20250617024709975_11111111111112_2025061300DEV_processed/rtk_pose_full.txt'
    wheel_encoder_file = '/home/roborock/下载/20hz_RTK采集数据/022c103c0090308121a2_022c103c0090308121a2-2025.6.17.10.48.0/000364.20250617024709975_11111111111112_2025061300DEV_processed/wheel_encoder.csv'
    imu_data_file = '/home/roborock/下载/20hz_RTK采集数据/022c103c0090308121a2_022c103c0090308121a2-2025.6.17.10.48.0/000364.20250617024709975_11111111111112_2025061300DEV_processed/imu.csv'
    plot_rtk_and_wheel_data(rtk_data_file, wheel_encoder_file, imu_data_file) 