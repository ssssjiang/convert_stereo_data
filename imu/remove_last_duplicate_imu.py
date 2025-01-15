#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import sys

def remove_last_duplicate_imu(input_bag_path, output_bag_path, imu_topic='/imu'):
    """
    移除指定ROS bag文件中/imu主题的最后一个重复消息。

    参数：
    - input_bag_path: 原始bag文件路径
    - output_bag_path: 新bag文件保存路径
    - imu_topic: 要处理的主题，默认为'/imu'
    """
    try:
        with rosbag.Bag(output_bag_path, 'w') as outbag:
            last_stamp = None
            duplicate_found = False
            # 读取所有消息
            for topic, msg, t in rosbag.Bag(input_bag_path).read_messages():
                if topic == imu_topic:
                    current_stamp = msg.header.stamp.to_sec()
                    if last_stamp is not None and current_stamp == last_stamp and not duplicate_found:
                        # 发现重复且尚未跳过
                        duplicate_found = True
                        print("跳过重复的IMU消息，时间戳: {:.6f} 秒".format(current_stamp))
                        continue  # 跳过当前重复消息
                    last_stamp = current_stamp
                outbag.write(topic, msg, t)
        print("成功创建新的bag文件（无重复IMU消息）：{}".format(output_bag_path))
    except Exception as e:
        print("处理bag文件时出错: {}".format(e))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python remove_last_duplicate_imu.py <输入bag文件> <输出bag文件>")
        print("示例: python remove_last_duplicate_imu.py a_newvio.bag a_newvio_clean.bag")
        sys.exit(1)
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    remove_last_duplicate_imu(input_bag, output_bag)
