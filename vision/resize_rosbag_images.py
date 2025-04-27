#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def resize_rosbag_images(input_bag_path, output_bag_path, scale_factor=0.5):
    """
    读取rosbag中的图像，将其resize为原来尺寸的一半，并保存到新的rosbag文件中。
    
    Args:
        input_bag_path (str): 输入rosbag文件路径
        output_bag_path (str): 输出rosbag文件路径
        scale_factor (float): 缩放因子，默认为0.5（原尺寸的一半）
    """
    if not os.path.exists(input_bag_path):
        print("错误：输入文件 {} 不存在".format(input_bag_path))
        return False
    
    if os.path.exists(output_bag_path):
        print("警告：输出文件 {} 已存在，将被覆盖".format(output_bag_path))
    
    bridge = CvBridge()
    
    # 打开输入和输出bag文件
    try:
        input_bag = rosbag.Bag(input_bag_path, 'r')
        output_bag = rosbag.Bag(output_bag_path, 'w')
        
        # 获取输入bag中的所有消息总数，用于显示进度
        total_messages = input_bag.get_message_count()
        processed_messages = 0
        
        print("开始处理，共有 {} 条消息...".format(total_messages))
        
        # 获取bag中的所有话题信息
        topics_info = input_bag.get_type_and_topic_info()
        topics = topics_info.topics
        
        # 找出所有图像话题
        image_topics = []
        for topic_name, topic_info in topics.items():
            if topic_info.msg_type == 'sensor_msgs/Image':
                image_topics.append(topic_name)
        
        print("找到图像话题: {}".format(", ".join(image_topics)))
        
        # 遍历输入bag中的所有消息
        for topic, msg, t in input_bag.read_messages():
            processed_messages += 1
            
            # 仅处理图像话题中的消息
            if topic in image_topics:
                try:
                    # 将ROS图像消息转换为OpenCV图像
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    
                    # 获取原始图像尺寸
                    height, width = cv_img.shape[:2]
                    
                    # 计算新的尺寸
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # 将图像resize为新尺寸
                    resized_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 将OpenCV图像转换回ROS图像消息
                    resized_msg = bridge.cv2_to_imgmsg(resized_img, encoding=msg.encoding)
                    
                    # 保留原始消息的header和其他元数据
                    resized_msg.header = msg.header
                    
                    # 将处理后的消息写入输出bag
                    output_bag.write(topic, resized_msg, t)
                except Exception as e:
                    print("处理图像时出错: {}".format(e))
                    # 如果出错，将原始消息写入输出bag
                    output_bag.write(topic, msg, t)
            else:
                # 对于非图像消息，直接写入输出bag
                output_bag.write(topic, msg, t)
            
            # 显示进度
            if processed_messages % 50 == 0 or processed_messages == total_messages:
                progress = (processed_messages / float(total_messages)) * 100
                print("进度: {:.1f}% ({}/{})".format(progress, processed_messages, total_messages))
        
        print("处理完成，已生成新的rosbag文件: {}".format(output_bag_path))
    
    except Exception as e:
        print("发生错误: {}".format(e))
        return False
    finally:
        # 关闭bag文件
        if 'input_bag' in locals():
            input_bag.close()
        if 'output_bag' in locals():
            output_bag.close()
    
    return True


def verify_resized_images(input_bag_path, output_bag_path, expected_scale_factor=0.5):
    """
    验证新生成的rosbag文件中的图像尺寸是否正确缩放
    
    Args:
        input_bag_path (str): 原始rosbag文件路径
        output_bag_path (str): 新生成的rosbag文件路径
        expected_scale_factor (float): 期望的缩放因子，默认为0.5
        
    Returns:
        bool: 验证结果，True表示所有图像都正确缩放，False表示存在不符合预期的图像
    """
    if not os.path.exists(input_bag_path) or not os.path.exists(output_bag_path):
        print("错误：无法进行验证，输入或输出文件不存在")
        return False
    
    bridge = CvBridge()
    try:
        input_bag = rosbag.Bag(input_bag_path, 'r')
        output_bag = rosbag.Bag(output_bag_path, 'r')
        
        # 获取bag中的所有话题信息
        input_topics_info = input_bag.get_type_and_topic_info()
        output_topics_info = output_bag.get_type_and_topic_info()
        
        # 找出所有图像话题
        input_image_topics = []
        for topic_name, topic_info in input_topics_info.topics.items():
            if topic_info.msg_type == 'sensor_msgs/Image':
                input_image_topics.append(topic_name)
                print("原始bag中找到图像话题: {} (消息数: {})".format(topic_name, topic_info.message_count))
        
        output_image_topics = []
        for topic_name, topic_info in output_topics_info.topics.items():
            if topic_info.msg_type == 'sensor_msgs/Image':
                output_image_topics.append(topic_name)
                print("新bag中找到图像话题: {} (消息数: {})".format(topic_name, topic_info.message_count))
        
        # 如果没有找到任何图像话题，直接报错
        if not input_image_topics:
            print("错误：原始bag中没有找到图像话题。请确认bag文件格式正确且包含sensor_msgs/Image类型的消息。")
            return False
        
        if not output_image_topics:
            print("错误：新bag中没有找到图像话题。处理过程可能存在问题。")
            return False
        
        # 创建一个话题到消息的映射
        input_msgs = {}
        output_msgs = {}
        
        # 统计消息数
        input_image_count = 0
        output_image_count = 0
        
        print("正在从原始bag中读取图像消息...")
        
        # 遍历原始bag中的图像消息
        for topic, msg, t in input_bag.read_messages(topics=input_image_topics):
            if topic not in input_msgs:
                input_msgs[topic] = []
            input_msgs[topic].append((msg, t))
            input_image_count += 1
            
            # 显示进度
            if input_image_count % 100 == 0:
                print("已读取 {} 个图像消息...".format(input_image_count))
        
        print("从原始bag中读取了 {} 个图像消息，分布在 {} 个话题中".format(input_image_count, len(input_msgs)))
        
        print("正在从新bag中读取图像消息...")
        
        # 遍历新bag中的图像消息
        for topic, msg, t in output_bag.read_messages(topics=output_image_topics):
            if topic not in output_msgs:
                output_msgs[topic] = []
            output_msgs[topic].append((msg, t))
            output_image_count += 1
            
            # 显示进度
            if output_image_count % 100 == 0:
                print("已读取 {} 个图像消息...".format(output_image_count))
        
        print("从新bag中读取了 {} 个图像消息，分布在 {} 个话题中".format(output_image_count, len(output_msgs)))
        
        # 如果没有找到任何图像消息，直接报错
        if input_image_count == 0:
            print("错误：原始bag中没有读取到图像消息。请确认bag文件格式正确。")
            return False
        
        if output_image_count == 0:
            print("错误：新bag中没有读取到图像消息。处理过程可能存在问题。")
            return False
        
        # 检查每个话题的消息数量是否一致
        for topic in input_msgs:
            if topic not in output_msgs:
                print("警告：输出bag中缺少话题 {}".format(topic))
                return False
            
            if len(input_msgs[topic]) != len(output_msgs[topic]):
                print("警告：话题 {} 的消息数量不一致，原始：{}，新的：{}".format(
                    topic, len(input_msgs[topic]), len(output_msgs[topic])))
                return False
        
        # 检查图像尺寸是否符合预期
        all_correct = True
        verified_count = 0
        error_count = 0
        
        print("开始验证图像尺寸...")
        
        # 为了简单起见，我们检查每个话题的前几个消息和最后几个消息
        samples_per_topic = 5
        
        for topic in input_msgs:
            print("正在验证话题 {} 的图像...".format(topic))
            topic_msgs = len(input_msgs[topic])
            
            # 如果消息数量很少，就全部检查
            if topic_msgs <= samples_per_topic * 2:
                sample_indices = range(topic_msgs)
            else:
                # 否则检查前几个和后几个
                sample_indices = list(range(samples_per_topic)) + list(range(topic_msgs - samples_per_topic, topic_msgs))
            
            for i in sample_indices:
                try:
                    # 获取原始图像和新图像
                    input_msg, input_time = input_msgs[topic][i]
                    output_msg, _ = output_msgs[topic][i]
                    
                    # 转换为OpenCV图像
                    input_img = bridge.imgmsg_to_cv2(input_msg, desired_encoding="passthrough")
                    output_img = bridge.imgmsg_to_cv2(output_msg, desired_encoding="passthrough")
                    
                    # 获取尺寸
                    input_height, input_width = input_img.shape[:2]
                    output_height, output_width = output_img.shape[:2]
                    
                    # 计算实际缩放比例
                    width_scale = output_width / float(input_width)
                    height_scale = output_height / float(input_height)
                    
                    # 设置允许的误差范围（考虑四舍五入）
                    tolerance = 0.01
                    
                    # 检查是否在预期范围内
                    if (abs(width_scale - expected_scale_factor) > tolerance or 
                        abs(height_scale - expected_scale_factor) > tolerance):
                        print("错误：话题 {} 的图像 #{} 尺寸不符合预期".format(topic, i+1))
                        print("  原始尺寸：{}x{}，新尺寸：{}x{}".format(
                            input_width, input_height, output_width, output_height))
                        print("  实际缩放比例：宽度={:.3f}，高度={:.3f}，预期={:.3f}".format(
                            width_scale, height_scale, expected_scale_factor))
                        all_correct = False
                        error_count += 1
                    else:
                        verified_count += 1
                        print("验证通过：话题 {} 的图像 #{} 尺寸符合预期".format(topic, i+1))
                        
                except Exception as e:
                    print("验证图像时出错 (话题 {}, 图像 #{}): {}".format(topic, i+1, e))
                    all_correct = False
                    error_count += 1
        
        if verified_count == 0:
            print("警告：没有成功验证任何图像！请检查代码逻辑和消息格式。")
            return False
        
        if all_correct:
            print("验证成功！所有检查的图像 ({} 个) 都正确缩放为原尺寸的 {:.1f}%".format(
                verified_count, expected_scale_factor * 100))
        else:
            print("验证完成，发现 {} 个错误，{} 个正确".format(error_count, verified_count))
        
        return all_correct
    
    except Exception as e:
        print("验证过程中发生错误: {}".format(e))
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return False
    finally:
        # 关闭bag文件
        if 'input_bag' in locals():
            input_bag.close()
        if 'output_bag' in locals():
            output_bag.close()


def main():
    if len(sys.argv) < 3:
        print("用法: python resize_rosbag_images.py <输入rosbag路径> <输出rosbag路径> [缩放因子]")
        print("缩放因子默认为0.5（原尺寸的一半）")
        return
    
    input_bag_path = sys.argv[1]
    output_bag_path = sys.argv[2]
    
    scale_factor = 0.5  # 默认缩放因子
    if len(sys.argv) > 3:
        try:
            scale_factor = float(sys.argv[3])
            if scale_factor <= 0:
                print("错误：缩放因子必须大于0")
                return
        except ValueError:
            print("错误：缩放因子必须是一个有效的浮点数")
            return
    
    # 执行图像缩放
    success = resize_rosbag_images(input_bag_path, output_bag_path, scale_factor)
    
    # 如果处理成功，进行验证
    if success:
        print("\n开始验证图像尺寸...")
        verify_resized_images(input_bag_path, output_bag_path, scale_factor)


if __name__ == "__main__":
    main() 