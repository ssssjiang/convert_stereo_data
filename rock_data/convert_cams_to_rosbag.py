#-*-coding:utf-8 -*-
import numpy as np
import cv2
import yaml
import os
import argparse
import rosbag
from sensor_msgs.msg import Image
from tqdm import tqdm
from apriltags_eth import make_default_detector

parser = argparse.ArgumentParser(description="Smart SLAM log converter.")
parser.add_argument('-i', '--input', type=str, default=os.getcwd(), help='in dir')
parser.add_argument('-m', '--merge_limit_ms', type=int, default=10)
parser.add_argument('-o', '--output', type=str, default="ros.bag")
parser.add_argument('-f', '--filter_dynamic', type=bool, default=True)
args = parser.parse_args()


class CamRecord:
    def __init__(self, time, path):
        self.time = time
        self.path = path


class CameraConfig:
    def __init__(self, img_path):
        file_names = os.listdir(img_path)
        file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

        self.img_file_list = [CamRecord(int(item.split('.')[0]), os.path.join(img_path, item)) for item in file_names if
                              item.find('png') >= 0 or item.find('jpg') >= 0]
        print('{} has {} images'.format(img_path, len(self.img_file_list)))


class MultiCam:
    def __init__(self, cam_list):
        self.cams = cam_list
        self.img_list = self.merge_list([item.img_file_list for item in cam_list])
        print('merge_result {} image pairs for {} cameras'.format(len(self.img_list), len(self.cams)))

    def merge_list(self, full_time_list):
        if (len(full_time_list) == 1):
            return [[item] for item in full_time_list[0]]
        idx_list = np.zeros((len(full_time_list)), dtype=np.int)
        max_list = np.zeros((len(full_time_list)), dtype=np.int)
        for ii in range(idx_list.shape[0]):
            max_list[ii] = len(full_time_list[ii])
        # 每轮找到一个最小的，剩下的距离最小值小于100ms则可合并，如果满合并，则成功一个元素
        # 如果不满合并，则合并成功的idx增加
        # 如果一个处理完毕，则返回
        done = False
        ret = []
        while not done:
            get_min = -1  # 任何一个idx为空，则返回-1
            min_time = 99999999999
            for ii in range(idx_list.shape[0]):
                time_img_list = full_time_list[ii]
                time_img_obj = time_img_list[idx_list[ii]]
                if min_time > time_img_obj.time:
                    min_time = time_img_obj.time
                    get_min = ii

            last_merge = []
            # try merge
            for ii in range(idx_list.shape[0]):
                time_img_list = full_time_list[ii]
                if ii == get_min:
                    last_merge.append(time_img_list[idx_list[ii]])
                    idx_list[get_min] += 1
                    continue
                while time_img_list[idx_list[ii]].time < min_time:
                    idx_list[ii] += 1
                if time_img_list[idx_list[ii]].time - args.merge_limit_ms < min_time:
                    # success merge
                    last_merge.append(time_img_list[idx_list[ii]])
                    idx_list[ii] += 1

            if (len(last_merge) == idx_list.shape[0]):
                ret.append(last_merge)

            for ii in range(idx_list.shape[0]):
                if idx_list[ii] >= max_list[ii]:
                    done = True
                    break


        return ret

    def __getitem__(self, index):
        return self.cams, self.img_list[index]

    def __len__(self):
        return len(self.img_list)


def do_job(path, time):
    print()
    cam_list = []

    path_list = os.listdir(path)

    if (len(path_list) == 1):
        cam_list.append(CameraConfig(os.path.join(path, path_list[0])))
    else:
        for ii in range(len(path_list)):
            cam_list.append(CameraConfig(os.path.join(path, path_list[ii])))

    def tag_same(a, b):
        a_id = set()
        b_id = set()
        for tag in a:
            a_id.add(tag.id)
        for tag in b:
            b_id.add(tag.id)

        common_id = a_id.intersection(b_id)

        if(len(common_id) == 0):
            return False

        corners_a = dict()
        for tag in a:
            corners_a[tag.id] = np.array(tag.corners)
        corners_b = dict()
        for tag in b:
            corners_b[tag.id] = np.array(tag.corners)

        diff = []
        for id in common_id:
            diff.append(corners_a[id] - corners_b[id])
        diff_np = np.array(diff)
        diff_metre = np.abs(diff_np).mean()

        if(diff_metre > 0.5):
            return False
        else:
            return True
        pass

    multi_cam = MultiCam(cam_list)
    # bridge = CvBridge()
    print(len(multi_cam.img_list))
    prev_cam_list = []
    detector = make_default_detector()
    with rosbag.Bag("ros.bag", 'w') as bag:
        for cams, images in tqdm(multi_cam):
            curr_cam_list = []
            for ii in range(len(cams)):
                curr_cam_list.append(detector.extract_tags(cv2.imread(images[ii].path)))

            if(len(prev_cam_list) == 0):
                prev_cam_list = curr_cam_list
                continue


            all_same = True
            for ii in range(len(cams)):
                prev_cam = prev_cam_list[ii]
                curr_cam = curr_cam_list[ii]
                if(not tag_same(prev_cam, curr_cam)):
                    all_same = False

            if not all_same:
                prev_cam_list = curr_cam_list
                continue
            print('good cam at time {}'.format(images[0].time))

            for ii in range(len(cams)):
                cv_image = cv2.imread(images[ii].path)
                ros_image = Image()
                # ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                ros_image.header.stamp.secs = int(images[ii].time / 1000)
                ros_image.header.stamp.nsecs = int(images[ii].time % 1000) * 1000 * 1000
                ros_image.encoding='bgr8'
                ros_image.height = cv_image.shape[0]
                ros_image.width = cv_image.shape[1]
                ros_image.data = cv_image.tobytes()
                bag.write('/image_{}'.format(ii), ros_image, ros_image.header.stamp)

            prev_cam_list = curr_cam_list

if __name__ == '__main__':
    # do_job('/home/libaoyu/mnt/sdb/c91_dataset_server/c91_dataset/lab_test/scene-1-day/#5-0025-70°/开灯-场景一', 0)
    do_job('/home/libaoyu/Downloads/camera', 0)
