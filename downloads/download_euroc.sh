#!/bin/bash

# 目标 URL
BASE_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"

# 本地保存目录
OUTPUT_DIR="/home/roborock/datasets/ijrr_euroc_mav_dataset"

# 使用 wget 下载
wget --recursive \
     --no-parent \
     --no-clobber \
     --continue \
     --cut-dirs=3 \
     --reject "index.html*" \
     --directory-prefix="${OUTPUT_DIR}" \
     "${BASE_URL}"
