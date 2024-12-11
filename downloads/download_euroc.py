import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 目标 URL
BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"
OUTPUT_DIR = "./ijrr_euroc_mav_dataset"

def download_file(url, save_path):
    """下载文件到本地"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def scrape_and_download(base_url, output_dir):
    """递归抓取网页链接并下载"""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href not in ["../", "./"]:
            full_url = urljoin(base_url, href)
            local_path = os.path.join(output_dir, href)

            if href.endswith("/"):  # 处理子目录
                if not os.path.exists(local_path):
                    os.makedirs(local_path)
                scrape_and_download(full_url, local_path)
            else:  # 下载文件
                print(f"Downloading {full_url} to {local_path}")
                download_file(full_url, local_path)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    scrape_and_download(BASE_URL, OUTPUT_DIR)
