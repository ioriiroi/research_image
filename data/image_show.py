import glob
import sys
import os
import cv2
import random

from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from src.JsonLoadAndWrite import openJson

def road_image_path(image_dir):
    all_image_paths = list(glob.glob("{}/*.jpg".format(image_dir))) # 画像パスを全て取得
    all_image_paths = natsorted(all_image_paths) # パスをソート
    return all_image_paths

data_dir = config.DATA_DIR
image_dir = config.ILLUST_FACE_DIR

all_image_paths = road_image_path(image_dir)
data_json = openJson(data_dir)

filenames = [os.path.splitext(os.path.basename(p))[0] for p in all_image_paths]
cnt = 0

while(cnt <= 10):
    i = random.randrange(len(all_image_paths))
    name = os.path.splitext(os.path.basename(all_image_paths[i]))[0]
    try:
        bookmark = data_json[name]["bookmark"]
    except:
        continue
    if bookmark <= 10:
        img = mpimg.imread(all_image_paths[i])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # HSVの個別チャンネルをプロット
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(hsv[:, :, 0], cmap='hsv')  # Hue
        plt.title("Hue")
        plt.subplot(1, 3, 2)
        plt.imshow(hsv[:, :, 1], cmap='gray')  # Saturation
        plt.title("Saturation")
        plt.subplot(1, 3, 3)
        plt.imshow(hsv[:, :, 2], cmap='gray')  # Value
        plt.title("Value")
        plt.tight_layout()
        plt.show()

        cnt += 1

