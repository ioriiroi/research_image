import glob
import sys
import os

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
image_dir = config.ILLUST_DIR

all_image_paths = road_image_path(image_dir)
data_json = openJson(data_dir)

filenames = [os.path.splitext(os.path.basename(p))[0] for p in all_image_paths]

for i in range(len(all_image_paths)):
    name = os.path.splitext(os.path.basename(all_image_paths[i]))[0]
    try:
        bookmark = data_json[name]["bookmark"]
    except:
        continue
    if 50 <= bookmark <= 1000000:
        img = mpimg.imread(all_image_paths[i])
        plt.imshow(img)
        plt.axis('off')  # 枠線・目盛りを消す
        plt.show()
