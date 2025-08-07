import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import csv
import glob
import cv2
import os
import sys
from natsort import natsorted

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config

image_dir = config.ILLUST_DIR
lbp = "data/lbpcascade_animeface.xml"

def road_image_path(image_dir):
    all_image_paths = list(glob.glob("{}/*.jpg".format(image_dir))) # 画像パスを全て取得
    all_image_paths = natsorted(all_image_paths) # パスをソート
    return all_image_paths

paths = road_image_path(image_dir)

for path in paths:
    img = cv2.imread(path)

    # カスケード型識別器の読み込み
    cascade = cv2.CascadeClassifier(lbp)

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # アニメ顔領域の探索
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 顔領域を赤色の矩形で囲む
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y+h), (0, 0, 200), 5)

    # 結果を出力（修正）
    cv2.imshow("Anime Face Detection", img)  # ウィンドウ名と画像を指定
    cv2.waitKey(0)  # キー入力待ち
    cv2.destroyAllWindows()  # ウィンドウを閉じる