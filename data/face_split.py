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

# 警告を非表示にする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全て, 1=INFO以外, 2=WARNING以外, 3=ERROR以外

image_dir = config.DOWNLOAD_DIR
output_dir = config.ILLUST_FACE_DIR
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
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

    # ファイル名（拡張子なし）を取得
    base_filename = os.path.splitext(os.path.basename(path))[0]
    
    # 顔が検出された場合
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # 顔領域を切り出し
            face_img = img[y:y+h, x:x+w]
            
            # 保存ファイル名を決定
            if len(faces) == 1:
                # 1つの場合：[ファイル名].jpg
                save_filename = f"{base_filename}.jpg"
            else:
                # 複数の場合：[ファイル名]_1.jpg, [ファイル名]_2.jpg...
                save_filename = f"{base_filename}_{i+1}.jpg"
            
            # 保存パスを作成
            save_path = os.path.join(output_dir, save_filename)
            
            # 顔画像を保存
            cv2.imwrite(save_path, face_img)
            print(f"顔画像を保存: {save_path}")
            
            # 元画像に矩形を描画（確認用）
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), 2)

    # # 結果を出力（修正）
    # cv2.imshow("Anime Face Detection", img)  # ウィンドウ名と画像を指定
    # cv2.waitKey(0)  # キー入力待ち
    # cv2.destroyAllWindows()  # ウィンドウを閉じる