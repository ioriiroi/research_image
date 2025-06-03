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
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- 自作のライブラリ ---
from model.model_normal import model_normal
import model.model_mobilenet as model_mobilenet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from lib.SetLabel import set_label
from src.JsonLoadAndWrite import openJson

mid = 4057
csv_path = "data.csv"
data_dir = config.DATA_DIR
image_dir = config.DOWNLOAD_DIR
test_dir = "testdata"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMAGE_SIZE = 198


def file_diff_check(image_path, data):
    for path in image_path:
        path = os.path.splitext(os.path.basename(path))[0]

        if path not in data:
            print(path)


def road_image_path(image_dir):
    all_image_paths = list(glob.glob("{}/*.jpg".format(image_dir))) # 画像パスを全て取得
    all_image_paths = natsorted(all_image_paths) # パスをソート

    return all_image_paths

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0

    return image

# def preprocess_image(path):
#     try:
#         # ファイルを読み込む
#         image = tf.io.read_file(path)
        
#         # 複数のデコード方法を試す
#         try:
#             # JPEGとしてデコード
#             image = tf.image.decode_jpeg(image, channels=3)
#         except tf.errors.InvalidArgumentError:
#             try:
#                 # PNGとしてデコード
#                 image = tf.image.decode_png(image, channels=3)
#             except tf.errors.InvalidArgumentError:
#                 # 汎用デコーダー
#                 image = tf.image.decode_image(image, channels=3, expand_animations=False)
        
#         # リサイズと正規化
#         image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#         image = tf.cast(image, tf.float32) / 255.0
        
#         return image
#     except Exception as e:
#         tf.print("画像処理エラー:", path)
#         # エラーが発生した場合は黒い画像を返す
#         return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)

def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        csv_list = [row for row in reader]

    return csv_list

# def set_label(csv_path, all_image_paths):
#     csv_list = load_csv(csv_path)

#     all_image_labels = []
#     for i in range(len(all_image_paths)):
#         path = os.path.split(all_image_paths[i])[1]
#         no = int(path[:3]) # 画像番号

#         like = int(csv_list[no][3]) # 画像番号に対するいいね数

#         # いいねが中央値より大きければ1 (データの取り方変えればいらなくなるかも)
#         if like > mid:
#             all_image_labels.append(1)
#         else:
#             all_image_labels.append(0)

#     return all_image_labels

def change_range(image,label):
    return 2*image-1, label


def show_graph(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

def main():   
    all_image_paths = road_image_path(image_dir)
    ds_path = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds_image = ds_path.map(preprocess_image, num_parallel_calls=AUTOTUNE) # 画像の読み込み

    data_json = openJson(data_dir)
    all_image_labels = set_label(data_json)

    ds_labels = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    ds_image_label = tf.data.Dataset.zip((ds_image, ds_labels))

    
    # """ 簡易的なテストケース """
    # test_image_paths = list(glob.glob("{}/*/*.jpeg".format(test_dir)))
    # test_image_paths = natsorted(test_image_paths)
    # test_image = tf.data.Dataset.from_tensor_slices(test_image_paths)
    # ds_test_image = test_image.map(preprocess_image)

    # test_label = []

    # for path in test_image_paths:
    #     path = os.path.split(path)[0]
    #     test_label.append(int(path[-1]))

    # ds_test_labels = tf.data.Dataset.from_tensor_slices(tf.cast(test_label, tf.int64))
    # test_image_label = tf.data.Dataset.zip((ds_test_image, ds_test_labels))

    # test_image_label = test_image_label.batch(BATCH_SIZE)
    # test_image_label = test_image_label.prefetch(buffer_size=AUTOTUNE)


    # 3. データをシャッフル
    ds_shuffled = ds_image_label.shuffle(buffer_size=1000, seed=42)
    
    # 4. データセットの総数を確認
    dataset_size = tf.data.experimental.cardinality(ds_shuffled).numpy()
    print(f"データセット総数: {dataset_size}")
    
    # 5. データを分割比率を設定
    # テスト:検証:訓練 = 2:1:7 の比率
    test_size = int(dataset_size * 0.2)  # 20%をテスト用
    val_size = int(dataset_size * 0.1)   # 10%を検証用
    train_size = dataset_size - test_size - val_size  # 残りを訓練用
    
    # 6. データセットを分割
    test_ds = ds_shuffled.take(test_size)  # テストデータを取得
    remaining_ds = ds_shuffled.skip(test_size)  # 残りのデータ
    val_ds = remaining_ds.take(val_size)  # 検証データを取得
    train_ds = remaining_ds.skip(val_size)  # 訓練データを取得
    
    # 7. データセットサイズを確認
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    
    print(f"訓練データ数: {train_ds_size}")
    print(f"検証データ数: {val_ds_size}")
    print(f"テストデータ数: {test_ds_size}")
    print(f"合計: {train_ds_size + val_ds_size + test_ds_size}")
    
    # 8. バッチ処理とプリフェッチの設定
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    

    # # (-1, 1)に正規化
    # # train_ds = train_ds.map(change_range)

    model = model_normal(IMAGE_SIZE)

    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, steps_per_epoch=train_ds_size // BATCH_SIZE,
              verbose=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0, patience=10,verbose=1)])

    show_graph(history)

    test_loss, test_acc = model.evaluate(
		test_ds,
		steps=1
    )

    print("test accuracy: {}".format(test_acc))
    print("test loss: {}".format(test_loss))

if __name__ == "__main__":
    main()