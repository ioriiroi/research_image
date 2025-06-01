import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import csv
import glob
import cv2
import os

from natsort import natsorted
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- 自作のライブラリ ---
import model.model_normal as model_normal
import model.model_mobilenet as model_mobilenet
import setting.config as config
from lib.JsonLoadAndWrite import openJson

mid = 4057
csv_path = "data.csv"
data_dir = config.ILLUST_DIR
test_dir = "testdata"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMAGE_SIZE = 198

def road_image_path(data_dir):
    all_image_paths = list(glob.glob("{}/*/*.jpg".format(data_dir))) # 画像パスを全て取得
    all_image_paths = natsorted(all_image_paths) # パスをソート

    return all_image_paths

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0

    return image

def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        csv_list = [row for row in reader]

    return csv_list

def set_label(csv_path, all_image_paths):
    csv_list = load_csv(csv_path)

    all_image_labels = []
    for i in range(len(all_image_paths)):
        path = os.path.split(all_image_paths[i])[1]
        no = int(path[:3]) # 画像番号

        like = int(csv_list[no][3]) # 画像番号に対するいいね数

        # いいねが中央値より大きければ1 (データの取り方変えればいらなくなるかも)
        if like > mid:
            all_image_labels.append(1)
        else:
            all_image_labels.append(0)

    return all_image_labels

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
    all_image_paths = road_image_path(data_dir)
    ds_path = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds_image = ds_path.map(preprocess_image) # 画像の読み込み

    all_image_labels = set_label(csv_path, all_image_paths) #各画像にラベル付

    ds_labels = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    ds_image_label = tf.data.Dataset.zip((ds_image, ds_labels))

    
    """ 簡易的なテストケース """
    test_image_paths = list(glob.glob("{}/*/*.jpeg".format(test_dir)))
    test_image_paths = natsorted(test_image_paths)
    test_image = tf.data.Dataset.from_tensor_slices(test_image_paths)
    ds_test_image = test_image.map(preprocess_image)

    test_label = []

    for path in test_image_paths:
        path = os.path.split(path)[0]
        test_label.append(int(path[-1]))

    ds_test_labels = tf.data.Dataset.from_tensor_slices(tf.cast(test_label, tf.int64))
    test_image_label = tf.data.Dataset.zip((ds_test_image, ds_test_labels))

    test_image_label = test_image_label.batch(BATCH_SIZE)
    test_image_label = test_image_label.prefetch(buffer_size=AUTOTUNE)

    # train_x, valid_x, train_y, valid_y = train_test_split(all_image_paths, all_image_labels, test_size=0.2)

    image_count = len(all_image_paths)
    val_size = int(image_count * 0.3)
    train_ds = ds_image_label.skip(val_size)
    val_ds = ds_image_label.take(val_size)

    train_ds_num = tf.data.experimental.cardinality(train_ds).numpy()
    val_ds_num = tf.data.experimental.cardinality(val_ds).numpy()

    print(image_count)
    print(train_ds_num, val_ds_num)


    train_ds = ds_image_label.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_ds_num))
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # (-1, 1)に正規化
    # train_ds = train_ds.map(change_range)

    model = model_normal(IMAGE_SIZE)

    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, steps_per_epoch=train_ds_num // BATCH_SIZE,
              verbose=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0, patience=10,verbose=1)])

    show_graph(history)

    test_loss, test_acc = model.evaluate(
		test_image_label,
		steps=1
    )

    print("test accuracy: {}".format(test_acc))
    print("test loss: {}".format(test_loss))

if __name__ == "__main__":
    main()