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
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Sequential
from keras.preprocessing import image
from keras import models
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D

# --- 自作のライブラリ ---
from model.model_normal import model_normal, model_normal_deep, model_normal_deep2, model_simple, model_balanced, model_deep_with_regularization
from model.model_mobilenet import model_mobilenet
from model.model_VGG16 import model_VGG16, model_EfficientNet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from lib.set_label import SetLabel
from src.JsonLoadAndWrite import openJson

# 警告を非表示にする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全て, 1=INFO以外, 2=WARNING以外, 3=ERROR以外
tf.get_logger().setLevel('ERROR')

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

data_dir = config.DATA_JSON_DIR
image_dir = config.DOWNLOAD_DIR
image_good_dir = config.ILLUST_GOOD_DIR
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMAGE_SIZE = 256


def file_diff_check(image_path, data):
    for path in image_path:
        path = os.path.splitext(os.path.basename(path))[0]

        if path not in data:
            print(path)

def road_image_path(image_dir):
    all_image_paths = list(glob.glob("{}/*.jpg".format(image_dir))) # 画像パスを全て取得
    all_image_paths = natsorted(all_image_paths) # パスをソート
    return all_image_paths


def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        csv_list = [row for row in reader]

    return csv_list

""" AIが作成 """
def check_class_balance(labels):
    """クラスの分布を確認し、クラス重みを計算する"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("クラスバランス:")
    for cls, count in zip(unique, counts):
        print(f"クラス {cls}: {count} サンプル ({count/total*100:.2f}%)")
    
    # クラス重みを計算
    class_weights = {}
    max_count = max(counts)
    for cls, count in zip(unique, counts):
        class_weights[int(cls)] = max_count / count
    
    return class_weights

def calculate_balanced_weights(labels):
    """より穏やかなクラス重みを計算する"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("クラスバランス:")
    for cls, count in zip(unique, counts):
        print(f"クラス {cls}: {count} サンプル ({count/total*100:.2f}%)")
    
    # 穏やかなクラス重みを計算
    class_weights = {}
    avg_count = sum(counts) / len(counts)
    for cls, count in zip(unique, counts):
        # 平方根スケーリングで穏やかな重みを計算
        weight = np.sqrt(avg_count / count)
        # 重みを制限して極端な値を避ける
        weight = min(max(weight, 0.5), 2.0)
        class_weights[int(cls)] = weight
        print(f"クラス {cls} の重み: {weight:.2f}")
    
    return class_weights

def simple_rotation_augmentation(image, min_angle, max_angle):
    """
    シンプルな回転拡張（tf.py_functionを使用）
    """
    def rotate_image(img, angle):
        # NumPyとPILを使用した回転
        from PIL import Image
        import numpy as np
        
        # TensorFlowテンソルをNumPy配列に変換
        img_np = img.numpy()
        
        # uint8に変換（PILで処理するため）
        if img_np.dtype == np.float32:
            img_np = (img_np * 255).astype(np.uint8)
        
        # PIL Imageに変換
        pil_img = Image.fromarray(img_np)
        
        # 回転
        rotated_pil = pil_img.rotate(angle, fillcolor=(0, 0, 0), expand=False)
        
        # NumPy配列に戻す
        rotated_np = np.array(rotated_pil)
        
        # float32に正規化
        if img.dtype == tf.float32:
            rotated_np = rotated_np.astype(np.float32) / 255.0
        
        return rotated_np
    
    # ランダムな角度を生成
    angle = tf.random.uniform([], min_angle, max_angle)
    
    # tf.py_functionを使用してPython関数を呼び出し
    rotated_image = tf.py_function(
        lambda img, ang: rotate_image(img, ang),
        [image, angle],
        tf.float32
    )
    
    # 形状を明示的に設定
    rotated_image.set_shape(image.shape)
    
    return rotated_image

def resize_with_padding_tf(image, size):
    """
    アスペクト比を保持してリサイズし、パディングで正方形(size x size)にする（TensorFlow ops）
    image: uint8 / float tensor with shape [H, W, 3]
    size: int (出力の長さ)
    """
    image = tf.convert_to_tensor(image)
    orig_shape = tf.shape(image)
    h = tf.cast(orig_shape[0], tf.float32)
    w = tf.cast(orig_shape[1], tf.float32)
    size_f = tf.cast(size, tf.float32)

    scale = size_f / tf.maximum(h, w)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    # リサイズ（補間は縮小時はAREA、拡大時はBILINEARを自動選択）
    resized = tf.image.resize(image, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)

    # パディング量を計算して中央寄せ
    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # padded は float または uint8 を受け取れる。ここでは中間値128（グレー）で埋める
    padded = tf.pad(resized,
                    [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    constant_values=128)

    # 念のため最終サイズを整える
    padded = tf.image.resize_with_crop_or_pad(padded, size, size)
    return padded

augment_layer = Sequential([
    layers.RandomRotation(0.4),
    layers.RandomTranslation(0, 0.2),
    layers.RandomTranslation(0.2, 0),
    layers.RandomZoom(0.2, 0.2),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomContrast(0.2),
], name="augmentation_layer")

def robust_preprocess(path, label, augment):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)

        # rgb -> hsv
        # image = tf.image.convert_image_dtype(image, tf.float32)  # 0-1 に正規化
        # image = tf.image.rgb_to_hsv(image)

        shape_tensor = tf.shape(image) # 画像サイズを取得
        max_size = tf.reduce_max(shape_tensor) # 画像の長辺を取得
        # アスペクト比を保ったままの切り取り
        image = tf.image.resize_with_crop_or_pad(image, max_size, max_size)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        # image = resize_with_padding_tf(image, IMAGE_SIZE)

        # augment フラグが True の場合、Keras の前処理レイヤーを適用
        if augment:
            # レイヤーはバッチ単位（[H,W,3] でも動作する）だが、tf.data.map内で確実に動かすため training=True を指定
            image = augment_layer(image, training=True)
        
        image = tf.cast(image, tf.float32) / 255.0
        image = preprocess_input(image)
        image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

        return image, label
    except Exception as e:
        tf.print("処理エラー:", path)
        return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32), label

def robust_preprocess_flip(path, label, augment):
    image, label = robust_preprocess(path, label, augment)
    image = tf.image.flip_left_right(image)  # 必ず反転
    return image, label
""""""

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

def visualize_confusion_matrix(model, test_ds, class_names=None):
    """
    モデルのテストデータに対する予測結果を混同行列として可視化する
    
    Parameters:
    - model: 評価するモデル
    - test_ds: テストデータセット
    - class_names: クラス名のリスト (省略可能)
    """
    import sklearn.metrics
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if class_names is None:
        class_names = ['クラス0', 'クラス1', 'クラス2', 'クラス3']
    
    # 予測と実際のラベルを収集
    y_pred = []
    y_true = []
    
    # テストデータセットに対する予測
    for images, labels in test_ds:
        predictions = model.predict(images)
        pred_classes = tf.argmax(predictions, axis=1)
        
        # バッチ処理されているため、結果をリストに追加
        y_pred.extend(pred_classes.numpy())
        y_true.extend(labels.numpy())
    
    # 混同行列を計算
    cm = confusion_matrix(y_true, y_pred)
    
    # 正規化された混同行列も計算
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 元の混同行列
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 正規化された混同行列
    # plt.subplot(1, 2, 2)
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
    #             xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel('predict')
    # plt.ylabel('actual')
    # plt.title('regular matrix')
    
    plt.tight_layout()
    plt.show()

def visualize_probability_scores(model, images, labels=None, class_names=None, max_plots=5, figsize=(12, 3)):
    """
    画像ごとの確率スコアを棒グラフで表示する。
    - images: バッチ画像（Tensor or numpy array）, shape=(N, H, W, C) または単一画像 (H,W,C)
    - labels: 真のラベル（オプション）
    - class_names: クラス名リスト（省略時はクラスインデックスを使用）
    - max_plots: 表示するサンプル数（バッチ内で先頭から）
    """
    import math

    # バッチ化されていない単一画像を対応
    imgs = images.numpy() if isinstance(images, tf.Tensor) else images
    is_single = imgs.ndim == 3
    if is_single:
        imgs = imgs[np.newaxis, ...]
        if labels is not None:
            labels = np.array([labels.numpy()]) if isinstance(labels, tf.Tensor) else np.array([labels])

    # 予測確率
    preds = model.predict(imgs)
    probs = tf.nn.softmax(preds, axis=1).numpy()

    num_plots = min(max_plots, imgs.shape[0])
    # 2行 (画像, 棒グラフ) x num_plots 列
    fig_width = figsize[0] * num_plots
    fig_height = figsize[1] * 2
    fig, axes = plt.subplots(2, num_plots, figsize=(fig_width, fig_height))
    # axes の形を統一
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=1)  # (2,1)

    for i in range(num_plots):
        ax_img = axes[0, i]
        ax_bar = axes[1, i]

        img = imgs[i]
        # uint8 (0-255) の場合はそのまま、float の場合は 0-1 を想定して表示
        if img.dtype == np.float32 or img.dtype == np.float64:
            disp_img = np.clip(img, 0.0, 1.0)
        else:
            disp_img = img.astype(np.uint8)
            # 正規化されていない(0-255)を0-1にすることで matplotlib の挙動を安定させる
            if disp_img.max() > 1:
                disp_img = (disp_img / 255.0).astype(np.float32)

        ax_img.imshow(disp_img)
        ax_img.axis('off')

        pred_cls = int(np.argmax(probs[i]))
        pred_prob = float(np.max(probs[i]))
        title = f"pred:{pred_cls} ({pred_prob:.2f})"
        if labels is not None:
            true_cls = int(labels[i].numpy()) if isinstance(labels[i], tf.Tensor) else int(labels[i])
            title = f"true:{true_cls}\n" + title
        ax_img.set_title(title)

        xs = np.arange(probs.shape[1])
        ax_bar.bar(xs, probs[i], color='tab:blue')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(xs)
        if class_names is not None:
            ax_bar.set_xticklabels(class_names, rotation=45, ha='right')
        else:
            ax_bar.set_xticklabels([str(x) for x in xs], rotation=45, ha='right')
        ax_bar.set_ylabel("probability")

    plt.tight_layout()
    plt.show()

def main():
    print(BATCH_SIZE)
    print(image_dir)
    print(data_dir)
    # 1. 画像パスとラベルを取得
    all_image_paths = road_image_path(image_dir)
    good_image_paths = road_image_path(image_good_dir)
    data_json = openJson(data_dir)
    print(len(data_json))

    # 画像ファイル名（拡張子なし）のリストを作成 例: 12345_a_b.jpg -> 12345
    image_names = []
    image_good_names = []
    for p in all_image_paths:
        filename = os.path.splitext(os.path.basename(p))[0]  # 拡張子を除去
        image_names.append(filename)
    for p in good_image_paths:
        filename = os.path.splitext(os.path.basename(p))[0]  # 拡張子を除去
        image_good_names.append(filename)
    

    paths_bookmarks = SetLabel.get_bookmark(image_dir, data_json, image_names) | SetLabel.get_bookmark(image_good_dir, data_json, image_good_names)

    image_paths, all_image_labels, CLASS_NUM = SetLabel.set_label_front_and_back(paths_bookmarks)
    # print(len(image_paths), len(all_image_labels))
    # for image, label in zip(image_paths, all_image_labels):
    #     print(image, label)
    
    # 2. パスとラベルのデータセットを作成（まだ画像は読み込まない）
    ds_path = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    ds_path_label = tf.data.Dataset.zip((ds_path, ds_labels))

    # 3. データをシャッフル
    ds_shuffled = ds_path_label.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)
    
    # 4. データセットの総数を確認
    dataset_size = tf.data.experimental.cardinality(ds_shuffled).numpy()
    print(f"データセット総数: {dataset_size}")
    
    # 5. データを分割比率を設定
    # テスト:検証:訓練 = 2:1:7 の比率
    test_size = int(dataset_size * 0.2)  # 20%をテスト用
    val_size = int(dataset_size * 0.1)   # 10%を検証用
    train_size = dataset_size - test_size - val_size  # 残りを訓練用
    
    # 6. データセットを分割（パスとラベルのペアを分割）
    test_ds = ds_shuffled.take(test_size)
    remaining_ds = ds_shuffled.skip(test_size)
    val_ds = remaining_ds.take(val_size)
    train_ds = remaining_ds.skip(val_size)

    # 6. 堅牢な前処理の適用
    # 通常画像データセット
    train_ds1 = train_ds.map(lambda path, label: robust_preprocess(path, label, augment=True), num_parallel_calls=AUTOTUNE)
    # 左右反転画像データセット
    train_ds2 = train_ds.map(lambda path, label: robust_preprocess_flip(path, label, augment=True), num_parallel_calls=AUTOTUNE)
    # 連結して2倍に
    train_ds = train_ds1.concatenate(train_ds2)
    
    val_ds = val_ds.map(
        lambda path, label: robust_preprocess(path, label, augment=False),
        num_parallel_calls=AUTOTUNE
    )
    
    test_ds = test_ds.map(
        lambda path, label: robust_preprocess(path, label, augment=False),
        num_parallel_calls=AUTOTUNE
    )
    
    # 7. データセットサイズを確認
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    
    print(f"訓練データ数: {train_ds_size}")
    print(f"検証データ数: {val_ds_size}")
    print(f"テストデータ数: {test_ds_size}")
    print(f"合計: {train_ds_size + val_ds_size + test_ds_size}")
    
    # 8. バッチ処理とプリフェッチの設定
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE)

    for images, labels in train_ds.take(1):
        print("images.shape:", images.shape)  # (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
        print("labels.shape:", labels.shape)  # (BATCH_SIZE,)

    # for image, label in train_ds.unbatch().take(20):
    #     plt.imshow(image.numpy())
    #     plt.title(f"label: {label.numpy()}")
    #     plt.axis('off')
    #     plt.show()
    # exit()

    model = model_EfficientNet(IMAGE_SIZE, CLASS_NUM)
    # model = model_mobilenet(IMAGE_SIZE)

    # 学習率スケジューラーの追加
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=100,
        decay_rate=0.96,
        staircase=False
    )

    class_weights = calculate_balanced_weights(all_image_labels)
    # モデルの構造確認
    # model.summary()

    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    for images, labels in train_ds.take(1):
        print("train images.shape:", images.shape)
    for images, labels in val_ds.take(1):
        print("val images.shape:", images.shape)
    for images, labels in test_ds.take(1):
        print("test images.shape:", images.shape)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.001,
        verbose=1
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=100,
        verbose=True,
        #class_weight=class_weights,
        callbacks=[reduce_lr, earlystop]
    )

    from sklearn.metrics import classification_report

    # 予測値と正解ラベルを集める
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    # --- 確率スコアの棒グラフを表示（テストデータの先頭バッチから最大5枚） ---
    for images, labels in test_ds.take(10):
        visualize_probability_scores(model, images[:5], labels[:5], class_names=[str(i) for i in range(CLASS_NUM)], max_plots=5)


    print(classification_report(y_true, y_pred, digits=4))

    show_graph(history)

    test_loss, test_acc = model.evaluate(test_ds)

    print("test accuracy: {}".format(test_acc))
    print("test loss: {}".format(test_loss))

    class_names = [0,1,2,3]
    visualize_confusion_matrix(model, test_ds, class_names)

if __name__ == "__main__":
    main()