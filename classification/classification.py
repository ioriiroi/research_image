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

# --- 自作のライブラリ ---
from model.model_normal import model_normal, model_normal_deep, model_normal_deep2, model_simple, model_balanced, model_deep_with_regularization
from model.model_mobilenet import model_mobilenet
from model.model_VGG16 import model_VGG16, model_VGG16_block5_conv3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from lib.SetLabel import set_label, set_label_devide2, set_label_interval
from src.JsonLoadAndWrite import openJson

# 警告を非表示にする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全て, 1=INFO以外, 2=WARNING以外, 3=ERROR以外
tf.get_logger().setLevel('ERROR')

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

data_dir = config.DATA_DIR
image_dir = config.ILLUST_DIR
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

def change_range(image,label):
    return 2*image-1, label

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

    
def safe_decode_image(image_bytes):
    """複数のデコード方法を試す堅牢な画像デコード関数"""
    try:
        # JPEG形式として試す
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        return image
    except tf.errors.InvalidArgumentError:
        try:
            # PNG形式として試す
            image = tf.image.decode_png(image_bytes, channels=3)
            return image
        except tf.errors.InvalidArgumentError:
            try:
                # 汎用デコーダーとして試す
                image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
                return image
            except:
                # すべて失敗した場合はエラーを発生
                raise ValueError("画像のデコードに失敗しました")

def robust_preprocess(path, label, augment):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        shape_tensor = tf.shape(image) # 画像サイズを取得
        max_size = tf.reduce_max(shape_tensor) # 画像の長辺を取得
        # アスペクト比を保ったままの切り取り
        image = tf.image.resize_with_crop_or_pad(image, max_size, max_size)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

        image = tf.cast(image, tf.float32) / 255.0
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

def main():
    print(BATCH_SIZE)
    # 1. 画像パスとラベルを取得
    all_image_paths = road_image_path(image_dir)
    data_json = openJson(data_dir)

    # 画像ファイル名（拡張子なし）のリストを作成
    image_names = set(os.path.splitext(os.path.basename(p))[0] for p in all_image_paths)
    image_paths = []
    bookmarks = []
    for id in data_json:
        if id in image_names:
            image_paths.append(f"{image_dir}/{id}.jpg")
            bookmarks.append(data_json[id]["bookmark"])

    image_paths, all_image_labels, CLASS_NUM = set_label_devide2(image_paths, bookmarks)
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
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)

    for images, labels in train_ds.take(1):
        print("images.shape:", images.shape)  # (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
        print("labels.shape:", labels.shape)  # (BATCH_SIZE,)

    # for image, label in train_ds.unbatch().take(10):
    #     plt.imshow(image.numpy())
    #     plt.title(f"label: {label.numpy()}")
    #     plt.axis('off')
    #     plt.show()
    # exit()

    model = model_deep_with_regularization(IMAGE_SIZE, CLASS_NUM)
    # model = model_mobilenet(IMAGE_SIZE)

    # 学習率スケジューラーの追加
    # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=5,
    #     min_lr=1e-8,
    #     verbose=1
    # )

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96,
        staircase=False
    )

    class_weights = calculate_balanced_weights(all_image_labels)
    model.summary()

    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=lr_scheduler),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    for images, labels in train_ds.take(1):
        print("train images.shape:", images.shape)
    for images, labels in val_ds.take(1):
        print("val images.shape:", images.shape)
    for images, labels in test_ds.take(1):
        print("test images.shape:", images.shape)
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=100,
        verbose=True,
        #class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            )
        ]
    )

    from sklearn.metrics import classification_report

    # 予測値と正解ラベルを集める
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    print(classification_report(y_true, y_pred, digits=4))

    show_graph(history)

    test_loss, test_acc = model.evaluate(test_ds)

    print("test accuracy: {}".format(test_acc))
    print("test loss: {}".format(test_loss))

    class_names = [0,1,2,3]
    visualize_confusion_matrix(model, test_ds, class_names)

if __name__ == "__main__":
    main()