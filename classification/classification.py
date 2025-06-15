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
from model.model_normal import model_normal, model_normal_deep, model_normal_deep2, model_simple, model_balanced, model_deep_with_regularization
from model.model_mobilenet import model_mobilenet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from lib.SetLabel import set_label
from src.JsonLoadAndWrite import openJson

# 警告を非表示にする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全て, 1=INFO以外, 2=WARNING以外, 3=ERROR以外
tf.get_logger().setLevel('ERROR')

data_dir = config.DATA_DIR
image_dir = config.DOWNLOAD_DIR
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
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
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        
        # データ拡張を追加
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])  # 余裕を持ってリサイズ
        image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])  # ランダムクロップ
        image = tf.image.random_flip_left_right(image)  # 左右反転
        image = tf.image.random_brightness(image, 0.1)  # 明るさをランダムに変更
        image = tf.image.random_contrast(image, 0.8, 1.2)  # コントラストをランダムに変更
        
        image = tf.cast(image, tf.float32) / 255.0
        return image
    except Exception as e:
        tf.print("画像処理エラー:", e)
        return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)


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

def apply_augmentation(image, label, pattern=0):
    """
    複数の拡張パターンを適用する関数
    pattern: 適用する拡張パターンの番号（0-4）
    """
    # 基本のリサイズ（すべてのパターンに適用）
    image = tf.image.resize(image, [IMAGE_SIZE+30, IMAGE_SIZE+30])
    
    # パターン別の拡張処理
    if pattern == 0:
        # パターン1: 基本的な拡張
        image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
    
    elif pattern == 1:
        # パターン2: 色調変更に重点
        image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.image.random_hue(image, 0.2)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.7, 1.3)
    
    elif pattern == 2:
        # パターン3: ジオメトリ変換に重点（回転なしバージョン）
        # 90度単位の回転（TensorFlowの標準機能）
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        
        # パディングとクロップでランダムシフト効果を出す
        padded = tf.pad(image, [[10, 10], [10, 10], [0, 0]], mode='REFLECT')
        offset_h = tf.random.uniform([], maxval=20, dtype=tf.int32)
        offset_w = tf.random.uniform([], maxval=20, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(
            padded, offset_h, offset_w, IMAGE_SIZE, IMAGE_SIZE
        )
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    
    elif pattern == 3:
        # パターン4: 明るさ・コントラスト変化に重点
        image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        
    elif pattern == 4:
        # パターン5: ぼかしと強い色調変更
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        # ガウスぼかしの代わりにダウンサンプリングとアップサンプリング
        image_small = tf.image.resize(image, [IMAGE_SIZE // 2, IMAGE_SIZE // 2])
        image = tf.image.resize(image_small, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.image.random_hue(image, 0.3)
        image = tf.image.random_saturation(image, 0.5, 1.5)
    
    # 正規化（すべてのパターンに適用）
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label
    
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

def robust_preprocess(path, label, augment=False):
    """さらに強化したエラーハンドリングを持つ前処理関数"""
    try:
        # ファイルの読み込み
        file_content = tf.io.read_file(path)
        
        # ファイルが空でないか確認
        file_size = tf.strings.length(file_content)
        if file_size == 0:
            tf.print("警告: 空のファイル", path)
            return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32), label
        
        # 複数のデコード方法を試す
        try:
            image = safe_decode_image(file_content)
        except:
            tf.print("画像デコードエラー:", path)
            return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32), label
        
        # 形状の確認と修正
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        
        # 無効な形状をチェック
        valid_shape = tf.logical_and(
            tf.greater(height, 0),
            tf.greater(width, 0)
        )
        
        # 無効な形状の場合は黒い画像を返す
        image = tf.cond(
            valid_shape,
            lambda: tf.identity(image),
            lambda: tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=image.dtype)
        )
        
        # リサイズ - 常に固定サイズにする
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        
        # 拡張を適用（トレーニングデータのみ）
        if augment:
            # ランダムに拡張パターンを選択
            pattern = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)
            image, label = apply_augmentation(image, label, pattern)
        else:
            # 拡張なしの場合は単純に正規化
            image = tf.cast(image, tf.float32) / 255.0
        
        # 最終的な形状チェック - 必ず正しい形状を持つことを保証
        image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        
        return image, label
        
    except Exception as e:
        tf.print("画像処理エラー:", e)
        # エラーの場合は黒い画像を返す
        return tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32), label
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
    # 1. 画像パスとラベルを取得
    all_image_paths = road_image_path(image_dir)
    data_json = openJson(data_dir)
    all_image_labels = set_label(data_json)
    
    # 2. パスとラベルのデータセットを作成（まだ画像は読み込まない）
    ds_path = tf.data.Dataset.from_tensor_slices(all_image_paths)
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
    train_ds = train_ds.map(
        lambda path, label: robust_preprocess(path, label, augment=True),
        num_parallel_calls=AUTOTUNE
    )
    
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
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    model = model_deep_with_regularization(IMAGE_SIZE, 4)
    # model = model_mobilenet(IMAGE_SIZE)

    # 学習率スケジューラーの追加
    # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=5,
    #     min_lr=1e-8,
    #     verbose=1
    # )

    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    class_weights = calculate_balanced_weights(all_image_labels)

    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=lr_scheduler),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=100,
        class_weight=class_weights,
        verbose=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # より長い忍耐値
                verbose=1
            )
        ]
    )

    show_graph(history)

    test_loss, test_acc = model.evaluate(test_ds)

    print("test accuracy: {}".format(test_acc))
    print("test loss: {}".format(test_loss))

    class_names = [0,1,2,3]
    visualize_confusion_matrix(model, test_ds, class_names)

if __name__ == "__main__":
    main()