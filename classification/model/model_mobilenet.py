import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from keras.optimizers import Adam

def model_mobilenet(IMAGE_SIZE):
    # MobileNet : 画像データに使われるディープラーニング手法
    mobile_net = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)

    mobile_net.trainable=True

    fine_tuning = 200

    for layer in mobile_net.layers[:fine_tuning]:
        layer.trainable = False

    # モデルの構築
    model = tf.keras.Sequential([
    mobile_net,
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
    ])

    return model