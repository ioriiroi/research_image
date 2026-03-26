import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Dropout, Flatten, Conv2D, BatchNormalization
from keras.models import Model

from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import EfficientNetB4, MobileNetV3Large

def model_VGG16(IMAGE_SIZE, num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # block5_conv1 (Conv2D)より前の重みを固定
    layer_names = [l.name for l in base_model.layers]
    idx = layer_names.index('block4_conv1')
    
    base_model.trainable = True
    for layer in base_model.layers[:idx]:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def model_EfficientNet(IMAGE_SIZE, num_classes):
    base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    layer_names = [l.name for l in base_model.layers]
    idx = layer_names.index('block5a_expand_conv')

    base_model.trainable = True
    for layer in base_model.layers[:idx]:  # 下位の層は凍結
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def model_mobilenet(IMAGE_SIZE, num_classes):
    # MobileNet : 画像データに使われるディープラーニング手法
    base_model = MobileNetV3Large(weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)

    # モデルの構築
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    return model