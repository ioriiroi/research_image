import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Dropout, Flatten, Conv2D, BatchNormalization
from keras.models import Model

from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import EfficientNetB4

def se_block(input_tensor, ratio=8):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([input_tensor, se])
    return x

def model_VGG16(IMAGE_SIZE, num_classes):
    base_model = VGG16(weights=None, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model.output
    # x = se_block(x)  # Attentionブロックを追加
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def model_EfficientNet(IMAGE_SIZE, num_classes):
    base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-50]:  # 下位の層は凍結
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model