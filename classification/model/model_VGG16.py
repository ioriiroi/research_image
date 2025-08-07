import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Dropout, Flatten, Conv2D, BatchNormalization
from keras.models import Model

from keras.applications.vgg16 import VGG16

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
    base_model = VGG16(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model.output
    # x = se_block(x)  # Attentionブロックを追加
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def model_VGG16_block5_conv3(IMAGE_SIZE, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    l2_num = 0.005

    x = base_model.get_layer('block2_pool').output

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model