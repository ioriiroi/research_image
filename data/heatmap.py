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

from classification.model.model_normal import model_normal, model_normal_deep, model_normal_deep2, model_simple, model_balanced, model_deep_with_regularization
from classification.model.model_mobilenet import model_mobilenet
from classification.model.model_VGG16 import model_VGG16, model_EfficientNet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config

IMAGE_SIZE = 256
CLASS_NUM = 2
IMAGE_PASS = config.ILLUST_DIR

model = model_deep_with_regularization(IMAGE_SIZE, CLASS_NUM)

img = image.load_img(f"{IMAGE_PASS}/121865201.jpg", target_size=(IMAGE_SIZE, IMAGE_SIZE))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
print("IMAGE: %s" % str(img.shape))

layers = model.layers[-10:-5]
layer_outputs = [layer.output for layer in layers]
activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)
# activation_model.summary()

activations = activation_model.predict(img)
# for i, activation in enumerate(activations):
#     print("%2d: %s" % (i, str(activation.shape)))

import math
import seaborn as sns

# プーリング層の出力のみに絞る (畳み込み層の出力も可視化できるが量が多くなるため)
activations = [(layer.name, activation) for layer, activation in zip(layers, activations) if isinstance(layer, Conv2D)]

for i, (name, activation) in enumerate(activations):
    num_of_image = activation.shape[3]
    max = np.max(activation[0])
    for j in range(0, num_of_image):
        plt.figure()
        sns.heatmap(activation[0, :, :, j], vmin=0, vmax=max, xticklabels=False, yticklabels=False, square=False)
        plt.savefig("%d_%d.png" % (i+1, j+1))
        plt.close()
exit()

# 出力層ごとに特徴画像を並べてヒートマップ画像として出力
for i, (name, activation) in enumerate(activations):
    num_of_image = activation.shape[3]
    cols = math.ceil(math.sqrt(num_of_image))
    rows = math.floor(num_of_image / cols)
    screen = []
    for y in range(0, rows):
        row = []
        for x in range(0, cols):
            j = y * cols + x
            if j < num_of_image:
                row.append(activation[0, :, :, j])
            else:
                row.append(np.zeros())
        screen.append(np.concatenate(row, axis=1))
    screen = np.concatenate(screen, axis=0)
    plt.figure()
    sns.heatmap(screen, xticklabels=False, yticklabels=False)
    plt.savefig("%s.png" % name)
    plt.close()
