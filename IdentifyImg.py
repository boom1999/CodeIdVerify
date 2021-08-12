# -*- coding: utf-8 -*-
# @Time : 2021/8/8 16:05
# @Author : lingz
# @Software: PyCharm

import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_dir = r'model.h5'
char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                                                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                                                 'w', 'x', 'y', 'z'] + ['A', 'B', 'C', 'D', 'E', 'F',
                                                                                        'G', 'H', 'I', 'J', 'K', 'L',
                                                                                        'M', 'N', 'O', 'P', 'Q', 'R',
                                                                                        'S', 'T', 'U', 'V', 'W', 'X',
                                                                                        'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z'] + ['A', 'B', 'C', 'D', 'E', 'F',
                                   'G', 'H', 'I', 'J', 'K', 'L',
                                   'M', 'N', 'O', 'P', 'Q', 'R',
                                   'S', 'T', 'U', 'V', 'W', 'X',
                                   'Y', 'Z']
char_to_int = dict((c, i) for i, c in enumerate(char_set))


def deNoising(image):
    """
    image.getpixel((x,y)):get RGB,return (*,*,*)
    :param image: RGB Img
    :return: Grayscale Img
    """
    # Set threshold to filter
    threshold = 128
    # Change to Grayscale
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            # For each pixel, if rgb>threshold,equals to light color==>bk_pixel,then set white(255,255,255)
            if r > threshold or g > threshold or b > threshold:
                r = 255
                g = 255
                b = 255
                image.putpixel((i, j), (r, g, b))
            # else,equals to dark color==>txt_pixel,set black(0,0,0)
            else:
                r = 0
                g = 0
                b = 0
                image.putpixel((i, j), (r, g, b))
    # Turn to grayscale Img
    image = image.convert('L')
    return image


def generateTrainData(filePath):
    """
    Generate data set, 0~9, a~z, A~Z turn to 0~61
    :param filePath: The Img in this folder need to be processed
    :return: x_data is data, shape=(num, 20, 80)
             y_data is label, shape=(num, 4)
    """
    # os.listdir(filePath) return list of files and folders
    # If the path has Chinese characters, it will be garbled, deal with unicode()
    train_file_name_list = os.listdir(filePath)
    x_data = []
    y_data = []

    # Process each Img
    for selected_train_file_name in train_file_name_list:
        if selected_train_file_name.endswith('.png'):
            # Get Img object, os.path.join(PathA, PathB) return PathA\PathB
            captcha_image = Image.open(os.path.join(filePath, selected_train_file_name))
            # deNoising
            captcha_image = deNoising(captcha_image)

            captcha_image_np = np.array(captcha_image)

            img_np = np.array(captcha_image_np)
            x_data.append(img_np)
            # split('.')[0] to get label from label.png
            y_data.append(np.array(list(selected_train_file_name.split('.')[0])))

    x_data = np.array(x_data).astype(np.float)
    y_data = np.array(y_data)

    # Temporarily store letters into array
    unicode_to_int = ['x']
    unicode_to_int_index = [0]

    for index_0, alphabet_list in enumerate(y_data):
        for index_1, value in enumerate(alphabet_list):
            if (alphabet_list[index_1]) in alphabet:
                unicode_to_int.append(alphabet_list[index_1])
                alphabet_list[index_1] = 0
                unicode_to_int_index.append(index_0)
                unicode_to_int_index.append(index_1)
    y_data = y_data.astype(np.int)

    del (unicode_to_int[0])
    del (unicode_to_int_index[0])

    for i in range(0, len(unicode_to_int)):
        y_data[unicode_to_int_index[2 * i]][unicode_to_int_index[2 * i + 1]] = char_to_int.get(unicode_to_int[i])

    return x_data, y_data


def preprocess(x, y):
    """
    Change to tensor
    :param x:
    :param y:
    :return:
    """
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    x = tf.expand_dims(x, -1)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


model = Sequential([
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.5),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.3),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(2480),
    layers.Dense(248),  # 4*62
    layers.Reshape([4, 62])
])

model.build(input_shape=[None, 20, 80, 1])
model.summary()
optimizer = optimizers.Adam(lr=0.43 * 1e-3)
