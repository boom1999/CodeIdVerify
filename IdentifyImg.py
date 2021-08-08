# -*- coding: utf-8 -*-
# @Time : 2021/8/8 16:05
# @Author : lingz
# @Software: PyCharm

import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_data_dir = r'train'
test_data_dir = r'test'
model_dir = r'model.h5'


def deNoising(image):
    """
    image.getpixel((x,y)):get RGB,return (*,*,*)
    :param image: RGB Img
    :return: Grayscale Img
    """
    # Set threshold to filter
    threshold = 127
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
    Generate data set
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

            # TODO: 验证灰度处理
            # Save the Img
            with open("./{}/{}".format(filePath, selected_train_file_name), "wb") as train:
                captcha_image.save(train, format="png")

            captcha_image_np = np.array(captcha_image)

            img_np = np.array(captcha_image_np)
            x_data.append(img_np)
            # split('.')[0] to get label from label.png
            # TODO: maybe .astype(np.int) is error
            y_data.append(np.array(list(selected_train_file_name.split('.')[0])))

    x_data = np.array(x_data).astype(np.float)
    y_data = np.array(y_data)
    return x_data, y_data


# Generate train set
(x_train, y_train) = generateTrainData(train_data_dir)
# Generate test set
(x_test, y_test) = generateTrainData(test_data_dir)
# (num of Img, 20:width, 80:height) (num of Img, 4)
print(x_train.shape, y_train.shape)


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


# load_dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.map(preprocess).batch(10)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(1)

model = Sequential([
    # 第一个卷积层
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # layers.Dropout(0.25),
    # 第二个卷积层
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # layers.Dropout(0.25),
    layers.Flatten(),

    # 全连接
    layers.Dense(128),
    layers.Dense(40),  # 因为这里我们4个数字，所以也就4*10可能性
    layers.Reshape([4, 10])
])

model.build(input_shape=[None, 20, 80, 1])
model.summary()
# 设置学习率
optimizer = optimizers.Adam(lr=1e-3)


def train():
    global model
    # If the model already exists, use it directly
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model('model.h5', compile=False)

    # Repeat training
    training_time = 20
    for epoch in range(training_time):
        for step, (x, y) in enumerate(train_db):
            if x.shape == (10, 20, 80, 1):
                with tf.GradientTape() as tape:
                    logits = model(x)
                    y_onehot = tf.one_hot(y, depth=10)
                    loss_ce = tf.losses.MSE(y_onehot, logits)
                    loss_ce = tf.reduce_mean(loss_ce)
                grads = tape.gradient(loss_ce, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, 'loss:', float(loss_ce))
    model.save('model.h5')


def test():
    model = tf.keras.models.load_model('model.h5', compile=False)
    for step, (x, y) in enumerate(test_db):
        if x.shape == (1, 20, 80, 1):
            logits = model(x)
            logits = tf.nn.softmax(logits)
            pred = tf.cast(tf.argmax(logits, axis=2), dtype=tf.int32)
            print('预测值：', pred[0].numpy(), '真实值：', y[0].numpy(), '是否相同：',
                  int(tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))) == 4)


if __name__ == '__main__':
    choice_flag = 0  # 0:train, 1: test
    if os.path.exists(model_dir) and choice_flag == 1:
        test()
    else:
        train()