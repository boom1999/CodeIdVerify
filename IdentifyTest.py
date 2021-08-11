# -*- coding: utf-8 -*-
# @Time : 2021/8/11 11:11
# @Author : lingz
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import IdentifyImg


def IdentifyTest():
    """
    :return: correct_rate = correct_quantity/total_number
    """
    correct_quantity = 0
    total_number = IdentifyImg.x_train.shape[0]
    IdentifyImg.model = tf.keras.models.load_model('model.h5', compile=False)
    for step, (x, y) in enumerate(IdentifyImg.test_db):
        if x.shape == (1, 20, 80, 1):
            logits = IdentifyImg.model(x)
            logits = tf.nn.softmax(logits)
            pred = tf.cast(tf.argmax(logits, axis=2), dtype=tf.int32)
            true_flag = int(tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))) == 4
            print('Pre：', np.array(IdentifyImg.char_set)[pred[0].numpy()], 'Real：',
                  np.array(IdentifyImg.char_set)[y[0].numpy()], 'Same：', true_flag)
            if true_flag:
                correct_quantity += 1
    correct_rate = correct_quantity / total_number
    return correct_rate


if __name__ == '__main__':
    correct_rate = IdentifyTest()
    print(correct_rate)
