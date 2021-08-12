# -*- coding: utf-8 -*-
# @Time : 2021/8/11 11:11
# @Author : lingz
# @Software: PyCharm

import tensorflow as tf
import IdentifyImg

train_data_dir = r'train'


def IdentifyTrain():
    """
    Save model.h5
    :return: None
    """
    # Repeat training
    training_time = 40
    for epoch in range(training_time):
        for step, (x, y) in enumerate(train_db):
            if x.shape == (64, 20, 80, 1):
                with tf.GradientTape() as tape:
                    logits = IdentifyImg.model(x)
                    y_one_hot = tf.one_hot(y, depth=62)
                    loss_ce = tf.losses.MSE(y_one_hot, logits)
                    loss_ce = tf.reduce_mean(loss_ce)
                grads = tape.gradient(loss_ce, IdentifyImg.model.trainable_variables)
                IdentifyImg.optimizer.apply_gradients(zip(grads, IdentifyImg.model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce))
    IdentifyImg.model.save('model.h5')
    print("Save model.h5 successfully!")


# Generate train set
(x_train, y_train) = IdentifyImg.generateTrainData(train_data_dir)
print(x_train.shape, y_train.shape)

# load_dataset
batch_size = 64
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.map(IdentifyImg.preprocess).batch(batch_size)

if __name__ == '__main__':
    IdentifyTrain()
