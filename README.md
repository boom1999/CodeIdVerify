# CodeIdVerify

Verified the code identification.

## This is CodeIdVerify ##

### Begin 👇 ###

> Run `GenerateImg.py`, change num to set train set num(90%) and test set num(10%).

> In `IdentifyImg.py`, some basic data set, deNoising(filter), generate data, preprocessing and generate train model.

> correct_rate now is reaching 93%

``` python
# model
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
``` 
