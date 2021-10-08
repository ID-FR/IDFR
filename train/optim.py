import numpy as np
import keras
import tensorflow as tf
from keras import layers, Model, Sequential, activations
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
sess = tf.Session(config=config)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train_mean = np.mean(x_train, axis=0)
x_test -= x_train_mean
x_train -= x_train_mean
model = keras.models.load_model(
    "/home/liyanni/1307/zwh/defense/models/cifar10/cifar10_ResNet56v2.h5"
)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model.evaluate(x_test, y_test)

logits = Model(inputs=model.input, outputs=model.get_layer("flatten_1").output)


logits.trainable = False
logits.name = "cnn_logits"
fx_train = logits.predict(x_train)
fx_test = logits.predict(x_test)

# classifier
# 拿到model的最后一层
classifier = Sequential()
classifier.add(model.layers[-1])
classifier.build((None, 256))
classifier.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["accuracy"])
classifier.trainable = False


#  cifar10 数据集 ID模型实现
def ID():
    x = layers.Input(shape=(32, 32, 3))
    # forword
    x1 = C2(x, 64)  # (32,32,64)
    x2 = forword_C3(x1, 128)  # (16,16,128)
    x3 = forword_C3(x2, 256)  # (8,8,256)
    x4 = forword_C3(x3, 256)  # (4,4,256)

    # backword
    x3 = F(x3, x4)  # (8,8,512)
    x3 = backword_C3(x3, 256)  # (8,8,256)
    x2 = F(x2, x3)  # (16,16,384)
    x2 = backword_C3(x2, 128)  # (16,16,128)
    x1 = F(x1, x2)  # (32,32,192)
    x1 = C2(x1, 64)  # (32,32,64)

    # x + noise
    noise = layers.Conv2D(filters=3, kernel_size=(1, 1))(x1)  # (32,32,3)
    y = layers.Add()([x, noise])

    # get logits
    y = logits(y)
    return Model(inputs=x, outputs=y)


def C2(x, out_channels):
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def forword_C3(x, out_channels):
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def backword_C3(x, out_channels):
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def F(input1, input2):
    input2 = layers.UpSampling2D(interpolation="bilinear")(input2)
    output = layers.concatenate([input1, input2], axis=-1)
    return output


# 定义回调函数
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.001
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("lr:", lr)
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
)
callbacks = [lr_reducer, lr_scheduler]

x_adv = np.load("../defense/adv_data/cifar10/cw/0.015x_adv.npy")
# label = np.append(y_train,y_test,axis=0)

x_train_adv = x_adv[:50000]
x_test_adv = x_adv[50000:]

model.evaluate(x_test_adv, y_test)

denoiser = ID()
denoiser.compile(loss="mean_absolute_error", optimizer=Adam(lr=lr_schedule(0)))

batch_size = 128
epochs = 100

denoiser.fit(
    x_train_adv,
    fx_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test_adv, fx_test),
    shuffle=True,
    callbacks=callbacks,
)

hidden = denoiser.predict(x_test)
classifier.evaluate(hidden, y_test)

denoiser.save("denoiser.h5")

fx_train_adv = denoiser.predict(x_train_adv)
fx_test_adv = denoiser.predict(x_test_adv)
fx_train = denoiser.predict(x_train)
fx_test = denoiser.predict(x_test)

train_group = []
(_, train_label), (_, test_label) = keras.datasets.cifar10.load_data()
train_label = train_label.reshape(-1)
for i in range(10):
    train_index = np.argwhere(train_label == i).reshape(-1)
    tmp = np.append(fx_train[train_index], fx_train_adv[train_index], axis=0)
    train_group.append(tmp)
# shuffle
for i in range(10):
    permutation = np.random.permutation(train_group[i].shape[0])
    train_group[i] = train_group[i][permutation]


def HR():
    x = layers.Input(shape=(256,))
    x1 = layers.Dense(256)(x)
    layers.Dropout(0.2)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Dense(128)(x1)
    layers.Dropout(0.2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x3 = layers.Dense(64)(x2)
    layers.Dropout(0.2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x4 = layers.Dense(32)(x3)
    layers.Dropout(0.2)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.Dense(64)(x4)
    layers.Dropout(0.2)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.add([x3, x4])
    x4 = layers.Dense(128)(x4)
    layers.Dropout(0.2)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.add([x2, x4])
    x4 = layers.Dense(256)(x4)
    layers.Dropout(0.2)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.add([x1, x4])
    x4 = classifier(x4)
    return Model(inputs=x, outputs=x4)


def generator():
    while True:
        x, y = [], []
        for _ in range(96):

            label = np.random.randint(10)
            length = len(train_group[label])

            v1 = np.random.randint(length)
            v2 = np.random.randint(length)
            v3 = np.random.randint(length)
            v4 = np.random.randint(length)

            rate = np.random.randint(1, 11, 4)
            rate = rate / float(sum(rate))
            a, b, c, d = [float(i) for i in rate]
            x.append(
                a * train_group[label][v1]
                + b * train_group[label][v2]
                + c * train_group[label][v3]
                + d * train_group[label][v4]
            )

            lb = np.zeros(10)
            lb[label] = 1
            y.append(lb)

        x = np.array(x)
        y = np.array(y)
        yield x, y


dataGen = generator()

print("Now to train HR model")
hr = HR()
hr.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=lr_schedule(0)),
    metrics=["accuracy"],
)
hr.fit_generator(
    dataGen,
    steps_per_epoch=500,
    epochs=80,
    verbose=1,
    validation_data=(fx_test, y_test),
    callbacks=callbacks,
)


classifier.evaluate(fx_test, y_test)

hr.evaluate(fx_test, y_test)
