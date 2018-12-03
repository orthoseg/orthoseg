# -*- coding: utf-8 -*-
"""
Unet code based on https://github.com/ternaus/kaggle_dstl_submission/tree/master/src

@author: Pieter Roggemans
"""

from __future__ import division

import numpy as np
import keras as kr

#from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import h5py

from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy

import datetime
import os

import random
import threading

from keras.models import model_from_json

'''
img_rows = 112          # -> renamed to input_height
img_cols = 112          # -> renamed to input_width
num_channels = 16       # -> renamed to n_channels
num_mask_channels = 1   # -> renamed to n_classes
'''

smooth = 1e-12

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_model(input_width=256, input_height=256, n_channels=3, n_classes=1,
              loss_mode='binary_crossentropy', learning_rate=1e-4,
              init_model_weights: bool = False, pretrained_weights_filepath: str = None):
    
    inputs = kr.layers.Input((input_width, input_height, n_channels))
    conv1 = kr.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = kr.layers.normalization.BatchNormalization(axis=-1)(conv1)
    #conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = kr.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = kr.layers.normalization.BatchNormalization(axis=-1)(conv1)
    #conv1 = kr.layers.advanced_activations.ELU()(conv1)
    pool1 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = kr.layers.normalization.BatchNormalization(axis=-1)(conv2)
    #conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = kr.layers.normalization.BatchNormalization(axis=-1)(conv2)
    #conv2 = kr.layers.advanced_activations.ELU()(conv2)
    pool2 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = kr.layers.normalization.BatchNormalization(axis=-1)(conv3)
    #conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    conv3 = kr.layers.normalization.BatchNormalization(axis=-1)(conv3)
    #conv3 = kr.layers.advanced_activations.ELU()(conv3)
    pool3 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = kr.layers.normalization.BatchNormalization(axis=-1)(conv4)
    #conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv4)
    conv4 = kr.layers.normalization.BatchNormalization(axis=-1)(conv4)
    #conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = kr.layers.normalization.BatchNormalization(axis=-1)(conv5)
    #conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = kr.layers.normalization.BatchNormalization(axis=-1)(conv5)
    #conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = kr.layers.Concatenate(axis=3)([kr.layers.UpSampling2D(size=(2, 2))(conv5), conv4])
    #up6 = merge([kr.layers.UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = kr.layers.normalization.BatchNormalization(axis=-1)(conv6)
    #conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = kr.layers.normalization.BatchNormalization(axis=-1)(conv6)
    #conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = kr.layers.Concatenate(axis=3)([kr.layers.UpSampling2D(size=(2, 2))(conv6), conv3])
    #up7 = kr.layers.merge([kr.layers.UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = kr.layers.normalization.BatchNormalization(axis=-1)(conv7)
    #conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = kr.layers.normalization.BatchNormalization(axis=-1)(conv7)
    #conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = kr.layers.Concatenate(axis=3)([kr.layers.UpSampling2D(size=(2, 2))(conv7), conv2])
    #up8 = kr.layers.merge([kr.layers.UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = kr.layers.normalization.BatchNormalization(axis=-1)(conv8)
    #conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = kr.layers.normalization.BatchNormalization(axis=-1)(conv8)
    #conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = kr.layers.Concatenate(axis=3)([kr.layers.UpSampling2D(size=(2, 2))(conv8), conv1])
    #up9 = kr.layers.merge([kr.layers.UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = kr.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = kr.layers.normalization.BatchNormalization(axis=-1)(conv9)
    #conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = kr.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)
    #crop9 = kr.layers.Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = kr.layers.normalization.BatchNormalization(axis=-1)(conv9)
    #conv9 = keras.layers.advanced_activations.ELU()(conv9)
    
    # TODO: crop9 vervangen door conv9 als batnormalisation terug aan
    conv10 = kr.layers.Conv2D(n_classes, 1, activation='sigmoid')(conv9)

    model = kr.models.Model(inputs=inputs, outputs=conv10)
    
    if loss_mode == "jaccard_coef_loss":
        loss_func = jaccard_coef_loss
    elif loss_mode == "binary_crossentropy":
        loss_func = 'binary_crossentropy'
    else:
        raise Exception(f"Unknown loss function: {loss_mode}")

    model.compile(optimizer=Nadam(lr=learning_rate), loss=loss_func, metrics=['binary_crossentropy', jaccard_coef_int])

    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

'''
def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])
    return X_batch, y_batch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb

        yield X_batch, y_batch[:, :, 16:16 + img_rows - 32, 16:16 + img_cols - 32]


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model
'''

'''
if __name__ == '__main__':
    data_path = '../data'
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask'])[:, 1]
    y_train = np.expand_dims(y_train, 1)
    print(y_train.shape)

    train_ids = np.array(f['train_ids'])

    batch_size = 128
    nb_epoch = 50

    history = History()
    callbacks = [
        history,
    ]

    suffix = 'structures_3_'
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.fit_generator(batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                        nb_epoch=nb_epoch,
                        verbose=1,
                        samples_per_epoch=batch_size * 400,
                        callbacks=callbacks,
                        nb_worker=8
                        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)

    suffix = 'structures_4_'
    model.compile(optimizer=Nadam(lr=1e-4), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.fit_generator(
        batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
        nb_epoch=nb_epoch,
        verbose=1,
        samples_per_epoch=batch_size * 400,
        callbacks=callbacks,
        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)
    f.close()
'''
