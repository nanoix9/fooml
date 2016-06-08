#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import fooml
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_classes = 10
nb_epoch = 12

def create_nn_v1(ds):

    X, y = ds

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    img_rows, img_cols = X.shape[2:4]

    # convert class vectors to binary class matrices
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    #print img_rows, img_cols

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #                  verbose=1, validation_data=(X_test, Y_test))
    #score = model.evaluate(X_test, Y_test, verbose=0)
    return model


def test1():
    foo = fooml.FooML()
    foo.show()

    data_img = 'mnist'
    foo.use_data(data_img)

    model = create_nn_v1(foo.get_train_data(data_img).train)

    #data_img = 'img'
    #foo.load_image(data_img, image_dir=, target=)

    foo.add_nn('clf', model, input=data_img, output='prob',
            train_opt=dict(batch_size=batch_size, nb_epoch=nb_epoch,
                      verbose=1) #, validation_data=(X_test, Y_test))
            )

    foo.evaluate('report', pred='prob')

    foo.show()
    foo.compile()
    foo.run_train()


def main():
    test1()
    return

if __name__ == '__main__':
    main()
