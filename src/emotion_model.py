
import csv
import os
import time
from scipy import misc

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, ZeroPadding2D




def emotion_model_v1(verbose=False):
    """https://www.kaggle.com/somshubramajumdar/digit-recognizer/deep-convolutional-network-using-keras"""
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 64
    nb_filters_3 = 128
    dropout = 0.25
    nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))  # 4096
    model.add(Dense(nb_classes, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model