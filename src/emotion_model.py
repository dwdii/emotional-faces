
import csv
import os
import time
from scipy import misc

import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, ZeroPadding2D




def emotion_model_v1(outputClasses, verbose=False):
    """https://www.kaggle.com/somshubramajumdar/digit-recognizer/deep-convolutional-network-using-keras"""
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 64
    nb_filters_3 = 128
    dropout = 0.25
    #nb_classes = 10
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
    model.add(Dense(128, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model
    
def emotion_model_v2(outputClasses, verbose=False):
    """https://www.kaggle.com/somshubramajumdar/digit-recognizer/deep-convolutional-network-using-keras"""
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 64
    nb_filters_3 = 128
    dropout = 0.25
    #nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
    
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))    

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model  

def emotion_model_v3(outputClasses, verbose=False):
    """https://www.kaggle.com/somshubramajumdar/digit-recognizer/deep-convolutional-network-using-keras"""
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 64
    nb_filters_3 = 128
    nb_filters_4 = 128
    dropout = 0.25
    #nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
    
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))    

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model        
    
def emotion_model_v4(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 64
    nb_filters_3 = 128
    nb_filters_4 = 128
    dropout = 0.25
    #nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
    
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))    

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model       

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def run_network(data, model, epochs=20, batch=256):
    """

    :param data: X_train, X_test, y_train, y_test
    :param model:
    :param epochs:
    :param batch:
    :return:
    """
    try:
        start_time = time.time()

        history = LossHistory()
        X_train, X_test, y_train, y_test = data

        y_trainC = np_utils.to_categorical(y_train )
        y_testC = np_utils.to_categorical(y_test)
        print y_trainC.shape
        print y_testC.shape

        print 'Training model...'
        model.fit(X_train, y_trainC, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_testC), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_testC, batch_size=16, verbose=0)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses