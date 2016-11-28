
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

def emotion_model_v3_1(outputClasses, verbose=False):
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
    #model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(1, 350, 350)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))    

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model    

def emotion_model_v3_2(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 3
    nb_filters_1 = 32
    nb_filters_2 = 32
    nb_filters_3 = 64
    nb_filters_4 = 128
    dropout = 0.25
    #nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    #model.add(ZeroPadding2D((1, 1), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(1, 350, 350)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))    

    #model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
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
    nb_filters_4 = 256
    nb_filters_5 = 256
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
    model.add(Convolution2D(nb_filters_5, nb_conv, nb_conv, activation="relu"))
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
    
def emotion_model_v5(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 20
    nb_filters_1 = 32
    #nb_filters_2 = 64
    #nb_filters_3 = 128
    #nb_filters_4 = 256
    #nb_filters_5 = 512
    #dropout = 0.25
    #nb_classes = 10
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((5, 5), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((10, 10)))
    #model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    #model.add(MaxPooling2D(strides=(2, 2)))
 
    model.add(Flatten())
    #model.add(Dropout(0.25))
    #model.add(Dense(nb_filters_5, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model    

def emotion_model_v6(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 30 # up from 20 to 30
    nb_filters_1 = 32
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((5, 5), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((10, 10)))
    #model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    #model.add(MaxPooling2D(strides=(2, 2)))
 
    model.add(Flatten())
    #model.add(Dropout(0.25))
    #model.add(Dense(nb_filters_5, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model       
    
def emotion_model_v7(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 40 # up from 30 to 40
    nb_filters_1 = 32
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((5, 5), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))

    #model.add(ZeroPadding2D((10, 10)))
    #model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    #model.add(MaxPooling2D(strides=(2, 2)))
 
    model.add(Flatten())
    #model.add(Dropout(0.25))
    #model.add(Dense(nb_filters_5, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model     

def emotion_model_v8(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 30 #  Back to 30 from 40
    nb_filters_1 = 32
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((5, 5), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(5, 5))) # 5,5 from 2,2

    #model.add(ZeroPadding2D((10, 10)))
    #model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
    #model.add(MaxPooling2D(strides=(2, 2)))
 
    model.add(Flatten())
    #model.add(Dropout(0.25))
    #model.add(Dense(nb_filters_5, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model     

def emotion_model_v9(outputClasses, verbose=False):
    nb_pool = 2
    nb_conv = 30 # up from 20 to 30
    nb_filters_1 = 32
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(ZeroPadding2D((5, 5), input_shape=(1, 350, 350), ))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
    #model.add(MaxPooling2D(strides=(2, 2)))

    model.add(ZeroPadding2D((5, 5)))
    model.add(Convolution2D(32, nb_conv, nb_conv, activation="relu"))
    model.add(MaxPooling2D(strides=(2, 2)))
 
    model.add(Flatten())
    #model.add(Dropout(0.25))
    #model.add(Dense(nb_filters_5, activation="relu"))
    model.add(Dense(outputClasses, activation="softmax"))

    if verbose:
        print (model.summary())

    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model       

def cnn_model_jhamski(outputClasses, input_shape=(3, 150, 150), verbose=False):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputClasses))
    model.add(Activation('softmax'))
    
    if verbose:
        print (model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def run_network(data, model, epochs=20, batch=256, verbosity=2):
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
                  validation_data=(X_test, y_testC), verbose=verbosity)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_testC, batch_size=16, verbose=0)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses