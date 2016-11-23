#
# Author: Daniel Dittenhafer
#
#     Created: Nov 20, 2016
#
# Description: Generative Adversarial Network with Keras
#
# Based on: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
__author__ = 'Daniel Dittenhafer'
import collections
import csv
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
#import tqdm

from keras.models import Model
from keras.optimizers import *
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.normalization import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

def buildGenerativeModel(optimizer = Adam(lr=1e-4), imgSize = (34, 34), verbose = False):
    # Build Generative model ...
    nch = 200
    x, y = imgSize
    xh, yh = x/2, y/2
    g_input = Input(shape=[100])
    H = Dense(nch*xh*yh, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=0)(H)
    H = Activation('relu')(H)
    H = Reshape( [nch, xh, yh] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=0)(H)
    H = Activation('relu')(H)
    H = Convolution2D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=0)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)
    
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    if verbose:
        generator.summary()
    
    return generator

def buildDiscriminativeModel(shape, dropout_rate = 0.25, optimizer = Adam(lr=1e-3), verbose=False):
    # Build Discriminative model ...
    d_input = Input(shape=shape)
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2,activation='softmax')(H)
    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer)
    if verbose:
        print discriminator.summary()
    return discriminator
    
def buildStackedGanModel(generator, discriminator, optimizer = Adam(lr=1e-4), verbose=False):
    # Build stacked GAN model
    gan_input = Input(shape=[100])
    H = generator(gan_input)
    gan_V = discriminator(H)
    theGan = Model(gan_input, gan_V)
    theGan.compile(loss='categorical_crossentropy', optimizer=optimizer)
    if verbose:
        print theGan.summary()
        
    return theGan
    
def plotGenerative(ndx, generator, n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    fig = plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    #plt.show()  
    fileName = "Gan-{0:03d}.png".format(ndx)
    
    # Retry the saving because the jupyter + docker volume mount don't play well.
    saved = False
    retry = 0
    while not saved:
        try:
            fig.savefig(fileName, dpi=fig.dpi)
            plt.close(fig)
            saved = True
        except: 
            print(ndx, "Error saving in plotGenerative, retrying: ", sys.exc_info()[0])
            retry += 1
            saved = False
            if retry > 5:
                plt.close(fig)
                break
            
def makeTrainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        


# Set up our main training loop
def train_for_n(X_train, GAN, generator, discriminator, nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    # set up loss storage vector
    #losses = {"d":[], "g":[]}

    for e in (range(nb_epoch)):   # tqdm
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        makeTrainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        #losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        makeTrainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        #losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            #plot_loss(losses)
            plotGenerative(e, generator)        