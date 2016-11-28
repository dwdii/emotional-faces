#
# Author: Daniel Dittenhafer
#
#     Created: Nov 3, 2016
#
# Description: Image Transformations for faces
#
# An example of each transformation can be found here: https://github.com/dwdii/emotional-faces/tree/master/data/transformed
#
__author__ = 'Daniel Dittenhafer'
import collections
import csv
import os
import random
import sys

from scipy import misc

import numpy as np

from scipy import misc
from scipy import ndimage
import cv2

def load_training_metadata(metadataFile, balanceViaRemoval = False, verbose=False):
    # Load the existing CSV so we can skip what we've already worked on
    emoDict = {}
    emoCounts = collections.defaultdict(int)
    with open(metadataFile, 'r') as csvfile:
        emoCsv = csv.reader(csvfile)
        headers = emoCsv.next()
        for row in emoCsv:
            emoLower = row[2].lower()
            emoDict[row[1]] = emoLower
            emoCounts[emoLower] += 1

    if verbose:
        print "Before Balancing"
        print "----------------"
        for e in emoCounts:
            print e, emoCounts[e]
            
    if balanceViaRemoval:
        balanaceViaRemoval(emoCounts, emoDict)

    if verbose:
        print
        print "After Balancing"
        print "----------------"
        for e in emoCounts:
            print e, emoCounts[e]
        
    return emoDict


def balanaceViaRemoval(emoCounts, emoDict, depth = 0):

    if(depth >= 2):
        return

    # First get mean items per category
    sum = len(emoDict)
    avgE = sum / len(emoCounts)

    # Determine categories for balancing.
    toBeBalanced = []
    for e in emoCounts.keys():
        if emoCounts[e] > avgE * 1.50:
            toBeBalanced.append(e)

    # iterate over categories to be balanced and do balancing.
    for b in toBeBalanced:
        candidatesForRemoval = []
        for f in emoDict.keys():
            if emoDict[f] == b:
                candidatesForRemoval.append(f)

        random.shuffle(candidatesForRemoval)
        candidatesForRemoval = candidatesForRemoval[avgE:]
        for c in candidatesForRemoval:
            del emoDict[c]

        emoCounts[b] = avgE

    balanaceViaRemoval(emoCounts, emoDict, depth + 1)


def emotionNumerics():
    emoNdx = {}
    emoNdx["anger"] = 0
    emoNdx["disgust"] = 1
    emoNdx["neutral"] = 2
    emoNdx["happiness"] = 3
    emoNdx["surprise"] = 4
    emoNdx["fear"] = 5
    emoNdx["sadness"] = 6
    emoNdx["contempt"] = 7
    return emoNdx

def numericEmotions():
    emoNdx = emotionNumerics()
    ndxEmo = {}
    for k in emoNdx:
        ndxEmo[emoNdx[k]] = k

    return ndxEmo

def load_data(metadataFile, imagesPath, categories = emotionNumerics(), verbose=True, verboseFreq = 200, maxData = None, imgSize = (350, 350), imgResize = None, theseEmotions = None):
    """Helper function to load the training/test data"""

    # Load the CSV meta data
    emoMetaData = load_training_metadata(metadataFile, True)
    total = len(emoMetaData)
    ndx = 0.0

    x, y = imgSize
    if imgResize is not None:
        x, y = imgResize
        
    if maxData is not None:
        total = maxData

    # Allocate containers for the data
    X_data = np.zeros([total, x, y])
    Y_data = np.zeros([total, 1], dtype=np.int8)

    # load the image bits based on what's in the meta data
    for k in emoMetaData.keys():
        
        if theseEmotions is None or emoMetaData[k] in theseEmotions:

            # Verbose status
            if verbose and ndx % verboseFreq == 0:
                msg = "{0:f}: {1}\r\n".format(ndx/total, k)
                sys.stdout.writelines(msg )
            
            # Load the file
            filepath = os.path.join(imagesPath, k)
            img = misc.imread(filepath, flatten = False) # flatten = True? 

            # Only accept images that are the appropriate size
            if img.shape == imgSize:
                
                # Resize if desired.
                if imgResize is not None:
                    img = misc.imresize(img, imgResize)
                    
                X_data[ndx] = img

                rawEmotion = emoMetaData[k]
                emotionKey = rawEmotion.lower()
                emotionNdx = categories[emotionKey]
                
                Y_data[ndx] = emotionNdx

                ndx += 1
                
        if maxData is not None and maxData <= ndx:
            break
            
    Y_data = Y_data[:ndx]
    X_data = X_data[:ndx]
    
    X_data = X_data.astype('float32')
    X_data /= 255.0

    return X_data, Y_data
    
def trainingTestSplit(xData, yData, ratio, verbose=False):
    
    # Split the data into Training and Test sets
    dataLen = xData.shape[0]
    trainNdx = int(dataLen * ratio)
    if verbose:
        print trainNdx
        
    X_train, X_test = np.split(xData, [trainNdx])
    Y_train, Y_test = np.split(yData, [trainNdx])
    
    X_train = X_train.reshape(X_train.shape[0], 1, xData.shape[1], xData.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, xData.shape[1], xData.shape[2])    
    
    if verbose:
        print "X_train: " + str(X_train.shape)
        print "X_test: " + str( X_test.shape)

        print "Y_train: " + str( Y_train.shape)
        print "Y_test: " + str(  Y_test.shape) 
    
    return X_train, X_test, Y_train, Y_test

def saveImg(destinationPath, prefix, filepath, imgData):
    """Helper function to enable a common way of saving the transformed images."""
    fileName = os.path.basename(filepath)
    destFile = destinationPath + "\\" + prefix + "-" + fileName
    misc.imsave(destFile, imgData)

def reflectY(img):

    tx = [[1, 0], [0, -1]]
    offset = [0, 350]
    img2 = ndimage.interpolation.affine_transform(img, tx, offset)

    return img2

def rotate5(img):

    img2 = cv2.resize(img, (385, 385), interpolation=cv2.INTER_AREA)

    # Rotate
    a = 5.0 * np.pi / 180.0
    tx = [[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]

    offset = [-10,25]  # [right, down] negatives go other direction
    img2 = ndimage.interpolation.affine_transform(img2, tx, offset)

    # Zoom
    img2 = img2[10:360, 10:360]

    return img2


def cvErode(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)


    return img_erosion


def cvDilate(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=1)

    return img_dilation

def cvDilate2(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=2)

    return img_dilation

def cvMedianBlur(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""

    img2 = cv2.medianBlur(img, 7 )

    return img2


def cvExcessiveSharpening(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec22/Sharpening"""
    kernel_sharpen_1 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    img2 = cv2.filter2D(img, -1, kernel_sharpen_1)
    return img2

def cvEdgeEnhancement(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec22/Sharpening"""
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0

    img2 = cv2.filter2D(img, -1, kernel_sharpen_3)
    return img2

def cvBlurMotion1(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec23/Embossing"""
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    img2 = cv2.filter2D(img, -1, kernel_motion_blur)
    return img2

def cvBlurMotion2(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec23/Embossing"""
    size = 30
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    img2 = cv2.filter2D(img, -1, kernel_motion_blur)
    return img2