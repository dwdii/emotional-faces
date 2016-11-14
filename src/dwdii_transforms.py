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
import csv
import os
import sys

from scipy import misc

import numpy as np

from scipy import misc
from scipy import ndimage
import cv2

def load_training_metadata(metadataFile):
    # Load the existing CSV so we can skip what we've already worked on
    emoDict = {}
    with open(metadataFile, 'r') as csvfile:
        emoCsv = csv.reader(csvfile)
        headers = emoCsv.next()
        for row in emoCsv:
            emoDict[row[1]] = row[2]

    return emoDict

def emotionNumerics():
    emoNdx = {}
    emoNdx["anger"] = 0
    emoNdx["disgust"] = 1
    emoNdx["neutral"] = 2
    emoNdx["happiness"] = 3
    emoNdx["surprise"] = 4
    emoNdx["fear"] = 5
    emoNdx["sadness"] = 6
    return emoNdx

def numericEmotions():
    emoNdx = emotionNumerics()
    ndxEmo = {}
    for k in emoNdx:
        ndxEmo[emoNdx[k]] = k

    return ndxEmo

def load_data(metadataFile, imagesPath, categories = emotionNumerics(), verbose=True, verboseFreq = 200, maxData = None, imgSize = (350, 350)):
    """Helper function to load the training/test data"""

    # Load the CSV meta data
    emoMetaData = load_training_metadata(metadataFile)
    total = len(emoMetaData)
    ndx = 0.0

    if maxData is not None:
        total = maxData

    # Allocate containers for the data
    X_data = np.zeros([total, 350, 350])
    Y_data = np.zeros([total, 1], dtype=np.int8)

    # load the image bits based on what's in the meta data
    for k in emoMetaData.keys():
        filepath = os.path.join(imagesPath, k)

        if verbose and ndx % verboseFreq == 0:
            msg = "{0:f}: {1}\r\n".format(ndx/total, k)
            sys.stdout.writelines(msg )

        img = misc.imread(filepath)

        if img.shape == imgSize:
            # Only accept images that are the appropriate size
            X_data[ndx] = img

            rawEmotion = emoMetaData[k]
            emotionKey = rawEmotion.lower()
            emotionNdx = categories[emotionKey]
            #Y_data = np.append(Y_data, emotionNdx, axis=0)
            Y_data[ndx] = emotionNdx

            ndx += 1

        if maxData is not None and maxData <= ndx:
            break

    X_data = X_data.astype('float32')

    return X_data, Y_data

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