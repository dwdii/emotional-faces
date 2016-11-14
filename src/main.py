#
# Author: Daniel Dittenhafer
#
#     Created: Sept 27, 2016
#
# Description: Main entry point for emotional faces data set creator.
#
__author__ = 'Daniel Dittenhafer'

import os
import operator
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from scipy import ndimage
import dwdii_transforms
import ms_emotionapi
#import emotion_model

_key = "2f1d134983cd4fa78f9f758fa1dd4d39"  # Here you have to paste your primary key

def list_files(path, recurse, onlyFilename = False):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        filepath = os.path.join(path, name)
        if os.path.isfile(filepath):

            if(onlyFilename):
                filepath = name

            files.append(filepath)
        elif recurse:
            files += list_files(filepath, recurse)

    return files


def labelFaces():
    """Function to iterate through a directory of images (faces) and call Microsoft Emotion API to determine
    the label for the face."""

    path = "C:\Code\Python\emotional-faces\data\Resize"
    emoFile = 'facialEmos.csv'

    theFiles = list_files(path, True)
    print(len(theFiles))

    # Load the existing CSV so we can skip what we've already worked on
    emoDict = {}
    with open(emoFile, 'r') as csvfile:
        emoCsv = csv.reader(csvfile)
        for row in emoCsv:
            emoDict[row[1]] = row[2]

    waitSecs = 60 / 20.0
    with open(emoFile, 'ab') as csvfile:
        emoCsv = csv.writer(csvfile, delimiter=',',
                            quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        # If we didn't load anything from the file, write the header line.
        if (0 == len(emoDict)):
            emoCsv.writerow(["id", "file", "emotion"])

        i = 0
        for f in theFiles:
            fileName = os.path.basename(f)
            if not fileName in emoDict:
                rj = ms_emotionapi.detectFacesFromDiskFile(f, _key)
                if 0 < len(rj):
                    scores = rj[0]["scores"]
                    emo = max(scores.iteritems(), key=operator.itemgetter(1))[0]

                    # Save the results
                    print(fileName + ": " + emo)
                    emoCsv.writerow([i, fileName, emo])
                    i += 1

                # throttle so Msft doesn't get mad at us...
                time.sleep(waitSecs)


def transformFaces(srcPath, destPath):

    start = 5924
    end = start + 1 #1500

    theFiles = list_files(srcPath, True)
    print(len(theFiles))

    for f in theFiles[start:end]:
        img = misc.imread(f)

        # 1. Reflect Y
        imgReflectY = dwdii_transforms.reflectY(img)
        dwdii_transforms.saveImg(destPath, "reflectY", f, imgReflectY)

        # 2. Rotate 10
        imgRotate5 = dwdii_transforms.rotate5(img)
        dwdii_transforms.saveImg(destPath, "rotate5", f, imgRotate5)

        # 3. cvErode
        newImg = dwdii_transforms.cvErode(img)
        dwdii_transforms.saveImg(destPath, "cvErode", f, newImg)

        # 4. cvDilate
        newImg = dwdii_transforms.cvDilate(img)
        dwdii_transforms.saveImg(destPath, "cvDilate", f, newImg)

        # 5. cvMedianBlur
        newImg = dwdii_transforms.cvMedianBlur(img)
        dwdii_transforms.saveImg(destPath, "cvMedianBlur", f, newImg)

        # 6. cvExcessiveSharpening
        newImg = dwdii_transforms.cvExcessiveSharpening(img)
        dwdii_transforms.saveImg(destPath, "cvExcessiveSharpening", f, newImg)

        # 7. cvBlurMotion1
        newImg = dwdii_transforms.cvBlurMotion1(img)
        dwdii_transforms.saveImg(destPath, "cvBlurMotion1", f, newImg)

        # 8. cvBlurMotion2
        newImg = dwdii_transforms.cvBlurMotion2(img)
        dwdii_transforms.saveImg(destPath, "cvBlurMotion2", f, newImg)

        # 9. cvEdgeEnhancement
        newImg = dwdii_transforms.cvEdgeEnhancement(img)
        dwdii_transforms.saveImg(destPath, "cvEdgeEnhancement", f, newImg)

        # 10. cvDilate2
        newImg = dwdii_transforms.cvDilate2(img)
        dwdii_transforms.saveImg(destPath, "cvDilate2", f, newImg)

        plt.imshow(newImg, cmap=cm.gray)
        plt.show()
        print "done"

def compareFolders():
    path = "C:\Code\Python\emotional-faces\data\Resize"
    path2 = "C:\Code\Other\\facial_expressions\images"

    efrFiles = list_files(path, True, True)
    feiFiles = list_files(path2, True, True)

    efrMissingInFei = []
    for f in efrFiles:
        if f not in feiFiles:
            print f
            efrMissingInFei.append(f)

    #print efrMissingInFei

    feiMissingInEfr = []
    for f in feiFiles:
        if f not in efrFiles:
            print f
            feiMissingInEfr.append(f)

    #print feiMissingInEfr

def compareLegendAndFiles():
    path2 = "C:\Code\Other\\facial_expressions\images"
    emoFile = 'C:\Code\Other\\facial_expressions\data\legend.csv'
    feiFiles = list_files(path2, True, True)
    waitSecs = 20

    # Load the existing CSV so we can skip what we've already worked on
    emoDict = {}
    with open(emoFile, 'r') as csvfile:
        emoCsv = csv.reader(csvfile)
        for row in emoCsv:
            emoDict[row[1]] = row[2]

    for f in feiFiles:
        if f not in emoDict:
            print f

            filepath = os.path.join(path2, f)
            rj = detectFacesFromDiskFile(filepath, _key)
            if 0 < len(rj):
                scores = rj[0]["scores"]
                emo = max(scores.iteritems(), key=operator.itemgetter(1))[0]

                # Save the results
                #print(fileName + ": " + emo)
                print "dwdii," + f + "," + emo

            # throttle so Msft doesn't get mad at us...
            time.sleep(waitSecs)

#def imageTransformPipeline():

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def main():
    """Our main function."""

    destPath = "C:\Code\Python\emotional-faces\data\\transformed"
    srcPath = "C:\Code\Python\emotional-faces\data\Resize"

    #labelFaces()

   #print "OpenCV version: " + cv2.__version__

    #transformFaces(srcPath, destPath)

    #compareFolders()
    #compareLegendAndFiles()

    X_data, Y_data = dwdii_transforms.load_data("C:\Code\Other\\facial_expressions\data\legend.csv",
                               "C:\Code\Other\\facial_expressions\images", maxData = 100, verboseFreq = 1)

    print X_data.shape

    to_categorical(Y_data)

    print "Done"


# This is the main of the program.
if __name__ == "__main__":
    main()