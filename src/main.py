#
# Author: Daniel Dittenhafer
#
#     Created: Sept 27, 2016
#
# Description: Main entry point for emotional faces data set creator.
#
__author__ = 'Daniel Dittenhafer'

import os
#import cv2
import requests
import operator
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from scipy import ndimage
import dwdii_transforms


_url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
_key = "2f1d134983cd4fa78f9f758fa1dd4d39"  # Here you have to paste your primary key
_maxNumRetries = 10

def list_files(path, recurse):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        filepath = os.path.join(path, name)
        if os.path.isfile(filepath):
            files.append(filepath)
        elif recurse:
            files += list_files(filepath, recurse)

    return files

# From: https://github.com/Microsoft/Cognitive-Emotion-Python/blob/master/Jupyter%20Notebook/Emotion%20Analysis%20Example.ipynb
def processRequest(json, data, headers, params):
    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        response = requests.request('post', _url, json=json, data=data, headers=headers, params=params)

        if response.status_code == 429:

            print("Message: %s" % (response.json()['error']['message']))

            if retries <= _maxNumRetries:
                time.sleep(1)
                retries += 1
                continue
            else:
                print('Error: failed after retrying!')
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
                result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower():
                    result = response.content
        else:
            print("Error code: %d" % (response.status_code))
            print("Message: %s" % (response.json()['error']['message']))

        break

    return result


def renderResultOnImage(result, img):
    """Display the obtained results onto the input image"""

    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        cv2.rectangle(img, (faceRectangle['left'], faceRectangle['top']),
                      (faceRectangle['left'] + faceRectangle['width'], faceRectangle['top'] + faceRectangle['height']),
                      color=(255, 0, 0), thickness=5)

    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]

        textToWrite = "%s" % (currEmotion)
        cv2.putText(img, textToWrite, (faceRectangle['left'], faceRectangle['top'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)

def detectFacesFromDiskFile(pathToFileInDisk, key):
    # Load raw image file into memory
    #pathToFileInDisk = r'D:\tmp\detection3.jpg'
    with open(pathToFileInDisk, 'rb') as f:
        data = f.read()

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = key
    headers['Content-Type'] = 'application/octet-stream'

    json = None
    params = None

    result = processRequest(json, data, headers, params)

    #if result is not None:
        # Load the original image from disk
        #data8uint = np.fromstring(data, np.uint8)  # Convert string to an unsigned int array
        #img = cv2.cvtColor(cv2.imdecode(data8uint, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        #renderResultOnImage(result, img)

        #ig, ax = plt.subplots(figsize=(15, 20))
        #ax.imshow(img)

    return result

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
                rj = detectFacesFromDiskFile(f, _key)
                if 0 < len(rj):
                    scores = rj[0]["scores"]
                    emo = max(scores.iteritems(), key=operator.itemgetter(1))[0]

                    # Save the results
                    print(fileName + ": " + emo)
                    emoCsv.writerow([i, fileName, emo])
                    i += 1

                # throttle so Msft doesn't get mad at us...
                time.sleep(waitSecs)

def transformFaces():
    destPath = "C:\Code\Python\emotional-faces\data\\transformed"
    srcPath = "C:\Code\Python\emotional-faces\data\Resize"
    start = 1000
    end = start + 1 #1500

    theFiles = list_files(srcPath, True)
    print(len(theFiles))

    for f in theFiles[start:end]:

        img2 = dwdii_transforms.reflectY(f)
        fileName = os.path.basename(f)
        destFile = destPath + "\\reflectY-" + fileName
        misc.imsave(destFile, img2)

        img2 = dwdii_transforms.rotate45(f)
        fileName = os.path.basename(f)
        destFile = destPath + "\\rotate45-" + fileName
        misc.imsave(destFile, img2)

        plt.imshow(img2, cmap=cm.gray)
        plt.show()
        print "done"


def main():
    """Our main function."""

    #labelFaces()

    transformFaces()


# This is the main of the program.
if __name__ == "__main__":
    main()