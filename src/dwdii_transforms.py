#
# Author: Daniel Dittenhafer
#
#     Created: Nov 3, 2016
#
# Description: Image Transformations for faces
#
__author__ = 'Daniel Dittenhafer'
import numpy as np
from scipy import misc
from scipy import ndimage

def reflectY(filepath):
    img = misc.imread(filepath)

    # Rotate
    # a = 15.0 * np.pi / 180.0
    # tx = [[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]

    tx = [[1, 0], [0, -1]]
    offset = [0, 350]
    img2 = ndimage.interpolation.affine_transform(img, tx, offset)

    return img2

def rotate45(filepath):
    img = misc.imread(filepath)


    # Rotate
    a = 45.0 * np.pi / 180.0
    tx = [[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]

    offset = [-5, 140]
    img = ndimage.interpolation.affine_transform(img, tx, offset)

    # Zoom
    tx = [[.75, 0], [0, .75]]
    img = ndimage.interpolation.affine_transform(img, tx)

    return img