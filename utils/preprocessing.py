#!/usr/bin/env python

import numpy as np

from stardist import fill_label_holes
from csbdeep.utils import normalize

def rescale_to_uint8(image):
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)


def preprocess_data(X, Y):
    X = [normalize(x,1,99.8,axis=(0,1)) for x in X]
    Y = [fill_label_holes(y) for y in Y]
    return X, Y