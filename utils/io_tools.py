#!/usr/bin/env python

import pickle
import os
import numpy as np
from tifffile import imread

def save_pickle(object, path):
    with open(path, "wb") as file:
        pickle.dump(object, file)

def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)
    
def load_data(data_dir):
    masks_dir = os.path.join(data_dir, 'masks')
    images_dir = os.path.join(data_dir, 'images')

    images_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    masks_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])

    X = []
    Y = []
    files = []
    for image_file, mask_file in zip(images_files, masks_files):
        image = imread(image_file)
        mask = imread(mask_file)

        # Append the image and mask to the lists
        X.append(image)
        Y.append(mask)
        files.append(image_file)

    return X, Y, files