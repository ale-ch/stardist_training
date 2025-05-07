#!/usr/bin/env python

import os
import numpy as np
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, Path, normalize

from stardist import fill_label_holes



def download_data(target_dir):
    download_and_extract_zip_file(
        url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = target_dir,
        verbose   = 1,
    )


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
        files.append(os.path.basename(image_file))

    return X, Y, files


def train_test_split(X, Y, filenames, val_prop=0.1, seed=42):
    rng = np.random.RandomState(seed)
    ind = rng.permutation(len(X))
    n_test = max(1, int(round(val_prop * len(ind))))
    ind_train, ind_test = ind[:-n_test], ind[-n_test:]
    X_test, Y_test = [X[i] for i in ind_test]  , [Y[i] for i in ind_test]
    X_train, Y_train = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    filenames_train, filenames_test = [filenames[i] for i in ind_train], [filenames[i] for i in ind_test]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_train))
    print('- test:     %3d' % len(X_test))

    return (X_train, Y_train), (X_test, Y_test), (filenames_train, filenames_test)


def train_val_split(X, Y, val_prop=0.15, seed=42):
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly

    X = [normalize(x,1,99.8,axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in Y]

    rng = np.random.RandomState(seed)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(val_prop * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    return (X_trn, Y_trn), (X_val, Y_val)




