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


def organize_data(train_data_dir, val_prop=0.15):
    masks_dir = os.path.join(train_data_dir, 'masks')
    images_dir = os.path.join(train_data_dir, 'images')

    fX = sorted(Path(images_dir).glob('*.tif'))
    fY = sorted(Path(masks_dir).glob('*.tif'))
    print(f"found {len(fX)} training images and {len(fY)} training masks")

    X = list(map(imread,map(str,fX)))
    Y = list(map(imread,map(str,fY)))

    print(f"X LEN: {len(X)}")
    print(f"Y LEN: {len(Y)}")

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly

    X = [normalize(x,1,99.8,axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in Y]

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(val_prop * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    return (X_trn, Y_trn), (X_val, Y_val)
