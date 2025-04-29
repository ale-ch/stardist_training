#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tifffile import imread
from datetime import datetime
from csbdeep.utils import Path, download_and_extract_zip_file, normalize

#from stardist.matching import matching_dataset
#from stardist import fill_label_holes, random_label_cmap, relabel_image_stardist, calculate_extents, gputools_available, _draw_polygons
#from stardist.models import Config2D, StarDist2D, StarDistData2D

from stardist import fill_label_holes, random_label_cmap

np.random.seed(42)

lbl_cmap = random_label_cmap()


if __name__ == '__main__':

    # Data loading and preparation

    download_and_extract_zip_file(
        url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = 'data',
        verbose   = 1,
    )

    masks_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/scripts/data/dsb2018/train/masks'
    images_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/scripts/data/dsb2018/train/images'

    fX = sorted(Path(images_dir).glob('*.tif'))
    fY = sorted(Path(masks_dir).glob('*.tif'))
    print(f"found {len(fX)} training images and {len(fY)} training masks")
    assert all(Path(x).name==Path(y).name for x,y in zip(fX,fY))

    fX_small, fY_small = fX[:10], fY[:10]
   
    # print(f"SHAPES: {fX_small.shape}, {fY_small.shape}")

    print("LOADING FILES")
    X_small = list(map(imread,map(str,fX_small)))
    Y_small = list(map(imread,map(str,fY_small)))

    print(f"SHAPES: {X_small[0].shape}, {Y_small[0].shape}")
  
    X = list(map(imread,map(str,fX)))
    Y = list(map(imread,map(str,fY)))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        ys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in Y]

    print(f"NORMALIZED IMAGES")

    # Split into training and validation

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))


