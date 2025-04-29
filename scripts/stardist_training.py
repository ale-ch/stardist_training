#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tifffile import imread
from datetime import datetime
from csbdeep.utils import Path, download_and_extract_zip_file, normalize

from stardist.matching import matching_dataset
from stardist import fill_label_holes, random_label_cmap, relabel_image_stardist, calculate_extents, gputools_available, _draw_polygons
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)

lbl_cmap = random_label_cmap()


if __name__ == '__main__':

    # Data loading and preparation

    download_and_extract_zip_file(
        url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = 'data',
        verbose   = 1,
    )

    fX = sorted(Path('D:/Softwares/MachineLearning/Akoya/training_dataset/images_256_ok/').glob('*.tif'))
    fY = sorted(Path('D:/Softwares/MachineLearning/Akoya/training_dataset/label_image_256_ok/').glob('*.tif'))
    print(f"found {len(fX)} training images and {len(fY)} training masks")
    assert all(Path(x).name==Path(y).name for x,y in zip(fX,fY))

    fX_small, fY_small = fX[:10], fY[:10]

    X_small = list(map(imread,map(str,fX_small)))
    Y_small = list(map(imread,map(str,fY_small)))

    fX = sorted(Path('D:/Softwares/MachineLearning/Akoya/training_dataset/images_256_ok/').glob('*.tif'))
    fY = sorted(Path('D:/Softwares/MachineLearning/Akoya/training_dataset/label_image_256_ok/').glob('*.tif'))
    assert all(Path(x).name==Path(y).name for x,y in zip(fX,fY))
    print(f"{len(fX)} files found")

    X = list(map(imread,map(str,tqdm(fX))))
    Y = list(map(imread,map(str,tqdm(fY))))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    # Split data into training and validation

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


    # Configure model

    conf = Config2D (
        n_rays       = 32,
        grid         = (2,2),
        n_channel_in = 1,
        train_batch_size = 2,
        train_epochs = 50, 
        train_steps_per_epoch = 50,
    )
    print(conf)
    vars(conf)

    # model = StarDist2D(conf, name='stardist_v3', basedir='models/instance_segmentation_2D_akoya')
  
    model = StarDist2D(conf, name='stardist', basedir='models')
  
    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))

    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")
    else:
        print(f"All good! (object sizes {median_size} fit into field of view {fov} of the neural network)")


    # Train model

    history = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                    epochs=400, steps_per_epoch=50)


    ## Optimize thresholds

    model.optimize_thresholds(X_val[::5], Y_val[::5])


    ## Create training summary

    # convert the history.history dict to a pandas DataFrame:     
    import csv
    import shutil
    import time
    lossData = pd.DataFrame(history.history) 
    model_path = "D:/04_instance_segmentation/stardist/models/instance_segmentation_2D_akoya/"
    model_name = "stardist_v3"
    if os.path.exists(model_path+"/"+model_name+"/Quality Control"):
    shutil.rmtree(model_path+"/"+model_name+"/Quality Control")

    os.makedirs(model_path+"/"+model_name+"/Quality Control")

    # The training evaluation.csv is saved (overwrites the Files if needed). 
    lossDataCSVpath = model_path+'/'+model_name+'/Quality Control/training_evaluation.csv'
    with open(lossDataCSVpath, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['loss','val_loss', 'learning rate'])
    for i in range(len(history.history['loss'])):
        writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])


    #pdf_export(trained=True, augmentation = Use_Data_augmentation, pretrained_model = Use_pretrained_model)

    #Create a pdf document with training summary
