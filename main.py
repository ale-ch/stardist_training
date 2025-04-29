#!/usr/bin/env python

import os
from scripts.get_data import download_data, organize_data
from scripts.conf_model import configure_model, instantiate_model


if __name__ == '__main__':


    base_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/test2'
    data_dir = os.path.join(base_dir, 'data')
    train_data_dir = os.path.join(base_dir, 'data', 'dsb2018', 'train')


    # '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/test2/data/dsb2018/train'
    
    print("Downlading data")
    download_data(data_dir)
    print("Data downloaded")

    (X_trn, Y_trn), (X_val, Y_val) = organize_data(train_data_dir)

    conf = configure_model()

    model = instantiate_model(conf)

    history = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), epochs=5, steps_per_epoch=4)
