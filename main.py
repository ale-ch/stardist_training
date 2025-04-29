#!/usr/bin/env python

import os
from scripts.get_data import download_data, organize_data
from scripts.conf_model import configure_model, instantiate_model


if __name__ == '__main__':
    base_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/test2'
    data_dir = os.path.join(base_dir, 'data')
    train_data_dir = os.path.join(base_dir, 'data', 'dsb2018', 'train')

    models_dir = os.path.join(base_dir, 'models')
    model_name = 'stardist'
    
    download_data(data_dir)

    (X_trn, Y_trn), (X_val, Y_val) = organize_data(train_data_dir)

    conf = configure_model()

    model = instantiate_model(conf, models_dir, model_name)

    history = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), epochs=5, steps_per_epoch=4)


    print(f"THRESHOLD OPTIMIZATION DATA: {X_val[::5]}")
    print(f"THRESHOLD OPTIMIZATION DATA: {Y_val[::5]}")

    model.optimize_thresholds(X_val[::5], Y_val[::5])
