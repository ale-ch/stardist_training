#!/usr/bin/env python

import os
import argparse
import numpy as np
import pickle
from scripts.get_data import download_data, load_data, train_val_split
from scripts.conf_model import configure_model, instantiate_model
from stardist.matching import matching_dataset

np.random.seed(42)


def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)

def load_pickle(path):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Deserialize the object from the file
        loaded_data = pickle.load(file)

    return loaded_data



def random_fliprot(img, mask): 
    axes = tuple(range(img.ndim)) 
    perm = np.random.permutation(axes)
    img = img.transpose(perm) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand()>.5:
            img = np.flip(img,axis = ax)
            mask = np.flip(mask,axis = ax)
    return img, mask 


def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2)
    return img


def augmenter(img,mask):
    """Augmentation for image,mask"""
    img, mask = random_fliprot(img, mask)
    img = random_intensity_change(img)
    return img, mask


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Stardist model with specified options.")

    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Use demo data (download and train on test2 dataset).'
    )
    parser.add_argument(
        '--base_dir', 
        type=str,
        default='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/',
        help='Base directory for data and models.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='stardist',
        help='Name of the model to use/save.'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5,
        help='Number of epochs to train the model.'
    )
    parser.add_argument(
        '--steps_per_epoch', 
        type=int, 
        default=4,
        help='Number of steps per epoch during training.'
    )
    parser.add_argument(
        '--augment', 
        action='store_true',
        help='Augment data during training.'
    )
    parser.add_argument(
        '--val_prop', 
        type=float, 
        default=0.1,
        help='Proportion of data to use for validation.'
    )
    
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    if args.demo:
        args.base_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test2'
        data_dir = os.path.join(args.base_dir, 'data')
        train_data_dir = os.path.join(data_dir, 'dsb2018', 'train')
        download_data(data_dir)
    else:
        data_dir = os.path.join(args.base_dir, 'data')
        train_data_dir = os.path.join(data_dir, 'train')

    models_dir = os.path.join(args.base_dir, 'models')


    print(f"Loading data from {train_data_dir}")
    X, Y = load_data(train_data_dir)

    (X_trn, Y_trn), (X_val, Y_val) = train_val_split(X, Y, val_prop=args.val_prop)

    conf = configure_model()
    model = instantiate_model(conf, models_dir, args.model_name)

    print("Training model")
    if args.augment:
        history = model.train(
            X_trn, 
            Y_trn,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            augmenter=augmenter
        )
    else:
        history = model.train(
            X_trn, 
            Y_trn,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch
        )

    print("Training complete")

    print("Optimizing thresholds")
    size = int(len(X_val) * 0.5)
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]

    # model.optimize_thresholds(X_val[::5], Y_val[::5])
    model.optimize_thresholds(X_val_opt, Y_val_opt)
    print("Optimization complete")


    print("Evaluation")

    print("Predicting instances")
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in X_val]
    print(f"Prediction complete. Length of Y_val_pred: {len(Y_val_pred)}")

    # taus = np.linspace(0.2, 0.9, 8)

    taus = [0.7, 0.8, 0.9]

    print(f"Taus: {taus}")

    print(f"Matching dataset")
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in taus]

    # Save the stats to a file

    stats_file = os.path.join(models_dir, f'stats_{args.model_name}.pkl')
    print(f"Saving stats to {stats_file}")
    # save_pickle(stats, stats_file)
    print(f"Saved stats to {stats_file}")

    # print(stats[taus.index(0.8)])

    # stats = [stats[taus.index(t)] for t in taus]



if __name__ == '__main__':
    main()
