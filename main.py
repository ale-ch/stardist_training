#!/usr/bin/env python

import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tifffile as tiff
import wandb 

from csbdeep.utils import normalize
from utils.get_data import download_data, load_data, train_val_split
from utils.conf_model import configure_model, instantiate_model
from stardist.matching import matching_dataset


def save_pickle(object, path):
    with open(path, "wb") as file:
        pickle.dump(object, file)

def load_pickle(path):
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def random_fliprot(img, mask): 
    axes = tuple(range(img.ndim)) 
    perm = np.random.permutation(axes)
    img = img.transpose(perm) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2)
    return img

def augmenter(img, mask):
    img, mask = random_fliprot(img, mask)
    img = random_intensity_change(img)
    return img, mask


def quality_control(model, test_imgs_dir, test_gt_dir, qc_outdir):
    for img_file, mask_file in zip(os.listdir(test_imgs_dir), os.listdir(test_gt_dir)):
        img_file = os.path.join(test_imgs_dir, img_file)  
        mask_file = os.path.join(test_gt_dir, mask_file)

        img = tiff.imread(img_file)
        mask = tiff.imread(mask_file)

        n_channel = 1 if img.ndim == 2 else img.shape[-1]
        axis_norm = (0, 1)

        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        img = normalize(img, 1, 99.8, axis=axis_norm)

        pred, details = model.predict_instances(img, verbose=True)

        filename = f"{os.path.basename(img_file).split('.')[0]}.jpg"
        output_path = os.path.join(qc_outdir, filename)
        print(f"Processing {img_file} and {mask_file}...")
        mask = tiff.imread(mask_file)
        pred = tiff.imread(img_file)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mask > 0, alpha=0.3, cmap='Blues')
        plt.imshow(pred > 0, cmap='Reds', alpha=0.2)
        print(f"Saving plot to {output_path}...")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}.")



def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Stardist model with specified options.")
    parser.add_argument('--demo', action='store_true', help='Use demo data (download and train on test2 dataset).')
    parser.add_argument('--base_dir', type=str, default='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/', help='Base directory for data and models.')
    parser.add_argument('--model_name', type=str, default='stardist', help='Name of the model to use/save.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model.')
    parser.add_argument('--steps_per_epoch', type=int, default=4, help='Number of steps per epoch during training.')
    parser.add_argument('--augment', action='store_true', help='Augment data during training.')
    parser.add_argument('--val_prop', type=float, default=0.1, help='Proportion of data to use for validation.')
    parser.add_argument('--val_prop_opt', type=float, default=1, help='Proportion of validation data to use for thresholds optimization.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


def main():
    args = _parse_args()

    np.random.seed(args.random_seed)

    run = wandb.init(
        project="stardist-training",
        name=args.model_name,
        config={
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "augment": args.augment,
            "val_prop": args.val_prop,
            "val_prop_opt": args.val_prop_opt,
        }
    )

    if args.demo:
        args.base_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test2'
        data_dir = os.path.join(args.base_dir, 'data')
        train_data_dir = os.path.join(data_dir, 'dsb2018', 'train')
        download_data(data_dir)
    else:
        data_dir = os.path.join(args.base_dir, 'data')
        train_data_dir = os.path.join(data_dir, 'train')
        test_imgs_dir = os.path.join(data_dir, 'test', 'images')
        test_gt_dir = os.path.join(data_dir, 'test', 'masks')
        

    models_dir = os.path.join(args.base_dir, 'models')
    

    print(f"Loading data from {train_data_dir}")
    X, Y = load_data(train_data_dir)
    (X_trn, Y_trn), (X_val, Y_val) = train_val_split(X, Y, val_prop=args.val_prop)

    conf = configure_model()
    model = instantiate_model(conf, models_dir, args.model_name)

    qc_outdir = os.path.join(args.base_dir, 'models', args.model_name, 'quality_control')
    os.makedirs(qc_outdir, exist_ok=True)

    print("Training model")
    if args.augment:
        history = model.train(
            X_trn, 
            Y_trn,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            augmenter=augmenter,
            seed=args.random_seed,
        )
    else:
        history = model.train(
            X_trn, 
            Y_trn,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.random_seed,
        )

    # Wandb log training metrics
    for epoch in range(args.epochs):
        for metric, values in history.history.items():
            if epoch < len(values):
                run.log({metric: values[epoch], "epoch": epoch})

    print("Training complete")

    print("Optimizing thresholds")
    size = int(len(X_val) * args.val_prop_opt)
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]
    model.optimize_thresholds(X_val_opt, Y_val_opt)
    print("Optimization complete")

    print("Evaluation")
    print("Predicting instances")
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in X_val]
    print(f"Prediction complete. Length of Y_val_pred: {len(Y_val_pred)}")

    taus = np.linspace(0.1, 0.9, 9)
    print(f"Taus: {taus}")
    print(f"Matching dataset")
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in taus]

    # Plot metrics
    fig1, ax1 = plt.subplots(figsize=(7,5))
    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()
    plt.close(fig1)

    # Plot counts
    fig2, ax2 = plt.subplots(figsize=(7,5))
    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()
    plt.close(fig2)

    wandb.log({
        "metrics_plot": wandb.Image(fig1),
        "counts_plot": wandb.Image(fig2)
    })

    wandb.finish()

    quality_control(model, test_imgs_dir, test_gt_dir, qc_outdir)



if __name__ == '__main__':
    main()
