#!/usr/bin/env python

import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tifffile as tiff
import json
import wandb 

from cellpose.utils import masks_to_outlines
from skimage.transform import rescale
from stardist import fill_label_holes
from csbdeep.utils import normalize
from utils.get_data import download_data, load_data, train_test_val_split
from utils.conf_model import configure_model, instantiate_model
from stardist.matching import matching_dataset


def save_pickle(object, path):
    with open(path, "wb") as file:
        pickle.dump(object, file)

def load_pickle(path):
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def normalize_image(image):
    """Normalize each channel of the image independently to [0, 255] uint8."""
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    return scaled_image.astype(np.uint8)


def rescale_to_uint8(image):
    # Rescale downsampled image to uint8
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    image = (image - min_val) / (max_val - min_val) * 255
    image = image.astype(np.uint8)

    return image


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

def default_augmenter(img, mask):
    img, mask = random_fliprot(img, mask)
    img = random_intensity_change(img)
    return img, mask


def plot_metrics(Y_val, Y_val_pred, taus):
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

    return fig1, fig2



def save_config_to_json(model_name, epochs, steps_per_epoch, learning_rate, augment,
                        val_prop, val_prop_opt, random_seed, files_train, files_test,
                        file_path="config.json"):
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": learning_rate,
        "augment": augment,
        "val_prop": val_prop,
        "val_prop_opt": val_prop_opt,
        "random_seed": random_seed,
        "files_train": files_train,
        "files_test": files_test
    }

    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)


def quality_control(model, X_test, Y_test, files_test, qc_outdir):
    for img, mask, file in zip(X_test, Y_test, files_test):

        filename = os.path.basename(file)
        output_path = os.path.join(qc_outdir, filename)

        img = normalize(img, 1, 99.8, axis=(0, 1))

        pred, details = model.predict_instances(img, verbose=True)

        outlines_test = masks_to_outlines(mask)
        outlines_test = np.array(outlines_test, dtype="float32")

        outlines_pred = masks_to_outlines(pred)
        outlines_pred = np.array(outlines_pred, dtype="float32")

        output_array = np.stack([img, mask, pred, outlines_test, outlines_pred], axis=0).astype("float32")
        output_array = normalize_image(output_array)
        output_array = np.array([
            rescale(output_array[0], scale=1, anti_aliasing=True), 
            rescale(output_array[1], scale=1, anti_aliasing=False),
            rescale(output_array[2], scale=1, anti_aliasing=False),
            rescale(output_array[3], scale=1, anti_aliasing=False),
            rescale(output_array[4], scale=1, anti_aliasing=False),
        ])
        output_array = rescale_to_uint8(output_array)

        pixel_microns = 0.34533768547788
        tiff.imwrite(
            output_path, 
            output_array, 
            imagej=True, 
            resolution=(1/pixel_microns, 1/pixel_microns), 
            metadata={'unit': 'um', 'axes': 'CYX', 'mode': 'composite'}
        )


def preprocess_data(X, Y):
    axis_norm = (0,1)   # normalize channels independently

    X = [normalize(x,1,99.8,axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in Y]

    return X, Y


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Stardist model with specified options.")
    parser.add_argument('--demo', action='store_true', help='Use demo data (download and train on test2 dataset).')
    parser.add_argument('--base_dir', 
                        type=str, 
                        default='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/', 
                        help='Base directory for the pipeline.'
    )
    parser.add_argument('--data_dir', 
                        type=str, 
                        help="Directory containing the data. The folder should contain subdirectories 'images' and 'masks'."
    )
    parser.add_argument('--pretrained', type=str, default='2D_versatile_fluo', help='Name of pretrained model to use.')
    parser.add_argument('--model_name', type=str, default='stardist', help='Name of the model to use/save.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model.')
    parser.add_argument('--steps_per_epoch', type=int, default=4, help='Number of steps per epoch during training.')
    parser.add_argument('--augment', action='store_true', help='Augment data during training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the model.')
    parser.add_argument('--test_prop', type=float, default=0.1, help='Proportion of hold out data.')
    parser.add_argument('--val_prop', type=float, default=0.1, help='Proportion of data to use for validation.')
    parser.add_argument('--val_prop_opt', type=float, default=1, help='Proportion of validation data to use for thresholds optimization.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


def main():
    args = _parse_args()

    np.random.seed(args.random_seed)
    np.random.RandomState(args.random_seed)

    run = wandb.init(
        project="stardist-training",
        name=args.model_name,
        config={
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "learning_rate": args.learning_rate,
            "augment": args.augment,
            "val_prop": args.val_prop,
            "val_prop_opt": args.val_prop_opt,
            "seed": args.random_seed,
        }
    )

    data_dir = os.path.join(args.base_dir, 'data')
    
    if args.demo:
        download_data(args.data_dir)
        # downloaded data goes to: data_dir/dsb2018/train and data_dir/dsb2018/test
    else:
        data_dir = args.data_dir
        

    seed_dir = os.path.join(args.base_dir, 'runs', str(args.random_seed))
    models_dir = os.path.join(seed_dir, 'models')
    cur_model_dir = os.path.join(models_dir, args.model_name)
    qc_outdir = os.path.join(cur_model_dir, 'quality_control')

    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    # os.makedirs(cur_model_dir, exist_ok=True)
    os.makedirs(qc_outdir, exist_ok=True)
    

    print(f"Loading data from {data_dir}")
    X, Y, files = load_data(data_dir)

    X, Y = preprocess_data(X, Y)

    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val), (files_train, files_test) = train_test_val_split(
        X, 
        Y,
        files,
        test_prop=args.test_prop, 
        val_prop=args.val_prop,
        seed=args.random_seed
    )


    save_config_to_json(args.model_name, args.epochs, args.steps_per_epoch, args.learning_rate,
                    args.augment, args.val_prop, args.val_prop_opt, args.random_seed,
                    files_train, files_test, file_path=os.path.join(cur_model_dir, "training_config.json"))



    size = int(len(X_val) * args.val_prop_opt)
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]


    conf = configure_model() if args.pretrained is None else None
    augmenter = default_augmenter if args.augment else None

    model = instantiate_model(
        models_dir,
        args.model_name, 
        conf, 
        args.learning_rate, 
        args.pretrained
    )


    print("Training model")

    history = model.train(
        X_train, 
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        augmenter=augmenter,
        seed=args.random_seed,
    )

    # Wandb log training metrics
    for epoch in range(args.epochs):
        for metric, values in history.history.items():
            if epoch < len(values):
                run.log({metric: values[epoch], "epoch": epoch})

    print("Training complete")

    print("Optimizing thresholds")
    model.optimize_thresholds(X_val_opt, Y_val_opt)
    print("Optimization complete")

    print("Evaluation")
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in X_val]
    taus = np.linspace(0.1, 0.9, 9)
    fig1, fig2 = plot_metrics(Y_val, Y_val_pred, taus)

    quality_control(model, X_test, Y_test, files_test, qc_outdir)  

    wandb.log({
        "metrics_plot": wandb.Image(fig1),
        "counts_plot": wandb.Image(fig2)
    })

    wandb.finish()


if __name__ == '__main__':
    main()
