#!/usr/bin/env python

import os
import argparse
import numpy as np
import wandb
import multiprocessing

from itertools import product

from utils.get_data import download_data, train_test_val_split
from utils.conf_model import configure_model, instantiate_model
from utils.preprocessing import preprocess_data
from utils.quality_control import quality_control, plot_metrics
from utils.data_augmentation import default_augmenter
from utils.io_tools import load_data
from utils.metadata_tools import save_config_to_json


def train_and_evaluate(config):
    model_name = config['model_name']
    base_dir = config['base_dir']
    data_dir = os.path.join(base_dir, 'data') if config['demo'] else config['data_dir']
    seed_dir = os.path.join(base_dir, 'runs', str(config['random_seed']))
    models_dir = os.path.join(seed_dir, 'models')
    cur_model_dir = os.path.join(models_dir, model_name)
    qc_outdir = os.path.join(cur_model_dir, 'quality_control')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(qc_outdir, exist_ok=True)

    if config['demo']:
        download_data(config['data_dir'])

    X, Y, files = load_data(data_dir)
    X, Y = preprocess_data(X, Y)

    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val), (files_train, files_test) = train_test_val_split(
        X, Y, files, test_prop=config['test_prop'], val_prop=config['val_prop'], seed=config['random_seed']
    )

    save_config_to_json(model_name, config['epochs'], config['steps_per_epoch'], config['learning_rate'],
                        config['augment'], config['early_stopping'], config['val_prop'], config['val_prop_opt'], config['random_seed'],
                        files_train, files_test, file_path=os.path.join(cur_model_dir, "training_config.json"))

    size = int(len(X_val) * config['val_prop_opt'])
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]

    conf = configure_model() if config['pretrained'] is None else None
    augmenter = default_augmenter if config['augment'] else None

    model = instantiate_model(models_dir, model_name, conf, config['learning_rate'], config['pretrained'], config['early_stopping'])

    run = wandb.init(
        project="stardist-training",
        name=model_name,
        config=config,
        reinit=True
    )

    history = model.train(
        X_train, 
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=config['epochs'],
        steps_per_epoch=config['steps_per_epoch'],
        augmenter=augmenter,
        seed=config['random_seed'],
    )

    for epoch in range(config['epochs']):
        for metric, values in history.history.items():
            if epoch < len(values):
                run.log({metric: values[epoch], "epoch": epoch})

    model.optimize_thresholds(X_val_opt, Y_val_opt)

    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in X_val]
    taus = np.linspace(0.1, 0.9, 9)
    fig1, fig2 = plot_metrics(Y_val, Y_val_pred, taus)

    quality_control(model, X_test, Y_test, files_test, qc_outdir)

    wandb.log({
        "metrics_plot": wandb.Image(fig1),
        "counts_plot": wandb.Image(fig2)
    })

    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel Stardist training with hyperparameter search.")
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='2D_versatile_fluo')
    parser.add_argument('--test_prop', type=float, default=0.1)
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--val_prop_opt', type=float, default=1.0)
    parser.add_argument('--random_seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs_list', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--steps_list', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--lr_list', nargs='+', type=float, default=[1e-4, 1e-3])
    parser.add_argument('--augment_list', nargs='+', type=str, default=['True', 'False'])
    parser.add_argument('--early_stopping_list', nargs='+', type=str, default=['True', 'False'])
    parser.add_argument('--demo', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    base_config = {
        "demo": args.demo,
        "base_dir": args.base_dir,
        "data_dir": args.data_dir,
        "pretrained": args.pretrained,
        "test_prop": args.test_prop,
        "val_prop": args.val_prop,
        "val_prop_opt": args.val_prop_opt
    }

    param_grid = {
        "epochs": args.epochs_list,
        "steps_per_epoch": args.steps_list,
        "learning_rate": args.lr_list,
        "augment": [x.lower() == 'true' for x in args.augment_list],
        "early_stopping": [x.lower() == 'true' for x in args.early_stopping_list],
        "random_seed": args.random_seeds
    }

    keys, values = zip(*param_grid.items())
    configs = []
    for v in product(*values):
        params = dict(zip(keys, v))
        run_config = base_config.copy()
        run_config.update(params)
        run_config["model_name"] = (
            f"stardist_e{params['epochs']}_s{params['steps_per_epoch']}_"
            f"lr{params['learning_rate']}_aug{params['augment']}_es{params['early_stopping']}_seed{params['random_seed']}"
        )
        configs.append(run_config)

    with multiprocessing.Pool(processes=min(len(configs), multiprocessing.cpu_count())) as pool:
        pool.map(train_and_evaluate, configs)

if __name__ == '__main__':
    main()
