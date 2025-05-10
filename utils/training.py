#!/usr/bin/env python

import os
import tensorflow as tf
import ast
import numpy as np
import wandb
from utils.metadata_tools import save_config_to_json
from utils.get_data import train_test_val_split, download_data
from utils.io_tools import load_data
from utils.conf_model import configure_model, instantiate_model
from utils.data_augmentation import default_augmenter
from utils.preprocessing import preprocess_data, rescale_to_uint8
from utils.quality_control import quality_control, plot_metrics
from itertools import product
from typing import List, Dict, Any
from utils.metadata_tools import get_compact_model_name

def generate_hyperparameter_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    param_grid = {}
    for key, value in config.items():
        if isinstance(value, dict):
            param_grid[key] = generate_hyperparameter_grid(value)
        elif isinstance(value, list):
            param_grid[key] = value
        elif isinstance(value, str) and '(' in value:
            values_tuple = ast.literal_eval(value)
            start, end, step = values_tuple

            if all([isinstance(v, float) for v in values_tuple]) and end - start == step:
                param_grid[key] = list(np.arange(start, end + step, step)[:-1])
            else:
                param_grid[key] = list(np.arange(start, end + step, step))
        else:
            param_grid[key] = [value]

    flat_grid = {}
    for key, value in param_grid.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_grid[f"{key}.{subkey}"] = subvalue
        else:
            flat_grid[key] = value

    keys = flat_grid.keys()
    values = flat_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    final_configs = []
    for combo in combinations:
        config_copy = config.copy()
        for key, value in combo.items():
            if '.' in key:
                main_key, sub_key = key.split('.')
                config_copy[main_key][sub_key] = value
            else:
                config_copy[key] = value
        final_configs.append(config_copy)

    return final_configs


def train_and_evaluate(config: Dict[str, Any]) -> None:
    model_name = get_compact_model_name(config)
    config['model_name'] = model_name

    base_dir = config['base_dir']
    data_dir = os.path.join(base_dir, 'data') if config.get('demo', False) else config['data_dir']
    seed_dir = os.path.join(base_dir, 'runs', str(config['random_seed']))
    models_dir = os.path.join(seed_dir, 'models')
    cur_model_dir = os.path.join(models_dir, model_name)
    qc_outdir = os.path.join(cur_model_dir, 'quality_control')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(qc_outdir, exist_ok=True)

    if config.get('demo', False):
        download_data(config['data_dir'])

    X, Y, files = load_data(data_dir)
    X, Y = preprocess_data(X, Y)

    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val), (files_train, files_test) = train_test_val_split(
        X, Y, files, test_prop=config['test_prop'], val_prop=config['val_prop'], seed=config['random_seed']
    )

    save_config_to_json(config, file_path=os.path.join(cur_model_dir, "training_config.json"))

    size = int(len(X_val) * config['val_prop_opt'])
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]

    architecture_conf = configure_model() if config['pretrained'] is None else None
    augmenter = default_augmenter if config['augment'] else None

    model = instantiate_model(models_dir, model_name, architecture_conf=architecture_conf, config=config)

    run = wandb.init(project="stardist-training", name=model_name, config=config, reinit=True)

    history = model.train(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=int(config['epochs']),
        steps_per_epoch=int(config['steps_per_epoch']),
        augmenter=augmenter,
        seed=int(config['random_seed']),
    )

    for epoch in range(int(config['epochs'])):
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
