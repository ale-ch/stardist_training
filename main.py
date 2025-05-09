#!/usr/bin/env python

import os
import ast
import argparse
import yaml
import json
import numpy as np
import wandb
import multiprocessing
import shutil
import tensorflow as tf
from itertools import product
from typing import Union, List, Dict, Any
from stardist.models import StarDist2D
from stardist.models import Config2D
from utils.get_data import train_test_val_split, download_data
from utils.io_tools import load_data
from utils.conf_model import configure_model
from utils.data_augmentation import default_augmenter
from utils.preprocessing import preprocess_data, rescale_to_uint8
from utils.quality_control import quality_control, plot_metrics
from utils.metadata_tools import save_config_to_json



def configure_model():
    conf = Config2D (
        n_rays       = 32,
        grid         = (2,2),
        n_channel_in = 1,
        train_batch_size = 2,
        train_epochs = 50, 
        train_steps_per_epoch = 50,
    )

    return conf

def get_compact_model_name(config: Dict[str, Any]) -> str:
    """Generate a compact but human-readable model name from config"""
    name_parts = [
        config['model_name'],
        f"e{config['epochs']}",
        f"lr{config['learning_rate']}".replace('.', '').replace('+', ''),
        f"aug{int(config['augment'])}",
        f"seed{config['random_seed']}"
    ]
    
    # Add key hyperparameters if they exist
    if 'early_stopping' in config:
        es = config['early_stopping']
        name_parts.append(f"es{es['patience']}p{es['min_delta']}")
    
    if 'train_reduce_lr' in config:
        lr = config['train_reduce_lr']
        name_parts.append(f"rlr{lr['factor']}p{lr['patience']}")
    
    return "_".join(name_parts)

def convert_to_float(value: Any) -> float:
    """Safely convert any value to float, handling tuples/lists from config"""
    if isinstance(value, (tuple, list)):
        return float(value[0])  # Use first value from range
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {value} to float")

def instantiate_model(
    models_dir: str,
    model_name: str,
    architecture_conf: Dict[str, Any] = None,
    config: Dict[str, Any] = None
) -> StarDist2D:
    """Instantiate StarDist2D model with configuration"""
    cur_model_dir = os.path.join(models_dir, model_name)
    os.makedirs(cur_model_dir, exist_ok=True)
    
    learning_rate = config.get('learning_rate')
    pretrained = config.get('pretrained')
    early_stopping = config.get('early_stopping', {})
    train_reduce_lr = config.get('train_reduce_lr', {})
    
    if pretrained is None:
        print(f"Instantiating new model '{model_name}' from scratch")
        model = StarDist2D(architecture_conf, name=model_name, basedir=models_dir)
    else:
        print(f"Instantiating model '{model_name}' from pretrained weights: {pretrained}")
        model_pretrained = StarDist2D.from_pretrained(pretrained)
        if os.path.exists(cur_model_dir):
            shutil.rmtree(cur_model_dir)
        shutil.copytree(model_pretrained.logdir, cur_model_dir)
        model = StarDist2D(None, name=model_name, basedir=models_dir)
    
    if learning_rate is not None:
        model.config.train_learning_rate = convert_to_float(learning_rate)
    
    if train_reduce_lr:
        for param in ['factor', 'patience', 'min_delta']:
            if param in train_reduce_lr:
                model.config.train_reduce_lr[param] = convert_to_float(train_reduce_lr[param])
    
    if early_stopping:
        early_stopping.setdefault('monitor', 'val_loss')
        early_stopping.setdefault('min_delta', 0)
        early_stopping.setdefault('patience', 10)
        early_stopping.setdefault('verbose', 1)
        early_stopping.setdefault('restore_best_weights', False)

        model.prepare_for_training()
        model.callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping['monitor'],
                min_delta=convert_to_float(early_stopping['min_delta']),
                patience=int(early_stopping['patience']),
                verbose=int(early_stopping['verbose']),
                restore_best_weights=bool(early_stopping['restore_best_weights']),
                mode=str(early_stopping['mode']),
            )
        )
    
    os.makedirs(os.path.join(cur_model_dir, 'quality_control'), exist_ok=True)
    return model

def save_config_to_json(
    config: Dict[str, Any],
    file_path: str
) -> None:
    """Save training configuration to JSON file"""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

def generate_hyperparameter_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters from config ranges"""
    param_grid = {}
    for key, value in config.items():
        if isinstance(value, dict):
            param_grid[key] = generate_hyperparameter_grid(value)
        elif isinstance(value, list):
            param_grid[key] = value
        elif isinstance(value, str) and '(' in value:
            values_tuple = ast.literal_eval(value)
            start, end, step = values_tuple
            param_grid[key] = list(np.arange(start, end + step, step)[:-1])
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
    """Train and evaluate a single model configuration"""
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

    save_config_to_json(
        config,
        file_path=os.path.join(cur_model_dir, "training_config.json")
    )

    size = int(len(X_val) * config['val_prop_opt'])
    sampled_idx = np.random.randint(0, high=len(X_val), size=size, dtype=int)
    X_val_opt = [X_val[i] for i in sampled_idx]
    Y_val_opt = [Y_val[i] for i in sampled_idx]

    architecture_conf = configure_model() if config['pretrained'] is None else None
    augmenter = default_augmenter if config['augment'] else None

    model = instantiate_model(
        models_dir, 
        model_name, 
        architecture_conf=architecture_conf, 
        config=config
    )

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stardist training with config file.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for outputs")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing training data")
    parser.add_argument('--config_file', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--workers', type=int, default=min(8, multiprocessing.cpu_count()), 
                       help="Number of parallel workers to use (default: min(8, CPU count))")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['base_dir'] = args.base_dir
    config['data_dir'] = args.data_dir

    configs = generate_hyperparameter_grid(config)
    print(f"Generated {len(configs)} hyperparameter combinations")
    
    # Save all configurations first
    configs_dir = os.path.join(config['base_dir'], 'configurations')
    os.makedirs(configs_dir, exist_ok=True)
    for i, cfg in enumerate(configs):
        save_config_to_json(cfg, os.path.join(configs_dir, f"config_{i}.json"))

    # Run in parallel with limited workers
    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.map(train_and_evaluate, configs)

if __name__ == '__main__':
    main()