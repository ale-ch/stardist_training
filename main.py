#!/usr/bin/env python

import os
import argparse
import yaml
import multiprocessing
from pydantic import ValidationError
from utils.config_validation import TrainingConfig
from utils.metadata_tools import save_config_to_json
from utils.training import generate_hyperparameter_grid, train_and_evaluate


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
        raw_config = yaml.safe_load(f)

    raw_config['base_dir'] = args.base_dir
    raw_config['data_dir'] = args.data_dir

    try:
        validated_config = TrainingConfig(**raw_config)
    except ValidationError as e:
        print("Configuration validation failed:")
        print(e.json(indent=2))
        exit(1)

    config_dict = validated_config.dict()
    configs = generate_hyperparameter_grid(config_dict)

    print(f"Generated {len(configs)} hyperparameter combinations")

    configs_dir = os.path.join(config_dict['base_dir'], 'configurations')
    os.makedirs(configs_dir, exist_ok=True)
    for i, cfg in enumerate(configs):
        save_config_to_json(cfg, os.path.join(configs_dir, f"config_{i}.json"))

    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.map(train_and_evaluate, configs)


if __name__ == '__main__':
    main()