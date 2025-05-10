#!/usr/bin/env python

import os
import json
from typing import Dict, Any

def get_compact_model_name(config: Dict[str, Any]) -> str:
    name_parts = [
        config['model_name'],
        f"e{config['epochs']}",
        f"lr{config['learning_rate']}".replace('.', '').replace('+', ''),
        f"aug{int(config['augment'])}",
        f"seed{config['random_seed']}"
    ]
    if 'early_stopping' in config:
        es = config['early_stopping']
        name_parts.append(f"es{es['patience']}p{es['min_delta']}")
    if 'train_reduce_lr' in config:
        lr = config['train_reduce_lr']
        name_parts.append(f"rlr{lr['factor']}p{lr['patience']}")
    return "_".join(name_parts)
    

def save_config_to_json(config: Dict[str, Any], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)