#!/usr/bin/env python

import json


# def save_config_to_json(model_name, config, files_train, files_test, file_path="config.json"):
#     config_to_save = {
#         "model_name": model_name,
#         "epochs": config["epochs"],
#         "steps_per_epoch": config["steps_per_epoch"],
#         "learning_rate": config["learning_rate"],
#         "augment": config["augment"],
#         "early_stopping": config.get("early_stopping", {}),
#         "train_reduce_lr": config.get("train_reduce_lr", {}),
#         "val_prop": config["val_prop"],
#         "val_prop_opt": config["val_prop_opt"],
#         "random_seed": config["random_seed"],
#         "files_train": files_train,
#         "files_test": files_test
#     }
# 
#     with open(file_path, "w") as f:
#         json.dump(config_to_save, f, indent=4)

    
import json
import os
from typing import Union, List, Dict, Any

def save_config_to_json(
    model_name: str,
    epochs: Union[int, List[int]],
    steps_per_epoch: Union[int, List[int]],
    learning_rate: Union[float, List[float]],
    augment: Union[bool, List[bool]],
    early_stopping: Dict[str, Any],
    random_seed: Union[int, List[int]],
    test_prop: float,
    val_prop: float,
    val_prop_opt: float,
    files_train: List[str],
    files_test: List[str],
    file_path: str = "config.json",
    **additional_params: Dict[str, Any]
) -> None:
    """
    Save training configuration to a JSON file with support for hyperparameter ranges.
    
    Args:
        model_name: Name of the model
        epochs: Number of epochs or list of possible values
        steps_per_epoch: Steps per epoch or list of possible values
        learning_rate: Learning rate or list of possible values
        augment: Whether to use augmentation or list of options
        early_stopping: Dictionary with early stopping configuration
        random_seed: Random seed or list of possible seeds
        test_prop: Proportion of data for testing
        val_prop: Proportion of data for validation
        val_prop_opt: Proportion of validation data to use for optimization
        files_train: List of training files
        files_test: List of test files
        file_path: Path to save the JSON file
        additional_params: Any additional parameters to include in the config
        
    Returns:
        None
    """
    # Create configuration dictionary with type hints
    config: Dict[str, Any] = {
        "model_name": model_name,
        "hyperparameters": {
            "epochs": epochs if isinstance(epochs, list) else [epochs],
            "steps_per_epoch": steps_per_epoch if isinstance(steps_per_epoch, list) else [steps_per_epoch],
            "learning_rate": learning_rate if isinstance(learning_rate, list) else [learning_rate],
            "augment": augment if isinstance(augment, list) else [augment],
            "random_seed": random_seed if isinstance(random_seed, list) else [random_seed]
        },
        "data_splitting": {
            "test_prop": test_prop,
            "val_prop": val_prop,
            "val_prop_opt": val_prop_opt,
            "files_train": files_train,
            "files_test": files_test
        },
        "early_stopping": early_stopping,
        **additional_params
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    
    # Save with pretty printing and sorted keys
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)
    
    # Verify the file was written
    if not os.path.exists(file_path):
        raise IOError(f"Failed to write configuration file at {file_path}")