#!/usr/bin/env python

import json


def save_config_to_json(model_name, config, files_train, files_test, file_path="config.json"):
    config_to_save = {
        "model_name": model_name,
        "epochs": config["epochs"],
        "steps_per_epoch": config["steps_per_epoch"],
        "learning_rate": config["learning_rate"],
        "augment": config["augment"],
        "early_stopping": config.get("early_stopping", {}),
        "train_reduce_lr": config.get("train_reduce_lr", {}),
        "val_prop": config["val_prop"],
        "val_prop_opt": config["val_prop_opt"],
        "random_seed": config["random_seed"],
        "files_train": files_train,
        "files_test": files_test
    }

    with open(file_path, "w") as f:
        json.dump(config_to_save, f, indent=4)
