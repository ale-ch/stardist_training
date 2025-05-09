#!/usr/bin/env python

import json

def save_config_to_json(model_name, epochs, steps_per_epoch, learning_rate, augment, early_stopping,
                        val_prop, val_prop_opt, random_seed, files_train, files_test,
                        file_path="config.json"):
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": learning_rate,
        "augment": augment,
        "early_stopping": early_stopping,
        "val_prop": val_prop,
        "val_prop_opt": val_prop_opt,
        "random_seed": random_seed,
        "files_train": files_train,
        "files_test": files_test
    }

    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)