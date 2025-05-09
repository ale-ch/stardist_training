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

def save_config_to_json(model_name, epochs_list, steps_list, lr_list, augment_list, early_stopping_list,
                        random_seeds, test_prop, val_prop, val_prop_opt, files_train, files_test,
                        file_path="config.json"):
    # Create a dictionary to hold the configuration
    config = {
        "model_name": model_name,
        "epochs_list": epochs_list,
        "steps_list": steps_list,
        "lr_list": lr_list,
        "augment_list": augment_list,
        "early_stopping_list": early_stopping_list,
        "random_seeds": random_seeds,
        "test_prop": test_prop,
        "val_prop": val_prop,
        "val_prop_opt": val_prop_opt,
        "files_train": files_train,
        "files_test": files_test
    }

    # Save the configuration to a JSON file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)

