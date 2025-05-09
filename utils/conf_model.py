#!/usr/bin/env python

import os
import numpy as np
import shutil 
import tensorflow as tf
from stardist.models import Config2D, StarDist2D, StarDistData2D

# from stardist.models import StarDist2D

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


def instantiate_model(
        models_dir, 
        model_name, 
        architecture_conf=None,
        config: dict = None, 
    ):

    cur_model_dir = os.path.join(models_dir, model_name)

    learning_rate = config.get('learning_rate')
    pretrained = config.get('pretrained' )
    early_stopping = config.get('early_stopping')
    train_reduce_lr = config.get('train_reduce_lr')

    if pretrained is None:
        print("instantiate_model: Instantiate model from scratch")
        model = StarDist2D(architecture_conf, name=model_name, basedir=models_dir)

    else:
        print("instantiate_model: Instantiate model from pretrained")
        cur_model_dir = os.path.join(models_dir, model_name)
        
        os.makedirs(cur_model_dir, exist_ok=True)

        model_pretrained = StarDist2D.from_pretrained(pretrained)
        shutil.copytree(model_pretrained.logdir, cur_model_dir, dirs_exist_ok=True)

        # create new model from folder (loading the  pretrained weights)
        model = StarDist2D(None, name=model_name, basedir=models_dir)
    
    if learning_rate is not None:
        model.config.train_learning_rate = learning_rate

    if train_reduce_lr is not None:
        print("train_reduce_lr: ", train_reduce_lr)
        model.config.train_reduce_lr["factor"] = train_reduce_lr["factor"]
        model.config.train_reduce_lr["patience"] = train_reduce_lr["patience"]
        model.config.train_reduce_lr["min_delta"] = train_reduce_lr["min_delta"]


    if early_stopping is not None:
        print("early_stopping: ", early_stopping)

        keys_list = ["monitor", "min_delta", "patience", "verbose", "baseline", "restore_best_weights", "start_from_epoch", "mode"]

        for key in keys_list:
            if key not in early_stopping:
                early_stopping[key] = None

        model.prepare_for_training()
        model.callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping["monitor"],
                min_delta=early_stopping["min_delta"],
                patience=early_stopping["patience"],
                verbose=early_stopping["verbose"],
                baseline=early_stopping["baseline"],
                restore_best_weights=early_stopping["restore_best_weights"],
                start_from_epoch=early_stopping["start_from_epoch"],
                mode=early_stopping["mode"],
            )
        )




    os.makedirs(os.path.join(cur_model_dir, 'quality_control'), exist_ok=True)

    print("Instantiated model")


    return model


#### Example ###

# https://forum.image.sc/t/how-to-continue-training/73482/5
# import numpy as np
# from stardist.models import StarDist2D
# import shutil 
# 
# # make a copy of a pretrained model into folder 'mymodel'
# model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
# shutil.copytree(model_pretrained.logdir, 'mymodel', dirs_exist_ok=True)
# 
# 
# # create new model from folder (loading the  pretrained weights)
# model = StarDist2D(None, 'mymodel')
# 
# # create new training data 
# X = np.zeros((16,128,128,1))
# X[:, 10:20,10:20] = 1.1
# Y = np.zeros((16,128,128), np.uint16)
# Y[:, 10:20,10:20] = 1
 
# # change some training params 
# model.config.train_patch_size = (128,128)
# model.config.train_batch_size = 16 
# model.config.train_learning_rate = 1e-5
# model.config.train_epochs = 10
# 
# # finetune on new data
# model.train(X,Y, validation_data=(X,Y))


# early_stopping["monitor"] = 'val_prob_loss'
# early_stopping["min_delta"] = 0.1
# early_stopping["patience"] = 0
# early_stopping["verbose"] = 0
# early_stopping["baseline"] = None
# early_stopping["restore_best_weights"] = False
# early_stopping["start_from_epoch"] = 0
# early_stopping["mode"] = 'min'

