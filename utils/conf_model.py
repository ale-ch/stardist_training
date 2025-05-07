#!/usr/bin/env python

import os
import numpy as np
import shutil 
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

    # model = StarDist2D(conf, name='stardist_v3', basedir='models/instance_segmentation_2D_akoya')

# def instantiate_model(conf, basedir, model_name, pretrained='2D_versatile_fluo'):
# 
#     if pretrained is None:
#         model = StarDist2D(conf, name=model_name, basedir=basedir)
#     else:
#         model = StarDist2D.from_pretrained(pretrained)
# 
#     print("Instantiated model") 
# 
#     return model


def instantiate_model(models_dir, model_name, conf=None, learning_rate: float = None, pretrained=None):
    print(f"instantiate_model: PRETRAINED: {pretrained}")
    cur_model_dir = os.path.join(models_dir, model_name)

    if pretrained is None:
        print("instantiate_model: Instantiate model from scratch")
        model = StarDist2D(conf, name=model_name, basedir=models_dir)

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
# 
# # change some training params 
# model.config.train_patch_size = (128,128)
# model.config.train_batch_size = 16 
# model.config.train_learning_rate = 1e-5
# model.config.train_epochs = 10
# 
# # finetune on new data
# model.train(X,Y, validation_data=(X,Y))
