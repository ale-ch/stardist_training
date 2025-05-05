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

def instantiate_model(conf, basedir, model_name, pretrained='2D_versatile_fluo'):

    if pretrained is None:
        model = StarDist2D(conf, name=model_name, basedir=basedir)
    else:
        model = StarDist2D.from_pretrained(pretrained)

    print("Instantiated model") 

    return model


def instantiate_model(conf, basedir, model_name, pretrained=True):
    if not pretrained:
        model = StarDist2D(conf, name=model_name, basedir=basedir)
    else:

        cur_model_dir = os.path.join(basedir, model_name)

        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, cur_model_dir, dirs_exist_ok=True)

        # create new model from folder (loading the  pretrained weights)
        model = StarDist2D(None, name=model_name, basedir=basedir)

    print("Instantiated model")


    return model
