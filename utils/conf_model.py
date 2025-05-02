#!/usr/bin/env python

from stardist.models import Config2D, StarDist2D, StarDistData2D

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

def instantiate_model(conf, basedir, model_name):
    model = StarDist2D(conf, name=model_name, basedir=basedir)
    print("Instantiated model") 

    return model

