#!/usr/bin/env python

from scripts.download_data import download_data
from scripts.conf_model import configure_model, instantiate_model


if __name__ == '__main__':

    download_data()

    conf = configure_model()
    model = instantiate_model(conf)
    history = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), epochs=20, steps_per_epoch=10)
