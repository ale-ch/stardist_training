#!/usr/bin/env python

import numpy as np

def random_fliprot(img, mask): 
    axes = tuple(range(img.ndim)) 
    perm = np.random.permutation(axes)
    img = img.transpose(perm) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    return img * np.random.uniform(0.6, 2)

def default_augmenter(img, mask):
    img, mask = random_fliprot(img, mask)
    img = random_intensity_change(img)
    return img, mask