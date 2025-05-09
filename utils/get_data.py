#!/usr/bin/env python

import random
from csbdeep.utils import download_and_extract_zip_file

def download_data(target_dir):
    download_and_extract_zip_file(
        url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = target_dir,
        verbose   = 1,
    )
    

def train_test_val_split(
    X: list, 
    Y: list, 
    filenames: list, 
    test_prop: float, 
    val_prop: float,
    seed=42
):
    assert len(X) == len(Y) == len(filenames), "All inputs must be of the same length"
    assert 0 <= test_prop <= 1, "test_prop must be between 0 and 1"
    assert 0 <= val_prop <= 1, "val_prop must be between 0 and 1"

    # Set the seed
    random.seed(seed)

    # Generate a shuffled list of indices
    indices = list(range(len(X)))
    random.shuffle(indices)

    # Compute split sizes
    n_total = len(X)
    n_test = int(n_total * test_prop)
    n_val = int(n_total * val_prop)
    n_train = n_total - n_test - n_val

    # Index splits
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    # Split data
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]

    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]

    files_train = [filenames[i] for i in train_indices]
    files_test = [filenames[i] for i in test_indices]

    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val), (files_train, files_test)
