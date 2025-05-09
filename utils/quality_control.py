#!/usr/bin/env python

import matplotlib.pyplot as plt
import os 
import numpy as np 
import tifffile as tiff

from utils.preprocessing import rescale_to_uint8
from csbdeep.utils import normalize
from stardist.matching import matching_dataset
from cellpose.utils import masks_to_outlines
from skimage.transform import rescale
from stardist.matching import matching_dataset

def plot_metrics(Y_val, Y_val_pred, taus):
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in taus]

    fig1, ax1 = plt.subplots(figsize=(7,5))
    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7,5))
    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()
    plt.close(fig2)

    return fig1, fig2


def quality_control(model, X_test, Y_test, files_test, qc_outdir):
    for img, mask, file in zip(X_test, Y_test, files_test):
        filename = os.path.basename(file)
        output_path = os.path.join(qc_outdir, filename)

        img = normalize(img, 1, 99.8, axis=(0, 1))
        pred, _ = model.predict_instances(img, verbose=True)

        outlines_test = np.array(masks_to_outlines(mask), dtype="float32")
        outlines_pred = np.array(masks_to_outlines(pred), dtype="float32")

        output_array = np.stack([img, mask, pred, outlines_test, outlines_pred], axis=0).astype("float32")
        output_array = rescale_to_uint8(output_array)
        output_array = np.array([rescale(ch, scale=1, anti_aliasing=(i==0)) for i, ch in enumerate(output_array)])
        output_array = rescale_to_uint8(output_array)

        pixel_microns = 0.34533768547788
        tiff.imwrite(
            output_path, 
            output_array, 
            imagej=True, 
            resolution=(1/pixel_microns, 1/pixel_microns), 
            metadata={'unit': 'um', 'axes': 'CYX', 'mode': 'composite'}
        )