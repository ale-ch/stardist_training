#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import numpy as np 
from csbdeep.utils import normalize
from stardist.models import StarDist2D
# from utils.preprocessing import rescale_to_uint8
from csbdeep.utils import normalize
from cellpose.utils import masks_to_outlines
from skimage.transform import rescale


def rescale_to_uint8(image):
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)


def quality_control(model, img, file, qc_outdir):
    filename = os.path.basename(file)
    output_path = os.path.join(qc_outdir, filename)

    img = normalize(img, 1, 99.8, axis=(0, 1))
    pred, _ = model.predict_instances(img, verbose=True)

    outlines_pred = np.array(masks_to_outlines(pred), dtype="float32")

    output_array = np.stack([img, pred, outlines_pred], axis=0).astype("float32")
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


def main():
    parser = argparse.ArgumentParser(description="Run StarDist2D prediction on a test image.")
    parser.add_argument('--model_name', required=True, help='Name of the trained StarDist2D model')
    parser.add_argument('--models_dir', required=True, help='Base directory wherein the model is stored')
    parser.add_argument('--test_imgs_dir', required=False, help='Path to directory with TIFF images to predict')
    parser.add_argument('--file', required=False, help='Path to input TIFF image')
    parser.add_argument('--outdir', required=True, help='Directory to save prediction')

    args = parser.parse_args()

    model_name = args.model_name
    models_dir = args.models_dir
    file = args.file
    outdir = args.outdir

    model_predictions_dir = os.path.join(outdir, model_name)
    os.makedirs(model_predictions_dir, exist_ok=True)
    print("Loading model...")
    model = StarDist2D(None, name=model_name, basedir=models_dir)
    print("Model loaded.")

    if not args.file:
        for file in os.listdir(args.test_imgs_dir):
            file_path = os.path.join(args.test_imgs_dir, file)  
            img = tiff.imread(file_path)
            quality_control(model, img, file, model_predictions_dir)
    else:
        file = args.file
        img = tiff.imread(file)
        quality_control(model, img, file, model_predictions_dir)


if __name__ == '__main__':
    main()
