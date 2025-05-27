import argparse
import os
import tifffile as tiff
import numpy as np 
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from utils.preprocessing import rescale_to_uint8
from csbdeep.utils import normalize
from stardist.matching import matching_dataset
from cellpose.utils import masks_to_outlines
from skimage.transform import rescale
from stardist.matching import matching_dataset


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
    

    # os.makedirs(model_predictions_dir, exist_ok=True)

    print("Loading model...")
    model = StarDist2D(None, name=model_name, basedir=models_dir)
    print("Model loaded.")

    if not args.file:
        for file in os.listdir(args.test_imgs_dir):
            # test_image_name = os.path.basename(file)
            # output_path = os.path.join(model_predictions_dir, file)
            output_path = os.path.join(outdir, file)

            file_path = os.path.join(args.test_imgs_dir, file)  
            img = tiff.imread(file_path)
            n_channel = 1 if img.ndim == 2 else img.shape[-1]
            axis_norm = (0, 1)

            if n_channel > 1:
                print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

            img = normalize(img, 1, 99.8, axis=axis_norm)

            labels, details = model.predict_instances(img, verbose=True)

            tiff.imwrite(output_path, labels)
            print(f"Prediction saved to {output_path}")
    else:
        file = args.file
        filename = os.path.basename(file)
        # output_path = os.path.join(model_predictions_dir, filename)
        output_path = os.path.join(outdir, filename)

        img = tiff.imread(file)
        n_channel = 1 if img.ndim == 2 else img.shape[-1]
        axis_norm = (0, 1)

        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        img = normalize(img, 1, 99.8, axis=axis_norm)

        labels, details = model.predict_instances(img, verbose=True)

        tiff.imwrite(output_path, labels)
        print(f"Prediction saved to {output_path}")


if __name__ == '__main__':
    main()
