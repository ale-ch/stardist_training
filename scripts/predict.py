import argparse
import os
from stardist.models import StarDist2D
import tifffile as tiff
from csbdeep.utils import normalize

def main():
    parser = argparse.ArgumentParser(description="Run StarDist2D prediction on a test image.")
    parser.add_argument('--model_name', required=True, help='Name of the trained StarDist2D model')
    parser.add_argument('--base_dir', required=True, help='Base directory wherein the model is stored')
    parser.add_argument('--indir', required=False, help='Path to directory with TIFF images to predict')
    parser.add_argument('--file', required=False, help='Path to input TIFF image')
    parser.add_argument('--outdir', required=True, help='Directory to save prediction')

    args = parser.parse_args()

    model_name = args.model_name
    base_dir = args.base_dir
    file = args.file
    outdir = args.outdir

    model_predictions_dir = os.path.join(outdir, model_name)
    

    os.makedirs(model_predictions_dir, exist_ok=True)

    print("Loading model...")
    model = StarDist2D(None, name=model_name, basedir=base_dir)
    print("Model loaded.")

    if not args.file:
        for file in os.listdir(args.indir):
            # test_image_name = os.path.basename(file)
            output_path = os.path.join(model_predictions_dir, file)

            file_path = os.path.join(args.indir, file)  
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
        output_path = os.path.join(model_predictions_dir, filename)

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
