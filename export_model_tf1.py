import os
import argparse
from stardist.models import StarDist2D

def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Stardist model with specified options.")

    parser.add_argument(
        '--base_dir', 
        type=str,
        default='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/',
        help='Base directory for data and models.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='stardist',
        help='Name of the model to use/save.'
    )
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = _parse_args()

    base_dir = args.base_dir
    model_name = args.model_name

    models_dir = os.path.join(base_dir, 'models')
    cur_model_dir = os.path.join(models_dir, model_name)

    print("Loading model")
    model = StarDist2D(None, name=model_name, basedir=models_dir)
    print("Loaded model")

    tf_model_outname = f"TF_{model_name}.zip"
    tf_model_path = os.path.join(cur_model_dir, tf_model_outname)

    print(f"Exporting model to {tf_model_path}")
    model.export_TF(fname=tf_model_path)
    print(f"Exported TF model to {tf_model_path}")
