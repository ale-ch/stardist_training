import os
import argparse
from stardist.models import StarDist2D

def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Stardist model with specified options.")

    parser.add_argument(
        '--base_dirs', 
        type=str,
        nargs='+',  # This allows multiple directory arguments
        default=['/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/'],
        help='List of base directories for data and models.'
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

    model_name = args.model_name
    base_dirs = args.base_dirs

    for base_dir in base_dirs:
        models_dir = os.path.join(base_dir, 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"Models directory not found: {models_dir}")
            continue
            
        # Get all model directories
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d))]
        
        if not model_dirs:
            print(f"No model directories found in {models_dir}")
            continue
            
        for model_dir in model_dirs:
            cur_model_dir = os.path.join(models_dir, model_dir)
            
            print(f"Loading model from {cur_model_dir}")
            try:
                model = StarDist2D(None, name=model_name, basedir=models_dir)
                print(f"Loaded model from {cur_model_dir}")
                
                tf_model_outname = f"TF_{model_name}.zip"
                tf_model_path = os.path.join(cur_model_dir, tf_model_outname)

                print(f"Exporting model to {tf_model_path}")
                model.export_TF(fname=tf_model_path)
                print(f"Exported TF model to {tf_model_path}")
                
            except Exception as e:
                print(f"Failed to process model in {cur_model_dir}: {str(e)}")