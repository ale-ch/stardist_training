import os
from stardist.models import StarDist2D

if __name__ == '__main__':

    base_dir = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/test2'

    models_dir = os.path.join(base_dir, 'models')

    model_name = 'stardist' 

    model = StarDist2D(None, name=model_name, basedir=models_dir)
    model.export_TF()
