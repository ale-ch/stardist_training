import tifffile as tiff
import pickle
import np

from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.matching import matching_dataset

def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)

def load_pickle(path):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Deserialize the object from the file
        loaded_data = pickle.load(file)

    return loaded_data



if __name__ == '__main__':
    print("Loading model")
    model = StarDist2D(None, name='stardist',basedir='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/training_test/models/')
    print("Loaded model")


    file = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/training_test/val.pkl'
    X_val, Y_val = load_pickle(file)

    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in X_val]

    taus = np.linspace(0.2, 0.9, 8)

    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in taus]

    print(stats[taus.index(0.8)])