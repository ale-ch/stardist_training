from stardist.models import StarDist2D
import tifffile as tiff
import pickle
from csbdeep.utils import normalize

def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)


if __name__ == '__main__':
    print("Loading model")
    model = StarDist2D(None, name='stardist',basedir='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/training_test/models/')
    print("Loaded model")


    file = '/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/training_test/data/dsb2018/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif'

    img = tiff.imread(file)

    n_channel = 1 if img.ndim == 2 else img.shape[-1]

    axis_norm = (0,1)

    # axis_norm = (0,1,2) # normalize channels jointly

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    img = normalize(img, 1,99.8, axis=axis_norm)

    labels, details = model.predict_instances(img, verbose = True)
    
    save_pickle(labels, 'labels.pkl')
