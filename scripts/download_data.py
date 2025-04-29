#!/usr/bin/env python

#import numpy as np
#import os
#import pandas as pd
#import sys
#import matplotlib.pyplot as plt
#from tifffile import imread
#from datetime import datetime
from csbdeep.utils import Path, download_and_extract_zip_file, normalize

#from stardist.matching import matching_dataset
#from stardist import fill_label_holes, random_label_cmap, relabel_image_stardist, calculate_extents, gputools_available, _draw_polygons
#from stardist.models import Config2D, StarDist2D, StarDistData2D

#np.random.seed(42)

#lbl_cmap = random_label_cmap()


if __name__ == '__main__':

    # Data loading and preparation

    download_and_extract_zip_file(
        url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = 'data',
        verbose   = 1,
    )

