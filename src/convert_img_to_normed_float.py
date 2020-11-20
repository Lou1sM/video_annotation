"""Preprocess MSVD video tensors. Convert from int to float, transpose to put
dimensions in the right order, and normalize by dataset mean and variance.
"""

import numpy as np
from skimage import img_as_float
import os

for i in range(1,1971):
    filepath = os.path.join('../data/unnormed_frames', 'vid{}_resized.npz'.format(i))
    try:
        float_img = img_as_float(np.ndarray.astype(np.load(filepath)['arr_0'].transpose(0,1,3,2), dtype=np.uint8))
    except KeyError:
        print(filepath)
        print(list(np.load(filepath).keys()))
    float_img = np.divide(np.add(float_img, [[[-.485]], [[-.456]], [[-.406]]]),[[[0.229]], [[0.224]], [[0.225]]])
    print(i)
    np.savez(os.path.join('../data/frames', 'vid{}.npz'.format(i)), float_img)
