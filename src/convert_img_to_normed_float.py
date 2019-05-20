import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt
import os

for filename in os.listdir('../data/frames/'):
    filepath = os.path.join('../data/frames', filename)
    float_img = img_as_float(np.ndarray.astype(np.load(filepath)['arr_0'][0].transpose(1,2,0), dtype=np.uint8))
    #plt.imshow(float_img)
    #plt.show()
    print(filename)
    np.save(os.path.join('../data/norm_frames', filename), float_img)
