import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt
import os

#for filename in os.listdir('../data/frames/'):
for i in range(1,1971):
    filepath = os.path.join('../data/frames', 'vid{}_resized.npz'.format(i))
    float_img = img_as_float(np.ndarray.astype(np.load(filepath)['arr_0'].transpose(0,2,3,1), dtype=np.uint8))
    #plt.imshow(float_img[0])
    #plt.show()
    print(i)
    np.savez(os.path.join('../data/norm_frames', 'vid{}.npz'.format(i)), float_img)
