import torch
import numpy as np
from silvio_img_preproc import image_preprocessing
from i3dpt import I3D

i3d = I3D(num_classes=400, modality='rgb')
i3d.load_state_dict(torch.load('../i3d_weights/model_rgb.pth'))

test_img = np.load('../data/frames/vid1.npz')['arr_0']
print(test_img.shape)
#preproc_img = image_preprocessing(test_img)
test_img1 = np.random.uniform(size=(18,3,1256,1256))
image_preprocessing(test_img1)

tensor_test_img = torch.tensor(test_img)
#outp = i3d(tensor_test_img)
#outp = i3d(test_img)
#outp = i3d(preproc_img)

