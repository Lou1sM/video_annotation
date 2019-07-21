import torch
import numpy as np
from silvio_img_preproc import image_preprocessing
from i3dpt import I3D

test_img = np.load('/home/louis/video_annotation/data/frames/vid1.npz')['arr_0']
tensor_test_img = torch.tensor(test_img).cuda()
cat_list = [
    torch.randn(1,3,224,224).cuda().double(), 
    tensor_test_img,
    torch.randn(1,3,224,224).cuda().double()]
test_img_padded = torch.cat(cat_list)
test_img_padded = test_img_padded.transpose(1,0).unsqueeze(0)

i3d = I3D(num_classes=400, modality='rgb').double().cuda()
i3d.load_state_dict(torch.load('/home/louis/video_annotation/i3d_weights/model_rgb.pth'))
i3d.eval()
#preproc_img = image_preprocessing(test_img)
test_img1 = np.random.uniform(size=(1,3,10,224,224))
sample_var = torch.autograd.Variable(torch.from_numpy(test_img1).cuda())
print(sample_var.shape)
#image_preprocessing(test_img1)

#outp = i3d(tensor_test_img)
#outp, _ = i3d(sample_var)
outp, _ = i3d(test_img_padded)
print(outp.shape)
#outp = i3d(preproc_img)

