import torch
import numpy as np
import os
from i3dpt import I3D

i3d = I3D(num_classes=400, modality='rgb').double().cuda()
i3d.load_state_dict(torch.load('/home/louis/video_annotation/i3d_weights/model_rgb.pth'))
i3d.eval()


#for filename in os.listdir('../data/frames/'):
for i in range(1,1971):
    framepath = '../data/frames/vid{}.npz'.format(i)
    print(framepath)
    video_array = np.load(framepath)['arr_0']
    tensor_test_img = torch.tensor(video_array).cuda()
    cat_list = [
        torch.randn(1,3,224,224).cuda().double(), 
        tensor_test_img,
        torch.randn(1,3,224,224).cuda().double()]
    test_img_padded = torch.cat(cat_list)
    test_img_padded = test_img_padded.transpose(1,0).unsqueeze(0)

    assert test_img_padded.shape == torch.Size([1,3,10,224,224])
    out = i3d.conv3d_1a_7x7(test_img_padded)
    out = i3d.maxPool3d_2a_3x3(out)
    out = i3d.conv3d_2b_1x1(out)
    out = i3d.conv3d_2c_3x3(out)
    out = i3d.maxPool3d_3a_3x3(out)
    out = i3d.mixed_3b(out)
    out = i3d.mixed_3c(out)
    out = i3d.maxPool3d_4a_3x3(out)
    out = i3d.mixed_4b(out)
    out = i3d.mixed_4c(out)
    out = i3d.mixed_4d(out)
    out = i3d.mixed_4e(out)
    out = i3d.mixed_4f(out)
    out = i3d.maxPool3d_5a_2x2(out)
    out = i3d.mixed_5b(out)
    out = i3d.mixed_5c(out)
    out = i3d.avg_pool(out)
    #i3dvec, _logit = i3d(test_img_padded)
    i3dvec = out.squeeze()
    i3dvec_np = i3dvec.cpu().detach().numpy()

    np.save('../data/i3dvecs/vid{}.npy'.format(i), i3dvec_np)


