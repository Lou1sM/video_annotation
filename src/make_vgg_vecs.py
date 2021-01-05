"""Pre-extract VGG feature vectors from each frame of videos. The VGG network
is frozen during training.
"""

import numpy as np
from pdb import set_trace
import torch
import torch.nn as nn
from torchvision import models


vgg = models.vgg19(pretrained=True).cuda()
num_ftrs = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(num_ftrs, 4096)

def vgg_vec_from_video_array(video_array):
    x = torch.tensor(video_array).float().cuda()
    x = vgg.features(x)
    x = vgg.avgpool(x)
    x = x.view(x.size(0), -1)
    x = vgg.classifier[0](x)
    vggvec = x.squeeze()
    vggvec_np = vggvec.cpu().detach().numpy()
    return vggvec_np


# VGG vecs for MSVD videos
for video_number in range(1,1971):
    framepath = f'../data/msvd/frames/vid{video_number}_resized.npz'
    video_array = np.load(framepath)['arr_0']
    vggvec_np = vgg_vec_from_video_array(video_array)
    vggpath = f'../data/msvd/vggvecs/vid{video_number}.npy'
    print('saving vggvec to',vggpath)
    np.save(vggpath, vggvec_np)


# VGG vecs for MSRVTT videos
for video_number in range(0,1):
    framepath = f'../data/msrvtt/frames/vid{video_number}_resized.npy'
    video_array = np.load(framepath)
    vggvec_np = vgg_vec_from_video_array(video_array)
    vggpath = f'../data/msrvtt/vggvecs/vid{video_number}.npy'
    print('saving vggvec to',vggpath)
    np.save(vggpath, vggvec_np)
