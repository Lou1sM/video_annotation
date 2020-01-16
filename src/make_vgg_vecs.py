import numpy as np
import os
from pdb import set_trace
import torch
import torch.nn as nn
from torchvision import models


vgg = models.vgg19(pretrained=True).cuda()
num_ftrs = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(num_ftrs, 4096)

for i in range(1,1971):
    framepath = f'/data1/louis/data/frames/vid{i}.npz'
    print(framepath)
    video_array = np.load(framepath)['arr_0']
    x = torch.tensor(video_array).float().cuda()
    x = vgg.features(x)
    x = vgg.avgpool(x)
    x = x.view(x.size(0), -1)
    x = vgg.classifier[0](x)
    vggvec = x.squeeze()
    vggvec_np = vggvec.cpu().detach().numpy()

    np.save(f'/data1/louis/data/vggvecs/vid{i}.npy', vggvec_np)
