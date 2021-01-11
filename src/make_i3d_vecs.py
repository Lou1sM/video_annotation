from i3dpt import I3D
import torch
import numpy as np
import os
from pdb import set_trace


m = I3D(num_classes=400,modality='rgb')
m.load_state_dict(torch.load('../i3d_weights/model_rgb.pth'))
m.eval()
m.cuda()

def get_i3d_feature_vec(i3d_net,inp):
    resized_inp = inp.transpose(0,1).unsqueeze(0)
    out = i3d_net.conv3d_1a_7x7(torch.cat([resized_inp,resized_inp[:,:,:1,:,:]],axis=2))
    out = i3d_net.maxPool3d_2a_3x3(out)
    out = i3d_net.conv3d_2b_1x1(out)
    out = i3d_net.conv3d_2c_3x3(out)
    out = i3d_net.maxPool3d_3a_3x3(out)
    out = i3d_net.mixed_3b(out)
    out = i3d_net.mixed_3c(out)
    out = i3d_net.maxPool3d_4a_3x3(out)
    out = i3d_net.mixed_4b(out)
    out = i3d_net.mixed_4c(out)
    out = i3d_net.mixed_4d(out)
    out = i3d_net.mixed_4e(out)
    out = i3d_net.mixed_4f(out)
    out = i3d_net.maxPool3d_5a_2x2(out)
    out = i3d_net.mixed_5b(out)
    out = i3d_net.mixed_5c(out)
    out = i3d_net.avg_pool(out)
    out=out.flatten()
    # Take vector from penultimate conv layer as feature vec
    out = out.cpu().detach().numpy()
    return out


if __name__ == "__main__":
    dset_dir = '../data/MSVD'
    for vid_num in range(1,1971):
        fpath = os.path.join(dset_dir,'frames',f'vid{vid_num}_resized.npy')
        x = torch.tensor(np.load(fpath)).float().cuda()
        try:
            feature_vec = get_i3d_feature_vec(m,x)
        except: set_trace()
        i3dpath = os.path.join(dset_dir,'i3dvecs',f'vid{vid_num}.npy')
        np.save(i3dpath,feature_vec)
        print(vid_num,feature_vec.shape)

    dset_dir = '../data/MSRVTT'
    for vid_num in range(10000):
        fpath = os.path.join(dset_dir,'frames',f'vid{vid_num}_resized.npy')
        x = torch.tensor(np.load(fpath)).float().cuda()
        feature_vec = get_i3d_feature_vec(m,x)
        i3dpath = os.path.join(dset_dir,'i3dvecs',f'vid{vid_num}.npy')
        np.save(i3dpath,feature_vec)
        print(vid_num)
