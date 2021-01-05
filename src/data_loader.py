import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
import json
from pdb import set_trace


def load_vgg_from_id(vid_id,video_data_dir):
    return np.load(os.path.join(video_data_dir,f'vggvecs/vid{vid_id}.npy'))

def vgg_lookup_table_from_range(start_idx, end_idx, video_data_dir):
    return {vid_id: load_vgg_from_id(vid_id) for vid_id in range(start_idx, end_idx)}

def vgg_lookup_table_from_ids(video_ids, video_data_dir):
    return {vid_id: load_vgg_from_id(vid_id, video_data_dir) for vid_id in video_ids}

def load_i3d_from_id(vid_id,video_data_dir):
    return np.load(os.path.join(video_data_dir,f'i3dvecs/vid{vid_id}.npy'))

def i3d_lookup_table_from_range(start_idx, end_idx, video_data_dir):
    return {vid_id: load_i3d_from_id(vid_id, video_data_dir) for vid_id in range(start_idx, end_idx)}

def i3d_lookup_table_from_ids(video_ids, video_data_dir):
    return {vid_id: load_i3d_from_id(vid_id, video_data_dir) for vid_id in video_ids}

def load_vid_from_id(video_data_dir,vid_id):
    return np.load(os.path.join(video_data_dir,f'frames/vid{vid_id}.npz'))['arr_0']

def video_lookup_table_from_range(start_idx, end_idx, video_data_dir):
    return {vid_id: load_vid_from_id(vid_id, video_data_dir) for vid_id in range(start_idx, end_idx)}

def video_lookup_table_from_ids(video_ids, video_data_dir):
    return {vid_id: load_vid_from_id(vid_id, video_data_dir) for vid_id in video_ids}

class LookupDataset(data.Dataset):
    def __init__(self, json_data, vgg_lookup_table, i3d_lookup_table, transform=None):
        self.dataset = json_data
        self.transform = transform
        self.vgg_lookup_table = vgg_lookup_table
        self.i3d_lookup_table = i3d_lookup_table

    def __getitem__(self, index):
        dp = self.dataset[index]
        video_id = dp['video_id']
        multiclass_inds = torch.tensor(dp['multiclass_inds'])
        vgg = self.vgg_lookup_table[video_id]
        try: i3d = torch.tensor([]) if self.i3d_lookup_table == None else self.i3d_lookup_table[video_id]
        except: set_trace()
        return vgg, multiclass_inds, video_id, i3d

    def __len__(self):
        return len(self.dataset)

    def close(self):
        self.archive.close()


def load_data_lookup(json_data, vgg_lookup_table, i3d_lookup_table, batch_size, shuffle, video_data_dir):
    new_data = LookupDataset(json_data, vgg_lookup_table, i3d_lookup_table, video_data_dir)
    new_data_loaded = data.DataLoader(new_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return new_data_loaded

def get_split_dls(json_data,splits,batch_size,shuffle,i3d,video_data_dir):
    train_data = json_data[0:splits[0]]
    val_data = json_data[splits[0]:splits[1]]
    test_data = json_data[splits[1]:splits[2]+1]
    train_table = vgg_lookup_table_from_ids([dp['video_id'] for dp in train_data],video_data_dir)
    train_i3d_table = i3d_lookup_table_from_ids([dp['video_id'] for dp in train_data],video_data_dir) if i3d else None
    train_dl = load_data_lookup(train_data,train_table,train_i3d_table,batch_size,shuffle,video_data_dir)
    val_table = vgg_lookup_table_from_ids([dp['video_id'] for dp in val_data],video_data_dir)
    val_i3d_table = i3d_lookup_table_from_ids([dp['video_id'] for dp in val_data],video_data_dir) if i3d else None
    val_dl = load_data_lookup(val_data, val_table,val_i3d_table,batch_size,shuffle,video_data_dir)
    test_table = vgg_lookup_table_from_ids([dp['video_id'] for dp in test_data],video_data_dir)
    test_i3d_table = i3d_lookup_table_from_ids([dp['video_id'] for dp in test_data],video_data_dir) if i3d else None
    test_dl = load_data_lookup(test_data,test_table,test_i3d_table,batch_size,shuffle,video_data_dir)
    return train_dl, val_dl, test_dl

class TestConvNet(torch.nn.Module):
    """Very simple CNN, for testing."""
    def __init__(self):
        super(TestConvNet, self).__init__()
        self.conv_layer = torch.nn.Conv2d(3,10,kernel_size=5)
        self.pool_layer = torch.nn.MaxPool2d(kernel_size=100, stride=100)
        self.fc_layer = torch.nn.Linear(20,10)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = x.view(-1, 20)
        x = self.fc_layer(x)
        return x

if __name__ == "__main__":
    dl,_,_ = get_split_dls('10dp.json',[4,6,11],batch_size=1,shuffle=True)
    vid_table = video_lookup_table_from_range(1,11)
    with open('10dp.json') as f: json_data=json.load(f)['dataset']
    #dl = load_data_lookup(json_data, vid_table, batch_size=1, shuffle=True)
    print("Number of batches:", len(dl), "\n")
    print(dl)
    for i, data in enumerate(dl):
        vid = data[0]
        multiclass_inds = data[2]
        print(vid.shape)
        print(multiclass_inds)
