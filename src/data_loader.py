"""Script for loading data from .h5 file. Functions for export
are load_data() and load_data_lookup(). The former takes a single
argument specifying the path of the h5 file to load from. It expects 
this h5 file to contain full video tensors. The latter takes two args,
one for the h5 file path and a range of videos to load into a lookup
table. It expects the h5 file path to contain video ids which it then
looks up in this table and returns. This is useful to reduce the h5
file size. 

The resulting Dataset objects generates 4-tuples of the form
(video, embedding_sequence, sequence_length, (i3d_vec))
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
import h5py as h5
import json
from pdb import set_trace
from skimage import img_as_float


def load_vid_from_id(vid_id, dataset):
    return np.load('/data1/louis/frames-resized/{}/vid{}.npz'.format(dataset, vid_id))['arr_0']

def video_lookup_table_from_range(start_idx, end_idx, dataset):
    return {vid_id: load_vid_from_id(vid_id, dataset) for vid_id in range(start_idx, end_idx)}

def video_lookup_table_from_ids(video_ids, cnn):
    return {vid_id: load_vid_from_id(vid_id, cnn) for vid_id in video_ids}

class VideoDataset(data.Dataset):
    """Dataset object that expects h5 file containing full video tensors, not ids."""
    def __init__(self, archive, transform=None):
        self.archive = h5.File(archive, 'r')
        self.videos = self.archive['videos']
        self.seq_lens = np.array(self.archive['seq_len'], dtype=np.int32)
        self.embeddings = self.archive['embeddings']
        self.transform = transform
        
    def __getitem__(self, index):
        video = self.videos[index]
        embedding_seq = self.embeddings[index]
        seq_len = self.seq_lens[index]
        if self.transform != None:
            video = self.transform(video)
        return video, embedding_seq, seq_len

    def __len__(self):
        return len(self.videos)

    def close(self):
        self.archive.close()


def load_data(h5file_path, batch_size, shuffle):
    """Load data from specified file path and return a Dataset that loads full video tensors.

    Each element is a 4-tuple of the form
    (video, embedding_sequence, sequence_length)
    """
    transforms.ToTensor()
    transform = transforms.Compose(
        [transforms.ToTensor()],
        )

    new_data = VideoDataset(h5file_path)
    new_data_loaded = data.DataLoader(new_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return new_data_loaded


class LookupDataset(data.Dataset):
    def __init__(self, archive, video_lookup_table, i3d_lookup_table, transform=None):
        with open(json_path) as f: self.json_list = json.load(f)
        self.transform = transform
        self.video_lookup_table = video_lookup_table

    def __getitem__(self, index):
        dp = self.json_list[idx]
        video_id = dp['video_id']
        multiclass_inds = dp['multiclass_inds']
        atoms = dp['atoms']
        #print('getting item for index', index)
        embedding_seq = self.embeddings[index]
        video = self.video_lookup_table[video_id]
        video_id = self.video_ids[index].astype(np.int32)
        return video, multiclass_inds, atoms, video_id

    def __len__(self):
        return len(self.json_list)

    def close(self):
        self.archive.close()


def load_data_lookup(h5file_path, video_lookup_table, batch_size, shuffle, i3d_lookup_table=None):

    transforms.ToTensor()
    transform = transforms.Compose(
        [transforms.ToTensor()],
        )

    new_data = LookupDataset(json_path, video_lookup_table=video_lookup_table)
    new_data_loaded = data.DataLoader(new_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return new_data_loaded


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
    import matplotlib.pyplot as plt

    new_data_loaded = load_data_lookup('/data1/louis/data/rdf_video_captions/10d-val.h5', batch_size=1, video_lookup_table=video_lookup_table, shuffle=False)    
    for epoch in range(1):
        print(epoch)
        print("Number of batches:", len(new_data_loaded), "\n")
        print(new_data_loaded)
        for i, data in enumerate(new_data_loaded):
            vid_id = data[4]
            vid = data[0]
            target_number_tensor = data[2]
            print(vid.shape)
            print(vid_id)
            print(target_number_tensor)
