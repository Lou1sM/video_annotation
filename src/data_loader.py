"""Script for loading data from .h5 file. Functions for export
are load_data() and load_data_lookup(). The former takes a single
argument specifying the path of the h5 file to load from. It expects 
this h5 file to contain full video tensors. The latter takes two args,
one for the h5 file path and a range of videos to load into a lookup
table. It expects the h5 file path to contain video ids which it then
looks up in this table and returns. This is useful to reduce the h5
file size. 

The resulting Dataset objects generates 4-tuples of the form
(video, embedding_sequence, sequence_length, eos_gt)
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
import h5py as h5
from skimage import img_as_float


def load_vid_from_id(vid_id):
    return np.load('../data/frames/vid{}_resized.npz'.format(vid_id))['arr_0']


def video_lookup_table_from_range(start_idx, end_idx):
    return {vid_id: load_vid_from_id(vid_id+1) for vid_id in range(start_idx, end_idx)}


class VideoDataset(data.Dataset):
    """Dataset object that expects h5 file containing full video tensors, not ids."""
    def __init__(self, archive, transform=None):
        self.archive = h5.File(archive, 'r')
        self.videos = self.archive['videos']
        self.seq_lens = np.array(self.archive['seq_len'], dtype=np.int32)
        self.embeddings = self.archive['embeddings']
        self.eos_gts = self.archive['eos_gt']
        self.transform = transform
        
    def __getitem__(self, index):
        video = self.videos[index]
        embedding_seq = self.embeddings[index]
        seq_len = self.seq_lens[index]
        eos_gt = self.eos_gts[index]
        if self.transform != None:
            video = self.transform(video)
        return video, embedding_seq, seq_len, eos_gt

    def __len__(self):
        return len(self.videos)

    def close(self):
        self.archive.close()



def load_data(h5file_path, batch_size, shuffle):
    """Load data from specified file path and return a Dataset that loads full video tensors.

    Each element is a 4-tuple of the form
    (video, embedding_sequence, sequence_length, eos_gt)
    """
    transforms.ToTensor()
    transform = transforms.Compose(
        [transforms.ToTensor()],
        )

    new_data = VideoDataset(h5file_path)
    new_data_loaded = data.DataLoader(new_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return new_data_loaded


class LookupDataset(data.Dataset):
    """Dataset object that expects videos as ids, and forms a lookup table for such ids.

    To save duplicating the same video for each caption that accompanies that video, the
    h5 file read by this object contains video ids in place of the full video tensor. For
    a given h5 file, the videos corresponding to video ids are loaded into memory as a 
    lookup table and stored in self.video_lookup_dict. The range of video ids that can 
    appear in the file must be passed as an argument to this init of this object. 

    Args:
      archive: full file path of the h5 file
      vid_range: range of video ids for building lookup table
    """

    def __init__(self, archive, video_lookup_table, transform=None):
        self.archive = h5.File(archive, 'r')
        self.seq_lens = np.array(self.archive['embedding_len'], dtype=np.int32)
        self.embeddings = self.archive['embeddings']
        self.video_ids = self.archive['videoId']
        self.transform = transform
        self.eos_gts = self.archive['eos_gt']
        self.video_lookup_table = video_lookup_table
        #{vid_id: load_vid_from_id(vid_id+1) for vid_id in range(vid_range[0], vid_range[1])}

    def __getitem__(self, index):
        embedding_seq = self.embeddings[index]
        seq_len = self.seq_lens[index]
        video_id = self.video_ids[index]
        video = self.video_lookup_table[video_id]
        eos_gt = self.eos_gts[index]
        if self.transform != None:
            video = self.transform(video)
        return video, embedding_seq, seq_len, eos_gt

    def __len__(self):
        return len(self.seq_lens)

    def close(self):
        self.archive.close()



def load_data_lookup(h5file_path, video_lookup_table, batch_size, shuffle):
    """Load data from specified file path and return a Dataset that uses a lookup table for videos.

    Each element returned is a 4-tuple of the form
    (video, embedding_sequence, sequence_length, eos_gt)
    
    The lookup table consumes ~15G memory for the full train set, ~1G for the full validation set
    and ~5G for the full test set. There are mini-datasets available by passing the --mini flag 
    when calling main.py.

    Args:
      h5_file_path: full path to a h5_file containing one dataset with key 'videoId' which stores
              video ids (ints), one dataset with key 'embeddings' which stores sequences of embed-
              dings (padded np arrays), and one dataset with key 'embedding_len' which stores the
              lengths of sequences of the embeddings (ints)

      vid_range: a tuple of two ints, specifying the range of videos to load in the video lookup
              table, default split given in MSVD is (1,1201) for train data, (1201,1301) for val-
              idation data and (1301,1971) for test data
      batch_size: int, number of dataset elements to return for each batch
      shuffle: bool, whether to shuffle entire dataset before each epoch
    """

    transforms.ToTensor()
    transform = transforms.Compose(
        [transforms.ToTensor()],
        )

    new_data = LookupDataset(h5file_path, video_lookup_table=video_lookup_table)
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

    #new_data_loaded = load_data_lookup('../data/dummy_data/train_data_dummy.h5', batch_size=2, vid_range=(1,1201), shuffle=True)    
    new_data_loaded = load_data_lookup('../data/mini/train_data.h5', batch_size=2, vid_range=(1,21), shuffle=True)    
    for epoch in range(5):
        print(epoch)
        print("Number of batches:", len(new_data_loaded), "\n")
        for i, data in enumerate(new_data_loaded):
            #print(i, type(data))
            print("Number of elements in each batch:",len(data), "\n")
            print(data[0].shape)
            print(data[1].shape)
            print(data[2].shape)
            print(data[3].shape)
            #print(data[0].type())
            #print(data[2])
            #print(data[0][0,0,0,0,0])
            ##im = data[0][0,0,:,:,:].type('torch.FloatTensor')
            #im = data[0][0,0,:,:,:]
            #print(im.shape)
            #im = np.transpose(im, axes=(2,1,0))
            #print(im.shape)
            #plt.imshow(im)
            #plt.show()
            #outp = test_net(data[0][:,0,:,:,:].type('torch.FloatTensor'))
            #outp = test_net(data[0][:,0,:,:,:])
            #print(type(data[0][0,:,:,:,:]))
            #print(type(data[0][0,:,:,:,:].double()))
            #test_net = test_net.float()
            test_net = TestConvNet().float()
            outp = test_net(data[0][0,:,:,:,:].float())
            #print("Input sample:\n")
            #print(data[0][0,0,:,:,:])
            #print("Test output:\n")
            #print(outp[0])
        break
