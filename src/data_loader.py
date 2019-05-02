"""Script for loading data from .h5 file. Function for export
is load_data() which takes a single argument specifying the path
of the .h5 file to load from. 

The resulting Dataset object generates triples of the form
(video, embedding_sequence, sequence_length)
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
import h5py as h5
from skimage import img_as_float


class ReadyDataset(data.Dataset):
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


def load_data(h5file_path, batch_size):
    """Load data from specified file path and return a Dataset.

    Each element is a triple of the form
    (video, embedding_sequence, sequence_length)
    """
    transforms.ToTensor()
    transform = transforms.Compose(
        [transforms.ToTensor()],
        )

    #new_data = ReadyDataset(h5file_path, transform)
    new_data = ReadyDataset(h5file_path)
    new_data_loaded = data.DataLoader(new_data, batch_size=batch_size)
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

    new_data_loaded = load_data('single_vid.h5', 2)    
    for i, data in enumerate(new_data_loaded):
        #print(i, type(data))
        #print(len(data))
        #print(data[0].shape)
        #print(data[1].shape)
        #print(data[2].shape)
        #print(data[0].type())
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
        print("Input sample:\n")
        print(data[0][0,0,:,:,:])
        print("Test output:\n")
        print(outp[0])
        break
