from pdb import set_trace
import math
import numpy as np
import random
from random import shuffle
import re
import string
import time
import unicodedata
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from early_stopper import EarlyStopper
from attention import Attention
import pretrainedmodels


class EncoderRNN(nn.Module):
    def __init__(self, ARGS, device):
        super(EncoderRNN, self).__init__()
        self.num_frames = ARGS.num_frames
        self.output_cnn_size = ARGS.output_cnn_size
        self.device = device
        self.batch_size = int(ARGS.batch_size)
        self.num_layers = int(ARGS.enc_layers)
        self.rnn_type = ARGS.enc_rnn
        self.cnn_type = ARGS.enc_cnn
        self.output_size = int(ARGS.enc_size)
        self.hidden_init = torch.randn(self.num_layers, 1, int(self.output_size), device=device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init

        #Encoder Recurrent layer
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.output_cnn_size, self.output_size, num_layers=self.num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.output_cnn_size, self.output_size, num_layers=self.num_layers)

        self.resize = nn.Linear(self.num_frames, 1)
        self.cnn_resize = nn.Linear(4032, self.output_cnn_size)

    def forward(self, input_, hidden=None):
        if hidden==None: hidden=self.initHidden()
        
        # pass the output of the vgg layers through the GRU cell
        outputs, hidden = self.rnn(input_, hidden)
        outputs = self.resize(outputs.view(self.num_frames,-1).transpose(0,1)).view(self.batch_size,-1)

        return outputs, hidden

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            if self.rnn_type == 'gru':
                return torch.ones(self.num_layers, self.batch_size, self.output_size, device=self.device)/(self.output_size**0.5)
            elif self.rnn_type == 'lstm':
                return (torch.ones(self.num_layers, self.batch_size, self.output_size, device=self.device)/(self.output_size**0.5), torch.ones(self.num_layers, self.batch_size, self.output_size, device=self.device)/(self.output_size**0.5))
                
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.output_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.output_size, device=self.device)/torch.norm(self.hidden_init,2)


class MLP(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,out_size)

    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
        
    
