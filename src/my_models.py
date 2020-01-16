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

        if self.cnn_type == "vgg_old":
            self.cnn = models.vgg19(pretrained=True)
            num_ftrs = self.cnn.classifier[6].in_features
            self.cnn.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)
        elif self.cnn_type == "vgg":
            self.cnn = models.vgg19(pretrained=True)
        elif self.cnn_type == "nasnet":
            model_name = 'nasnetalarge'
            self.cnn = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            print(ARGS.enc_cnn)

        #Encoder Recurrent layer
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.output_cnn_size, self.output_size, num_layers=self.num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.output_cnn_size, self.output_size, num_layers=self.num_layers)

        self.resize = nn.Linear(self.num_frames, 1)
        self.cnn_resize = nn.Linear(4032, self.output_cnn_size)


    def cnn_vector(self, input_):
        if self.cnn_type == "vgg_old":
            x = self.cnn(input_)
        elif self.cnn_type == "vgg":
            x = self.cnn.features(input_)
            x = self.cnn.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.cnn.classifier[0](x)
        elif self.cnn_type == "nasnet":
            x = self.cnn.features(input_)
            x = self.cnn.relu(x)
            x = self.cnn.avg_pool(x)
            x = x.view(x.size(0),-1)
        return x

    def forward(self, input_, hidden):
        
        # pass the input through the cnn
        cnn_outputs = torch.zeros(self.num_frames, input_.shape[1], self.output_cnn_size, device=self.device)

        for i, inp in enumerate(input_):
            embedded = self.cnn_vector(inp)

        # pass the output of the vgg layers through the GRU cell
        outputs, hidden = self.rnn(cnn_outputs, hidden)
        outputs = self.resize(outputs.view(self.num_frames,-1).transpose(0,1)).view(self.batch_size,-1)

        return outputs, hidden

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.output_size, device=self.device)
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

