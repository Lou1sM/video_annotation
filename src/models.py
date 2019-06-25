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
        self.hidden_size = ARGS.enc_size
        self.device = device
        self.batch_size = ARGS.batch_size
        self.num_layers = ARGS.enc_layers
        self.rnn_type = ARGS.enc_rnn
        self.cnn_type = ARGS.enc_cnn
        self.output_size = ARGS.ind_size
        self.hidden_init = torch.randn(self.num_layers, 1, self.hidden_size, device=device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init
        self.i3d = ARGS.i3d
        self.i3d_size = 1024 if self.i3d else 0

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
            self.rnn = nn.GRU(self.output_cnn_size+self.i3d_size, self.hidden_size, num_layers=self.num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.output_cnn_size+self.i3d_size, self.hidden_size, num_layers=self.num_layers)

        # Resize outputs to ind_size
        self.resize = nn.Linear(self.hidden_size, ARGS.ind_size)
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


    def forward(self, input_, i3d_vec, hidden):
        
        # pass the input through the cnn
        cnn_outputs = torch.zeros(self.num_frames, input_.shape[1], self.output_cnn_size+self.i3d_size, device=self.device)

        for i, inp in enumerate(input_):
            embedded = self.cnn_vector(inp)
            if self.i3d:
                embedded = torch.cat([embedded, i3d_vec], dim=1)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        outputs, hidden = self.rnn(cnn_outputs, hidden)
        outputs = self.resize(outputs)

        return outputs, hidden

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/torch.norm(self.hidden_init,2)



class DecoderRNN(nn.Module):
    
    def __init__(self, ARGS, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = ARGS.dec_size
        self.num_layers = ARGS.dec_layers
        self.dropout_p = ARGS.dropout
        self.batch_size = ARGS.batch_size
        self.device = device
        self.output_size = ARGS.ind_size+1

        self.rnn = nn.GRU(ARGS.ind_size, self.hidden_size, num_layers=self.num_layers)
        self.attention = Attention(ARGS.ind_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_init = torch.randn(self.num_layers, 1, self.hidden_size, device=device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init

        # Resize from GRU size to embedding size
        self.out1 = nn.Linear(self.hidden_size, ARGS.ind_size)
        self.eos = nn.Sequential(nn.Linear(ARGS.ind_size, 1), nn.Sigmoid())


    def forward(self, input_, hidden, input_lengths, encoder_outputs):
        
        if self.training:
            drop_input = self.dropout(input_)
        else:
            drop_input = input_
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)

        # permute dimensions to have batch_size in the first
        #enc_perm shape: (batch_size, num_frames, ind_size)
        enc_perm = encoder_outputs.permute(1,0,2)
        output_perm = output.permute(1,0,2)

        # apply attention
        output = self.out1(output_perm)
        output, attn = self.attention(output, enc_perm)

        if not self.training:
            output = (1-self.dropout_p)*output

        return output, hidden

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/torch.norm(self.hidden_init,2)


class DecoderRNN_openattn(nn.Module):
    
    def __init__(self, ARGS):
        super(DecoderRNN_openattn, self).__init__()
        self.hidden_size = ARGS.dec_size
        self.num_layers = ARGS.dec_layers
        self.dropout_p = ARGS.dropout
        self.batch_size = ARGS.batch_size
        self.device = ARGS.device
        self.output_size = ARGS.ind_size

        self.rnn = nn.GRU(ARGS.ind_size, self.hidden_size, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_init = torch.randn(self.num_layers, 1, self.hidden_size, device=ARGS.device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init

        # Resize from GRU size to embedding size
        self.out1 = nn.Linear(self.hidden_size, ARGS.ind_size)
        self.out2 = nn.Linear(2*ARGS.ind_size, self.output_size)
        self.eos = nn.Sequential(nn.Linear(2*ARGS.ind_size, 1), nn.Sigmoid())


    def get_attention_context_concatenation(self, input_, hidden, input_lengths, encoder_outputs):
        # apply dropout
        if self.training:
            drop_input = self.dropout(input_)
        else:
            drop_input = input_
        
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        enc_perm = encoder_outputs.permute(1,2,0)
        output_perm = output.permute(1,0,2)

        output = self.out1(output_perm)
        attn_weights = torch.bmm(output, enc_perm).permute(0,2,1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_enc_outp = torch.matmul(enc_perm, attn_weights).permute(0,2,1)
        attn_concat_outp = torch.cat([output, weighted_enc_outp], dim=2)

        return attn_concat_outp, hidden
        

    def forward(self, input_, hidden, input_lengths, encoder_outputs):
        attn_concat_output, hidden = self.get_attention_context_concatenation(input_, hidden, input_lengths, encoder_outputs)

        output = self.out2(attn_concat_output)
        if not self.training:
            output = (1-self.dropout_p)*output
        return output, hidden

    def eos_preds(self, input_, hidden, input_lengths, encoder_outputs):
        attn_concat_output, hidden = self.get_attention_context_concatenation(input_, hidden, input_lengths, encoder_outputs)

        eos_pred = self.eos(attn_concat_output)
        return eos_pred, hidden
   

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/torch.norm(self.hidden_init,2)

       
class NumIndEOS(DecoderRNN_openattn):
 
    def __init__(self, ARGS):
        super(NumIndEOS, self).__init__(ARGS)
        self.output_size = 1
        self.out2 = nn.Linear(2*ARGS.ind_size, self.output_size)


