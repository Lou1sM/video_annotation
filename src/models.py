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
        self.hidden_size = int(ARGS.enc_size)
        self.device = device
        self.batch_size = int(ARGS.batch_size)
        self.num_layers = int(ARGS.enc_layers)
        self.rnn_type = ARGS.enc_rnn
        self.cnn_type = ARGS.enc_cnn
        self.output_size = int(ARGS.ind_size)
        self.hidden_init = torch.randn(self.num_layers, 1, int(self.hidden_size), device=device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init
        self.attn_type = 'dot'

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
            self.rnn = nn.GRU(self.output_cnn_size, self.hidden_size, num_layers=self.num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.output_cnn_size, self.hidden_size, num_layers=self.num_layers)

        #self.resize = nn.Linear(self.hidden_size, ARGS.ind_size+1)
        if self.attn_type == 'ff':
            self.resize = nn.Linear(self.hidden_size, ARGS.dec_size)
        elif self.attn_type == 'dot':
            self.resize = nn.Linear(self.hidden_size, ARGS.ind_size)
        self.resize = nn.Linear(self.hidden_size, ARGS.dec_size)
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
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        outputs, hidden = self.rnn(cnn_outputs, hidden)
        outputs = self.resize(outputs)

        return outputs, hidden

    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            if self.rnn_type == 'gru':
                return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)
            elif self.rnn_type == 'lstm':
                return (torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5), torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5))
                
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/torch.norm(self.hidden_init,2)


class DecoderRNN(nn.Module):
    
    def __init__(self, ARGS, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = ARGS.dec_size
        self.num_layers = ARGS.dec_layers
        #self.dropout_p = ARGS.dropout
        self.batch_size = ARGS.batch_size
        self.device = device
        self.output_size = ARGS.ind_size+1

        self.rnn = nn.GRU(ARGS.ind_size, self.hidden_size, num_layers=self.num_layers)
        self.attention = Attention(ARGS.ind_size)
        #self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_init = torch.randn(self.num_layers, 1, self.hidden_size, device=device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init

        # Resize from GRU size to embedding size
        self.out1 = nn.Linear(self.hidden_size, ARGS.ind_size)


    def forward(self, input_, hidden, input_lengths, encoder_outputs):
        
        #if self.training:
        #    drop_input = self.dropout(input_)
        #else:
        #    drop_input = input_
        
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

        #if not self.training:
            #output = (1-self.dropout_p)*output

        output = self.out1(output)

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
        self.hidden_size = int(ARGS.dec_size)
        self.num_layers = int(ARGS.dec_layers)
        #self.dropout_p = ARGS.dropout
        self.batch_size = int(ARGS.batch_size)
        self.device = ARGS.device
        self.output_size = int(ARGS.ind_size)
        self.rnn_type = ARGS.dec_rnn

        #self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_init = torch.randn(self.num_layers, 1, int(self.hidden_size), device=ARGS.device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.dec_init
        self.attn_type = 'dot'

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.output_size, self.hidden_size, num_layers=self.num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.output_size, self.hidden_size, num_layers=self.num_layers)

        self.resize = nn.Linear(self.hidden_size, ARGS.ind_size)
        self.out = nn.Linear(2*self.hidden_size, self.output_size)
        self.dummy_out =nn.Linear(1,self.output_size)


    def get_attention_context_concatenation(self, input_, hidden, input_lengths, encoder_outputs):
        
        max_seq_len = input_.shape[0]
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        enc_perm = encoder_outputs.permute(1,2,0)
        output_perm = output.permute(1,0,2)

        longest_in_batch = output.shape[0]
        if self.attn_type == 'ff':
            ff_attn_weights = torch.zeros(longest_in_batch, 8, encoder_outputs.shape[1], device=self.device)
            for i in range(longest_in_batch):
                for j in range(8):
                    ff_attn_input = torch.cat([output[i], encoder_outputs[j]], dim=1)
                    new_weight = self.attention_layer(ff_attn_input)
                    ff_attn_weights[i,j] = new_weight.squeeze(-1)
            ff_enc_perm = encoder_outputs.transpose(0,1)
            ff_attn_weights = ff_attn_weights.permute(2,0,1)
            ff_attn_weights = F.softmax(ff_attn_weights, dim=2)
            attn_contexts = torch.bmm(ff_attn_weights, ff_enc_perm)
            attn_concat_outp = torch.cat([attn_contexts, output_perm], dim=-1)
        
        elif self.attn_type == 'dot':
            attn_weights = torch.bmm(output_perm, enc_perm).permute(0,2,1)
            attn_weights = F.softmax(attn_weights, dim=1)
            weighted_enc_outp = torch.matmul(enc_perm, attn_weights).permute(0,2,1)
            attn_concat_outp = torch.cat([output_perm, weighted_enc_outp], dim=2)

        return attn_concat_outp, hidden
        

    def forward(self, input_, hidden, input_lengths, encoder_outputs):
        #attn_concat_outp, hidden = self.get_attention_context_concatenation(input_, hidden, input_lengths, encoder_outputs)
        hidden = torch.randn(1,self.batch_size,self.hidden_size).cuda()

        #output = self.out(attn_concat_outp)
        largest_in_batch = int(torch.max(input_lengths).item())
        #dummy_inp = torch.randn([self.batch_size,largest_in_batch,1])
        output = self.dummy_out(torch.randn([self.batch_size,largest_in_batch,1]).cuda())
        
        return output, hidden


    def initHidden(self):
        if self.init_type == 'zeroes':
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit':
            if self.rnn_type == 'gru':
                return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)
            elif self.rnn_type == 'lstm':
                return (torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5), torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5))
        elif self.init_type == 'learned':
            return self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        elif self.init_type == 'unit_learned':
            return  self.hidden_init+torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/torch.norm(self.hidden_init,2)

       
class NumIndRegressor(nn.Module):
    
    def __init__(self, ARGS, device):
        super(NumIndRegressor, self).__init__()
        #sizes = [ARGS.enc_size] + ARGS.reg_sizes + [1]
        #self.linears= nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        #final_linear = nn.Linear(sizes[-1], 1)
        #self.linears.append(final_linear)
        self.dropout_p = ARGS.dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.activation = nn.LeakyReLU()
        #self.single = nn.Linear(50,1)
        self.l1 = nn.Linear(ARGS.enc_size, ARGS.reg_sizes[0])

        self.l2 = nn.Linear(ARGS.reg_sizes[0], ARGS.reg_sizes[1])
        self.l3 = nn.Linear(ARGS.reg_sizes[1], 1)

    def forward(self, x):
        #for linear in self.linears:
        #    print(x[0,0])
        #    x = linear(x)
        #    print(x[0,0])
        #    x = self.activation(x)
        #    print(x[0,0])
        #    x = self.dropout(x)
        #    print(x[0,0])
        #return x.squeeze()
        #print(self.l1)
        #print(x.shape)
        x = self.l1(x)
        print('l1',x)
        x = self.activation(x)
        print('l1',x)
        #if self.training:
        #    x = self.dropout(x)
        #else:
        #    x = x*(1-self.dropout_p)

        x = self.l2(x)
        print('l2',x)
        x = self.activation(x)
        print('l2',x)
        print(x)
        #if self.training:
        #    x = self.dropout(x)
        #else:
        #    x = x*(1-self.dropout_p)

        x = self.l3(x)
        print('l3',x)
        x = self.activation(x)
        print('l3',x)
        return x.squeeze()






