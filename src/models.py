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
        self.i3d = ARGS.i3d and not ARGS.i3d_after
        self.i3d_size = 1024 if self.i3d else 0
        self.attn_type = ARGS.attn_type

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

        #self.resize = nn.Linear(self.hidden_size, ARGS.ind_size+1)
        if self.attn_type == 'ff':
            self.resize = nn.Linear(self.hidden_size, ARGS.dec_size)
        elif self.attn_type == 'dot':
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
        self.hidden_size = ARGS.dec_size
        self.num_layers = ARGS.dec_layers
        self.dropout_p = ARGS.dropout
        self.batch_size = ARGS.batch_size
        self.device = ARGS.device
        self.output_size = ARGS.ind_size

        self.i3d = ARGS.i3d and ARGS.i3d_after
        self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_init = torch.randn(self.num_layers, 1, self.hidden_size, device=ARGS.device)
        self.hidden_init.requires_grad = True
        self.init_type = ARGS.enc_init
        self.attn_type = ARGS.attn_type

        if self.i3d:
            self.rnn = nn.GRU(ARGS.ind_size+1024, self.hidden_size, num_layers=self.num_layers)
        else:
            self.rnn = nn.GRU(ARGS.ind_size, self.hidden_size, num_layers=self.num_layers)

        if self.attn_type == 'dot':
            self.resize = nn.Linear(self.hidden_size, ARGS.ind_size)
            self.out = nn.Linear(2*ARGS.ind_size, self.output_size)
        elif self.attn_type == 'ff':
            self.attention_layer = nn.Sequential(nn.Linear(2*ARGS.dec_size, 1), nn.Tanh())
            self.out = nn.Linear(2*self.hidden_size, self.output_size)

        self.eos1 = nn.Sequential(nn.Linear(2*ARGS.ind_size, ARGS.ind_size), nn.ReLU())
        self.eos2 = nn.Sequential(nn.Linear(ARGS.ind_size, 1), nn.Sigmoid())


    def get_attention_context_concatenation(self, input_, hidden, input_lengths, encoder_outputs, i3d):
        # apply dropout
        max_seq_len = input_.shape[0]
        if self.i3d:
            batch_size = input_.shape[1]
            i3d_expanded = i3d.repeat(1, max_seq_len)
            i3d_expanded = i3d_expanded.view(batch_size, max_seq_len, 1024)
            i3d_expanded = i3d_expanded.transpose(0,1)
            input_ = torch.cat([input_, i3d_expanded], dim=-1)
        if self.training:
            drop_input = self.dropout(input_)
        else:
            drop_input = (1-self.dropout_p)*input_
        
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        enc_perm = encoder_outputs.permute(1,2,0)
        output_perm = output.permute(1,0,2)

        longest_in_batch = output.shape[0]
        if self.attn_type == 'ff':
            ff_attn_weights = torch.zeros(longest_in_batch, 8, encoder_outputs.shape[1], device=self.device)
            for i in range(longest_in_batch):
                for j in range(8):
                    #print(output[i].shape, encoder_outputs[j].shape)
                    ff_attn_input = torch.cat([output[i], encoder_outputs[j]], dim=1)
                    #print('ffat', ff_attn_input.shape)
                    new_weight = self.attention_layer(ff_attn_input)
                    ff_attn_weights[i,j] = new_weight.squeeze(-1)
                    #print(ff_attn_weights[i,j])
            ff_enc_perm = encoder_outputs.transpose(0,1)
            ff_attn_weights = ff_attn_weights.permute(2,0,1)
            ff_attn_weights = F.softmax(ff_attn_weights, dim=2)
            #print('post trans', ff_attn_weights.shape)
            #print(ff_attn_weights.sum(dim=2))
            #print(torch.ones(self.batch_size, longest_in_batch, device=self.device))
            #print(ff_attn_weights.sum(dim=2) == torch.ones(self.batch_size, longest_in_batch, device=self.device))
            #assert (ff_attn_weights.sum(dim=2) == torch.ones(self.batch_size, longest_in_batch, device=self.device)).all()
            attn_contexts = torch.bmm(ff_attn_weights, ff_enc_perm)
            attn_concat_outp = torch.cat([attn_contexts, output_perm], dim=-1)
        
        elif self.attn_type == 'dot':
            output_perm = self.resize(output_perm)
            attn_weights = torch.bmm(output_perm, enc_perm).permute(0,2,1)
            attn_weights = F.softmax(attn_weights, dim=1)
            weighted_enc_outp = torch.matmul(enc_perm, attn_weights).permute(0,2,1)
            attn_concat_outp = torch.cat([output_perm, weighted_enc_outp], dim=2)

        #print(attn_concat_outp.shape)
        return attn_concat_outp, hidden
        

    def forward(self, input_, hidden, input_lengths, encoder_outputs, i3d):
        attn_concat_outp, hidden = self.get_attention_context_concatenation(input_, hidden, input_lengths, encoder_outputs, i3d)

        #if self.i3d:
            #attn_concat_outp = torch.cat([attn_concat_outp, i3d], dim=-1)
        output = self.out(attn_concat_outp)
        
        return output, hidden

    def eos_preds(self, input_, hidden, input_lengths, encoder_outputs):
        attn_concat_outp, hidden = self.get_attention_context_concatenation(input_, hidden, input_lengths, encoder_outputs)

        eos_pred = self.eos1(attn_concat_outp)
        eos_pred = self.eos2(eos_pred)
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

       
class NumIndRegressor(nn.Module):
    
    def __init__(self, ARGS, device):
        super(NumIndRegressor, self).__init__()
        #sizes = [ARGS.enc_size] + ARGS.reg_sizes + [1]
        #self.linears= nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        #final_linear = nn.Linear(sizes[-1], 1)
        #self.linears.append(final_linear)
        self.dropout_p = ARGS.dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.activation = nn.ReLU()
        #self.single = nn.Linear(50,1)
        if ARGS.i3d:
            self.l1 = nn.Linear(ARGS.enc_size + 1024, ARGS.reg_sizes[0])
        else:
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
        x = self.l1(x)
        x = self.activation(x)
        if self.training:
            x = self.dropout(x)
        else:
            x = x*(1-self.dropout_p)

        x = self.l2(x)
        x = self.activation(x)
        if self.training:
            x = self.dropout(x)
        else:
            x = x*(1-self.dropout_p)

        x = self.l3(x)
        x = self.activation(x)
        return x.squeeze()






