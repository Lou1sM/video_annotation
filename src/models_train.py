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
    
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.num_frames = args.num_frames
        self.output_cnn_size = args.output_cnn_size
        self.hidden_size = args.enc_size
        self.device = device
        self.batch_size = args.batch_size
        self.num_layers = args.enc_layers
        self.rnn_type = args.enc_rnn
        self.cnn_type = args.enc_cnn
        self.output_size = args.ind_size

        if args.enc_cnn == "vgg_old":
            self.cnn = models.vgg19(pretrained=True)
            num_ftrs = self.cnn.classifier[6].in_features
            self.cnn.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)
        elif args.enc_cnn == "vgg":
            self.cnn = models.vgg19(pretrained=True)
        elif args.enc_cnn == "nasnet":
            model_name = 'nasnetalarge'
            self.cnn = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            print(args.enc_cnn)

        #Encoder Recurrent layer
        self.gru = nn.GRU(self.output_cnn_size, self.hidden_size, num_layers=self.num_layers)

        # Resize outputs to ind_size
        self.resize = nn.Linear(self.hidden_size, args.ind_size)
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


    def forward(self, input, hidden):

        # pass the input through the cnn 
        cnn_outputs = torch.zeros(self.num_frames, input.shape[1], self.output_cnn_size, device=self.device)
        for i, inp in enumerate(input):
            embedded = self.cnn_vector(inp)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        outputs, hidden = self.gru(cnn_outputs, hidden)
        outputs = self.resize(outputs)

        return outputs, hidden

    def initHidden(self, zeroes=True):
        if zeroes:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)


class DecoderRNN(nn.Module):
    
    def __init__(self, args, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.dec_size
        self.num_layers = args.dec_layers
        self.dropout_p = args.dropout
        self.batch_size = args.batch_size
        self.device = device
        self.output_size = args.ind_size

        self.gru = nn.GRU(args.ind_size, self.hidden_size, num_layers=self.num_layers)
        self.attention = Attention(args.ind_size)
        self.dropout = nn.Dropout(self.dropout_p)

        # Resize from GRU size to embedding size
        self.out1 = nn.Linear(self.hidden_size, args.ind_size)

    def forward(self, input, hidden, input_lengths, encoder_outputs):
        # apply dropout
        drop_input = self.dropout(input)
        
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)

        # permute dimensions to have batch_size in the first
        #enc_perm shape: (batch_size, num_frames, ind_size)
        enc_perm = encoder_outputs.permute(1,0,2)
        output_perm = output.permute(1,0,2)

        # apply attention
        output = self.out1(output_perm)
        output, attn = self.attention(output, enc_perm)

        return output, hidden

    def initHidden(self, zeroes):
        if zeroes:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.ones(self.num_layers, self.batch_size, self.hidden_size, device=self.device)/(self.output_size**0.5)


def train_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):

    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(args.enc_zeroes).to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)

    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)

    if use_teacher_forcing:

        if args.enc_dec_hidden_init:
            decoder_hidden = encoder_hidden#.expand(decoder.num_layers, args.batch_size, args.ind_size)
        else:
            #decoder_hidden = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size).to(device)
            decoder_hidden = decoder.initHidden(args.dec_zeroes)
        
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous())

        norms = [torch.norm(decoder_outputs[i][j]).item() for i in range(decoder.batch_size) for j in range(int(target_number_tensor[i].item()))]

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        target_tensor_perm = target_tensor.permute(1,0,2)
        
        arange = torch.arange(0, decoder_outputs.shape[1], step=1).expand(args.batch_size, -1).to(args.device)
        lengths = target_number_tensor.expand(decoder_outputs.shape[1], args.batch_size).long().to(args.device)
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.float().unsqueeze(2)

        loss = criterion(decoder_outputs*mask, target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask)
        mse_rescale = (args.ind_size*args.batch_size*decoder_outputs.shape[1])/torch.sum(target_number_tensor)
        loss = loss*mse_rescale
        
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(args.batch_size, decoder_outputs.shape[1], 1, device=args.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        norm_loss = F.relu(((args.norm_threshold*torch.ones(args.batch_size, decoder_outputs.shape[1], device=args.device)) - output_norms)*mask).mean()*mse_rescale
        total_loss = loss +  norm_loss*args.lmbda

    else:
        loss = 0 
        decoder_hidden_0 = encoder_hidden#.expand(decoder.num_layers,args.batch_size, args.ind_size)
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) 
                l_loss += criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
            loss += l_loss/float(l)
        loss /= float(b)

    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), norm_loss.item(), norms


def train(args, encoder, decoder, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    # where to save the image of the loss funcitons
    loss_plot_file_path = '../data/loss_plots/loss_{}.png'.format(exp_name)
    checkpoint_path = '../checkpoints/chkpt{}.pt'.format(exp_name)

    v = 1
    for param in encoder.cnn.parameters():
        #if v <= args.cnn_layers_to_freeze*2: # Assuming each layer has two params
            #param.requires_grad = False
        param.requires_grad = False
        v += 1

    if encoder_optimizer == None:
        encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        if args.optimizer == "SGD":
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "RMS":
            encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if decoder_optimizer == None:
        if args.optimizer == "SGD":
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "RMS":
            decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    
    epoch_train_losses = []
    epoch_train_norm_losses = []
    epoch_val_losses = []
    epoch_val_norm_losses = []
    for epoch_num in range(args.max_epochs):
        total_norms = []
        total_val_norms = []
        encoder.train()
        decoder.train()
        batch_train_losses = []
        batch_train_norm_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            video_id = training_triplet[4].float().to(device)
            new_train_loss, new_train_norm_loss, norms1 = train_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion, device=device)
            print(iter_, new_train_loss, new_train_norm_loss)
            total_norms += norms1
            
            batch_train_losses.append(new_train_loss)
            batch_train_norm_losses.append(new_train_norm_loss)
            if args.quick_run:
                break
        encoder.eval()
        decoder.eval()
        batch_val_losses = []
        batch_val_norm_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            new_val_loss, new_val_norm_loss, val_norms = eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, dec_criterion=criterion, device=device)
            batch_val_losses.append(new_val_loss)
            batch_val_norm_losses.append(new_val_norm_loss)
            print('val', iter_, new_val_loss, new_val_norm_loss)

            total_val_norms += val_norms
            if args.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_train_norm_loss = sum(batch_train_norm_losses)/len(batch_train_norm_losses)
        try:
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
       
        new_epoch_val_norm_loss = sum(batch_val_norm_losses)/len(batch_val_norm_losses)
        
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_train_norm_losses.append(new_epoch_train_norm_loss)

        epoch_val_losses.append(new_epoch_val_loss)
        epoch_val_norm_losses.append(new_epoch_val_norm_loss)
        print(epoch_train_losses)
        print(epoch_val_losses)
        print(epoch_train_norm_losses)
        print(epoch_val_norm_losses)
        save_dict = {'encoder':encoder, 'decoder':decoder, 'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        save = (new_epoch_val_loss < 5.1) and args.chkpt
        EarlyStop(new_epoch_val_loss, save_dict, filename=checkpoint_path, save=save)
        
        print('val_loss', new_epoch_val_loss)
        if EarlyStop.early_stop:
            break 
   
    return EarlyStop.early_stop


def eval_on_batch(mode, args, input_tensor, target_tensor, target_number_tensor=None, eos_target=None, encoder=None, decoder=None, regressor=None, eos=None, dec_criterion=None, reg_criterion=None, eos_criterion=None, device='cpu'):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "eval_eos", "test"}"""

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    if mode == "eval_reg":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        return reg_loss.item()

    elif mode == "eval_seq2seq":
        decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)

        if args.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, args.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size, device=device)
        dec_loss = 0
        denom = 0
        l2_distances = []
        total_dist = torch.zeros([args.ind_size], device=device).float()
        norms =[] 
        norm_loss = 0
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, new_decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) 
                new_norm = torch.norm(decoder_output).item()
                #if new_norm > 0.1:
                norms.append(new_norm)
                decoder_hidden = new_decoder_hidden
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                norm_loss += F.relu(1-mean_norm)
                
                l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                #single_dec_input = decoder_output
                single_dec_input = target_tensor[l,b].unsqueeze(0).unsqueeze(0)
                denom += 1
                l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
                dist = decoder_output.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                l2_distances.append(l2)
                #print('\tcomputing norm between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l2))
                #print('\tcomputing loss between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l_loss))
            #dec_loss += l_loss/float(l)
            dec_loss += l_loss

        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        dec_loss = dec_loss*args.ind_size/torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
        
        return dec_loss.item(), norm_loss.item(), norms

    elif mode == "eval_eos":
        eos_input = torch.zeros(1, args.batch_size, args.ind_size, device=eos.device).to(device)
        eos_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
        eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
        eos_outputs, hidden = eos(input=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
        eos_target = eos_target[:eos_outputs.shape[0],:].permute(1,0)
        eos_outputs = eos_outputs.squeeze(2).permute(1,0)
        
        arange = torch.arange(0, eos_outputs.shape[1], step=1).expand(args.batch_size, eos_outputs.shape[1]).cuda()
        lengths = target_number_tensor.expand(eos_outputs.shape[1], args.batch_size).long().cuda()
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.cuda().float()
        eos_target = torch.argmax(eos_target, dim=1)
        
        eos_loss = eos_criterion(eos_outputs*mask, eos_target*mask)
        return eos_loss.item()


