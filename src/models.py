import numpy as np
import random
from random import shuffle
import re
import string
import time
import unicodedata
import utils

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from early_stopper import EarlyStopper
from attention import Attention

##############################
#
# ENCODER
#
##############################

class EncoderRNN(nn.Module):
    
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.num_frames = args.num_frames
        self.output_vgg_size = args.output_vgg_size
        #self.hidden_size = args.ind_size
        self.hidden_size = args.enc_size
        self.device = device
        self.batch_size = args.batch_size
        self.num_layers = args.enc_layers

        #Use VGG, NOTE: param.requires_grad are set to True by default
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        #Changes the size of VGG output to the GRU size
        self.vgg.classifier[6] = nn.Linear(num_ftrs, self.output_vgg_size)

        #Encoder Recurrent layer
        self.gru = nn.GRU(self.output_vgg_size, self.hidden_size, num_layers=self.num_layers)

        # Resize outputs to ind_size
        self.resize = nn.Linear(self.hidden_size, args.ind_size)

    def forward(self, input, hidden):

        # pass the input throught the vgg layers 
        #print(self.output_vgg_size, self.hidden_size, self.num_layers)
        vgg_outputs = torch.zeros(self.num_frames, input.shape[1], self.output_vgg_size, device=self.device)
        for i, inp in enumerate(input):
            embedded = self.vgg(inp)#.view(1, self.batch_size, -1)
            vgg_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        #print(vgg_outputs.shape)
        outputs, hidden = self.gru(vgg_outputs, hidden)
        
        outputs = self.resize(outputs)

        #print(" \n NETWORK INPUTS (first element in batch) \n")
        #print(input[0,0,:,:,0])
        #print(" \n NETWORK INPUTS (second element in batch) \n")
        #print(input[0,1,:,:,0])
        #print(" \n VGG OUTPUTS (first element in batch) \n")
        #print(vgg_outputs[:, 0])
        #print(" \n VGG OUTPUTS (second element in batch) \n")
        #print(vgg_outputs[:, 1])
        #print(" \n OUTPUTS (first element in batch \n")
        #print(outputs[:, 0])
        #print(" \n OUTPUTS (second element in batch \n")
        #print(outputs[:, 1])

        return outputs, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)




##############################
#
# DECODER
#
##############################

class DecoderRNN(nn.Module):
    
    def __init__(self, args, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.dec_size
        self.max_length = args.max_length
        self.num_layers = args.dec_layers
        self.dropout_p = args.dropout
        self.batch_size = args.batch_size
        self.device = device

        self.gru = nn.GRU(args.ind_size, self.hidden_size, num_layers=self.num_layers)
        #self.attention = Attention(self.hidden_size)
        self.attention = Attention(args.ind_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.out1 = nn.Linear(self.hidden_size, args.ind_size)
        #self.out2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.out3 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, input_lengths, encoder_outputs):
        # apply dropout
        #print('input', input)
        drop_input = self.dropout(input)
        
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)

        # permute dimensions to have batch_size in the first
        #enc_perm shape: (batch_size, num_frames, ind_size)
        enc_perm = encoder_outputs.permute(1,0,2)
        #output_perm shape: (batch_size, max_length, ind_size)
        output_perm = output.permute(1,0,2)

        # apply attention
        #output_perm, attn = self.attention(output_perm, enc_perm)
        output = self.out1(output_perm)
        #print('output b4 attn', output)
        output, attn = self.attention(output, enc_perm)
        # permute back to (max_length, batch_size, ind_size)
        #output = output_perm.permute(1,0,2)

        #output = self.out2(output)
        #output = self.out3(output)
        # output = self.out4(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)




##############################
#
# TRAIN FUNCTIONS
#
##############################


def train_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):

    #encoder.train()
    #decoder.train()

    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)

    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)


    if use_teacher_forcing:

        #if args.enc_layers == args.dec_layers:
        if args.enc_dec_hidden_init:
            decoder_hidden = encoder_hidden#.expand(decoder.num_layers, args.batch_size, args.ind_size)
        else:
            decoder_hidden = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size).to(device)
            #decoder_hidden = torch.zeros(args.dec_layers, args.batch_size, args.ind_size)
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous())

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        target_tensor_perm = target_tensor.permute(1,0,2)

        #loss = criterion(decoder_outputs*mask, target_tensor[:,:decoder_outputs.shape[1],:]*mask)
        loss = criterion(decoder_outputs, target_tensor_perm[:,:decoder_outputs.shape[1],:])
        #print('\tcomputing loss between {} and {}, result: {}'.format(decoder_outputs.squeeze()[0,0].item(), target_tensor.squeeze()[0,0].item(), loss))
        mse_rescale = (args.ind_size*args.batch_size*decoder_outputs.shape[1])/torch.sum(target_number_tensor)
        #print(args.batch_size*decoder_outputs.shape[1]/torch.sum(target_number_tensor))
        #print('unscaled_loss', loss)
        loss = loss*mse_rescale
        #print('rescaled_loss', loss)

    else:

        loss = 0 
        decoder_hidden_0 = encoder_hidden#.expand(decoder.num_layers,args.batch_size, args.ind_size)
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) #input_lengths=torch.tensor([target_number_tensor[b]])
                l_loss += criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
            loss += l_loss/float(l)
        loss /= float(b)

    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train(args, encoder, decoder, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    # where to save the image of the loss funcitons
    loss_plot_file_path = '../data/loss_plots/loss_seq2seq{}.png'.format(exp_name)
    checkpoint_path = '../checkpoints/chkpt{}.pt'.format(exp_name)

    if encoder_optimizer == None:
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

    # Freeze vgg layers that we don't want to train
    v = 1
    for param in encoder.vgg.parameters():
        if v <= args.vgg_layers_to_freeze*2: # Assuming each layer has two params
            param.requires_grad = False
        v += 1


    epoch_train_losses = []
    epoch_val_losses = []
    for epoch_num in range(args.max_epochs):
        encoder.train()
        decoder.train()
        batch_train_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            video_id = training_triplet[4].float().to(device)
            #print('input tensor on train', input_tensor.squeeze()[0,0,0,0].item())
            new_train_loss = train_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion, device=device)
            #print(video_id)
            #print(target_tensor.squeeze()[0])
            print(iter_, new_train_loss)
            #print(input_tensor[0,0,0,0,0])

            batch_train_losses.append(new_train_loss)
            if args.quick_run:
                break
        encoder.eval()
        decoder.eval()
        batch_val_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            #print('input tensor on eval', input_tensor.squeeze()[0,0,0,0].item())
            new_val_loss = eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, dec_criterion=criterion, device=device)
            batch_val_losses.append(new_val_loss)
            print(iter_, new_val_loss)

            if args.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_val_losses.append(new_epoch_val_loss)
        utils.plot_losses(epoch_train_losses, epoch_val_losses, loss_plot_file_path)
        save_dict = {'encoder':encoder, 'decoder':decoder, 'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        save = (new_epoch_val_loss < 0.5)
        print(save)
        EarlyStop(new_epoch_val_loss, save_dict, filename=checkpoint_path, save=save)
        #checkpoint = torch.load(checkpoint_path)
        #reloaded_encoder = checkpoint['encoder']
        #reloaded_decoder = checkpoint['decoder']
        #reload_val_losses = []
        #for iter_, training_triplet in enumerate(val_generator):
        #    input_tensor = training_triplet[0].float().transpose(0,1).to(device)
        #    target_tensor = training_triplet[1].float().transpose(0,1).to(device)
        #    target_number = training_triplet[2].float().to(device)
        #    new_reload_val_loss = eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=reloaded_encoder, decoder=reloaded_decoder, dec_criterion=criterion, device=device)
        #    reload_val_losses.append(new_reload_val_loss)

        #    if args.quick_run:
        #        break

        #reload_val_loss = sum(reload_val_losses)/len(reload_val_losses)
        print('val_loss', new_epoch_val_loss)
        #print('reloaded_val_loss', reload_val_loss)
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min

    return new_epoch_val_loss



##############################
#
# VALIDATION FUNCTION
#
##############################

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

        #if encoder.hidden_size == decoder.hidden_size:
        if args.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, args.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size, device=device)
        #decoder_hidden_0 = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size, device=device)
        #if torch.cuda.is_available():
            #decoder_input = decoder_input.cuda()
        dec_loss = 0
        denom = 0
        l2_distances = []
        total_dist = torch.zeros([50], device=device).float()
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, new_decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) #input_lengths=torch.tensor([target_number_tensor[b]])
                decoder_hidden = new_decoder_hidden
                l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
                denom += 1
                #print(decoder_output.squeeze(), target_tensor[l,b].squeeze())
                l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
                dist = decoder_output.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                #print('new l2 at', l, b, l2)
                l2_distances.append(l2)
                #print('\tcomputing norm between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l2))
                #print('\tcomputing loss between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l_loss))
            #dec_loss += l_loss/float(l)
            dec_loss += l_loss

        #print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        #dec_loss /= float(b) 
        #print('dist', total_dist)
        dec_loss = dec_loss*50/torch.sum(target_number_tensor)
        if dec_loss < 2.5:
            #print(decoder_output.squeeze())
            #print(target_tensor[l,b].squeeze())
            #print('l2:', torch.norm(decoder_output-target_tensor[l,b],2).item())
            pass
        return dec_loss.item()

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


