from pdb import set_trace
import json
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

import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from early_stopper import EarlyStopper
from attention import Attention
from get_pred import get_pred_loss
import pretrainedmodels


def make_mask(target_number_tensor):
    longest_in_batch = int(torch.max(target_number_tensor).item())
    batch_size = target_number_tensor.shape[0]
    device = target_number_tensor.device
    arange = torch.arange(0, longest_in_batch, step=1).expand(batch_size, -1).to(device)
    lengths = target_number_tensor.expand(longest_in_batch, batch_size).long().to(device)
    lengths = lengths.permute(1,0)
    mask = arange < lengths 
    mask = mask.float().unsqueeze(2)
    assert (torch.sum(mask, dim=1) == target_number_tensor.unsqueeze(1)).all()
    return mask


def train_on_batch(ARGS, input_tensor, multiclass_inds, atoms, encoder, multiclassifier, ind_mlp_dict, optimizer, criterion, device, train):
    if train: 
        encoder.train()
        multiclassifier.train()
    else:
        encoder.eval()
        multiclassifier.eval()
    encoder.batch_size = ARGS.batch_size
    encoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    encoding = encoder(input_tensor)
    mutliclassif_logits = multiclassifier(encoding)
    multiclass_loss = criterion(multiclassif_logits,multiclass_inds)

    pred_loss = get_pred_loss(video_ids,encoding,json_data_dict,ind_dict,mlp_dict)
    loss = multiclass_loss + ARGS.lmbda*pred_loss
    if train: total_loss.backward(); optimizer.step(); optimizer.zero_grad()

    return round(loss.item(),5)

def train(ARGS, encoder, multiclassifier, json_data_dict, ind_dict, mlp_dict, train_dl, val_dl, exp_name, device, optimizer=None):
    
    for param in encoder.cnn.parameters():
        param.requires_grad = False
    if optimizer == None:
        encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        params_list = [encoder.parameters(), decoder.parameters(), multiclassifier.parameters()] + [ind.parameters() for ind in ind_dict.values()] + [mlp.parameters() for mlp in mlp_dict.values()]
        optimizer = optim.Adam([{'params': params, 'lr':ARGS.learning_rate, 'wd':ARGS.weight_decay} for params in params_list])

    EarlyStop = EarlyStopper(patience=ARGS.patience, verbose=True)
    
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch_num in range(ARGS.max_epochs):
        epoch_start_time = time.time()
        batch_train_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            multiclass_inds = training_triplet[1].float().transpose(0,1).to(device)
            atoms = training_triplet[2].float().to(device)
            video_ids = training_triplet[3].float().to(device)
            new_train_loss = train_on_batch(
                ARGS, 
                input_tensor, 
                multiclass_inds, 
                encoder=encoder, 
                multiclassifier=multiclassifier, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=device, train=True)
            if new_train_loss == 0: set_trace()
            print('Batch:', iter_, 'dec loss:', new_train_loss, 'norm loss', new_train_norm_loss)
            
            batch_train_losses.append(new_train_loss)
            if ARGS.quick_run:
                break
        batch_val_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            multiclass_inds = training_triplet[1].float().transpose(0,1).to(device)
            atoms = training_triplet[2].float().to(device)
            video_ids = training_triplet[3].float().to(device)
            new_val_loss= eval_on_batch(ARGS, input_tensor, multiclass_inds, video_ids=video_ids, encoder=encoder, multiclassifier=multiclassifier, criterion=criterion, mlp_dict=mlp_dict, json_data_dict=json_data_dict, device=device, train=False)
            batch_val_losses.append(new_val_loss)

            if ARGS.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        try:
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
       
        new_epoch_val_norm_loss = sum(batch_val_norm_losses)/len(batch_val_norm_losses)
        
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_train_norm_losses.append(new_epoch_train_norm_loss)

        epoch_val_losses.append(new_epoch_val_loss)
        save_dict = {'encoder':encoder, 'multiclassifier':multiclassifier, 'ind_dict': ind_dict, 'mlp_dict': mmlp_dict, 'optimizer': optimizer}
        save = not ARGS.no_chkpt and new_epoch_val_loss < 0.01 and random.random() < 0.1
        EarlyStop(new_epoch_val_loss, save_dict, exp_name=exp_name, save=save)
        
        print('val_loss', new_epoch_val_loss)
        if EarlyStop.early_stop:
            break 
   
        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')
    losses = {'train': epoch_train_losses, 'train_norm': epoch_train_norm_losses,'val': epoch_val_losses,'val_norm': epoch_val_norm_losses}

    return losses, EarlyStop.early_stop


def eval_on_batch(ARGS, input_tensor, target_tensor, multiclass_inds, video_ids, encoder, criterion, mlp_dict, json_data_dict, device):
    encoder.eval()
    multiclass_inds.eval()
    encoder.batch_size = ARGS.eval_batch_size

    encoder_hidden = encoder.initHidden()#.to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    denom = torch.tensor([0]).float().to(device)
    l2_distances = []
    total_dist = torch.zeros([ARGS.ind_size], device=device).float()
    for b in range(ARGS.eval_batch_size):
        dec_out_list = []
        single_dec_input = decoder_input[:, b].view(1, 1, -1)
        if ARGS.dec_rnn == 'gru':
            try: single_dec_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            except: set_trace()
        elif ARGS.dec_rnn == 'lstm':
            single_dec_hidden = (decoder_hidden_0[0][:, b].unsqueeze(1), decoder_hidden_0[1][:, b].unsqueeze(1))
        l_loss = 0
        try:
            for l in range(target_number_tensor[b].int()):
                decoder_output, single_dec_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=single_dec_hidden) 
                dec_out_list.append(decoder_output)
                arange = torch.arange(0, l, step=1).expand(1, -1).to(ARGS.device)
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                norm_loss += (mean_norm.item()-1)**2
                emb_pred = decoder_output

                if ARGS.setting == "embeddings":
                    l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0), batch_size=1)
                single_dec_input = target_tensor[l,b].unsqueeze(0).unsqueeze(0)
                denom += 1
                l2 = torch.norm(emb_pred.squeeze()-target_tensor[l,b].squeeze(),2).item()
                dist = emb_pred.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                l2_distances.append(l2)
        except: 
            set_trace()
        dec_out_tensor = torch.cat(dec_out_list, dim=1).to(device)
        dec_loss += l_loss*ARGS.ind_size/float(l)
    index_tensor = (target_number_tensor.long()-1).unsqueeze(0).unsqueeze(-1)
    index_tensor = index_tensor.squeeze()
    
    print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
    dec_loss /= torch.sum(target_number_tensor)
    norm_loss /= torch.sum(target_number_tensor)
    
    return round(dec_loss.item(),5), round(norm_loss.item(),5)
