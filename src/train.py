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


def train_on_batch(ARGS, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    encoder.batch_size = ARGS.batch_size
    decoder.batch_size = ARGS.batch_size

    use_teacher_forcing = True if random.random() < ARGS.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #encoder_hidden = encoder.initHidden()#.to(device)
    #encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs, encoder_hidden = None,None

    decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)
    if ARGS.enc_dec_hidden_init:
        decoder_hidden = encoder_hidden
    else:
        decoder_hidden = decoder.initHidden()
 
    if use_teacher_forcing:
       
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden)

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        
        longest_in_batch = decoder_outputs.shape[1]
        target_tensor_perm = target_tensor[:longest_in_batch].permute(1,0,2)
        arange = torch.arange(0, longest_in_batch, step=1).expand(ARGS.batch_size, -1).to(ARGS.device)
        lengths = target_number_tensor.expand(longest_in_batch, ARGS.batch_size).long().to(ARGS.device)
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.float().unsqueeze(2)
        assert (mask == make_mask(target_number_tensor)).all()

        decoder_outputs_masked = decoder_outputs*mask
        emb_preds = decoder_outputs_masked


        loss = criterion(emb_preds*mask, target_tensor_perm[:,:longest_in_batch,:]*mask, ARGS.batch_size)
       
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, longest_in_batch, 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        norm_criterion = nn.MSELoss()
        norm_loss = norm_criterion(ARGS.norm_threshold*torch.ones(ARGS.batch_size, longest_in_batch, device=ARGS.device), output_norms)

        packing_rescale = ARGS.batch_size * longest_in_batch/torch.sum(target_number_tensor) 
        loss = loss*packing_rescale
        norm_loss = norm_loss*packing_rescale

    else:
        loss = 0 
        norm_loss = 0
        decoder_hidden_0 = encoder_hidden
        decoder_hidden_0 = decoder_hidden
        for b in range(ARGS.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            if ARGS.dec_rnn == 'gru':
                single_dec_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            elif ARGS.dec_rnn == 'lstm':
                single_dec_hidden = (decoder_hidden_0[0][:, b].unsqueeze(1), decoder_hidden_0[1][:, b].unsqueeze(1))
            loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, single_dec_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=single_dec_hidden) 
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                #norm_loss += F.relu(1-mean_norm)
                norm_loss += (mean_norm.item()-1)**2
                loss += criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0), batch_size=1)
                single_dec_input = decoder_output

        loss /= torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
    
    total_loss = loss + ARGS.lmbda_norm*norm_loss
    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return round(loss.item(),3), round(norm_loss.item(),3)


def train_on_batch_pred(ARGS, input_tensor, target_tensor, target_number_tensor, video_ids, encoder, decoder, encoder_optimizer, decoder_optimizer, json_data_dict, mlp_dict, device):

    encoder.train()
    decoder.train()
    encoder.batch_size = ARGS.batch_size

    use_teacher_forcing = True if random.random() < ARGS.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)

    if use_teacher_forcing:
        if ARGS.enc_dec_hidden_init:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = decoder.initHidden()
        
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous())

        norms = [torch.norm(decoder_outputs[i][j]).item() for i in range(decoder.batch_size) for j in range(int(target_number_tensor[i].item()))]

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        target_tensor_perm = target_tensor.permute(1,0,2)
        arange = torch.arange(0, decoder_outputs.shape[1], step=1).expand(ARGS.batch_size, -1).to(ARGS.device)
        lengths = target_number_tensor.expand(decoder_outputs.shape[1], ARGS.batch_size).long().to(ARGS.device)
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.float().unsqueeze(2)

        assisted_embeddings = (decoder_outputs + ARGS.pred_embeddings_assist*target_tensor[:decoder_outputs.shape[1],:,:].permute(1,0,2))/(1+ARGS.pred_embeddings_assist)
        assert (ARGS.pred_embeddings_assist==0) == torch.all(torch.eq(assisted_embeddings, decoder_outputs))
        if ARGS.pred_normalize:
            assisted_embeddings = F.normalize(assisted_embeddings, p=2, dim=-1)

        loss = get_pred_loss(video_ids, assisted_embeddings, json_data_dict, mlp_dict, margin=ARGS.pred_margin, device=device)
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, decoder_outputs.shape[1], 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        norm_criterion = nn.MSELoss()
        norm_loss = norm_criterion(ARGS.norm_threshold*torch.ones(ARGS.batch_size, decoder_outputs.shape[1], device=ARGS.device), output_norms)

        packing_rescale = ARGS.batch_size * decoder_outputs.shape[1]/torch.sum(target_number_tensor) 
        norm_loss = norm_loss*packing_rescale
        loss = loss*packing_rescale
        total_loss = loss + ARGS.lmbda_norm*norm_loss
        
    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    #return round(loss.item(), 3), round(norm_loss.item(),3)
    return loss.item(), norm_loss.item()


def make_mlp_dict_from_pickle(fname,grad=False):
    mlp_dict = {}
    print(mlp_dict)
    weight_dict = torch.load(fname)
    for rel, weights in weight_dict.items():
        hidden_lyr = nn.Linear(weights["hidden_weights"].shape[1], weights["hidden_bias"].shape[0])
        hidden_lyr.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=grad)
        hidden_lyr.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=grad)
        output_lyr = nn.Linear(weights["output_weights"].shape[1], weights["output_bias"].shape[0])
        output_lyr.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=grad)
        output_lyr.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=grad)
        mlp_dict[rel] = nn.Sequential(hidden_lyr, nn.ReLU(), output_lyr)
    return mlp_dict

def train(ARGS, encoder, decoder, transformer, dataset, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    
    json_data_dict, mlp_dict = None,None
    if ARGS.setting == 'preds':
        gt_file_path = f'/data1/louis/data/rdf_video_captions/{dataset}.json'
        mlp_weights_file_path = f'/data1/louis/data/{dataset}-mlps.pickle'

        with open(gt_file_path, 'r') as f:
            json_data_list = json.load(f)
            json_data_dict = {v['video_id']: v for v in json_data_list}

        mlp_dict = make_mlp_dict_from_pickle(mlp_weights_file_path)
    if ARGS.setting in ['preds', 'embeddings']:
        v = 1
        for param in encoder.cnn.parameters():
            param.requires_grad = False
            v += 1
        if encoder_optimizer == None:
            encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        if decoder_optimizer == None:
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
    elif ARGS.setting == 'transformer':
        transformer_optimizer =optim.Adam(transformer.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay) 

    mse = nn.MSELoss()
    def criterion(network_output, ground_truth, batch_size):
        return ARGS.ind_size*(mse(network_output, ground_truth))

    EarlyStop = EarlyStopper(patience=ARGS.patience, verbose=True)
    
    epoch_train_losses = []
    epoch_train_norm_losses = []
    epoch_val_losses = []
    epoch_val_norm_losses = []

    for epoch_num in range(ARGS.max_epochs):
        epoch_start_time = time.time()
        batch_train_losses = []
        batch_train_norm_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            video_ids = training_triplet[3].float().to(device)
            if ARGS.setting == 'embeddings':
                new_train_loss, new_train_norm_loss= train_on_batch(
                    ARGS, 
                    input_tensor, 
                    target_tensor, 
                    target_number, 
                    encoder=encoder, 
                    decoder=decoder, 
                    encoder_optimizer=encoder_optimizer, 
                    decoder_optimizer=decoder_optimizer, 
                    criterion=criterion, 
                    device=device)
            elif ARGS.setting == 'preds':
                new_train_loss, new_train_norm_loss = train_on_batch_pred(
                    ARGS, 
                    input_tensor, 
                    target_tensor,
                    target_number, 
                    video_ids, 
                    encoder, 
                    decoder, 
                    encoder_optimizer, 
                    decoder_optimizer, 
                    json_data_dict, 
                    mlp_dict, 
                    device)
            else:
                print('Unrecognized setting: {}'.format(ARGS.setting))
            print('Batch:', iter_, 'dec loss:', new_train_loss, 'norm loss', new_train_norm_loss)
            
            batch_train_losses.append(new_train_loss)
            batch_train_norm_losses.append(new_train_norm_loss)
            if ARGS.quick_run:
                break
        batch_val_losses = []
        batch_val_norm_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            video_ids = training_triplet[3].float().to(device)
            new_val_loss, new_val_norm_loss = eval_on_batch(ARGS, input_tensor, target_tensor, target_number, video_ids=video_ids, encoder=encoder, decoder=decoder, transformer=transformer, dec_criterion=criterion, mlp_dict=mlp_dict, json_data_dict=json_data_dict, device=device)
            batch_val_losses.append(new_val_loss)
            batch_val_norm_losses.append(new_val_norm_loss)

            if ARGS.quick_run:
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
        save_dict = {'encoder':encoder, 'decoder':decoder, 'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        save = not ARGS.no_chkpt
        EarlyStop(new_epoch_val_loss, save_dict, exp_name=exp_name, save=save)
        
        print('val_loss', new_epoch_val_loss)
        if EarlyStop.early_stop:
            break 
   
        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')
    losses = {'train': epoch_train_losses, 'train_norm': epoch_train_norm_losses,'val': epoch_val_losses,'val_norm': epoch_val_norm_losses}

    return losses, EarlyStop.early_stop


def eval_on_batch(ARGS, input_tensor, target_tensor, target_number_tensor, video_ids=None, encoder=None, decoder=None, transformer=None, dec_criterion=None, mlp_dict=None, json_data_dict=None, device='cpu'):
    
    if ARGS.setting == "transformer":
        cnn = models.vgg19(pretrained=True).cuda()
        v = 1
        for param in cnn.parameters():
            param.requires_grad = False
            v += 1
 
        transformer.eval()
        cnn_outputs = torch.zeros(8, input_tensor.shape[1], 4096, device=device)
        for i, inp in enumerate(input_tensor):
            x = cnn.features(inp)
            x = cnn.avgpool(x)
            x = x.view(x.size(0), -1)
            x = cnn.classifier[0](x)
            cnn_outputs[i] = x
        cnn_outputs = cnn_outputs.permute(1,0,2)

        target_tensor_perm = target_tensor.permute(1,0,2)
        for b in range(ARGS.batch_size):
            growing_output = torch.zeros(1, 1, ARGS.ind_size, device=device)
            next_transformer_preds = growing_output
            l=0
            while True:
                l+=1
                next_transformer_preds = transformer(cnn_outputs[b].unsqueeze(0), next_transformer_preds)
                if l == target_number_tensor[b].int():
                    break
                next_transformer_preds = torch.cat([next_transformer_preds, growing_output], dim=1)
            t_loss = dec_criterion(next_transformer_preds, target_tensor_perm[b,:target_number_tensor[b].int()].unsqueeze(0), batch_size=1)
            output_norms = torch.norm(next_transformer_preds, dim=-1)
            norm_criterion = nn.MSELoss()
            norm_loss = norm_criterion(ARGS.norm_threshold*torch.ones(ARGS.batch_size, target_number_tensor[b].int(), device=device), output_norms)
            return t_loss.item(), norm_loss.item()

    elif ARGS.setting in ["embeddings", "preds"]:
        encoder.eval()
        decoder.eval()
        encoder.batch_size = ARGS.eval_batch_size

        encoder_hidden = encoder.initHidden()#.to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)

        if ARGS.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, ARGS.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = decoder.initHidden()
        dec_loss = torch.tensor([0]).float().to(device)
        norm_loss = torch.tensor([0]).float().to(device)
        denom = torch.tensor([0]).float().to(device)
        l2_distances = []
        total_dist = torch.zeros([ARGS.ind_size], device=device).float()
        decoder.batch_size=1
        for b in range(ARGS.eval_batch_size-1):
            dec_out_list = []
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            if ARGS.dec_rnn == 'gru':
                single_dec_hidden = decoder_hidden_0[:, b].unsqueeze(1)
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
            if ARGS.setting == "preds":
                dec_loss += get_pred_loss(video_ids[b].unsqueeze(0), dec_out_tensor, json_data_dict, mlp_dict, margin=ARGS.pred_margin, device=device)
            elif ARGS.setting == "embeddings":
                dec_loss += l_loss*ARGS.ind_size/float(l)
        index_tensor = (target_number_tensor.long()-1).unsqueeze(0).unsqueeze(-1)
        index_tensor = index_tensor.squeeze()
        
        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        dec_loss /= torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
        
        return round(dec_loss.item(),3), round(norm_loss.item(),3)
