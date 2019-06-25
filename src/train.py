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


def train_on_batch_eos(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, eos_target, encoder, eos_decoder, optimizer, criterion):
    
    encoder_hidden = encoder.initHidden()
    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
    eos_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=eos_decoder.device)
    eos_hidden = eos_decoder.initHidden()
    eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
    eos_preds, hidden = eos_decoder.eos_preds(input_=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
    
    #loss = criterion(outputs.squeeze(2), eos_target[:outputs.shape[0],:])
    #loss = criterion(outputs.squeeze(2), eos_target.long()-1)
    loss = criterion(eos_preds.squeeze(2), eos_target-1)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_on_batch(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, eos_target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, eos_criterion, device):

    use_teacher_forcing = True if random.random() < ARGS.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)

    decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)

    if use_teacher_forcing:

        if ARGS.enc_dec_hidden_init:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = decoder.initHidden()
        
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous())

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        target_tensor_perm = target_tensor.permute(1,0,2)
        
        arange = torch.arange(0, decoder_outputs.shape[1], step=1).expand(ARGS.batch_size, -1).to(ARGS.device)
        lengths = target_number_tensor.expand(decoder_outputs.shape[1], ARGS.batch_size).long().to(ARGS.device)
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.float().unsqueeze(2)

        #loss = criterion(decoder_outputs[:,:,:-1]*mask, target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask, ARGS.batch_size)
        loss = criterion(decoder_outputs*mask, target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask, ARGS.batch_size)
        #cut_eos_target = eos_target[:,:decoder_outputs.shape[1]]
        #print(torch.sum(cut_eos_target, dim=1))
        #assert (torch.sum(cut_eos_target, dim=1).long() == torch.ones(ARGS.batch_size, device=ARGS.device)).all()
        #eos_loss = eos_criterion(eos_pred.squeeze(-1), eos_target-1)
        cos = nn.CosineEmbeddingLoss()
        #print(cos(decoder_outputs*mask, target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask, torch.ones(1, device=decoder.device)))
        #mse_rescale = (ARGS.ind_size*ARGS.batch_size*decoder_outputs.shape[1])/torch.sum(target_number_tensor)
        
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, decoder_outputs.shape[1], 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        norm_loss = F.relu(((ARGS.norm_threshold*torch.ones(ARGS.batch_size, decoder_outputs.shape[1], device=ARGS.device)) - output_norms)*mask).mean()

        packing_rescale = ARGS.batch_size * decoder_outputs.shape[1]/torch.sum(target_number_tensor) 
        loss = loss*packing_rescale
        norm_loss = norm_loss*packing_rescale

        #total_loss = (loss + ARGS.lmbda_norm*norm_loss + ARGS.lmbda_eos*eos_loss)
        total_loss = loss + ARGS.lmbda_norm*norm_loss
    else:
        loss = 0 
        decoder_hidden_0 = encoder_hidden
        for b in range(ARGS.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) 
                l_loss += criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
            loss += l_loss/float(l)
        loss /= float(b)

    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), norm_loss.item()


def train_on_batch_pred(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, video_ids, encoder, decoder, encoder_optimizer, decoder_optimizer, json_data_dict, mlp_dict, device):

    use_teacher_forcing = True if random.random() < ARGS.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
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

        loss = get_pred_loss(video_ids, assisted_embeddings, json_data_dict, mlp_dict, neg_weight=ARGS.neg_pred_weight, log_pred=ARGS.log_pred, device=device)
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, decoder_outputs.shape[1], 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        norm_loss = F.relu(((ARGS.norm_threshold*torch.ones(ARGS.batch_size, decoder_outputs.shape[1], device=ARGS.device)) - output_norms)*mask).mean()

        packing_rescale = ARGS.batch_size * decoder_outputs.shape[1]/torch.sum(target_number_tensor) 
        norm_loss = norm_loss*packing_rescale
        total_loss = loss + ARGS.lmbda_norm*norm_loss
        
    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item(), norm_loss.item()


def train(ARGS, encoder, decoder, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    
    loss_plot_file_path = '../data/loss_plots/loss_{}.png'.format(exp_name)
    checkpoint_path = '../checkpoints/{}.pt'.format(exp_name)

    mlp_dict = {}
    json_data_dict = None
    if ARGS.setting == 'preds':
        with open('/data2/commons/rdf_video_captions/{}d-det.json.neg'.format(ARGS.ind_size), 'r') as f:
            json_data_list = json.load(f)
            json_data_dict = {v['videoId']: v for v in json_data_list}
                       
        weight_dict = torch.load("../data/{}d-mlps.pickle".format(ARGS.ind_size))
        for relation, weights in weight_dict.items():
            hidden_layer = nn.Linear(weights["hidden_weights"].shape[0], weights["hidden_bias"].shape[0])
            hidden_layer.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=False)
            hidden_layer.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=False)
            output_layer = nn.Linear(weights["output_weights"].shape[0], weights["output_bias"].shape[0])
            output_layer.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=False)
            output_layer.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=False)
            if ARGS.sigmoid_mlp:
                mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer, nn.Sigmoid()) 
            else:
                mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer)

    v = 1
    for param in encoder.cnn.parameters():
        #if v <= ARGS.cnn_layers_to_freeze*2: # Assuming each layer has two params
            #param.requires_grad = False
        param.requires_grad = False
        v += 1

    if encoder_optimizer == None:
        encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        if ARGS.optimizer == "SGD":
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        elif ARGS.optimizer == "Adam":
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        elif ARGS.optimizer == "RMS":
            encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)

    if decoder_optimizer == None:
        if ARGS.optimizer == "SGD":
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        elif ARGS.optimizer == "Adam":
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        elif ARGS.optimizer == "RMS":
            decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)

    if ARGS.loss_func == 'mse':
        mse = nn.MSELoss()
        def criterion(network_output, ground_truth, batch_size):
            return ARGS.ind_size*(mse(network_output, ground_truth))
    elif ARGS.loss_func == 'cos':
        cos = nn.CosineEmbeddingLoss()
        def criterion(network_output, ground_truth, batch_size):
            return cos(network_output, ground_truth, target=torch.ones(batch_size, 1,1, device=ARGS.device))

    eos_criterion = nn.CrossEntropyLoss()

    EarlyStop = EarlyStopper(patience=ARGS.patience, verbose=True)
    
    epoch_train_losses = []
    epoch_train_norm_losses = []
    epoch_train_eos_losses = []
    epoch_val_losses = []
    epoch_val_norm_losses = []
    epoch_val_eos_losses = []
    for epoch_num in range(ARGS.max_epochs):
        encoder.train()
        decoder.train()
        batch_train_losses = []
        batch_train_norm_losses = []
        batch_train_eos_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            eos_target = training_triplet[3].long().to(device)
            video_ids = training_triplet[4].float().to(device)
            i3d_vec = training_triplet[5].float().to(device)
            if ARGS.i3d:
                assert i3d_vec.shape == torch.Size([ARGS.batch_size, 1024])
            else:
                assert i3d_vec.shape == torch.Size([ARGS.batch_size])
            print('i3d batch vec', i3d_vec.shape)
            if ARGS.setting == 'embeddings':
                new_train_loss, new_train_norm_loss = train_on_batch(
                    ARGS, 
                    input_tensor, 
                    target_tensor, 
                    target_number, 
                    i3d_vec, 
                    eos_target=target_number.long(), 
                    encoder=encoder, 
                    decoder=decoder, 
                    encoder_optimizer=encoder_optimizer, 
                    decoder_optimizer=decoder_optimizer, 
                    criterion=criterion, 
                    eos_criterion=eos_criterion, 
                    device=device)
            elif ARGS.setting == 'preds':
                new_train_loss, new_train_norm_loss = train_on_batch_pred(
                    ARGS, 
                    input_tensor, 
                    target_tensor,
                    target_number, 
                    i3d_vec, 
                    video_ids, 
                    encoder, 
                    decoder, 
                    encoder_optimizer, 
                    decoder_optimizer, 
                    json_data_dict, 
                    mlp_dict, 
                    device)
            elif ARGS.setting == 'eos':
                new_train_loss = train_on_batch_eos(
                    ARGS, 
                    input_tensor, 
                    target_tensor,
                    target_number, 
                    i3d_vec, 
                    eos_target=target_number.long(), 
                    encoder=encoder, 
                    eos_decoder=decoder, 
                    optimizer=decoder_optimizer, 
                    criterion=eos_criterion)
                new_train_norm_loss = -1
            print('Batch:', iter_, 'dec loss:', new_train_loss, 'norm loss', new_train_norm_loss)
            #total_norms += norms1
            
            batch_train_losses.append(new_train_loss)
            batch_train_norm_losses.append(new_train_norm_loss)
            if ARGS.quick_run:
                break

        encoder.eval()
        decoder.eval()
        batch_val_losses = []
        batch_val_norm_losses = []
        batch_val_eos_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            eos_target = training_triplet[3].float().to(device).transpose(0,1).to(device)
            video_ids = training_triplet[4].float().to(device)
            i3d_vec = training_triplet[5].float().to(device)
            new_val_loss, new_val_norm_loss, new_val_eos_loss = eval_on_batch("eval_seq2seq", ARGS, input_tensor, target_tensor, target_number, i3d_vec, video_ids=video_ids, eos_target=eos_target, encoder=encoder, decoder=decoder, dec_criterion=criterion, eos_criterion=eos_criterion, mlp_dict=mlp_dict, json_data_dict=json_data_dict, device=device)
            batch_val_losses.append(new_val_loss)
            batch_val_norm_losses.append(new_val_norm_loss)
            batch_val_eos_losses.append(new_val_eos_loss)
            print('val', iter_, new_val_loss, new_val_norm_loss)

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
   
    losses = {  'train': epoch_train_losses,
                'train_norm': epoch_train_norm_losses,
                'train_eos': epoch_train_eos_losses,
                'val': epoch_val_losses,
                'val_norm': epoch_val_norm_losses,
                'val_eos': epoch_val_eos_losses}
    if not ARGS.mini:
        EarlyStop.save_to_disk(exp_name)
    assert EarlyStop.val_loss_min == min(losses['val'])
    return losses, EarlyStop.early_stop


def eval_on_batch(mode, ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, video_ids=None, eos_target=None, encoder=None, decoder=None, regressor=None, eos=None, dec_criterion=None, reg_criterion=None, eos_criterion=None, mlp_dict=None, json_data_dict=None, device='cpu'):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "eval_eos", "test"}"""

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
    
    if mode == "eval_reg":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        return reg_loss.item()

    elif mode == "eval_seq2seq":
        decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)

        if ARGS.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, ARGS.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = decoder.initHidden()
        dec_loss = 0
        norm_loss = 0
        eos_loss = 0
        denom = 0
        eos_preds_list = []
        l2_distances = []
        total_dist = torch.zeros([ARGS.ind_size], device=device).float()
        for b in range(ARGS.batch_size):
            dec_out_list = []
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, new_decoder_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) 
                dec_out_list.append(decoder_output)
                if ARGS.setting == "eos":
                    new_eos_component, _hidden = decoder.eos_preds(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous())
                    eos_preds_list.append(new_eos_component)
                decoder_hidden = new_decoder_hidden
                arange = torch.arange(0, l, step=1).expand(1, -1).to(ARGS.device)
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                norm_loss += F.relu(1-mean_norm)
                
                if ARGS.setting == "embeddings":
                    l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0), batch_size=1)
                single_dec_input = target_tensor[l,b].unsqueeze(0).unsqueeze(0)
                denom += 1
                l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
                dist = decoder_output.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                l2_distances.append(l2)
            dec_out_tensor = torch.cat(dec_out_list, dim=1)
            if ARGS.setting == "preds":
                dec_loss += get_pred_loss(video_ids[b].unsqueeze(0), dec_out_tensor, json_data_dict, mlp_dict, neg_weight=ARGS.neg_pred_weight, log_pred=ARGS.log_pred, device=device)
            elif ARGS.setting == "embeddings":
                dec_loss += l_loss*ARGS.ind_size/float(l)
            elif ARGS.setting == "eos":
                eos_preds_tensor = torch.cat(eos_preds_list, dim=1)
                dec_loss += eos_criterion(eos_preds_tensor.squeeze(2), target_number_tensor[b].unsqueeze(0).long()-1)

        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        dec_loss /= torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
        
        return dec_loss.item(), norm_loss.item(), -1

    elif mode == "eval_eos":
        eos_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=eos.device).to(device)
        eos_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
        eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
        eos_outputs, hidden = eos(input_=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
        eos_target = eos_target[:eos_outputs.shape[0],:].permute(1,0)
        eos_outputs = eos_outputs.squeeze(2).permute(1,0)
        
        arange = torch.arange(0, eos_outputs.shape[1], step=1).expand(ARGS.batch_size, eos_outputs.shape[1]).cuda()
        lengths = target_number_tensor.expand(eos_outputs.shape[1], ARGS.batch_size).long().cuda()
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.cuda().float()
        eos_target = torch.argmax(eos_target, dim=1)
        
        eos_loss = eos_criterion(eos_outputs*mask, eos_target*mask)
        return eos_loss.item()


