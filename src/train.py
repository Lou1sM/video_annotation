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


#torch.manual_seed(0)

#cnn = pretrainedmodels.__dict__['vgg'](num_classes=1000, pretrained='imagenet')
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


def train_on_batch_transformer(ARGS, input_tensor, target_tensor, target_number_tensor, eos_target, transformer, optimizer, criterion):
    #(batch_size, time_step, vector_size)

    cnn = models.vgg19(pretrained=True).cuda()
    v = 1
    for param in cnn.parameters():
        #if v <= ARGS.cnn_layers_to_freeze*2: # Assuming each layer has two params
            #param.requires_grad = False
        param.requires_grad = False
        v += 1
    
    optimizer.zero_grad()

    longest_in_batch = torch.max(target_number_tensor).int()
    target_tensor = target_tensor[:longest_in_batch]
    eos_target = eos_target[:,:longest_in_batch]
    cnn_outputs = torch.zeros(8, target_tensor.shape[1], 4096, device='cuda')

    for i, inp in enumerate(input_tensor):
        x = cnn.features(inp)
        x = cnn.avgpool(x)
        x = x.view(x.size(0), -1)
        x = cnn.classifier[0](x)
        cnn_outputs[i] = x

    cnn_outputs = cnn_outputs.permute(1,0,2)
    transformer.cuda()
    #criterion = nn.MSELoss()
    target_tensor_perm = target_tensor.permute(1,0,2)
    transformer_preds = transformer(cnn_outputs, target_tensor_perm)
    arange = torch.arange(0, longest_in_batch, step=1).expand(ARGS.batch_size, -1).to(ARGS.device)
    lengths = target_number_tensor.expand(longest_in_batch, ARGS.batch_size).long().to(ARGS.device)
    lengths = lengths.permute(1,0)
    mask = arange < lengths 
    mask = mask.float().unsqueeze(2)

    transformer_preds_masked = transformer_preds*mask
    loss = criterion(transformer_preds_masked, target_tensor_perm, batch_size=ARGS.batch_size)

    inv_byte_mask = mask.byte()^1
    inv_mask = inv_byte_mask.float()
    assert (mask+inv_mask == torch.ones(ARGS.batch_size, longest_in_batch, 1, device=ARGS.device)).all()
    output_norms = torch.norm(transformer_preds, dim=-1)
    mask = mask.squeeze(2)
    inv_mask = inv_mask.squeeze(2)
    output_norms = output_norms*mask + inv_mask
    mean_norm = output_norms.mean()
    #if ARGS.norm_loss == 'relu':
    #norm_loss = F.relu(((ARGS.norm_threshold*torch.ones(ARGS.batch_size, longest_in_batch, device=ARGS.device)) - output_norms)*mask).mean()
    #elif ARGS.norm_loss == 'mse':
    norm_criterion = nn.MSELoss()
    norm_loss = norm_criterion(ARGS.norm_threshold*torch.ones(ARGS.batch_size, longest_in_batch, device=ARGS.device), output_norms)

    packing_rescale = ARGS.batch_size*longest_in_batch/torch.sum(target_number_tensor) 
    loss = loss*packing_rescale
    norm_loss = norm_loss*packing_rescale

    #total_loss = (loss + ARGS.lmbda_norm*norm_loss + ARGS.lmbda_eos*eos_loss)
    #bin_criterion = nn.BCEWithLogitsLoss()
    #eos_preds= transformer_preds[:,:,-1].squeeze()
    #eos_loss = bin_criterion(eos_preds, eos_target.float())
    #eos_loss = 0
    total_loss = loss + ARGS.lmbda_norm*norm_loss + reg_loss
    total_loss.backward()
    optimizer.step()
    return round(loss.item(), 3), round(norm_loss.item(),3)


def train_on_batch_eos(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, eos_target, encoder, eos_decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    encoder.train()
    eos_decoder.train()
    encoder_hidden = encoder.initHidden()
    #optimizer = optim.Adam(eos_decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
    eos_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=eos_decoder.device)
    eos_hidden = eos_decoder.initHidden()
    eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
    eos_preds, hidden = eos_decoder.eos_preds(input_=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden, i3d=i3d_vec)
    longest_in_batch = eos_preds.shape[1]
    
    mask = make_mask(target_number_tensor)
    #loss = criterion(outputs.squeeze(2), eos_target[:outputs.shape[0],:])
    loss = criterion((eos_preds*mask).squeeze(2), eos_target[:,:longest_in_batch])
    print((eos_preds*mask)[0], eos_target[0,:longest_in_batch])
    index_tensor = (target_number_tensor.long()-1).unsqueeze(0).unsqueeze(-1)
    index_tensor = index_tensor.squeeze()
    #print(eos_preds.squeeze().shape)
    #print(index_tensor.shape)
    if ARGS.reweight_eos:
        logits_at_ones = eos_preds.squeeze().gather(1,index_tensor.view(-1,1))
        #print(logits_at_ones)
        scaled_logits_at_ones = torch.mul(logits_at_ones.squeeze(),target_number_tensor.squeeze()-2)
        #print(scaled_logits_at_ones)
        #print(longest_in_batch)
        scaled_logits_at_ones = scaled_logits_at_ones/(longest_in_batch-1)
        #print(scaled_logits_at_ones)
        eos_criterion = nn.BCEWithLogitsLoss(reduce=False)
        loss_at_ones = eos_criterion(logits_at_ones.squeeze(), torch.ones(ARGS.batch_size, device=ARGS.device))
        #print(loss_at_ones)
        scaled_loss_at_ones = torch.mul(loss_at_ones.squeeze(),target_number_tensor.squeeze()-2)
        #print(scaled_loss_at_ones)
        scaled_loss_at_ones = scaled_loss_at_ones/(longest_in_batch-1)
        #print(scaled_loss_at_ones)
        #ones_loss = -scaled_logits_at_ones.mean()
        ones_loss = scaled_loss_at_ones.mean()
        #print('ones', ones_loss.item(), 'normal', loss.item())
        total_loss = loss+ones_loss
    else:
        total_loss = loss
    total_loss.backward()
    #print(list(eos_decoder.parameters())[0].grad)
    for param in eos_decoder.parameters():
        #print(param.requires_grad)
        #print(param.data)
        #print(param.grad)
        pass
    encoder_optimizer.step()
    decoder_optimizer.step()

    return round(loss.item(),3)


def train_on_batch(ARGS, input_tensor, target_tensor, target_number_tensor, eos_target, i3d_vec, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, eos_criterion, device):
    encoder.train()
    decoder.train()

    use_teacher_forcing = True if random.random() < ARGS.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden()#.to(device)
    #print('before', encoder_hidden.shape)
    encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
    #print('after', encoder_hidden.shape)

    #concat_target = torch.cat([target_tensor, eos_target.float().transpose(1,0).unsqueeze(-1)], dim=2).permute(1,0,2)
    decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)
    if ARGS.enc_dec_hidden_init:
        decoder_hidden = encoder_hidden
    else:
        decoder_hidden = decoder.initHidden()
 
    if use_teacher_forcing:
       
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        #if ARGS.i3d and ARGS.i3d_after:
        #    decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous(), i3d=i3d_vec)
        #else:
        decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden, i3d=i3d_vec)

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
        #emb_preds = decoder_outputs_masked[:,:,:-1]
        emb_preds = decoder_outputs_masked
        #eos_preds = decoder_outputs_masked[:,:,-1]

        #eos_target = eos_target[:,:longest_in_batch].float()
        #bin_criterion = nn.BCEWithLogitsLoss()

        #print('mask', mask.shape)
        #print('embs', emb_preds.shape)
        #print((mask*emb_preds).shape)
        #print('ttp', target_tensor_perm.shape)
        #print((target_tensor_perm*mask).shape)
        #print('eos', eos_preds.shape)
        #print('eosf', eos_preds_flat.shape)
        #print(target_number_tensor)
        #eos_loss = bin_criterion(eos_preds, eos_target.float())
        #loss = criterion(decoder_outputs*mask, target_tensor_perm[:,:longest_in_batch,:]*mask, ARGS.batch_size)
        loss = criterion(emb_preds*mask, target_tensor_perm[:,:longest_in_batch,:]*mask, ARGS.batch_size)
        #cut_eos_target = eos_target[:,:longest_in_batch]
        #assert (torch.sum(cut_eos_target, dim=1).long() == torch.ones(ARGS.batch_size, device=ARGS.device)).all()
        #cos = nn.CosineEmbeddingLoss()
        #print(cos(decoder_outputs*mask, target_tensor_perm[:,:longest_in_batch,:]*mask, torch.ones(1, device=decoder.device)))
        #mse_rescale = (ARGS.ind_size*ARGS.batch_size*longest_in_batch)/torch.sum(target_number_tensor)
        
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, longest_in_batch, 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        #if ARGS.norm_loss == 'relu':
        #    norm_loss = F.relu(((ARGS.norm_threshold*torch.ones(ARGS.batch_size, longest_in_batch, device=ARGS.device)) - output_norms)*mask).mean()
        #elif ARGS.norm_loss == 'mse':
        norm_criterion = nn.MSELoss()
        norm_loss = norm_criterion(ARGS.norm_threshold*torch.ones(ARGS.batch_size, longest_in_batch, device=ARGS.device), output_norms)

        packing_rescale = ARGS.batch_size * longest_in_batch/torch.sum(target_number_tensor) 
        loss = loss*packing_rescale
        norm_loss = norm_loss*packing_rescale

        #total_loss = (loss + ARGS.lmbda_norm*norm_loss + ARGS.lmbda_eos*eos_loss)
        #total_loss = eos_loss
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
                decoder_output, single_dec_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=single_dec_hidden, i3d=i3d_vec) 
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

    #return round(loss.item(),3), round(norm_loss.item(),3), round(eos_loss.item(),3)
    return round(loss.item(),3), round(norm_loss.item(),3), -1


def train_on_batch_pred(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, video_ids, encoder, decoder, encoder_optimizer, decoder_optimizer, json_data_dict, mlp_dict, device):

    encoder.train()
    decoder.train()

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
        decoder_outputs, decoder_hidden = decoder(input_=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous(), i3d=i3d_vec)

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

        loss = get_pred_loss(video_ids, assisted_embeddings, json_data_dict, mlp_dict, neg_weight=ARGS.neg_pred_weight, margin=ARGS.pred_margin, device=device)
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        assert (mask+inv_mask == torch.ones(ARGS.batch_size, decoder_outputs.shape[1], 1, device=ARGS.device)).all()
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        output_norms = output_norms*mask + inv_mask
        mean_norm = output_norms.mean()
        #norm_loss = F.relu(((ARGS.norm_threshold*torch.ones(ARGS.batch_size, decoder_outputs.shape[1], device=ARGS.device)) - output_norms)*mask).mean()
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


def train(ARGS, encoder, decoder, transformer, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    
    loss_plot_file_path = '../experiments/{}/{}_lossplot.png'.format(exp_name, exp_name)
    eos_loss_plot_file_path = '../experiments/{}/{}_eos_lossplot.png'.format(exp_name, exp_name)

    mlp_dict = {}
    json_data_dict = None
    if ARGS.setting == 'preds':
        if ARGS.exp_name.startswith('jade') or True:
            gt_file_path = '../data/rdf_video_captions/{}d-det.json.neg'.format(ARGS.ind_size)
        else:
            gt_file_path = '/data2/commons/rdf_video_captions/{}d-det.json.neg'.format(ARGS.ind_size)

        with open(gt_file_path, 'r') as f:
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

    if ARGS.setting in ['preds', 'embeddings', 'eos']:
        v = 1
        for param in encoder.cnn.parameters():
            #if v <= ARGS.cnn_layers_to_freeze*2: # Assuming each layer has two params
                #param.requires_grad = False
            param.requires_grad = False
            v += 1
        if encoder_optimizer == None:
            encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
        if decoder_optimizer == None:
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)
    elif ARGS.setting == 'transformer':
        transformer_optimizer =optim.Adam(transformer.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay) 

    #if ARGS.loss_func == 'mse':
    mse = nn.MSELoss()
    def criterion(network_output, ground_truth, batch_size):
        return ARGS.ind_size*(mse(network_output, ground_truth))
    #elif ARGS.loss_func == 'cos':
    #    cos = nn.CosineEmbeddingLoss()
    #    def criterion(network_output, ground_truth, batch_size):
    #        return cos(network_output, ground_truth, target=torch.ones(batch_size, 1,1, device=ARGS.device))

    #eos_criterion = nn.CrossEntropyLoss()
    #eos_criterion = nn.MSELoss()
    #eos_criterion = nn.BCEWithLogitsLoss(reduce=False)
    eos_criterion = nn.BCEWithLogitsLoss()

    EarlyStop = EarlyStopper(patience=ARGS.patience, verbose=True)
    
    epoch_train_losses = []
    epoch_train_norm_losses = []
    epoch_train_eos_losses = []
    epoch_val_losses = []
    epoch_val_norm_losses = []
    epoch_val_eos_losses = []
    for epoch_num in range(ARGS.max_epochs):
        #transformer.train()
        batch_train_losses = []
        batch_train_norm_losses = []
        batch_train_eos_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            eos_target = training_triplet[3].float().to(device)
            video_ids = training_triplet[4].float().to(device)
            i3d_vec = training_triplet[5].float().to(device)
            if ARGS.i3d:
                assert i3d_vec.shape == torch.Size([ARGS.batch_size, 1024])
            else:
                assert i3d_vec.shape == torch.Size([ARGS.batch_size])
            if ARGS.setting == 'embeddings':
                new_train_loss, new_train_norm_loss, new_train_eos_loss = train_on_batch(
                    ARGS, 
                    input_tensor, 
                    target_tensor, 
                    target_number, 
                    eos_target,
                    i3d_vec, 
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
                new_train_eos_loss = -1
            elif ARGS.setting == 'eos':
                new_train_eos_loss = train_on_batch_eos(
                    ARGS, 
                    input_tensor, 
                    target_tensor,
                    target_number, 
                    i3d_vec, 
                    #eos_target=target_number.long(), 
                    eos_target=eos_target,
                    encoder=encoder, 
                    eos_decoder=decoder, 
                    encoder_optimizer=encoder_optimizer, 
                    decoder_optimizer=decoder_optimizer, 
                    criterion=eos_criterion)
                new_train_loss = new_train_norm_loss = -1
            elif ARGS.setting == 'transformer':
                new_train_loss, new_train_norm_loss = train_on_batch_transformer(
                    ARGS, 
                    input_tensor, 
                    target_tensor,
                    target_number,
                    eos_target,
                    transformer=transformer,
                    optimizer=transformer_optimizer, 
                    criterion=criterion)
                new_train_eos_loss = -1
            else:
                print('Unrecognized setting: {}'.format(ARGS.setting))
            print('Batch:', iter_, 'dec loss:', new_train_loss, 'norm loss', new_train_norm_loss, 'eos oss:', new_train_eos_loss)
            #total_norms += norms1
            
            batch_train_losses.append(new_train_loss)
            batch_train_norm_losses.append(new_train_norm_loss)
            batch_train_eos_losses.append(new_train_eos_loss)
            if ARGS.quick_run:
                break
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
            new_val_loss, new_val_norm_loss, new_val_eos_loss = eval_on_batch(ARGS, input_tensor, target_tensor, target_number, i3d_vec, video_ids=video_ids, eos_target=eos_target, encoder=encoder, decoder=decoder, transformer=transformer, dec_criterion=criterion, eos_criterion=eos_criterion, mlp_dict=mlp_dict, json_data_dict=json_data_dict, device=device)
            batch_val_losses.append(new_val_loss)
            batch_val_norm_losses.append(new_val_norm_loss)
            batch_val_eos_losses.append(new_val_eos_loss)
            print('val', iter_, new_val_loss, new_val_norm_loss, new_val_eos_loss)

            if ARGS.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_train_norm_loss = sum(batch_train_norm_losses)/len(batch_train_norm_losses)
        new_epoch_train_eos_loss = sum(batch_train_eos_losses)/len(batch_train_eos_losses)
        try:
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
            new_epoch_val_eos_loss = sum(batch_val_eos_losses)/len(batch_val_eos_losses)
        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
            new_epoch_val_eos_loss = sum(batch_val_eos_losses)/len(batch_val_eos_losses)
       
        new_epoch_val_norm_loss = sum(batch_val_norm_losses)/len(batch_val_norm_losses)
        
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_train_norm_losses.append(new_epoch_train_norm_loss)
        epoch_train_eos_losses.append(new_epoch_train_eos_loss)

        epoch_val_losses.append(new_epoch_val_loss)
        epoch_val_norm_losses.append(new_epoch_val_norm_loss)
        epoch_val_eos_losses.append(new_epoch_val_eos_loss)
        save_dict = {'encoder':encoder, 'decoder':decoder, 'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        save = not ARGS.no_chkpt
        if ARGS.setting == 'eos':
            EarlyStop(new_epoch_val_eos_loss, save_dict, exp_name=exp_name, save=save)
        else:
            EarlyStop(new_epoch_val_loss, save_dict, exp_name=exp_name, save=save)
        
        print('val_loss', new_epoch_val_loss)
        utils.plot_losses(epoch_train_losses, epoch_val_losses, 'MSE', loss_plot_file_path)
        utils.plot_losses(epoch_train_eos_losses, epoch_val_eos_losses, 'MSE', eos_loss_plot_file_path)
        if EarlyStop.early_stop:
            break 
   
    losses = {  'train': epoch_train_losses,
                'train_norm': epoch_train_norm_losses,
                'train_eos': epoch_train_eos_losses,
                'val': epoch_val_losses,
                'val_norm': epoch_val_norm_losses,
                'val_eos': epoch_val_eos_losses}
    #if not ARGS.mini:
        #EarlyStop.save_to_disk(exp_name)
    #assert EarlyStop.val_loss_min == min(losses['val'])
    return losses, EarlyStop.early_stop


def eval_on_batch(ARGS, input_tensor, target_tensor, target_number_tensor, i3d_vec, video_ids=None, eos_target=None, encoder=None, decoder=None, transformer=None, regressor=None, eos=None, dec_criterion=None, reg_criterion=None, eos_criterion=None, mlp_dict=None, json_data_dict=None, device='cpu'):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "eval_eos", "test"}"""
    
    if ARGS.setting == "transformer":
        cnn = models.vgg19(pretrained=True).cuda()
        v = 1
        for param in cnn.parameters():
            #if v <= ARGS.cnn_layers_to_freeze*2: # Assuming each layer has two params
                #param.requires_grad = False
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
            #for l in range(target_number_tensor[b].int()):
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
            #eos_loss = torch.tensor([0])
            eos_loss = torch.tensor([0])
            return t_loss.item(), norm_loss.item(), eos_loss.item()

    elif ARGS.setting in ["embeddings", "preds", 'eos']:
        encoder.eval()
        decoder.eval()

        encoder_hidden = encoder.initHidden()#.to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, i3d_vec, encoder_hidden)
        decoder_input = torch.zeros(1, ARGS.batch_size, ARGS.ind_size, device=decoder.device).to(device)

        if ARGS.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, ARGS.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = decoder.initHidden()
        dec_loss = torch.tensor([0]).float().to(device)
        norm_loss = torch.tensor([0]).float().to(device)
        eos_loss = torch.tensor([0]).float().to(device)
        denom = torch.tensor([0]).float().to(device)
        eos_preds_list = []
        l2_distances = []
        total_dist = torch.zeros([ARGS.ind_size], device=device).float()
        eos_preds_tensor = torch.zeros(29, ARGS.batch_size, device=device).float()
        for b in range(ARGS.batch_size):
            dec_out_list = []
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            if ARGS.dec_rnn == 'gru':
                single_dec_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            elif ARGS.dec_rnn == 'lstm':
                single_dec_hidden = (decoder_hidden_0[0][:, b].unsqueeze(1), decoder_hidden_0[1][:, b].unsqueeze(1))
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, single_dec_hidden = decoder(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=single_dec_hidden, i3d=i3d_vec[b]) 
                dec_out_list.append(decoder_output)
                if ARGS.setting == "eos":
                    new_eos_component, _hidden = decoder.eos_preds(input_=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous(), i3d=i3d_vec)
                    eos_preds_list.append(new_eos_component)
                    eos_preds_tensor[l,b] = new_eos_component
                arange = torch.arange(0, l, step=1).expand(1, -1).to(ARGS.device)
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                #norm_loss += F.relu(1-mean_norm)
                norm_loss += (mean_norm.item()-1)**2
                #print('output_norms val', output_norm.shape)

                #emb_pred = decoder_output[:,:,:-1]
                emb_pred = decoder_output
                #eos_pred = decoder_output[:,:,-1]
                #eos_preds_list.append(eos_pred)

                if ARGS.setting == "embeddings":
                    l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0), batch_size=1)
                    #l_loss += dec_criterion(emb_pred, target_tensor[l, b].unsqueeze(0).unsqueeze(0), batch_size=1)
                single_dec_input = target_tensor[l,b].unsqueeze(0).unsqueeze(0)
                denom += 1
                #l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
                l2 = torch.norm(emb_pred.squeeze()-target_tensor[l,b].squeeze(),2).item()
                #dist = decoder_output.squeeze()-target_tensor[l,b].squeeze()
                dist = emb_pred.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                l2_distances.append(l2)
            dec_out_tensor = torch.cat(dec_out_list, dim=1).to(device)
            if ARGS.setting == "preds":
                #dec_loss += get_pred_loss(video_ids[b].unsqueeze(0), dec_out_tensor, json_data_dict, mlp_dict, neg_weight=ARGS.neg_pred_weight, device=device)
                dec_loss += get_pred_loss(video_ids[b].unsqueeze(0), dec_out_tensor, json_data_dict, mlp_dict, neg_weight=ARGS.neg_pred_weight, margin=ARGS.pred_margin, device=device)
            elif ARGS.setting == "embeddings":
                dec_loss += l_loss*ARGS.ind_size/float(l)
            elif ARGS.setting == "eos":
                #eos_preds_tensor = torch.cat(eos_preds_list, dim=1)
                #dec_loss += eos_criterion(eos_preds_tensor.squeeze(2), target_number_tensor[b].unsqueeze(0).long()-1)
                #dec_loss += eos_criterion(eos_preds_tensor, target_number_tensor[b].unsqueeze(0).long()-1)
                #dec_loss += eos_criterion(eos_preds_tensor, eos_target)
                pass
        eos_loss += eos_criterion(eos_preds_tensor, eos_target)
        index_tensor = (target_number_tensor.long()-1).unsqueeze(0).unsqueeze(-1)
        index_tensor = index_tensor.squeeze()
        #print(eos_preds.squeeze().shape)
        #print(index_tensor.shape)
        eos_criterion = nn.BCEWithLogitsLoss(reduce=False)
        logits_at_ones = eos_preds_tensor.squeeze().transpose(0,1).gather(1,index_tensor.view(-1,1))
        #print(logits_at_ones)
        loss_at_ones = eos_criterion(logits_at_ones.squeeze(), torch.ones(ARGS.batch_size, device=device))
        #print(target_number_tensor)
        #print(loss_at_ones)
        scaled_logits_at_ones = torch.mul(logits_at_ones.squeeze(),target_number_tensor.squeeze()-2)
        scaled_loss_at_ones = torch.mul(loss_at_ones.squeeze(),target_number_tensor.squeeze()-2)
        #print(scaled_loss_at_ones)
        scaled_logits_at_ones = scaled_logits_at_ones/(target_number_tensor[b].int())
        scaled_loss_at_ones = scaled_loss_at_ones/(target_number_tensor[b].int())
        #print(scaled_loss_at_ones)
        ones_loss = -scaled_logits_at_ones.mean()
        ones_loss = scaled_loss_at_ones.mean()
        eos_loss += ones_loss

        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        if ARGS.setting == 'eos':
            dec_loss /= ARGS.batch_size

        else:
            dec_loss /= torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
        
        return round(dec_loss.item(),3), round(norm_loss.item(),3), round(eos_loss.item(),3)


def train_reg(ARGS, encoder, regressor, train_generator, val_generator, device):
    
    loss_plot_file_path = '../experiments/{}/{}_lossplot.png'.format(ARGS.exp_name, ARGS.exp_name)

    v = 1
    for param in encoder.cnn.parameters():
        #if v <= ARGS.cnn_layers_to_freeze*2: # Assuming each layer has two params
            #param.requires_grad = False
        param.requires_grad = False
        v += 1
    
    enc_optimizer = optim.Adam(encoder.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay) 
    reg_optimizer = optim.Adam(regressor.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay) 
    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=ARGS.patience, verbose=True)
    batch_train_losses = []
    batch_val_losses = []
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch_num in range(ARGS.max_epochs):
        print('Epoch:', epoch_num)
        for iter_, train_batch in enumerate(train_generator):
            input_ = train_batch[0].float().transpose(0,1).to(device)
            target_number = train_batch[2].float().to(device)
            i3d_vec = train_batch[5].float().to(device)

            enc_optimizer.zero_grad()
            reg_optimizer.zero_grad()
            
            encoder.train()
            regressor.train()
            encoder_hidden = encoder.initHidden().to(device)
            cnn_outputs = torch.zeros(encoder.num_frames, input_.shape[1], encoder.output_cnn_size+encoder.i3d_size, device=encoder.device)

            for i, inp in enumerate(input_):
                embedded = encoder.cnn_vector(inp)
                if encoder.i3d:
                    embedded = torch.cat([embedded, i3d_vec], dim=1)
                cnn_outputs[i] = embedded 

            # pass the output of the vgg layers through the GRU cell
            encoder_outputs, encoder_hidden = encoder.rnn(cnn_outputs, encoder_hidden)
            #print('prernn', cnn_outputs.shape)
            #print(encoder.rnn)
            #print('postrnn', encoder_outputs.shape)
            #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

            vid_vec = encoder_outputs.mean(dim=0)
            print(vid_vec.shape, i3d_vec.shape)
            if ARGS.i3d:
                vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
            reg_pred = regressor(vid_vec)
            print('train', target_number, reg_pred)
           
            train_loss = criterion(reg_pred, target_number)
            #print(reg_pred)
            #print(target_number)
            train_loss.backward()
            for param in regressor.parameters():
                #print(param.grad)
                pass
            enc_optimizer.step()
            reg_optimizer.step()
            batch_train_losses.append(train_loss.item())

            print(iter_, train_loss.item())
            if ARGS.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        epoch_train_losses.append(new_epoch_train_loss)

        for iter_, val_batch in enumerate(val_generator):
            input_ = val_batch[0].float().transpose(0,1).to(device)
            target_number = val_batch[2].float().to(device)
            i3d_vec = val_batch[5].float().to(device)

            encoder.eval()
            regressor.eval()
            encoder_hidden = encoder.initHidden().to(device)
            cnn_outputs = torch.zeros(encoder.num_frames, input_.shape[1], encoder.output_cnn_size+encoder.i3d_size, device=encoder.device)

            for i, inp in enumerate(input_):
                embedded = encoder.cnn_vector(inp)
                if encoder.i3d:
                    embedded = torch.cat([embedded, i3d_vec], dim=1)
                cnn_outputs[i] = embedded 

            # pass the output of the vgg layers through the GRU cell
            encoder_outputs, encoder_hidden = encoder.rnn(cnn_outputs, encoder_hidden)
            #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

            vid_vec = encoder_outputs.mean(dim=0)
            if ARGS.i3d:
                vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
            reg_pred = regressor(vid_vec)
           
            val_loss = criterion(reg_pred, target_number)
            batch_val_losses.append(val_loss.item())

            print(iter_, val_loss.item())
        new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        epoch_val_losses.append(new_epoch_val_loss)
        #print(epoch_train_losses)
        #print(epoch_val_losses)

        save_dict = {'encoder': encoder, 'encoder_optimizer': enc_optimizer, 'regressor':regressor, 'regressor_optimizer':reg_optimizer}
        save = not ARGS.no_chkpt
        EarlyStop(new_epoch_val_loss, save_dict, exp_name=ARGS.exp_name, save=save)

        if EarlyStop.early_stop:
            break 

        utils.plot_losses(epoch_train_losses, epoch_val_losses, 'MSE', loss_plot_file_path)
    utils.plot_losses(epoch_train_losses, epoch_val_losses, 'MSE', loss_plot_file_path)
