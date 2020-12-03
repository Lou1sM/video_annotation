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
from get_pred import get_pred_loss
import pretrainedmodels


def train_on_batch(ARGS, training_example, encoder, multiclassifier, dataset_dict, optimizer, criterion, device, train):
    input_tensor = training_example[0].float().transpose(0,1).to(device)
    multiclass_inds = training_example[1].float().to(device)
    video_ids = training_example[2].to(device)
    i3d = training_example[3].float().to(device)
    if train:
        encoder.train()
        multiclassifier.train()
    else:
        encoder.eval()
        multiclassifier.eval()
    encoding, encoder_hidden = encoder(input_tensor)
    if ARGS.i3d: encoding = torch.cat([encoding,i3d],dim=-1)
    if ARGS.bottleneck: encoding = torch.randn_like(encoding)

    multiclassif_logits = multiclassifier(encoding)
    multiclass_loss = criterion(multiclassif_logits,multiclass_inds)

    pred_loss = get_pred_loss(video_ids,encoding,dataset_dict,testing=False)
    loss = multiclass_loss + ARGS.lmbda*pred_loss
    if train: loss.backward(); optimizer.step(); optimizer.zero_grad()
    return round(multiclass_loss.item(),5), round(pred_loss.item(),5)

def train(ARGS, encoder, multiclassifier, dataset_dict, train_dl, val_dl, optimizer, exp_name, device, train):
    EarlyStop = EarlyStopper(patience=ARGS.patience)

    epoch_train_multiclass_losses = []
    epoch_val_multiclass_losses = []
    epoch_train_pred_losses = []
    epoch_val_pred_losses = []

    criterion = nn.BCEWithLogitsLoss()
    for epoch_num in range(ARGS.max_epochs):
        epoch_start_time = time.time()
        batch_train_multiclass_losses = []
        batch_train_pred_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_example in enumerate(train_dl):
            new_train_multiclass_loss, new_train_pred_loss = train_on_batch(
                ARGS,
                training_example,
                encoder=encoder,
                multiclassifier=multiclassifier,
                dataset_dict=dataset_dict,
                optimizer=optimizer,
                criterion=criterion,
                device=device, train=True)
            print('Batch:', iter_, 'multiclass loss:', new_train_multiclass_loss, 'pred loss:', new_train_pred_loss)
            batch_train_multiclass_losses.append(new_train_multiclass_loss)
            batch_train_pred_losses.append(new_train_pred_loss)
            if ARGS.quick_run:
                break
        batch_val_multiclass_losses = []
        batch_val_pred_losses = []
        for iter_, valing_triplet in enumerate(val_dl):
            new_val_multiclass_loss, new_val_pred_loss = train_on_batch(ARGS, training_example, encoder=encoder, multiclassifier=multiclassifier, optimizer=None, criterion=criterion, dataset_dict=dataset_dict, device=device, train=False)

            batch_val_multiclass_losses.append(new_val_multiclass_loss)
            batch_val_pred_losses.append(new_val_pred_loss)

            if ARGS.quick_run:
                break

        epoch_train_multiclass_loss = sum(batch_train_multiclass_losses)/len(batch_train_multiclass_losses)
        epoch_train_pred_loss = sum(batch_train_pred_losses)/len(batch_train_pred_losses)
        try:
            epoch_val_multiclass_loss = sum(batch_val_multiclass_losses)/len(batch_val_multiclass_losses)
            epoch_val_pred_loss = sum(batch_val_pred_losses)/len(batch_val_pred_losses)
        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
        save_dict = {'encoder':encoder, 'multiclassifier':multiclassifier, 'ind_dict': dataset_dict['ind_dict'], 'mlp_dict': dataset_dict['mlp_dict'], 'optimizer': optimizer}
        #save = not ARGS.no_chkpt and new_epoch_val_loss < 0.01 and random.random() < 0.1
        EarlyStop(epoch_val_multiclass_loss+epoch_val_pred_loss, save_dict, exp_name=exp_name, save=not ARGS.no_chkpt)

        print('val_multiclass_loss', epoch_val_multiclass_loss,'val_pred_loss', epoch_val_pred_loss)
        if EarlyStop.early_stop:
            break

        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')
