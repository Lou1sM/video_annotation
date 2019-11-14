import os
import numpy as np
import torch

class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = None

    def __call__(self, val_loss, model, exp_name, save=True):

        if val_loss < self.val_loss_min - 0.01:
            if save: self.save_checkpoint(val_loss, model, exp_name)
            self.val_loss_min, self.model, self.counter  = val_loss, model, 0
        else:
            self.counter += 1
            print('Val loss was {}, no improvement on best of {}'.format(val_loss, self.val_loss_min))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                filepath = '../checkpoints/{}.pt'.format(exp_name)

    def save_checkpoint(self, val_loss, model_dict, exp_name):
        '''Saves model when validation loss decrease.'''
        if exp_name.startswith('jade'):
            if exp_name.endswith('d'): filename = '../jade_checkpoints/{}.pt'.format(exp_name)
            else: filename = '../jade_checkpoints/{}.pt'.format(exp_name[:-2])
        else: filename = '/data1/louis/checkpoints/{}.pt'.format(exp_name)
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model to {filename} ...')
        try: torch.save(model_dict,filename)
        except FileNotFoundError: print("Can't save file to {} because the directory doesn't exist.".format(filename))
        self.val_loss_min = val_loss
        
