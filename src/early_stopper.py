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

        if val_loss < self.val_loss_min:
            if save:
                self.save_checkpoint(val_loss, model, exp_name)
            self.val_loss_min = val_loss
            self.best_model = model
            self.counter = 0
            print(self.counter)
        else:
            self.counter += 1
            print('Val loss was {}, no improvement on best of {}'.format(val_loss, self.val_loss_min))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                filepath = '../checkpoints/{}.pt'.format(exp_name)
                #self.save_to_disk(filepath)

    def save_checkpoint(self, val_loss, model_dict, exp_name):
        '''Saves model when validation loss decrease.'''
        filename = '/data2/louis/checkpoints/{}.pt'.format(exp_name)
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model to {} ...'.format(self.val_loss_min, val_loss, filename))
        try:
            torch.save(model_dict,filename)
        except FileNotFoundError:
            print("Can't save file to {} because the directory doesn't exist.".format(filename))
        self.val_loss_min = val_loss
        
    def save_to_disk(self, exp_name):
        filepath = os.path.join('/data2/louis/checkpoints', '{}.pt'.format(exp_name))
        print('Saving final model to {}'.format(filepath) )
        torch.save(self.best_model, filepath)
