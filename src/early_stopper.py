import numpy as np
from dl_utils.misc import check_dir
import torch

class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = None

    def __call__(self, val_loss, model, exp_name, save=True):

        #if val_loss < self.val_loss_min - 0.01:
        if val_loss < self.val_loss_min:
            if save: self.save_checkpoint(val_loss, model, exp_name)
            self.val_loss_min, self.model, self.counter = val_loss, model, 0
        else:
            self.counter += 1
            print('Val loss was {}, no improvement on best of {}'.format(val_loss, self.val_loss_min))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model_dict, exp_name):
        '''Saves model when validation loss decreases.'''
        check_dir('../data/checkpoints')
        filename = '../data/checkpoints/{}.pt'.format(exp_name)
        print(f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model to {filename} ...')
        try: torch.save(model_dict,filename)
        except FileNotFoundError: print(f"Can't save file to {filename} because the directory doesn't exist.")
        self.val_loss_min = val_loss
