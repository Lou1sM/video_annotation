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

    def __call__(self, val_loss, model, filename, save=True):

        if val_loss < self.val_loss_min:
            if save:
                self.save_checkpoint(val_loss, model, filename)
            self.val_loss_min = val_loss
            self.counter = 0
            print(self.counter)
        else:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model_dict, filename='../checkpoints/chkpt.pt'):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model to {} ...'.format(self.val_loss_min, val_loss, filename))
        torch.save(model_dict,filename)
        self.val_loss_min = val_loss
        

