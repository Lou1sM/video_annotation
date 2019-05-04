import os
import utils
import options
import models
import torch
from data_loader import load_data

SUMMARY_DIR = '../data/logs'
BEST_FILE_PATH = os.path.join(SUMMARY_DIR, "best.txt")
class HyperParamSet():
    def __init__(self, param_dict):
        self.ind_size = 300
        self.dec_size = param_dict['dec_size']
        self.num_frames = 8
        self.batch_size = param_dict['batch_size']
        self.learning_rate = param_dict['lr']
        self.optimizer = param_dict['opt']
        self.max_length = 10
        self.weight_decay = param_dict['weight_decay']
        self.dropout = 0
        self.shuffle = True
        self.max_epochs = 200
        self.patience = 10
        

def train_with_hyperparams(model, param_dict, exp_name=None, best_val_loss=0):
    args = HyperParamSet(param_dict)

    if exp_name == None:
        exp_name = utils.get_datetime_stamp()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = models.EncoderRNN(args, device).to(device)
    decoder = models.DecoderRNN(args, device).to(device)
    regressor = models.NumIndRegressor(args,device).to(device)
    h5_train_generator = load_data('../data/datasets/four_vids.h5', args.batch_size, shuffle=args.shuffle)
    h5_val_generator = load_data('../data/datasets/four_vids.h5', args.batch_size, shuffle=args.shuffle)

    if model == 'seq2seq':
        val_loss = models.train_iters_seq2seq(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name)
    elif model == 'reg':
        val_loss = models.train_iters_reg(args, encoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name)
 
#    if model == "seq2seq":
#        train_func = models.train_iters_seq2seq
#    elif model == "reg":
#        train_func = models.reg
# 
#    val_loss = train_func(args, encoder, decoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name)
#
    summary_file_path = os.path.join(SUMMARY_DIR, "{}.txt".format(exp_name))
    summary_file = open(summary_file_path, 'w')
    summary_file.write("Val loss:" + ': ' + str(round(val_loss, 3)))
    for key in sorted(param_dict.keys()):
        summary_file.write(key + ': ' +  str(param_dict[key]))
   
    if val_loss < best_val_loss:
        with open(BEST_FILE_PATH, 'w') as summary_file:
            summary_file.write("Val loss:" + ': ' + str(val_loss))
            for key in sorted(param_dict.keys()):
                summary_file.write(key + ': ' +  str(param_dict[key]))
    
    return val_loss


def grid_search(dec_sizes, batch_sizes, lrs, opts, weight_decays):
    it = 0
    best_val_loss = float('inf')
    for dec_size in dec_sizes:
        for batch_size in batch_sizes:
            for lr in lrs:
                for opt in opts:
                    for weight_decay in weight_decays:
                        param_dict = {
                            'dec_size': dec_size,
                            'batch_size': batch_size,
                            'lr': lr,
                            'opt': opt,
                            'weight_decay': weight_decay}
                        new_val_loss = train_with_hyperparams('reg', param_dict, it, best_val_loss)
                        if new_val_loss < best_val_loss:
                            best_it = it
                            best_val_loss = new_val_loss
                        it += 1


if __name__=="__main__":

    dec_sizes = [1,2,3,4]
    batch_sizes = [2]
    lrs = [1e-3]
    opts = ['Adam']
    weight_decays = [0]

    grid_search(dec_sizes, batch_sizes, lrs, opts, weight_decays)

