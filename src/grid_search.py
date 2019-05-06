import os
import sys
import utils
import options
import models
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range


SUMMARY_DIR = '../data/logs'

def get_best_dev_file_path(dev):
    return os.path.join(SUMMARY_DIR, "best_on_device{}.txt".format(dev))


"""
def load_vid_from_id(vid_id):
    return np.load('../data/frames/vid{}_resized.npz'.format(vid_id))['arr_0']


def video_lookup_table_from_range(start_idx, end_idx):
    return {vid_id: load_vid_from_id(vid_id+1) for vid_id in range(start_idx, end_idx)}
"""

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
        self.patience = 2
        self.vgg_layers_to_freeze = 17
        self.quick_run = True
        

def train_with_hyperparams(model, train_table, val_table, param_dict, exp_name=None, best_val_loss=0, device="cuda"):

    args = HyperParamSet(param_dict)
    if exp_name == None:
        exp_name = utils.get_datetime_stamp()
    if mini:
        h5_train_generator = load_data_lookup('../data/mini/train_data.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup('../data/mini/val_data.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)
    else:
        h5_train_generator = load_data('../data/dummy_data/train_data.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data('../data/dummy_data/val_data.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)


    encoder = models.EncoderRNN(args, device).to(device)
    if model == 'seq2seq':
        decoder = models.DecoderRNN(args, device).to(device)
        val_loss = models.train_iters_seq2seq(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name, device=device)
    elif model == 'reg':
        regressor = models.NumIndRegressor(args, device).to(device)
        val_loss = models.train_iters_reg(args, encoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name, device=device)
    elif model == 'eos':
        eos = models.NumIndEOS(args, device).to(device)
        val_loss = models.train_iters_eos(args, encoder, eos, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name, device=device)

    summary_file_path = os.path.join(SUMMARY_DIR, "{}.txt".format(exp_name))
    summary_file = open(summary_file_path, 'w')
    summary_file.write("Val loss:" + ': ' + str(round(val_loss, 3)))
    for key in sorted(param_dict.keys()):
        summary_file.write(key + ': ' +  str(param_dict[key]))
   
    if val_loss < best_val_loss:
        best_dev_file_path = get_best_dev_file_path(device)
        with open(best_dev_file_path, 'w') as summary_file:
            summary_file.write("Val loss:" + ': ' + str(val_loss))
            for key in sorted(param_dict.keys()):
                summary_file.write(key + ': ' +  str(param_dict[key]))
    
    return val_loss


def grid_search(model, dec_sizes, batch_sizes, lrs, opts, weight_decays):
    cuda_devs = ["cuda: {}".format(i) for i in range(torch.cuda.device_count())]
    print("Available devices:")
    if cuda_devs == []:
        cuda_devs = ['cpu']
    for d in cuda_devs:
        print(d)
    if cuda_devs == []:
        cuda_devs = ['cpu']
    it = 0
 
    
    if mini:
        train_table = video_lookup_table_from_range(1,11)
        val_table = video_lookup_table_from_range(1201,1211)
    else:
        train_table = video_lookup_table_from_range(1,1201)
        val_table = video_lookup_table_from_range(1201,1301)
        #h5_train_generator = load_data('../data/datasets/four_vids.h5', vid_range=(1,1201), batch_size=args.batch_size, shuffle=args.shuffle)
        #h5_val_generator = load_data('../data/datasets/four_vids.h5', vid_range=(1201,1301), batch_size=args.batch_size, shuffle=args.shuffle)

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
                        next_available_device = cuda_devs[it%len(cuda_devs)]
                        print("Executing run on {}".format(next_available_device))
                        new_val_loss = train_with_hyperparams(model, train_table, val_table, param_dict, it, best_val_loss, device=next_available_device)
                        if new_val_loss < best_val_loss:
                            best_it = it
                            best_val_loss = new_val_loss
                        it += 1


if __name__=="__main__":

    dec_sizes = [1,2,3,4]
    batch_sizes = [3]
    lrs = [1e-3]
    opts = ['Adam']
    weight_decays = [0]

    if len(sys.argv) == 1:
        mini = False
        print("running grid search on full dataset")
    elif sys.argv[1] == 'm':
        mini = True
        print("running grid search on mini dataset")
    else:
        print("Unrecognized argument")
        sys.exit()
    grid_search('eos', dec_sizes, batch_sizes, lrs, opts, weight_decays)

