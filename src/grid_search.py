import os
import sys
import utils
import options
import models
import train
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, i3d_lookup_table_from_range
from main import run_experiment

DATETIME_STAMP=utils.get_datetime_stamp()

def get_best_dev_file_path(dev):
    #return os.path.join(SUMMARY_DIR, "best_on_device{}.txt".format(dev))
    return "../data/logs/best_on_device{}.txt".format(dev)


class HyperParamSet():
    def __init__(self, param_dict):
        self.batch_size = 100
        #self.batch_size = param_dict['batch_size']
        self.no_chkpt=MINI
        self.enc_cnn = "vgg"
        self.enc_dec_hidden_init = param_dict['enc_dec_hidden_init']
        #self.enc_init = param_dict['rnn_init']
        self.enc_init = 'unit'
        self.enc_layers = param_dict['enc_layers']
        self.enc_rnn = param_dict['enc_rnn']
        self.enc_size = param_dict['enc_size']
        self.exp_name = param_dict['exp_name']
        #self.dec_init = param_dict['rnn_init']
        self.dec_init = 'unit'
        self.dec_layers = param_dict['dec_layers']
        self.dec_rnn = param_dict['dec_rnn']
        self.dec_size = param_dict['dec_size']
        #self.dec_size = param_dict['dec_size']
        self.dropout = param_dict['dropout'] 
        self.i3d=param_dict['i3d']
        self.i3d_after=param_dict['i3d_after']
        self.ind_size = 10 if MINI else 10
        self.learning_rate = param_dict['lr']
        self.lmbda_norm = param_dict['lmbda_norm']
        self.max_epochs = 1 if MINI else 1000
        self.num_frames = 8
        self.mini = MINI
        self.max_length = 29
        self.neg_pred_weight=param_dict['neg_pred_weight']
        self.norm_threshold = param_dict['norm_threshold']
        #self.optimizer = param_dict['opt']
        self.optimizer = 'Adam'
        self.output_cnn_size = 4096
        self.shuffle = True
        self.patience = 1 if MINI else 7 
        self.pred_embeddings_assist = 1.0
        self.quick_run = False
        self.reload=None
        self.setting='embeddings'
        self.teacher_forcing_ratio = param_dict['teacher_forcing_ratio']
        self.verbose = False
        self.weight_decay = 0
        #self.loss_func = param_dict['loss_func']
        #self.lmbda=1.
        

def ask_user_yes_no(question):
    #print('asking', question)
    answer = input(question+'y/n')
    if answer == 'y':
        return True
    elif answer == 'n':
        return False
    else:
        print("Please answer 'y' or 'n'")
        return ask_user_yes_no(question)

def train_with_hyperparams(train_table, val_table, test_table, i3d_train_table, i3d_val_table, i3d_test_table, param_dict, exp_name=None, device="cuda"):

    if exp_name == None:
        exp_name = utils.get_datetime_stamp()

    if os.path.isdir('../experiments/{}'.format(exp_name)):
        overwrite = ask_user_yes_no('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name))
        #overwrite = input('An experiment with name {} has already been run, do you want to overwrite?y/n'.format(exp_name))
        if overwrite:
            pass
        else:
            print('Skipping experiment with these parameters')
            return -1, None
    else: 
        os.mkdir('../experiments/{}'.format(exp_name))


    ARGS = HyperParamSet(param_dict)
    ARGS.device = device
    
    if ARGS.i3d:
        i3d_train_table1 = i3d_train_table
        i3d_val_table1 = i3d_val_table
        i3d_test_table1 = i3d_test_table
    else:
        i3d_train_table1 = i3d_val_table1 = i3d_test_table1 = None

    test_info_dict = run_experiment(
                    exp_name,
                    ARGS,
                    train_table=train_table,
                    val_table=val_table,
                    test_table=test_table,
                    i3d_train_table=i3d_train_table1,
                    i3d_val_table=i3d_val_table1,
                    i3d_test_table=i3d_test_table1)

    print(test_info_dict)
    return test_info_dict



def grid_search(enc_sizes, dec_sizes, enc_cnn, enc_dec_hidden_inits, lrs, lmbda_norms, rnn_inits, dropouts, enc_layers, dec_layers, teacher_forcing_ratio, i3ds, i3d_afters, norm_thresholds, checkpoint_path=None):
    cuda_devs = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
    print(cuda_devs)
    print("Available devices:")
    if cuda_devs == []:
        cuda_devs = ['cpu']
    for d in cuda_devs:
        print(d)
    it = 0
 
    print("Loading video lookup tables..")
    if MINI:
        #train_table = val_table = test_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=enc_cnn)
        train_table = val_table = test_table  = video_lookup_table_from_range(1,7, cnn=enc_cnn)
        i3d_train_table = i3d_val_table = i3d_test_table  = i3d_lookup_table_from_range(1,7)
        batch_sizes = [2]
        enc_sizes = dec_sizes = [50]
        enc_layers = dec_layers = [1]
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/50d.6dp.h5'
    else:
        train_table = video_lookup_table_from_range(1,1201, cnn=enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=enc_cnn)
        test_table = video_lookup_table_from_range(1301,1971, cnn=enc_cnn)
        i3d_train_table = i3d_lookup_table_from_range(1,1201)
        i3d_val_table = i3d_lookup_table_from_range(1201,1301)
        i3d_test_table = i3d_lookup_table_from_range(1301,1971)
        #dataset = '10d-det'
        #train_file_path = os.path.join('../data/rdf_video_captions', 'train_{}.h5'.format(dataset))
        #val_file_path = os.path.join('../data/rdf_video_captions', 'val_{}.h5'.format(dataset))
        #test_file_path = os.path.join('../data/rdf_video_captions', 'test_{}.h5'.format(dataset))
        #print('Using dataset: {}'.format(dataset))

        #train_table = video_lookup_table_from_range(1,1201, cnn=enc_cnn)
        #val_table = video_lookup_table_from_range(1201,1301, cnn=enc_cnn)

    #best_val_loss = float('inf')
    best_f1 = -1
    best_exp_name = None 
    dropout = 0.0
    for enc_size in enc_sizes:
        for dec_size in dec_sizes:
            for enc_dec_hidden_init in enc_dec_hidden_inits:
                for enc_rnn in enc_rnns:
                    for dec_rnn in dec_rnns:
                        if enc_dec_hidden_init and ((enc_size != dec_size) or (enc_rnn != dec_rnn)):
                            continue
                        for neg_pred_weight in neg_pred_weights:
                            for lr in lrs:
                                for lmbda_norm in lmbda_norms:
                                    for rnn_init in rnn_inits:
                                        for dec_layer in dec_layers:
                                            for enc_layer in enc_layers:
                                                for i3d in i3ds:
                                                    for i3d_after in i3d_afters:
                                                        for norm_threshold in norm_thresholds:
                                                            name_str=DATETIME_STAMP+'-search:'+str(it)
                                                            print(name_str)
                                                            param_dict = {
                                                                'enc_size': enc_size,
                                                                'exp_name': name_str,
                                                                'dec_size': dec_size,
                                                                'enc_rnn': enc_rnn,
                                                                'dec_rnn': dec_rnn,
                                                                'enc_dec_hidden_init': enc_dec_hidden_init,
                                                                'lr': lr,
                                                                'lmbda_norm': lmbda_norm, 
                                                                'rnn_init': rnn_init,
                                                                'dropout': dropout,
                                                                'dec_layers': dec_layer,
                                                                'enc_layers': enc_layer, 
                                                                'teacher_forcing_ratio': teacher_forcing_ratio,
                                                                'i3d': i3d,
                                                                'i3d_after': i3d_after,
                                                                'neg_pred_weight': neg_pred_weight,
                                                                'norm_threshold': norm_threshold}
                                                            next_available_device = cuda_devs[it%len(cuda_devs)]
                                                            print("Executing run on {}".format(next_available_device))
                                                            print("Parameters:")
                                                            for key in sorted(param_dict.keys()):
                                                                print('\t'+key+': '+str(param_dict[key]))
                                                            #name_str='-'.join([k+str(v) for k,v in param_dict.items() if k in BEING_TESTED])
                                                            #new_accuracy, new_train_info_dict = train_with_hyperparams(train_table, val_table, param_dict, exp_name=name_str, device=next_available_device)
                                                            new_test_info_dict = train_with_hyperparams(train_table, val_table, test_table, i3d_train_table, i3d_val_table, i3d_test_table, param_dict, exp_name=name_str, device=next_available_device)
                                                            new_f1 = new_test_info_dict['f1']
                                                            if new_f1 > best_f1:
                                                                best_it = it
                                                                best_f1 = new_f1
                                                                best_exp_name = name_str
                                                                write_best_info(best_name=best_exp_name[:-8], best_params=param_dict, best_results=new_test_info_dict)
                                                            it += 1
                return best_exp_name, best_f1

def write_best_info(best_name, best_params, best_results):
    
    fname = '../experiments/{}-best.txt'.format(best_name)
    with open(fname, 'w') as writefile:
        writefile.write('PARAMS:\n')
        for k,v in best_params.items():
            writefile.write(k+': '+str(v)+'\n')
        writefile.write('\nRESULTS:\n')
        for k,v in best_results.items():
            writefile.write(k+': '+str(v)+'\n')
     


if __name__=="__main__":

    #dec_sizes = [1,2,3,4]
    enc_sizes = [2000]
    dec_sizes = [1500]
    enc_cnn = "vgg"
    #enc_rnns = ['gru', 'lstm']
    enc_rnns = ['gru']
    dec_rnns = ['gru']
    enc_dec_hidden_inits =[False,True]
    rnn_inits = ['zeroes', 'unit', 'learned', 'unit_learned']
    #batch_sizes = [64]
    lrs = [1e-3]
    lmbda_norms =[1.0]
    #opts = ['Adam']
    dropouts = [0.0]
    enc_layers = [2]
    dec_layers = [2]
    teacher_forcing_ratio = 1.0
    #loss_funcs = ['mse', 'cos']
    i3ds = [True, False]
    i3d_afters = [True, False]
    lmbdas = [0., 1.]
    neg_pred_weights = [0.05]
    norm_thresholds = [1.]
    norm_losses = ['mse', 'relu']

    BEING_TESTED = ['rnn_inits', 'loss_func', 'norm_threshold']

    if len(sys.argv) == 1:
        MINI = False
        print("running grid search on full dataset")
    elif sys.argv[1] == 'm':
        batch_sizes = [2]
        lrs = [1e-3]
        enc_sizes = dec_sizes = [50]
        enc_layers = dec_layers = [1]
        MINI = True
        batch_sizes = [3]
        print("running grid search on mini dataset")
    else:
        print("Unrecognized argument")
        sys.exit()

    best_exp_name, best_f1 = grid_search(
        enc_sizes=enc_sizes, 
        dec_sizes=dec_sizes, 
        enc_cnn=enc_cnn, 
        enc_dec_hidden_inits=enc_dec_hidden_inits, 
        #batch_sizes=batch_sizes, 
        lrs=lrs, 
        lmbda_norms = lmbda_norms,
        #opts=opts, 
        rnn_inits=rnn_inits, 
        dropouts=dropouts, 
        enc_layers=enc_layers, 
        dec_layers=dec_layers, 
        teacher_forcing_ratio=teacher_forcing_ratio,
        i3ds=i3ds,
        i3d_afters=i3d_afters,
        norm_thresholds=norm_thresholds)

    print('Best f1 of {}, obtained on run number {}.'.format(best_f1, best_exp_name))
