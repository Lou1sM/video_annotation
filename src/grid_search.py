import os
import sys
import utils
import options
import models_train
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from main import run_experiment


DATA_DIR = '/home/eleonora/video_annotation/data'

def get_best_dev_file_path(dev):
    #return os.path.join(SUMMARY_DIR, "best_on_device{}.txt".format(dev))
    return "../data/logs/best_on_device{}.txt".format(dev)


class HyperParamSet():
    def __init__(self, param_dict):
        self.rrn_init = 'det'
        self.verbose = False
        self.ind_size = 50 if MINI else 10
        #self.dec_size = param_dict['dec_size']
        self.num_frames = 8
        self.batch_size = param_dict['batch_size']
        self.learning_rate = param_dict['lr']
        self.optimizer = param_dict['opt']
        self.max_length = 29
        self.dropout = param_dict['dropout'] 
        self.shuffle = True
        self.max_epochs = 10000
        self.patience = 1 if MINI else 15 
        self.cnn_layers_to_freeze = 19
        self.output_cnn_size = 4096
        self.quick_run = False
        self.enc_layers = param_dict['enc_layers']
        self.dec_layers = param_dict['dec_layers']
        self.teacher_forcing_ratio = param_dict['teacher_forcing_ratio']
        self.enc_size = param_dict['enc_size']
        self.dec_size = param_dict['dec_size']
        self.enc_rnn = param_dict['enc_rnn']
        self.dec_rnn = param_dict['dec_rnn']
        self.enc_dec_hidden_init = param_dict['enc_dec_hidden_init']
        self.enc_cnn = "vgg"
        self.enc_init = param_dict['rnn_init']
        self.dec_init = param_dict['rnn_init']
        self.reload_path = None
        self.weight_decay = 0
        self.loss_func = param_dict['loss_func']
        self.norm_threshold = param_dict['norm_threshold']
        self.lmbda=1.
        self.chkpt=not MINI
        

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

def train_with_hyperparams(model, train_table, val_table, test_table, train_file_path, val_file_path, test_file_path, param_dict, exp_name=None, device="cuda"):

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
    ARGS.model = model
    
    accuracy, test_info_dict = run_experiment(
                    exp_name,
                    ARGS,
                    train_file_path=train_file_path,
                    val_file_path=val_file_path,
                    test_file_path=test_file_path,
                    train_table=train_table,
                    val_table=val_table,
                    test_table=test_table)

    return accuracy, test_info_dict



def grid_search(model, enc_sizes, dec_sizes, enc_cnn, enc_dec_hidden_inits, batch_sizes, lrs, opts, rnn_inits, dropouts, enc_layers, dec_layers, teacher_forcing_ratio, norm_thresholds, checkpoint_path=None):
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
        train_table = val_table = test_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=enc_cnn)
        #train_table = val_table = test_table  = video_lookup_table_from_range(1,4, cnn=enc_cnn)
        batch_sizes = [2]
        enc_sizes = dec_sizes = [50]
        enc_layers = dec_layers = [1]
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/50d.6dp.h5'
        print('Using dataset: ../data/rdf_video_captions/50d.6dp')
    else:
        train_table = video_lookup_table_from_range(1,1201, cnn=enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=enc_cnn)
        test_table = video_lookup_table_from_range(1301,1971, cnn=enc_cnn)
    
        dataset = '10d-det'
        train_file_path = os.path.join('../data/rdf_video_captions', 'train_{}.h5'.format(dataset))
        val_file_path = os.path.join('../data/rdf_video_captions', 'val_{}.h5'.format(dataset))
        test_file_path = os.path.join('../data/rdf_video_captions', 'test_{}.h5'.format(dataset))
        print('Using dataset: {}'.format(dataset))

        #train_table = video_lookup_table_from_range(1,1201, cnn=enc_cnn)
        #val_table = video_lookup_table_from_range(1201,1301, cnn=enc_cnn)

    #best_val_loss = float('inf')
    best_accuracy = -1
    best_exp_name = None 
    dropout = 0.0
    for enc_size in enc_sizes:
        for dec_size in dec_sizes:
            for enc_dec_hidden_init in enc_dec_hidden_inits:
                for enc_rnn in enc_rnns:
                    for dec_rnn in dec_rnns:
                        if enc_dec_hidden_init and ((enc_size != dec_size) or (enc_rnn != dec_rnn)):
                            continue
                        for batch_size in batch_sizes:
                            for lr in lrs:
                                for opt in opts:
                                    for rnn_init in rnn_inits:
                                        for loss_func in loss_funcs:
                                            for dec_layer in dec_layers:
                                                for enc_layer in enc_layers:
                                                    for norm_threshold in norm_thresholds:
                                                        param_dict = {
                                                            'enc_size': enc_size,
                                                            'dec_size': dec_size,
                                                            'enc_rnn': enc_rnn,
                                                            'dec_rnn': dec_rnn,
                                                            'enc_dec_hidden_init': enc_dec_hidden_init,
                                                            'batch_size': batch_size,
                                                            'lr': lr,
                                                            'opt': opt,
                                                            'rnn_init': rnn_init,
                                                            'dropout': dropout,
                                                            'dec_layers': dec_layer,
                                                            'enc_layers': enc_layer, 
                                                            'teacher_forcing_ratio': teacher_forcing_ratio,
                                                            'loss_func': loss_func,
                                                            'norm_threshold': norm_threshold}
                                                        next_available_device = cuda_devs[it%len(cuda_devs)]
                                                        print("Executing run on {}".format(next_available_device))
                                                        print("Parameters:")
                                                        for key in sorted(param_dict.keys()):
                                                            print('\t'+key+': '+str(param_dict[key]))
                                                        name_str='_batch'+str(batch_size)+'_lr'+str(lr)+'_enc'+str(enc_layer)+'_dec'+str(dec_layer)+'_tfratio'+str(teacher_forcing_ratio)+'_rnnInit'+str(rnn_init)+'_'+opt
                                                        name_str='-'.join([k+str(v) for k,v in param_dict.items() if k in BEING_TESTED])
                                                        print(name_str)
                                                        #new_accuracy, new_train_info_dict = train_with_hyperparams(model, train_table, val_table, param_dict, exp_name=name_str, device=next_available_device)
                                                        new_accuracy, new_train_info_dict = train_with_hyperparams(model, train_table, val_table, test_table, train_file_path, val_file_path, test_file_path, param_dict, exp_name=name_str, device=next_available_device)
                                                        if new_accuracy > best_accuracy:
                                                            best_it = it
                                                            best_accuracy = new_accuracy
                                                            best_exp_name = name_str
                                                        it += 1
                return best_exp_name

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
    batch_sizes = [64]
    lrs = [1e-3]
    opts = ['Adam']
    dropouts = [0.0]
    enc_layers = [2]
    dec_layers = [2]
    teacher_forcing_ratio = 1.0
    vgg_layers_to_train = [0]
    loss_funcs = ['mse', 'cos', 'mse', 'cos']
    lmbdas = [0., 1.]
    norm_thresholds = [1.]

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

    best_exp_name = grid_search(
        model = 'seq2seq', 
        enc_sizes=enc_sizes, 
        dec_sizes=dec_sizes, 
        enc_cnn=enc_cnn, 
        enc_dec_hidden_inits=enc_dec_hidden_inits, 
        batch_sizes=batch_sizes, 
        lrs=lrs, 
        opts=opts, 
        rnn_inits=rnn_inits, 
        dropouts=dropouts, 
        enc_layers=enc_layers, 
        dec_layers=dec_layers, 
        teacher_forcing_ratio=teacher_forcing_ratio,
        norm_thresholds=norm_thresholds)

    #best_exp_name = '0'
#    grid_search('eos', dec_sizes, batch_sizes, lrs, opts, weight_decays, enc_layers, dec_layers, teacher_forcing_ratio, checkpoint_path='../checkpoints/chkpt{}.pt'.format(best_exp_name))
#
#    with open("checkpoint.out", 'w') as f:
#        f.write('../checkpoints/chkpt{}.pt'.format(best_exp_name))
#
#    ckpt_path = '../checkpoints/chkpt{}.pt'.format(best_exp_name)
#
#    test_table = video_lookup_table_from_range(1301,1969)
#    num_lines = 1969 - 1301
#    h5_test_generator = load_data_lookup('../data/rdf_video_captions/test_50d.h5', video_lookup_table=test_table, batch_size=num_lines, shuffle=False)
#    print(h5_test_generator)
#    for t in h5_test_generator:
#        print(t)
#    models_train.get_test_output(ckpt_path, h5_test_generator[0], num_datapoints= num_lines, ind_size=50, use_eos=True, device='cuda')
#

    """
    if mini:
        h5_train_generator = load_data_lookup(os.path.join(DATA_DIR, 'rdf_video_captions/50d_overfitting.h5'), video_lookup_table=train_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
        h5_val_generator = load_data_lookup(os.path.join(DATA_DIR, 'rdf_video_captions/50d_overfitting.h5'), video_lookup_table=val_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
    else:
        h5_train_generator = load_data_lookup(os.path.join(DATA_DIR,'rdf_video_captions/train_50d.h5'), video_lookup_table=train_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
        h5_val_generator = load_data_lookup(os.path.join(DATA_DIR, 'rdf_video_captions/val_50d.h5'), video_lookup_table=val_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)

    if model == 'seq2seq':
        encoder = models_train.EncoderRNN(ARGS, device).to(device)
        decoder = models_train.DecoderRNN(ARGS, device).to(device)
        val_loss = models_train.train(ARGS, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif model == 'reg':
        if checkpoint_path == None: 
            print("\nPath to the encoder weights checkpoints needed\n")
            sys.exit()
        checkpoint = torch.load(checkpoint_path)
        print("\n CHECKPOINT \n")
        print(checkpoint)
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        regressor = models_train.NumIndRegressor(ARGS, device).to(device)
        #val_loss = models_train.train_iters_reg(ARGS, encoder, decoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif model == 'eos':
        if checkpoint_path == None: 
            print("\nPath to the encoder weights checkpoints needed\n")
            sys.exit()
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        eos = models_train.NumIndEOS(ARGS, device).to(device)
        #val_loss = models_train.train_iters_eos(ARGS, encoder, decoder, encoder_optimizer=None, decoder_optimizer=None, eos, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)

    #summary_file_path = os.path.join(DATA_DIR, "logs/{}.txt".format(exp_name))
    summary_file_path = "../data/logs/{}.txt".format(exp_name)
    summary_file = open(summary_file_path, 'w')
    summary_file.write("Val loss:" + ': ' + str(round(val_loss, 3)) + '\n')
    for key in sorted(param_dict.keys()):
        summary_file.write(key + ': ' +  str(param_dict[key]) + '\n')
   
    if val_loss < best_val_loss:
        best_dev_file_path = get_best_dev_file_path(device)
        with open(best_dev_file_path, 'w') as summary_file:
            summary_file.write("Val loss:" + ': ' + str(val_loss) + '\n')
            for key in sorted(param_dict.keys()):
                summary_file.write(key + ': ' +  str(param_dict[key]) + '\n')

        with open('../data/logs/best.txt', 'w') as summary_file:
            summary_file.write("Val loss:" + ': ' + str(val_loss) + '\n')
            for key in sorted(param_dict.keys()):
                summary_file.write(key + ': ' +  str(param_dict[key]) + '\n')

    return val_loss, exp_name

    """

