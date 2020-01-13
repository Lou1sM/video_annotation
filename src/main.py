from gensim.models import KeyedVectors
import csv
import sys
import os
import json
from utils import get_datetime_stamp, asMinutes
from time import time
import options
import models
import train
import data_loader 
from get_output import write_outputs_get_info
from pdb import set_trace


def run_experiment(exp_name, ARGS, json_data, train_dl, val_dl, test_dl, w2v):

    ARGS.eval_batch_size = min(ARGS.batch_size,100)
    print(ARGS)
    inds,preds,json_data_dict = json_data['inds'],json_data['preds'],json_data['dataset']

    set_trace()
    if ARGS.reload:
        reload_file_path = '/data1/louis/checkpoints/{}.pt'.format(ARGS.reload)
        reload_file_path = ARGS.reload
        print('Reloading model from {}'.format(reload_file_path))
        saved_model = torch.load(reload_file_path)
        encoder = saved_model['encoder']
        multiclassifier = saved_model['multiclassifier']
        encoder.batch_size = ARGS.batch_size
        optimizer = saved_model['optimizer']
    else: 
        encoder = models.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
        multiclassifier = models.MLP(ARGS.enc_size,ARGS.classif_size,len(inds))
        mlp_dict = {pred: models.MLP(ARGS.enc_size + arity*ARGS.ind_size,ARGS.mlp_size,1).to(ARGS.device) for pred,arity in preds}
        ind_dict = {ind: torch.nn.parameter(w2v[ind],device=ARGS.device) for ind in inds}
        optimizer = None
    
    global TRAIN_START_TIME; TRAIN_START_TIME = time()
    print('\nTraining the model')
    train_info, _ = train.train(ARGS, encoder, multiclassifier, json_data_dict, ind_dict, mlp_dict, train_dl=train_dl, val_dl=val_dl, exp_name=exp_name, device=ARGS.device, optimizer=optimizer, train=True)
       
    global EVAL_START_TIME; EVAL_START_TIME = time()
    if ARGS.no_chkpt: print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
    else:
        print("Reloading best network version for outputs")
        filename = '../jade_checkpoints/{}.pt'.format(exp_name[:-2] )if exp_name.startswith('jade') else '/data1/louis/checkpoints/{}.pt'.format(exp_name) 
        checkpoint = torch.load(filename)
        encoder,decoder = checkpoint['encoder'], checkpoint['decoder']

    """
    print('\nComputing outputs on val set')
    val_output_info = write_outputs_get_info(encoder, decoder, ARGS=ARGS, dataset=dataset, data_generator=val_generator, exp_name=exp_name, dset_fragment='val', setting=ARGS.setting)
    val_output_info['dset_fragment'] = 'val'
    print('\nComputing outputs on train set')
    train_output_info = write_outputs_get_info(encoder, decoder, ARGS=ARGS, dataset=dataset, data_generator=train_generator, exp_name=exp_name, dset_fragment='train', setting=ARGS.setting)
    train_output_info['dset_fragment'] = 'train'
    fixed_thresh = ((train_output_info['thresh']*1200)+(val_output_info['thresh']*100))/1300
    print('\nComputing outputs on test set')
    test_output_info = write_outputs_get_info(encoder, decoder, ARGS=ARGS, dataset=dataset, data_generator=test_generator, exp_name=exp_name, dset_fragment='test', fixed_thresh=fixed_thresh, setting=ARGS.setting)
    test_output_info['dset_fragment'] = 'test'

    summary_filename = '../experiments/{}/{}_summary.txt'.format(exp_name, exp_name) 
    with open(summary_filename, 'w') as summary_file:
        summary_file.write('Experiment name: {}\n'.format(exp_name))
        summary_file.write('\tTrain\tVal\tTest\n')
        for k in ['dset_fragment', 'l2_distance', 'tp', 'fn', 'fp', 'tn', 'thresh', 'best_acc', 'acchalf', 'legit_f1', 'f1half', 'inf_acc', 'inf_acchalf', 'avg_pos_prob', 'avg_neg_prob']: 
            summary_file.write(k+'\t'+str(train_output_info[k])+'\t'+str(val_output_info[k])+'\t'+str(test_output_info[k])+'\n')
        summary_file.write('\nParameters:\n')
        for key in options.IMPORTANT_PARAMS:
            summary_file.write(str(key) + ": " + str(vars(ARGS)[key]) + "\n")
    summary_csvfilename = '../experiments/{}/{}_summary.csv'.format(exp_name, exp_name) 
    fieldnames = list(sorted(vars(ARGS).keys())) + list(sorted(val_output_info.keys()))
    with open(summary_csvfilename, 'w') as csvfile:
        val_output_info.update(vars(ARGS))
        dictwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dictwriter.writeheader()
        dictwriter.writerow(val_output_info)

    print(val_output_info)
    return val_output_info
    """


def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y': return True
    elif answer == 'n': return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))
    

def main():
    if ARGS.mini: 
        ARGS.exp_name = 'try'
        splits = [4,6,11] 
        json_path = f"{ARGS.dataset}_10dp.json"
        ARGS.batch_size = min(2, ARGS.batch_size)
        ARGS.enc_size, ARGS.dec_size  = 50, 51
        ARGS.enc_layers = ARGS.dec_layers = 1
        ARGS.no_chkpt = True
        if ARGS.max_epochs == 1000:
            ARGS.max_epochs = 1
        w2v = KeyedVectors.load_word2vec_format('/home/louis/model.bin',binary=True,limit=20000)
    else:
        json_path = f"{ARGS.dataset}_final.json"
        splits =  [1200,1300,1970] if ARGS.dataset=='msvd' else [6517,7010,10000]
    
    exp_name = get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if not os.path.isdir('../experiments/{}'.format(exp_name)): os.mkdir('../experiments/{}'.format(exp_name))
    elif ARGS.exp_name == 'try' or ARGS.exp_name.startswith('jade') or ARGS.overwrite: pass
    elif not get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name)):
        print('Please rerun command with a different experiment name')
        sys.exit()

    with open(json_path) as f: json_data=json.load(f)

    global LOAD_START_TIME; LOAD_START_TIME = time() 
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data['dataset'],splits,ARGS.batch_size,ARGS.shuffle)
    
    run_experiment(exp_name, ARGS, json_data, train_dl, val_dl, test_dl, w2v)
    print(f'Load Time: {asMinutes(TRAIN_START_TIME-LOAD_START_TIME)}\nTrain Time: {asMinutes(EVAL_START_TIME-TRAIN_START_TIME)}\nEval Time: {asMinutes(time() - EVAL_START_TIME)}\nTotal Time: {asMinutes(time()-LOAD_START_TIME)}')


if __name__=="__main__":
    ARGS = options.load_arguments()

    import torch
    torch.manual_seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.cuda_visible_devices
    import numpy as np
    np.random.seed(ARGS.seed)
    main()
