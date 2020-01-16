import csv
import sys
import os
import json
from pdb import set_trace
from collections import OrderedDict
from torch import optim
from utils import get_datetime_stamp, asMinutes,get_w2v_vec
from time import time
from gensim.models import KeyedVectors

import options
import my_models
import train
import data_loader 
from semantic_parser import tuplify
from get_metrics import compute_dset_fragment_scores


def main():
    print(ARGS)

    global LOAD_START_TIME; LOAD_START_TIME = time() 
    exp_name = get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if not os.path.isdir('../experiments/{}'.format(exp_name)): os.mkdir('../experiments/{}'.format(exp_name))
    elif ARGS.exp_name == 'try' or ARGS.exp_name.startswith('jade') or ARGS.overwrite: pass
    elif not get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name)):
        print('Please rerun command with a different experiment name')
        sys.exit()
    
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
        w2v = KeyedVectors.load_word2vec_format('/data1/louis/w2v_vecs.bin',binary=True,limit=2000)
    else:
        splits =  [1200,1300,1970] if ARGS.dataset=='MSVD' else [6517,7010,10000]
        json_path = f"{ARGS.dataset}_final.json"
        print('Loading w2v model...')
        w2v = KeyedVectors.load_word2vec_format('/data1/louis/w2v_vecs.bin',binary=True)
    with open(json_path) as f: json_data=json.load(f)

    inds,classes,relations,json_data_list = json_data['inds'],json_data['classes'],json_data['relations'],json_data['dataset']
    for dp in json_data_list:
        dp['pruned_atoms_with_synsets'] = [tuplify(a) for a in dp['pruned_atoms_with_synsets']]
        dp['lcwa'] = [tuplify(a) for a in dp['lcwa']]
    json_data_dict = {dp['video_id']:dp for dp in json_data_list}
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data_list,splits,ARGS.batch_size,ARGS.shuffle,ARGS.i3d)

    if ARGS.reload:
        reload_file_path = '/data1/louis/checkpoints/{}.pt'.format(ARGS.reload)
        reload_file_path = ARGS.reload
        print('Reloading model from {}'.format(reload_file_path))
        saved_model = torch.load(reload_file_path)
        encoder = saved_model['encoder']
        multiclassifier = saved_model['multiclassifier']
        ind_dict = saved_model['ind_dict']
        mlp_dict = saved_model['mlp_dict']
        encoder.batch_size = ARGS.batch_size
        optimizer = saved_model['optimizer']
    else: 
        print('Initializing new networks...')
        encoder = my_models.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
        encoding_size = ARGS.enc_size + 1024 if ARGS.i3d else ARGS.enc_size
        multiclassifier = my_models.MLP(encoding_size,ARGS.classif_size,len(inds)).to(ARGS.device)
        mlp_dict = {}
        class_dict = {tuple(c): my_models.MLP(encoding_size + ARGS.ind_size,ARGS.mlp_size,1).to(ARGS.device) for c in classes}
        relation_dict = {tuple(r): my_models.MLP(encoding_size + 2*ARGS.ind_size,ARGS.mlp_size,1).to(ARGS.device) for r in relations}
        #class_dict = {tuple(c): my_models.MLP(ARGS.enc_size + ARGS.ind_size,ARGS.mlp_size,1).to(ARGS.device) for c in classes}
        #relation_dict = {tuple(r): my_models.MLP(ARGS.enc_size + 2*ARGS.ind_size,ARGS.mlp_size,1).to(ARGS.device) for r in relations}
        mlp_dict = {'classes':class_dict, 'relations':relation_dict}
        ind_dict = {tuple(ind): torch.nn.Parameter(torch.tensor(get_w2v_vec(ind,w2v),device=ARGS.device,dtype=torch.float32)) for ind in inds}
    
        encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        params_list = [encoder.parameters(), multiclassifier.parameters()] + [ind for ind in ind_dict.values()] + [mlp.parameters() for mlp in mlp_dict['classes'].values()] + [mlp.parameters() for mlp in mlp_dict['relations'].values()]
        optimizer = optim.Adam([{'params': params, 'lr':ARGS.learning_rate, 'wd':ARGS.weight_decay} for params in params_list])
    for param in encoder.cnn.parameters():
        param.requires_grad = False

    dataset_dict = {'dataset':json_data_dict,'ind_dict':ind_dict,'mlp_dict':mlp_dict}
    global TRAIN_START_TIME; TRAIN_START_TIME = time()
    print('\nTraining the model')
    train.train(ARGS, encoder, multiclassifier, dataset_dict, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, exp_name=exp_name, device=ARGS.device, train=True)
       
    global EVAL_START_TIME; EVAL_START_TIME = time()
    if ARGS.no_chkpt: print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
    else:
        checkpoint_path = '/data1/louis/checkpoints/{}.pt'.format(exp_name) 
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        multiclassifier = checkpoint['multiclassifier']
        dataset_dict['ind_dict']  = checkpoint['ind_dict']
        dataset_dict['mlp_dict']  = checkpoint['mlp_dict']
        print("Reloading best network version for outputs")

    encoder.batch_size=1
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data['dataset'],splits,batch_size=1,shuffle=False,i3d=ARGS.i3d)
    val_classification_scores, val_prediction_scores, val_perfects = compute_dset_fragment_scores(val_dl,encoder,multiclassifier,dataset_dict,'val',ARGS.i3d)
    train_classification_scores, train_prediction_scores, train_perfects = compute_dset_fragment_scores(train_dl,encoder,multiclassifier,dataset_dict,'train',ARGS.i3d)
    #fixed_thresh = ((train_output_info['thresh']*1200)+(val_output_info['thresh']*100))/1300
    test_classification_scores, test_prediction_scores, test_perfects = compute_dset_fragment_scores(test_dl,encoder,multiclassifier,dataset_dict,'test',ARGS.i3d)

    perfects={}
    for vid_id,num_atoms in test_perfects.items():
        if num_atoms < 2: continue
        assert num_atoms == len(json_data_dict[vid_id]['pruned_atoms_with_synsets'])
        perfects[vid_id]=json_data_dict[vid_id]['pruned_atoms_with_synsets']
    perfects_path = f'../experiments/{exp_name}/perfects.json'
    #if not os.path.isfile(perfects_path): os.mkdir(perfects_path)
    open(perfects_path,'a').close()
    with open(perfects_path,'w') as f: json.dump(perfects,f)
    summary_filename = '../experiments/{}/{}_summary.txt'.format(exp_name, exp_name) 
    with open(summary_filename, 'w') as summary_file:
        summary_file.write('Experiment name: {}\n'.format(exp_name))
        summary_file.write('\tTrain\tVal\tTest\n')
        for k in ['dset_fragment', 'tp', 'fn', 'fp', 'tn', 'f1', 'thresh', 'best_acc', 'acchalf', 'f1half', 'avg_pos_prob', 'avg_neg_prob']: 
            summary_file.write(k+'\t'+str(train_classification_scores[k])+'\t'+str(val_classification_scores[k])+'\t'+str(test_classification_scores[k])+'\n')
        for k in ['dset_fragment', 'tp', 'fn', 'fp', 'tn', 'f1', 'thresh', 'best_acc', 'acchalf', 'f1half', 'avg_pos_prob', 'avg_neg_prob']: 
            summary_file.write(k+'\t'+str(train_prediction_scores[k])+'\t'+str(val_prediction_scores[k])+'\t'+str(test_prediction_scores[k])+'\n')
        summary_file.write('\nParameters:\n')
        for key in options.IMPORTANT_PARAMS:
            summary_file.write(str(key) + ": " + str(vars(ARGS)[key]) + "\n")
    print(f'Load Time: {asMinutes(TRAIN_START_TIME-LOAD_START_TIME)}\nTrain Time: {asMinutes(EVAL_START_TIME-TRAIN_START_TIME)}\nEval Time: {asMinutes(time() - EVAL_START_TIME)}\nTotal Time: {asMinutes(time()-LOAD_START_TIME)}')


def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y': return True
    elif answer == 'n': return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))

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
