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


def main(args):
    print(args)

    global LOAD_START_TIME; LOAD_START_TIME = time()
    if args.mini:
        args.exp_name = 'try'
    exp_name = get_datetime_stamp() if args.exp_name == "" else args.exp_name
    exp_dir = os.path.join(args.data_dir,exp_name)
    if not os.path.isdir(exp_dir): os.makedirs(exp_dir)
    elif args.exp_name == 'try' or args.exp_name.startswith('jade') or args.overwrite: pass
    elif not get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name)):
        print('Please rerun command with a different experiment name')
        sys.exit()

    w2v_path = os.path.join(args.data_dir,'w2v_vecs.bin')
    if args.mini:
        splits = [4,6,11]
        json_path = f"{args.dataset}_10dp.json"
        args.batch_size = min(2, args.batch_size)
        args.enc_size, args.dec_size = 50, 51
        args.enc_layers = args.dec_layers = 1
        args.no_chkpt = True
        if args.max_epochs == 1000:
            args.max_epochs = 1
        w2v = KeyedVectors.load_word2vec_format(w2v_path,binary=True,limit=2000)
    else:
        splits = [1200,1300,1970] if args.dataset=='MSVD' else [6517,7010,10000]
        json_path = f"{args.dataset}_final.json"
        print('Loading w2v model...')
        w2v = KeyedVectors.load_word2vec_format(w2v_path,binary=True,limit=args.w2v_limit)
    with open(json_path) as f: json_data=json.load(f)

    inds,classes,relations,json_data_list = json_data['inds'],json_data['classes'],json_data['relations'],json_data['dataset']
    for dp in json_data_list:
        dp['pruned_atoms_with_synsets'] = [tuplify(a) for a in dp['pruned_atoms_with_synsets']]
        dp['lcwa'] = [tuplify(a) for a in dp['lcwa']]
    json_data_dict = {dp['video_id']:dp for dp in json_data_list}
    video_data_dir = os.path.join(args.data_dir,args.dataset)
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data_list,splits,args.batch_size,args.shuffle,args.i3d,video_data_dir=video_data_dir)

    if args.reload:
        reload_file_path = args.reload
        print('Reloading model from {}'.format(reload_file_path))
        saved_model = torch.load(reload_file_path)
        encoder = saved_model['encoder']
        multiclassifier = saved_model['multiclassifier']
        ind_dict = saved_model['ind_dict']
        mlp_dict = saved_model['mlp_dict']
        encoder.batch_size = args.batch_size
        optimizer = saved_model['optimizer']
    else:
        print('Initializing new networks...')
        encoder = my_models.EncoderRNN(args, args.device).to(args.device)
        encoding_size = args.enc_size + 4096 if args.i3d else args.enc_size
        multiclassifier = my_models.MLP(encoding_size,args.classif_size,len(inds)).to(args.device)
        mlp_dict = {}
        class_dict = {tuple(c): my_models.MLP(encoding_size + args.ind_size,args.mlp_size,1).to(args.device) for c in classes}
        relation_dict = {tuple(r): my_models.MLP(encoding_size + 2*args.ind_size,args.mlp_size,1).to(args.device) for r in relations}
        mlp_dict = {'classes':class_dict, 'relations':relation_dict}
        ind_dict = {tuple(ind): torch.nn.Parameter(torch.tensor(get_w2v_vec(ind[0],w2v),device=args.device,dtype=torch.float32)) for ind in inds}

        #encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        params_list = [encoder.parameters(), multiclassifier.parameters()] + [ind for ind in ind_dict.values()] + [mlp.parameters() for mlp in mlp_dict['classes'].values()] + [mlp.parameters() for mlp in mlp_dict['relations'].values()]
        optimizer = optim.Adam([{'params': params, 'lr':args.learning_rate, 'wd':args.weight_decay} for params in params_list])

    dataset_dict = {'dataset':json_data_dict,'ind_dict':ind_dict,'mlp_dict':mlp_dict}
    global TRAIN_START_TIME; TRAIN_START_TIME = time()
    print('\nTraining the model')
    train.train(args, encoder, multiclassifier, dataset_dict, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, exp_name=exp_name, device=args.device, train=True)

    global EVAL_START_TIME; EVAL_START_TIME = time()
    if args.no_chkpt: print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
    else:
        checkpoint_path = os.path.join(args.data_dir,'checkpoints/{}.pt'.format(args.exp_name))
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        multiclassifier = checkpoint['multiclassifier']
        dataset_dict['ind_dict'] = checkpoint['ind_dict']
        dataset_dict['mlp_dict'] = checkpoint['mlp_dict']
        print("Reloading best network version for outputs")

    encoder.batch_size=1
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data['dataset'],splits,batch_size=1,shuffle=False,i3d=args.i3d,video_data_dir=video_data_dir)
    val_classification_scores, val_prediction_scores, val_perfects = compute_dset_fragment_scores(val_dl,encoder,multiclassifier,dataset_dict,'val',args)
    train_classification_scores, train_prediction_scores, train_perfects = compute_dset_fragment_scores(train_dl,encoder,multiclassifier,dataset_dict,'train',args)
    #fixed_thresh = ((train_output_info['thresh']*1200)+(val_output_info['thresh']*100))/1300
    test_classification_scores, test_prediction_scores, test_perfects = compute_dset_fragment_scores(test_dl,encoder,multiclassifier,dataset_dict,'test',args)

    summary_filename = os.path.join(exp_dir,'{}_summary.txt'.format(exp_name, exp_name))
    with open(summary_filename, 'w') as summary_file:
        summary_file.write('Experiment name: {}\n'.format(exp_name))
        summary_file.write('\tTrain\tVal\tTest\n')
        for k in ['dset_fragment', 'tp', 'fn', 'fp', 'tn', 'f1', 'thresh', 'best_acc', 'acchalf', 'f1half', 'avg_pos_prob', 'avg_neg_prob']:
            summary_file.write(k+'\t'+str(train_classification_scores[k])+'\t'+str(val_classification_scores[k])+'\t'+str(test_classification_scores[k])+'\n')
        for k in ['dset_fragment', 'tp', 'fn', 'fp', 'tn', 'f1', 'thresh', 'best_acc', 'acchalf', 'f1half', 'avg_pos_prob', 'avg_neg_prob']:
            summary_file.write(k+'\t'+str(train_prediction_scores[k])+'\t'+str(val_prediction_scores[k])+'\t'+str(test_prediction_scores[k])+'\n')
        summary_file.write('\nParameters:\n')
        for key in options.IMPORTANT_PARAMS:
            summary_file.write(str(key) + ": " + str(vars(args)[key]) + "\n")
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

    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.cuda_visible_devices
    import numpy as np
    np.random.seed(ARGS.seed)
    main(ARGS)
