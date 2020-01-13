import os
import operator
import numpy as np
import subprocess
from compute_probs import compute_probs_for_dataset
from pdb import set_trace


def compute_scores_for_thresh(positive_probs, negative_probs, thresh):

    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])

    prec = tp/(tp+fp+1e-4)
    rec = tp/(tp+fn+1e-4)
    f1 = 2/((1/(prec+1e-4))+(1/(rec+1e-4)))
    acc = (tp+tn)/(tp+fp+fn+tn)

    return tp, fp, fn, tn, f1, acc


def find_best_thresh_from_probs(outputs_json, gt_json, mlp_dict):
       
    positive_probs, negative_probs = compute_probs_for_dataset(dl,json_data_dict,ind_dict,mlp_dict)
       
    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)

    tphalf, fphalf, fnhalf, tnhalf, f1half, acchalf = compute_scores_for_thresh(positive_probs, negative_probs, 0.5)
    
    print("\nSearching thresholds")
    best_f1 = -1
    for thresh in np.linspace(avg_neg_prob, avg_pos_prob, num=10):
        tp, fp, fn, tn, f1, acc = compute_scores_for_thresh(positive_probs, negative_probs, thresh)
        if f1>best_f1:
            best_thresh = thresh
            best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn
            best_f1 = f1
            best_acc = acc

    best_metric_data = {'thresh': best_thresh, 'tp':best_tp, 'fp':best_fp, 'fn':best_fn, 'tn':best_tn, 'f1':best_f1, 'best_acc':best_acc, 'tphalf':tphalf, 'fphalf':fphalf, 'fnhalf':fnhalf, 'tnhalf':tnhalf, 'f1half':f1half, 'acchalf':acchalf, 'avg_pos_prob':avg_pos_prob, 'avg_neg_prob':avg_neg_prob}

    return best_metric_data, positive_probs, negative_probs



if __name__ == "__main__":
    import sys
    import json
    from train import make_mlp_dict_from_pickle
    exp_name = sys.argv[1]
    mlp_dict = make_mlp_dict_from_pickle(f'/data1/louis/data/MSVD-wordnet-25d-mlps.pickle', sigmoid=True)
    with open('/data1/louis/data/rdf_video_captions/MSVD-wordnet-25d.json') as f: gt_json = {d['video_id']: d for d in json.load(f)}
    with open(f'../experiments/{exp_name}/{exp_name}-test_outputs.txt') as f: exp_outputs = json.load(f)
    best_metric_data, positive_probs, negative_probs, inference_probs, error_dict = find_best_thresh_from_probs(exp_outputs,gt_json,mlp_dict)
    set_trace()
