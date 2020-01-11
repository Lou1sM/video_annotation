import os
import operator
import numpy as np
import subprocess
from compute_probs import compute_probs_for_dataset
from pdb import set_trace


def compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, thresh):

    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])
    infp = len([p for p in inference_probs if p>thresh])
    infn = len(inference_probs) - infp

    prec = tp/(tp+fp+1e-4)
    rec = tp/(tp+fn+1e-4)
    f1 = 2/((1/(prec+1e-4))+(1/(rec+1e-4)))
    acc = (tp+tn)/(tp+fp+fn+tn)
    inf_acc = infp/len(inference_probs)

    return tp, fp, fn, tn, infp, infn, f1, acc, inf_acc


def find_best_thresh_from_probs(outputs_json, gt_json, mlp_dict):
       
    positive_probs, negative_probs, inference_probs, error_dict = compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device='cuda')
       
    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)
    avg_inf_prob = sum(inference_probs)/len(inference_probs)

    tphalf, fphalf, fnhalf, tnhalf, infphalf, infnhalf, f1half, acchalf, inf_acchalf = compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, 0.5)
    
    print("\nSearching thresholds")
    best_f1 = -1
    #for thresh in np.concatenate([np.array([0.]),np.arange(avg_neg_prob-.01, avg_pos_prob+0.1, 1e-3)]):
    for thresh in np.linspace(avg_neg_prob, avg_pos_prob, num=10):
        tp, fp, fn, tn, infp, infn, f1, acc, inf_acc = compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, thresh)
        if f1>best_f1:
            best_thresh = thresh
            best_tp, best_fp, best_fn, best_tn, best_infp, best_infn = tp, fp, fn, tn, infp, infn
            best_f1 = f1
            best_acc = acc
            best_inf_acc = inf_acc

    best_metric_data = {'thresh': best_thresh, 'tp':best_tp, 'fp':best_fp, 'fn':best_fn, 'tn':best_tn, 'f1':best_f1, 'best_acc':best_acc, 'inf_acc': best_inf_acc, 'tphalf':tphalf, 'fphalf':fphalf, 'fnhalf':fnhalf, 'tnhalf':tnhalf, 'infphalf': infphalf, 'infnhalf': infnhalf, 'f1half':f1half, 'acchalf':acchalf, 'inf_acchalf': inf_acchalf, 'avg_pos_prob':avg_pos_prob, 'avg_neg_prob':avg_neg_prob, 'avg_inf_prob': avg_inf_prob}

    return best_metric_data, positive_probs, negative_probs, inference_probs, error_dict



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
