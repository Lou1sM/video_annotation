import os
import operator
import numpy as np
import subprocess
import sys
import json
from compute_probs import compute_probs_for_dataset
from pdb import set_trace


def compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, thresh):

    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])
    inf_acc = len([p for p in inference_probs if p>thresh])/len(inference_probs)

    prec = tp/(tp+fp+1e-4)
    rec = tp/(tp+fn+1e-4)
    f1 = 2/((1/(prec+1e-4))+(1/(rec+1e-4)))
    acc = (tp+tn)/(tp+fp+fn+tn)

    return tp, fp, fn, tn, f1, acc, inf_acc


def find_best_thresh_from_probs(outputs_json, gt_json, mlp_dict):
       
    positive_probs, negative_probs, inference_probs, error_dict = compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device='cuda')
       
    threshes,tps,fps,fns,tns,f1s,accs,inf_accs = [[]]*8
    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)
    best_f1 = -1

    print("\nSearching thresholds")
    tphalf, fphalf, fnhalf, tnhalf, f1half, acchalf, inf_acchalf = compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, 0.5)
    for thresh in np.concatenate([np.array([0.]),np.arange(avg_neg_prob-.01, avg_pos_prob+0.1, 1e-3)]):
        tp, fp, fn, tn, f1, acc, inf_acc = compute_scores_for_thresh(positive_probs, negative_probs, inference_probs, thresh)
        threshes.append(thresh)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        f1s.append(f1)
        accs.append(acc)
        inf_accs.append(inf_acc)
        if f1>best_f1:
            best_thresh = thresh
            best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn
            best_f1 = f1
            best_acc = acc
            best_inf_acc = inf_acc

    best_metric_data = {'thresh': best_thresh, 'tp':best_tp, 'fp':best_fp, 'fn':best_fn, 'tn':best_tn, 'f1':best_f1, 'best_acc':best_acc, 'inf_acc': best_inf_acc, 'tphalf':tphalf, 'fphalf':fphalf, 'fnhalf':fnhalf, 'tnhalf':tnhalf, 'f1half':f1half, 'acchalf':acchalf, 'inf_acchalf': inf_acchalf, 'avg_pos_prob':avg_pos_prob, 'avg_neg_prob':avg_neg_prob}

    return best_metric_data, positive_probs, negative_probs, inference_probs, error_dict



if __name__ == "__main__":
    exp_name = sys.argv[1]
    print(find_best_thresh_from_outputs_file(exp_name))
