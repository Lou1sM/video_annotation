import os
import operator
import numpy as np
import subprocess
import sys
import json
from compute_probs import compute_probs_for_dataset


def compute_scores_for_thresh(positive_probs, negative_probs, thresh):

    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])

    prec = tp/(tp+fp+1e-4)
    rec = tp/(tp+fn+1e-4)
    f1 = 2/((1/(prec+1e-4))+(1/(rec+1e-4)))
    acc = (tp+tn)/(tp+fp+fn+tn)

    return tp, fp, fn, tn, prec, rec, f1, acc


def find_best_thresh_from_probs(exp_name, dset_fragment, ind_size, dataset, mlp_dict, gt_json):
    
    prob_file_name = "../experiments/{}/{}-{}probabilities.json".format(exp_name, exp_name, dset_fragment)
    try:
        with open(prob_file_name, 'r') as prob_file:
            data = json.load(prob_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        emb_file_path = f"../experiments/{exp_name}/{exp_name}-{dset_fragment}_outputs.txt"
        gt_file_path = f"/data1/louis/data/rdf_video_captions/{dataset}.json"
    emb_file_path = f"../experiments/{exp_name}/{exp_name}-{dset_fragment}_outputs.txt"
    with open(emb_file_path, 'r') as emb_file:
        outputs_json = json.load(emb_file)
        
    positive_probs, negative_probs, error_dict = compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device='cuda')
       
    threshes,tps,fps,fns,tns,f1s,accs = [[]]*7
    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)
    best_f1 = -1

    print("\nSearching thresholds")
    tphalf, fphalf, fnhalf, tnhalf, prechalf, rechalf, f1half, acchalf = compute_scores_for_thresh(positive_probs, negative_probs, 0.5)
    for thresh in np.concatenate([np.array([0.]),np.arange(avg_neg_prob-.01, avg_pos_prob+0.1, 1e-3)]):
        tp, fp, fn, tn, prec, rec, f1, acc = compute_scores_for_thresh(positive_probs, negative_probs, thresh)
        threshes.append(thresh)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        f1s.append(f1)
        accs.append(acc)
        if f1>best_f1:
            best_thresh = thresh
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
            best_f1 = f1
            best_acc = acc

    total_metric_data = {'thresh': threshes, 'tp': tps, 'fp': fps, 'fn': fns, 'tn':tns, 'f1': f1s, 'acc': accs}
    best_metric_data = {'thresh': best_thresh, 'tp': best_tp, 'tphalf': tphalf, 'fp': best_fp, 'fphalf': fphalf, 'fn': best_fn, 'fnhalf': fnhalf, 'tn':best_tn, 'tnhalf': tnhalf, 'f1': best_f1, 'f1half': f1half, 'best_acc': best_acc, 'acchalf': acchalf, 'avg_pos_prob': avg_pos_prob, 'avg_neg_prob': avg_neg_prob}

    with open('../experiments/{}/{}-{}metrics.json'.format(exp_name, dset_fragment, exp_name), 'w') as jsonfile:
        json.dump(total_metric_data, jsonfile)

    return best_metric_data, total_metric_data, positive_probs, negative_probs, error_dict



if __name__ == "__main__":
    exp_name = sys.argv[1]
    print(find_best_thresh_from_outputs_file(exp_name))
