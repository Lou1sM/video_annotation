"""Compute scores for results from a specified experiment. Metrics computed are
tp,fp,fn,tn,f1,accuracy. Each are computed twice, once using a fixed threshold
of 0.5 and once using the best available threshold. Also computed are the
average probability assigned to positive facts and negative facts respectively.
"""

import json
import numpy as np
from get_pred import compute_probs_for_dataset
from utils import acc_f1_from_binary_confusion_mat


def compute_dset_fragment_scores(dl,encoder,multiclassifier,dataset_dict,fragment_name,ARGS):
    """Compute performance metrics for a train/val/test dataset fragment. First
    executes forward pass of network to get outputs corresponding to true and
    false individuals and predicates; then thresholds and computes metrics.
    """

    pos_classifs,neg_classifs,pos_preds,neg_preds,perfects,acc,f1 = compute_probs_for_dataset(dl,encoder,multiclassifier,dataset_dict,ARGS.i3d)
    classif_scores = find_best_thresh_from_probs(pos_classifs,neg_classifs)
    pred_scores = find_best_thresh_from_probs(pos_preds,neg_classifs)
    classif_scores['dset_fragment'] = fragment_name
    pred_scores['dset_fragment'] = fragment_name
    for vid_id,num_atoms in perfects.items():
        if num_atoms < 2: continue
        assert num_atoms == len(dataset_dict['dataset'][vid_id]['pruned_atoms_with_synsets'])
        perfects[vid_id]=dataset_dict['dataset'][vid_id]['pruned_atoms_with_synsets']
    perfects_path = f'../experiments/{ARGS.exp_name}/train_perfects.json'
    open(perfects_path,'a').close()
    with open(perfects_path,'w') as f: json.dump(perfects,f)
    return classif_scores, pred_scores, perfects, acc, f1

def compute_scores_for_thresh(positive_probs, negative_probs, thresh):
    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)

    return tp, fp, fn, tn, f1, acc

def find_best_thresh_from_probs(positive_probs, negative_probs):
    """Compute accuracy and f1 by thresholding probabilities for positive and
    negative atoms. Use both a fixed threshold of 0.5 (reported in the paper)
    and also search for the threshold that gives the highest f1.
    """

    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)
    tphalf, fphalf, fnhalf, tnhalf, f1half, acchalf = compute_scores_for_thresh(positive_probs, negative_probs, 0.0)

    best_f1 = -1
    for thresh in np.linspace(avg_neg_prob, avg_pos_prob, num=10):
        tp, fp, fn, tn, f1, acc = compute_scores_for_thresh(positive_probs, negative_probs, thresh)
        if f1>best_f1:
            best_thresh = thresh
            best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn
            best_f1 = f1
            best_acc = acc

    return {'thresh': best_thresh,
            'tp':best_tp,
            'fp':best_fp,
            'fn':best_fn,
            'tn':best_tn,
            'f1':best_f1,
            'best_acc':best_acc,
            'tphalf':tphalf,
            'fphalf':fphalf,
            'fnhalf':fnhalf,
            'tnhalf':tnhalf,
            'f1half':f1half,
            'acchalf':acchalf,
            'avg_pos_prob':avg_pos_prob,
            'avg_neg_prob':avg_neg_prob
            }
