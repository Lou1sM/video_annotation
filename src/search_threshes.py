import operator
import numpy as np
import subprocess
import sys
import json


def compute_f1_for_thresh(positive_probs, negative_probs, thresh):

    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])

    prec = tp/(tp+fp+1e-4)
    rec = tp/(tp+fn+1e-4)
    f1 = 2/((1/(prec+1e-4))+(1/(rec+1e-4)))

    return tp, fp, fn, tn, prec, rec, f1


def find_best_thresh_from_probs(exp_name, dset_fragment):
    
    prob_file_name = "../experiments/{}/{}-{}probabilities.json".format(exp_name, exp_name, dset_fragment)
    try:
        with open(prob_file_name, 'r') as prob_file:
            data = json.load(prob_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        emb_file_name = "../experiments/{}/{}-{}_outputs.txt".format(exp_name, exp_name, dset_fragment)
        #emb_file_name = "../experiments/{}/10d-det-{}_{}_outputs.txt".format(exp_name, dset_fragment, exp_name)
        print("Can't read from probabilites file {}. Computing probabilities from scratch using embeddings at {}".format(prob_file_name, emb_file_name))
        data = json.loads(subprocess.Popen(["vc-eval", emb_file_name], stdout=subprocess.PIPE).communicate()[0].decode())
        with open(prob_file_name, 'w') as prob_file:
            json.dump(data, prob_file)
    f1_by_thresh = {}
       
    threshes = []
    tps = []
    fps = []
    fns = []
    tns = []
    f1s = []
    positive_probs = data['probabilities']['pos']
    negative_probs = data['probabilities']['neg']
    avg_pos_prob = data['avg-probabilities']['pos'] 
    avg_neg_prob = data['avg-probabilities']['neg'] 
    assert avg_pos_prob - (sum(positive_probs)/len(positive_probs)) < 1e-3
    assert avg_neg_prob - (sum(negative_probs)/len(negative_probs)) < 1e-3
    best_f1 = 0

    print("\nSearching thresholds")
    for thresh in np.arange(avg_neg_prob-.01, avg_pos_prob+0.1, 3e-5):
        tp, fp, fn, tn, prec, rec, f1 = compute_f1_for_thresh(positive_probs, negative_probs, thresh)
        threshes.append(thresh)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        f1s.append(f1)
        if f1>best_f1:
            best_thresh = thresh
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
            best_f1 = f1

    total_metric_data = {'thresh': threshes, 'tp': tps, 'fp': fps, 'fn': fns, 'tn':tns, 'f1': f1s}
    best_metric_data = {'thresh': best_thresh, 'tp': best_tp, 'fp': best_fp, 'fn': best_fn, 'tn':best_tn, 'f1': best_f1, 'pat_norm': data['avg-embedding-norm'], 'pat_distance': data['avg-distance'], 'avg_pos_prob': avg_pos_prob, 'avg_neg_prob': avg_neg_prob}

    with open('../experiments/{}/{}-{}metrics.json'.format(exp_name, dset_fragment, exp_name), 'w') as jsonfile:
        json.dump(total_metric_data, jsonfile)

    return best_metric_data, total_metric_data, positive_probs, negative_probs



if __name__ == "__main__":
    exp_name = sys.argv[1]
    find_best_thresh_from_outputs_file(exp_name)
