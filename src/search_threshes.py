import os
import operator
import numpy as np
import subprocess
import sys
import json


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


def find_best_thresh_from_probs(exp_name, dset_fragment, ind_size):
    
    
    prob_file_name = "../experiments/{}/{}-{}probabilities.json".format(exp_name, exp_name, dset_fragment)
    try:
        with open(prob_file_name, 'r') as prob_file:
            data = json.load(prob_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        emb_file_path = "../experiments/{}/{}-{}_outputs.txt".format(exp_name, exp_name, dset_fragment)
        gt_file_path = "../data/rdf_video_captions/{}d-det.json.neg".format(ind_size)
        model_file_path = "../rrn-models/model-{}d-det.state".format(ind_size)
        assert os.path.isfile(emb_file_path)
        assert os.path.isfile(gt_file_path)
        assert os.path.isfile(model_file_path)
        print("Can't read from probabilites file {}. Computing probabilities from scratch using embeddings at {}".format(prob_file_name, emb_file_path))
        #data = json.loads(subprocess.Popen(["vc-eval", emb_file_path], stdout=subprocess.PIPE).communicate()[0].decode())
        #data = json.loads(subprocess.check_output("./embedding-gen/run-eval.sh %s %s %s" % ('../data/rdf_video_captions/10d-det.json.neg', '../experiments/try/try-val_outputs.txt', '../rrn-models/model-10d-det.state'), shell=True).decode())
        data = json.loads(subprocess.check_output("./embedding-gen/run-eval.sh %s %s %s" % (gt_file_path, emb_file_path, model_file_path), shell=True))
        with open(prob_file_name, 'w') as prob_file:
            json.dump(data, prob_file)
    f1_by_thresh = {}
       
    threshes = []
    tps = []
    fps = []
    fns = []
    tns = []
    f1s = []
    accs = []
    positive_probs = data['probabilities']['pos']
    negative_probs = data['probabilities']['neg']
    avg_pos_prob = data['avg-probabilities']['pos'] 
    avg_neg_prob = data['avg-probabilities']['neg'] 
    assert avg_pos_prob - (sum(positive_probs)/len(positive_probs)) < 1e-3
    assert avg_neg_prob - (sum(negative_probs)/len(negative_probs)) < 1e-3
    best_f1 = 0

    print("\nSearching thresholds")
    tphalf, fphalf, fnhalf, tnhalf, prechalf, rechalf, f1half, acchalf = compute_scores_for_thresh(positive_probs, negative_probs, 0.5)
    for thresh in np.arange(avg_neg_prob-.01, avg_pos_prob+0.1, 3e-5):
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
    best_metric_data = {'thresh': best_thresh, 'tp': best_tp, 'tphalf': tphalf, 'fp': best_fp, 'fphalf': fphalf, 'fn': best_fn, 'fnhalf': fnhalf, 'tn':best_tn, 'tnhalf': tnhalf, 'f1': best_f1, 'f1half': f1half, 'best_acc': best_acc, 'acchalf': acchalf, 'pat_norm': data['avg-embedding-norm'], 'pat_distance': data['avg-distance'], 'avg_pos_prob': avg_pos_prob, 'avg_neg_prob': avg_neg_prob}

    with open('../experiments/{}/{}-{}metrics.json'.format(exp_name, dset_fragment, exp_name), 'w') as jsonfile:
        json.dump(total_metric_data, jsonfile)

    return best_metric_data, total_metric_data, positive_probs, negative_probs



if __name__ == "__main__":
    exp_name = sys.argv[1]
    print(find_best_thresh_from_outputs_file(exp_name))
