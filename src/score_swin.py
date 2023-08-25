import json
from utils import tuplify, acc_f1_from_binary_confusion_mat
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d','--dset',type=str,required=True,choices=['msvd','msrvtt'])
parser.add_argument('--recompute_counts','-t',action='store_true')
parser.add_argument('--verbose','-v',action='store_true')
ARGS = parser.parse_args()


with open(f'../data/{ARGS.dset}/swin_{ARGS.dset}_parsed_linked.json') as f: d=json.load(f)
if ARGS.recompute_counts:
    with open(f'../data/{ARGS.dset}/{ARGS.dset.upper()}_final.json') as f: all_gts_=json.load(f)
    all_gts = all_gts_['dataset']

    def counts_dict_by_id(full_list):
        names_by_id = {x[1]:x[0] for x in full_list} # this also amounts to removing nonunique ids
        all_ids = np.array([x[1] for x in full_list])
        all_ids_counts = np.bincount(all_ids)
        counts_by_id = {k:all_ids_counts[k] for k in names_by_id.keys()}
        return names_by_id, counts_by_id

    inds_list = sum([x[1:] for dp in all_gts for x in dp['atoms_with_synsets']],[])
    unique_inds_by_id_, ind_counts = counts_dict_by_id(inds_list)
    ind_ids_to_keep = [c for c,v in ind_counts.items() if v>50]
    INDS = [(unique_inds_by_id_[x],x) for x in ind_ids_to_keep]

    classes_list = [x[0] for dp in all_gts for x in dp['atoms_with_synsets'] if len(x)==2]
    unique_classes_by_id_, class_counts = counts_dict_by_id(classes_list)
    class_ids_to_keep = [c for c,v in class_counts.items() if v>50]
    CLASSES = [(unique_classes_by_id_[x],x) for x in class_ids_to_keep]

    relations_list = [x[0] for dp in all_gts for x in dp['atoms_with_synsets'] if len(x)==3]
    unique_relations_by_id_, relation_counts = counts_dict_by_id(relations_list)
    relation_ids_to_keep = [c for c,v in relation_counts.items() if v>50]
    RELATIONS = [(unique_relations_by_id_[x],x) for x in relation_ids_to_keep]

    new_gts = dict(all_gts_, **{'inds':INDS,'classes':CLASSES,'relations':RELATIONS})
    #unique_relations_by_id_, relation_counts = counts_dict_by_id(relations_list)

    with open('new_{ARGS.dset}_final.json','w') as f:
        json.dump(new_gts,f)

else:
    with open(f'../data/{ARGS.dset}/new_{ARGS.dset}_final.json') as f: all_gts_=json.load(f)
    all_gts = all_gts_['dataset']
    INDS = all_gts_['inds']
    CLASSES = all_gts_['classes']
    RELATIONS = all_gts_['relations']

all_accs = []
all_f1s = []

def in_vocab(triple):
    assert len(triple) in (2,3)
    if len(triple)==2:
        return triple[0] in CLASSES and triple[1] in INDS
    else:
        return triple[0] in RELATIONS and triple[1] in INDS and triple[2] in INDS

for dpoint in d:
    gt_ = [x for x in all_gts if x['video_id']==int(dpoint['video_id'].split('_')[0][3:])]
    assert len(gt_) == 1
    gt = gt_[0]
    y = tuplify(gt['pruned_atoms_with_synsets'])
    y_neg = tuplify(gt['lcwa'])
    pred_ = tuplify(dpoint['atoms_with_synsets'])
    pred = [p for p in pred_ if in_vocab(p)]
    excluded_pred = [p for p in pred_ if not in_vocab(p)]
    assert set(tuplify([tuplify(p) for p in pred+excluded_pred])) == set(tuplify([tuplify(p) for p in pred_]))
    wrongos = [p for p in y if p in excluded_pred]
    if len(wrongos)>0:
        print(wrongos)

    tp = len([item for item in y if item in pred_])
    fp = len([item for item in y_neg if item in pred])
    fn = len([item for item in y if item not in pred_])
    tn = len([item for item in y_neg if item not in pred])
    if tp+fp+fn+tn > 0:
        acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        all_accs.append(acc)
        all_f1s.append(f1)
    else:
        print(888)

print('acc',np.array(all_accs).mean())
print('f1',np.array(all_f1s).mean())
