import json
from train import make_mlp_dict_from_pickle
from search_threshes import find_best_thresh_from_probs
import argparse
from pdb import set_trace
from copy import deepcopy


gt_json_fname = f'/data1/louis/data/rdf_video_captions/MSVD-wordnet-25d.json'
with open(gt_json_fname) as f: gt_json_as_dict = {dp['video_id']: dp for dp in json.load(f)}

outputs_json_fname = f'/data1/louis/data/rdf_video_captions/test_gt.json'
with open(outputs_json_fname) as f: outputs = json.load(f)

gt_outputs = []
for dp in outputs[:10]:
    new = deepcopy(dp)
    new['embeddings'] = gt_json_as_dict[dp['video_id']]['gt_embeddings']
    gt_outputs.append(new)

sames = [dp['embeddings'] == gt_json_as_dict[dp['video_id']]['gt_embeddings'] for dp in gt_outputs]
try: assert sum(sames) == len(sames)
except: set_trace()
set_trace()
mlp_fname = f'/data1/louis/data/MSVD-wordnet-25d-mlps.pickle'
mlp_dict = make_mlp_dict_from_pickle(mlp_fname,sigmoid=True)

res,*_ = find_best_thresh_from_probs(outputs_json=gt_outputs,gt_json=gt_json_as_dict, mlp_dict=mlp_dict)

print(res)

