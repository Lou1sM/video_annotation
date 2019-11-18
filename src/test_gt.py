import json
from train import make_mlp_dict_from_pickle
from search_threshes import find_best_thresh_from_probs
import argparse



gt_json_fname = f'/data1/louis/data/rdf_video_captions/MSVD-wordnet-25d.json'
mlp_fname = f'/data1/louis/data/MSVD-wordnet-25d-mlps.pickle'
with open(gt_json_fname) as f: gt_json_as_list=json.load(f)
gt_json_as_dict = {dp['video_id']: dp for dp in gt_json_as_list}

mlp_dict = make_mlp_dict_from_pickle(mlp_fname,sigmoid=True)
res,*_ = find_best_thresh_from_probs('gt','',25,mlp_dict=mlp_dict,gt_json=gt_json_as_dict)

print(res)

