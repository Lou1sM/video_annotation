import json
from search_threshes import find_best_thresh_from_probs
from train import make_mlp_dict_from_pickle
from pdb import set_trace


with open('/data1/louis/data/rdf_video_captions/MSVD-wordnet-25d.json') as f: gt_json = json.load(f)
gt_json = {dp['video_id']: dp for dp in gt_json}
mlp_dict = make_mlp_dict_from_pickle('/data1/louis/data/MSVD-wordnet-25d-mlps.pickle')

results, *_ = find_best_thresh_from_probs('test','test',25,mlp_dict,gt_json)
print(results)
