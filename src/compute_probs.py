from pdb import set_trace
import re
import pandas as pd
import sys
import torch.nn as nn
import json
import torch
import numpy as np
import torch.nn.functional as F
from utils import get_pred_sub_obj, make_prediction


def update_row(error_row):
    total = error_row['total']
    error_row = {k:v/total for k,v in error_row.items() if k != 'total'}
    error_row['total'] = total
    return error_row


def compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device):
    pos_predictions, neg_predictions, inf_predictions, errors_by_object  = [], [], [], {}

    for dpoint in outputs_json:
        video_id = dpoint['video_id']
        assert isinstance(gt_json,dict)
        try: gt = gt_json[int(video_id)]
        except: set_trace()
        atoms,inferences,lcwa = gt['facts'], gt['inferences'], gt['lcwa']
        for atom in atoms:
            embedding,predname,subname,objname  = get_pred_sub_obj(atom,gt,dpoint)
            isclass = predname.startswith('c')
            assert predname.startswith('c') or predname.startswith('r')
            assert predname.startswith('r') != isclass
            new_pos_prediction = make_prediction(embedding.to(device),predname,isclass,mlp_dict,device).item()
            pos_predictions.append(new_pos_prediction)
            for ind in filter(lambda x: x,[predname,subname,objname]):
                if ind not in errors_by_object.keys():
                    errors_by_object[ind] = {'subject_pos': 0, 'object_pos': 0, 'predicate_pos': 0, 'subject_neg': 0, 'object_neg': 0, 'predicate_neg': 0, 'total_errors_pos':0, 'total_errors_neg':0, 'total_errors': 0, 'total':0}
                errors_by_object[ind]['total'] += 1

            if new_pos_prediction < .5:
                errors_by_object[subname]['subject_pos'] += 1
                errors_by_object[subname]['total_errors_pos'] += 1
                errors_by_object[subname]['total_errors'] += 1
                errors_by_object[predname]['predicate_pos'] += 1
                errors_by_object[predname]['total_errors_pos'] += 1
                errors_by_object[predname]['total_errors'] += 1
                if objname:
                    errors_by_object[objname]['object_pos'] += 1
                    errors_by_object[objname]['total_errors_pos'] += 1
                    errors_by_object[objname]['total_errors'] += 1

        for negatom in lcwa:
            embedding,predname,subname,objname = get_pred_sub_obj(negatom,gt,dpoint)
            predname = predname[1:]
            isclass = predname.startswith('c')
            assert predname.startswith('c') or predname.startswith('r')
            assert predname.startswith('r') != isclass
            new_neg_prediction = make_prediction(embedding.to(device),predname,isclass,mlp_dict,device).item()
            neg_predictions.append(new_neg_prediction)
            for ind in filter(lambda x: x, [predname,subname,objname]):
                if ind not in errors_by_object.keys():
                    errors_by_object[ind] = {'subject_pos': 0, 'object_pos': 0, 'predicate_pos': 0, 'subject_neg': 0, 'object_neg': 0, 'predicate_neg': 0, 'total_errors_pos':0, 'total_errors_neg':0, 'total_errors': 0, 'total':0}
                errors_by_object[ind]['total'] += 1

            if new_neg_prediction > .5:
                errors_by_object[subname]['subject_neg'] += 1
                errors_by_object[subname]['total_errors_neg'] += 1
                errors_by_object[subname]['total_errors'] += 1
                errors_by_object[predname]['total_errors_neg'] += 1
                errors_by_object[predname]['total_errors'] += 1
                errors_by_object[predname]['predicate_neg'] += 1
                if objname:
                    errors_by_object[objname]['object_neg'] += 1
                    errors_by_object[objname]['total_errors_neg'] += 1
                    errors_by_object[objname]['total_errors'] += 1

        for inference in inferences:
            embedding,predname,subname,objname  = get_pred_sub_obj(inference,gt,dpoint)
            isclass = predname.startswith('c')
            try: assert predname.startswith('c') or predname.startswith('r')
            except: set_trace()
            assert predname.startswith('r') != isclass
            new_inf_prediction = make_prediction(embedding.to(device),predname,isclass,mlp_dict,device).item()
            inf_predictions.append(new_inf_prediction)
    errors_by_object = {k: update_row(v) for k,v in errors_by_object.items()}
    return pos_predictions, neg_predictions, inf_predictions, errors_by_object


if __name__ == "__main__":
    
    exp_name = sys.argv[1]
    with open('../experiments/{}/{}-train_outputs.txt'.format(exp_name, exp_name)) as f:
        outputs_json = json.load(f)

    with open('/data4/patrick/wordnet-data-gen/out/msvd.0.json') as f:
        gt = [json.load(f)]
        gt = {g['video_id']: g for g in gt}

    mlp_dict = {}
    weight_dict = torch.load("/data1/louis/data/10d-mlps.pickle")
    for relation, weights in weight_dict.items():
        hidden_layer = nn.Linear(weights["hidden_weights"].shape[0], weights["hidden_bias"].shape[0])
        hidden_layer.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=False)
        hidden_layer.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=False)
        output_layer = nn.Linear(weights["output_weights"].shape[0], weights["output_bias"].shape[0])
        output_layer.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=False)
        output_layer.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=False)
        mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer, nn.Sigmoid()) 

    pos_predictions, neg_predictions, errors_by_object = compute_probs_for_dataset(outputs_json, gt, mlp_dict, 'cuda')

    df = pd.DataFrame(errors_by_object).T.sort_values(by='total_errors', ascending=False)
    df.to_csv('../experiments/{}/{}errors_by_obj.csv'.format(exp_name, exp_name))
    print(pos_predictions[:100])
    print(neg_predictions[:100])
    print(df.head())
