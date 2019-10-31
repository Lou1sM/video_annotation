import pandas as pd
import sys
import torch.nn as nn
import json
import torch
import numpy as np
import torch.nn.functional as F


def update_row(error_row):
    total = error_row['total']
    error_row = {k:v/total for k,v in error_row.items() if k != 'total'}
    error_row['total'] = total
    return error_row


def compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device):
    total_count = 0
    missed_count = 0
    pos_predictions = []
    neg_predictions = []
    errors_by_object = {}

    for dpoint in outputs_json:
        video_id = int(dpoint['videoId'])
        gt = gt_json[video_id]
        
        triples = gt['caption']
        ntriples = gt['negatives']

        for triple in triples:
            total_count += 1
            sub, relation, obj = triple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                missed_count += 1
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = gt['individuals'].index(sub)
            obj_pos = gt['individuals'].index(obj)
            sub_embedding = torch.tensor(dpoint['embeddings'][sub_pos], device=device)
            obj_embedding = torch.tensor(dpoint['embeddings'][obj_pos], device=device)
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            new_pos_prediction = mlp(sub_obj_concat).item()
            pos_predictions.append(new_pos_prediction)
            for ind in [sub,relation,obj]:
                if ind not in errors_by_object.keys():
                    errors_by_object[ind] = {'subject_pos': 0, 'object_pos': 0, 'predicate_pos': 0, 'subject_neg': 0, 'object_neg': 0, 'predicate_neg': 0, 'total_errors_pos':0, 'total_errors_neg':0, 'total_errors': 0, 'total':0}
                errors_by_object[ind]['total'] += 1

            if new_pos_prediction < .5:
                errors_by_object[sub]['subject_pos'] += 1
                errors_by_object[sub]['total_errors_pos'] += 1
                errors_by_object[sub]['total_errors'] += 1
                errors_by_object[obj]['object_pos'] += 1
                errors_by_object[obj]['total_errors_pos'] += 1
                errors_by_object[obj]['total_errors'] += 1
                errors_by_object[relation]['predicate_pos'] += 1
                errors_by_object[relation]['total_errors_pos'] += 1
                errors_by_object[relation]['total_errors'] += 1

        for ntriple in ntriples:
            total_count += 1
            sub, relation, obj = ntriple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                missed_count += 1
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = gt['individuals'].index(sub)
            obj_pos = gt['individuals'].index(obj)
            sub_embedding = torch.tensor(dpoint['embeddings'][sub_pos], device=device)
            obj_embedding = torch.tensor(dpoint['embeddings'][obj_pos], device=device)
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            new_neg_prediction = mlp(sub_obj_concat).item()
            neg_predictions.append(new_neg_prediction)
            for ind in [sub,relation,obj]:
                if ind not in errors_by_object.keys():
                    errors_by_object[ind] = {'subject_pos': 0, 'object_pos': 0, 'predicate_pos': 0, 'subject_neg': 0, 'object_neg': 0, 'predicate_neg': 0, 'total_errors_pos':0, 'total_errors_neg':0, 'total_errors': 0, 'total':0}
                errors_by_object[ind]['total'] += 1

            if new_neg_prediction > .5:
                errors_by_object[sub]['subject_neg'] += 1
                errors_by_object[sub]['total_errors_neg'] += 1
                errors_by_object[sub]['total_errors'] += 1
                errors_by_object[obj]['object_neg'] += 1
                errors_by_object[obj]['total_errors_neg'] += 1
                errors_by_object[obj]['total_errors'] += 1
                errors_by_object[relation]['predicate_neg'] += 1
                errors_by_object[relation]['total_errors_neg'] += 1
                errors_by_object[relation]['total_errors'] += 1

    print('Missed:', missed_count)
    print('Total:', total_count)
    errors_by_object = {k: update_row(v) for k,v in errors_by_object.items()}
    return pos_predictions, neg_predictions, errors_by_object



if __name__ == "__main__":
    
    exp_name = sys.argv[1]

    with open('/data1/louis/experiments/{}/{}-test_outputs.txt'.format(exp_name, exp_name)) as f:
        outputs_json = json.load(f)

    with open('/data1/louis/data/rdf_video_captions/MSRVTT-10d-det.json.neg') as f:
        gt = json.load(f)
        gt = {g['videoId']: g for g in gt}

    mlp_dict = {}
    weight_dict = torch.load("../data/10d-mlps.pickle")
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
