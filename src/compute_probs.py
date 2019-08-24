import torch.nn as nn
import json
import torch
import numpy as np
import torch.nn.functional as F


def compute_probs_for_dataset(outputs_json, gt_json, mlp_dict, device):
    
    pos_predictions = []
    neg_predictions = []
    print(len(outputs_json))

    for dpoint in outputs_json:
        video_id = int(dpoint['videoId'])
        gt = gt_json[video_id]
        
        triples = gt['caption']
        ntriples = gt['negatives']

        for triple in triples:
            sub, relation, obj = triple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = gt['individuals'].index(sub)
            obj_pos = gt['individuals'].index(obj)
            #print(dpoint['embeddings'])
            sub_embedding = torch.tensor(dpoint['embeddings'][sub_pos], device=device)
            obj_embedding = torch.tensor(dpoint['embeddings'][obj_pos], device=device)
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            new_pos_prediction = mlp(sub_obj_concat).item()
            pos_predictions.append(new_pos_prediction)
 
        for ntriple in ntriples:
            sub, relation, obj = ntriple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = gt['individuals'].index(sub)
            obj_pos = gt['individuals'].index(obj)
            sub_embedding = torch.tensor(dpoint['embeddings'][sub_pos], device=device)
            obj_embedding = torch.tensor(dpoint['embeddings'][obj_pos], device=device)
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            new_neg_prediction = mlp(sub_obj_concat).item()
            neg_predictions.append(new_neg_prediction)
 
    return pos_predictions, neg_predictions



if __name__ == "__main__":
    with open('../experiments/5.39/5.39-val_outputs.txt') as f:
        outputs_json = json.load(f)

    with open('../data/rdf_video_captions/MSVD-10d-det.json.neg') as f:
        gt = json.load(f)
        #gt = {int(g['videoId']): g for g in gt}
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

    pos_predictions, neg_predictions = compute_probs_for_dataset(outputs_json, gt, mlp_dict, 'cuda')
    print(pos_predictions[:100])
    print(neg_predictions[:100])
