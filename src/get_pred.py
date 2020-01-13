import json
import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace
import re
from utils import make_prediction
import math

torch.manual_seed(0)

def get_pred_loss(video_ids, context_vecs, json_data_dict, ind_dict, mlp_dict, training, margin=1, device='cuda'):
    loss = torch.tensor([0.], device=device)
    num_atoms = 0
    for video_id, context_vec in zip(video_ids,context_vecs):
        dpoint = json_data_dict[str(int(video_id.item()))]
        atoms,lcwa = dpoint['pruned_atoms_with_synsets'], dpoint['lcwa']
        try: neg_weight = float(len(atoms))/len(lcwa)
        except ZeroDivisionError: pass # Don't define neg_weight because it shouldn't be needed
        truth_values = [True]*len(atoms) + [False]*len(lcwa)
        if truth_values == []: continue
        num_atoms += len(truth_values)
        for tfatom,truth_value in zip(atoms+lcwa,truth_values):
            if len(tfatom) == 2:
                predname,subname = tfatom
                embedding = torch.cat([context_vec, ind_dict[subname]])
            elif len(tfatom) == 3:
                predname,subname,objname = tfatom
                context_embedding = torch.cat([context_vec, ind_dict[subname],ind_dict[objname]])
            else: set_trace()
            if len(items) == 3:
                obj_pos = dpoint['individuals'].index(objname)
                obj_embedding = embeddings[batch_idx,obj_pos]
            mlp = mlp_dict[predname]
            prediction = mlp[context_embedding]
            if training:
                if truth_value: loss += F.relu(-prediction+margin)
                else: loss += neg_weight*F.relu(prediction+margin)
                if math.isnan(loss.item()): set_trace()
                if math.isnan(loss.item()/num_atoms): set_trace()
            else:
                if truth_value: pos_predictions.append(prediction)
                else: loss += neg_predictions.append(prediction)
    if training: return loss if num_atoms == 0 else loss/num_atoms
    else: return pos_predictions, neg_predictions

def compute_probs_from_dataset(dl,json_data_dict,ind_dict,mlp_dict):
    pos_predictions, neg_predictions = [], []
    for d in d:
        video_tensor = d[0]
        video_ids = d[2]
        context_vecs = encoder(video_tensor)
        new_pos_predictions, new_neg_predictions = get_pred_loss(video_ids, context_vecs, json_data_dict, ind_dict, mlp_dict, training=False)
        pos_predictions += new_pos_predictions
        neg_predictions += new_neg_predictions
    return pos_predictions, neg_predictions
        
       

if __name__ == "__main__":

    with open('/data2/commons/rdf_video_captions/10d.dev.json', 'r') as f:
        json_data_dict = json.load(f)

    weight_dict = torch.load("mlp-weights.pickle")
    mlp_dict = {}
    for relation, weights in weight_dict.items():
        hidden_layer = nn.Linear(weights["hidden_weights"].shape[0], weights["hidden_bias"].shape[0])
        hidden_layer.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=False)
        hidden_layer.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=False)
        output_layer = nn.Linear(weights["output_weights"].shape[0], weights["output_bias"].shape[0])
        output_layer.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=False)
        output_layer.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=False)
        mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer, nn.Sigmoid()) 
