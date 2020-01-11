import json
import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace
import re
from utils import make_prediction
import math

torch.manual_seed(0)

def get_pred_loss(video_ids, embeddings, json_data_dict, mlp_dict, margin, device):
    loss = torch.tensor([0.], device=device)
    num_atoms = 0
    for batch_idx, video_id in enumerate(video_ids):
        dpoint = json_data_dict[str(int(video_id.item()))]
        atoms,inferences,lcwa = dpoint['facts'], dpoint['inferences'], dpoint['lcwa']
        try: neg_weight = float(len(atoms))/len(lcwa)
        except ZeroDivisionError: pass # Don't define neg_weight because it shouldn't be needed
        neg_loss = torch.tensor([0.], device=device)
        truth_values = [True]*len(atoms) + [False]*len(lcwa)
        if truth_values == []: continue
        num_atoms += len(truth_values)
        for atomstr,truth_value in zip(atoms+lcwa,truth_values):
            assert atomstr.startswith('~') != truth_value
            if not truth_value: atomstr = atomstr[1:]
            items = re.split('\(|\)|,',atomstr)[:-1]
            if len(items) == 2:
                predname,subname = items
                assert predname.startswith('c') #Should be unary
            elif len(items) == 3:
                predname,subname,objname = items
                try: assert predname.startswith('r')
                except: set_trace()
            else: set_trace()
            sub_pos = dpoint['individuals'].index(subname)
            embedding = embeddings[batch_idx,sub_pos]
            if len(items) == 3:
                obj_pos = dpoint['individuals'].index(objname)
                obj_embedding = embeddings[batch_idx,obj_pos]
                embedding = torch.cat([embedding, obj_embedding])
            prediction = make_prediction(embedding,predname,len(items)==2,mlp_dict,device)
            if truth_value: loss += F.relu(-prediction+margin)
            else: loss += neg_weight*F.relu(prediction+margin)
            if math.isnan(loss.item()): set_trace()
            if math.isnan(loss.item()/num_atoms): set_trace()
    return loss if num_atoms == 0 else loss/num_atoms
       

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
