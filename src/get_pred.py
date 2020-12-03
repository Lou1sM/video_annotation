import torch
import torch.nn.functional as F
from pdb import set_trace

torch.manual_seed(0)

def get_pred_loss(video_ids, encodings, dataset_dict, testing, margin=1, device='cuda'):
    loss = torch.tensor([0.], device=device)
    num_atoms = 0
    if testing: pos_predictions, neg_predictions = [],[]
    json_data_dict,ind_dict,pred_mlps = dataset_dict['dataset'],dataset_dict['ind_dict'],dataset_dict['mlp_dict']
    for video_id, encoding in zip(video_ids,encodings):
        dpoint = json_data_dict[video_id.item()]
        atoms = dpoint['pruned_atoms_with_synsets']
        lcwa = dpoint['lcwa'][:len(atoms)]
        try: neg_weight = float(len(atoms))/len(lcwa)
        except ZeroDivisionError: pass # Don't define neg_weight because it shouldn't be needed
        truth_values = [True]*len(atoms) + [False]*len(lcwa)
        if truth_values == []: continue
        num_atoms += len(truth_values)
        # Use GT inds to test predictions for GT atoms
        for tfatom,truth_value in zip(atoms+lcwa,truth_values):
            arity = len(tfatom)-1
            if arity == 1:
                # Unary predicate
                predname,subname = tfatom
                context_embedding = torch.cat([encoding, ind_dict[subname]])
            elif arity == 2:
                # Binary predicate
                predname,subname,objname = tfatom
                context_embedding = torch.cat([encoding, ind_dict[subname],ind_dict[objname]])
            else: set_trace()
            mlp = pred_mlps['classes' if arity==1 else 'relations'][predname]
            prediction = mlp(context_embedding)
            if testing:
                if truth_value: pos_predictions.append(prediction.item())
                else: neg_predictions.append(prediction.item())
            else:
                if truth_value: loss += F.relu(-prediction+margin)
                else: loss += neg_weight*F.relu(prediction+margin)
    if testing:return pos_predictions, neg_predictions
    else: return loss if num_atoms == 0 else loss/num_atoms

def compute_probs_for_dataset(dl,encoder,multiclassifier,dataset_dict,use_i3d):
    pos_classifications, neg_classifications, pos_predictions, neg_predictions, perfects = [],[],[],[],{}
    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        multiclass_inds = d[1].to('cuda')
        video_ids = d[2].to('cuda')
        i3d = d[3].float().to('cuda')
        encoding, enc_hidden = encoder(video_tensor)
        if use_i3d: encoding = torch.cat([encoding,i3d],dim=-1)
        multiclassif = multiclassifier(encoding)
        assert (multiclass_inds==1).sum() + (multiclass_inds==0).sum() == multiclass_inds.shape[1]
        multiclass_inds = multiclass_inds.type(torch.bool)
        new_pos_classifications,new_neg_classifications = multiclassif[multiclass_inds], multiclassif[~multiclass_inds]
        new_pos_predictions, new_neg_predictions = get_pred_loss(video_ids, encoding, dataset_dict, testing=True)
        if (new_pos_classifications>0).all() and (new_neg_classifications<0).all() and all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[int(video_ids[0].item())] = len(new_pos_predictions)
        #if all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[video_ids[0].item()] = len(new_pos_predictions)
        pos_predictions += new_pos_predictions
        neg_predictions += new_neg_predictions
        pos_classifications += new_pos_classifications.tolist()
        neg_classifications += new_neg_classifications.tolist()
    return pos_classifications, neg_classifications, pos_predictions, neg_predictions, perfects
