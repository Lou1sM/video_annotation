import torch
import numpy as np
from utils import acc_f1_from_binary_confusion_mat
import torch.nn.functional as F
from pdb import set_trace
from dl_utils.label_funcs import unique_labels

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
                context_embedding = torch.cat([encoding, ind_dict[subname[1]]])
            elif arity == 2:
                # Binary predicate
                predname,subname,objname = tfatom
                context_embedding = torch.cat([encoding, ind_dict[subname[1]],ind_dict[objname[1]]])
            else: set_trace()
            mlp = pred_mlps['classes' if arity==1 else 'relations'][predname[1]]
            prediction = mlp(context_embedding)
            if testing:
                if truth_value: pos_predictions.append(prediction.item())
                else: neg_predictions.append(prediction.item())
            else:
                if truth_value: loss += F.relu(-prediction+margin)
                else: loss += neg_weight*F.relu(prediction+margin)
    if testing:return pos_predictions, neg_predictions
    else: return loss if num_atoms == 0 else loss/num_atoms

def convert_atoms_to_ids_only(atoms_list):
    return [tuple([item[1] for item in atom]) for atom in atoms_list]

def inference(video_ids, encodings, ind_multiclassifications, dataset_dict):
    json_data_dict = dataset_dict['dataset']
    ind_dict = dataset_dict['ind_dict']
    pred_mlps  = dataset_dict['mlp_dict']
    annotations_by_id = {}
    for video_id, encoding, vid_ind_c in zip(video_ids,encodings,ind_multiclassifications):
        print('evaluating', video_id.item())
        annotation = []
        inds_to_use = [list(ind_dict.items())[i] for i,c in enumerate(vid_ind_c) if c>0.5]
        for subj_id, subj_vector in inds_to_use:
            for class_id, class_mlp in pred_mlps['classes'].items():
                context_embedding = torch.cat([encoding, subj_vector])
                if class_mlp(context_embedding) > 0.5:
                    annotation.append((class_id,subj_id))
            for obj_id, obj_vector in inds_to_use:
                if obj_id==subj_id: continue # Assume no reflexive predicates
                for rel_id, rel_mlp in pred_mlps['relations'].items():
                    context_embedding = torch.cat([encoding, subj_vector, obj_vector])
                    if rel_mlp(context_embedding) > 0.5:
                        annotation.append((class_id,subj_id,obj_id))

        # Compare to GT and compute scores
        dpoint = json_data_dict[video_id.item()]
        atoms = dpoint['pruned_atoms_with_synsets']
        if len(atoms)==0:
            acc = -1
            f1 = -1
        else:
            lcwa = dpoint['lcwa']
            atoms = convert_atoms_to_ids_only(atoms)
            lcwa = convert_atoms_to_ids_only(lcwa)
            tp = len([a for a in atoms if a in annotation])
            fp = len([a for a in lcwa if a in annotation])
            fn = len([a for a in atoms if a not in annotation])
            tn = len([a for a in lcwa if a not in annotation])
            acc,f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)

        annotations_by_id[video_id] = {'annotation': annotation,
                                       'acc': acc,
                                       'f1': f1}

    return annotations_by_id


def compute_probs_for_dataset(dl,encoder,multiclassifier,dataset_dict,use_i3d):
    pos_classifications, neg_classifications, pos_predictions, neg_predictions, perfects = [],[],[],[],{}
    all_accs = []
    all_f1s = []
    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        multiclass_inds = d[1].to('cuda')
        video_ids = d[2].to('cuda')
        i3d = d[3].float().to('cuda')
        encodings, enc_hidden = encoder(video_tensor)
        if use_i3d: encodings = torch.cat([encodings,i3d],dim=-1)
        multiclassif = multiclassifier(encodings)
        #assert (multiclass_inds==1).sum() + (multiclass_inds==0).sum() == multiclass_inds.shape[1]
        assert unique_labels(multiclass_inds).issubset(set([0,1]))
        multiclass_inds = multiclass_inds.type(torch.bool)
        new_pos_classifications,new_neg_classifications = multiclassif[multiclass_inds], multiclassif[~multiclass_inds]
        new_pos_predictions, new_neg_predictions = get_pred_loss(video_ids, encodings, dataset_dict, testing=True)
        annotations_by_id = inference(video_ids, encodings, multiclassif, dataset_dict)
        all_accs += [vid['acc'] for vid in annotations_by_id.values()]
        all_f1s += [vid['f1'] for vid in annotations_by_id.values()]
        if (new_pos_classifications>0).all() and (new_neg_classifications<0).all() and all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[int(video_ids[0].item())] = len(new_pos_predictions)
        #if all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[video_ids[0].item()] = len(new_pos_predictions)
        pos_predictions += new_pos_predictions
        neg_predictions += new_neg_predictions
        pos_classifications += new_pos_classifications.tolist()
        neg_classifications += new_neg_classifications.tolist()
    total_acc = np.array([x for x in all_accs if x!=-1]).mean()
    total_f1 = np.array([x for x in all_f1s if x!=-1]).mean()
    return pos_classifications, neg_classifications, pos_predictions, neg_predictions, perfects, total_acc, total_f1
