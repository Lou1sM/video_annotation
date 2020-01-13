import random
from pdb import set_trace
import re
import json
import numpy as np
from nltk.corpus import wordnet
from collections import OrderedDict
from semantic_parser import tuplify

#with open('/home/louis/video_annotation/nlp/msvd_filtered_linked_merged_parsed_captions.json') as f: d=json.load(f)
with open('msvd_linked_parsed_captions.json') as f: d=json.load(f)

ignore_preds = ['take','do','be','have']
ignore_inds = ['it','piece','thing','one','group','other']
all_atoms_as_lists  = [a for dp in d for a in dp['atoms_with_synsets']]
all_inds = [tuple(i) for dp in d for a in dp['atoms_with_synsets'] for i in a[1:]]
unique_individuals = set(all_inds)
all_predicates = [tuple(a[0]) for dp in d for a in dp['atoms_with_synsets']]
unique_predicates = set(all_predicates)
pred_counts = {p:all_predicates.count(p) for p in unique_predicates}
unique_predicates = [p for p in unique_predicates if pred_counts[p]>50 and p[0] not in ignore_preds]
counts = {p: all_inds.count(p) for p in unique_individuals}
unique_individuals = [i for i in unique_individuals if counts[i]>50 and i[0] not in ignore_inds]
winds = [i[0] for i in unique_individuals]
wpreds = [p[0] for p in unique_predicates]

inddict = OrderedDict({item[1]:np.arange(len(unique_individuals))==i for i,item in enumerate(unique_individuals)})

def isnoun(offset):
    try: assert isinstance(offset,int)
    except: print(f'{offset} should be an int')
    try: wordnet.synset_from_pos_and_offset('n',offset); return True
    except: return False

#inds_by_id = {dp['video_id']: set([tuple(a[0]) for a in dp['atoms_with_synsets'] if len(a) == 2 and tuple(a[0]) in unique_individuals and isnoun(a[0][1])]) for dp in d }
inds_by_id = {dp['video_id']: set([tuple(i) for a in dp['atoms_with_synsets'] for i in a[1:] if tuple(i) in unique_individuals]) for dp in d }

def corrupt_atom_at_idx(atom,idx):
    new_ind = random.choice([ind for ind in unique_individuals if ind!=atom[idx]])
    return [new_ind if i == idx else o for i,o in enumerate(atom)]
    
def corrupt_atom(atom):
    return [corrupt_atom_at_idx(atom,idx) for idx in range(1,len(atom))]

def corrupt_atom_pred(atom):
    new_pred = random.choice([pred for pred in unique_predicates if pred!=atom[0]])
    return [tuple([new_pred] + list(atom[1:])) for new_pred in unique_predicates if new_pred!=atom[0]]
    
def compute_lcwa(atoms):
    return [corruption for atom in atoms for corruption in corrupt_atom_pred(atom)]
        
lab_by_id = {k: [unique_individuals.index(ind) for ind in v] for k,v in inds_by_id.items()}
for dp in d:
    dp['pruned_atoms_with_synsets'] = [tuple(tuplify(a)) for a in dp['atoms_with_synsets'] if tuple(a[0]) in unique_predicates and tuple(a[1]) in unique_individuals and (len(a) == 2 or tuple(a[2]) in unique_individuals)]
    dp['multiclass_inds'] = list(lab_by_id[dp['video_id']])
    dp['pruned_atoms'] = [a for a in dp['atoms'] if a[0] in wpreds and (len(a) == 2 and a[1] in winds) or (len(a)==3 and a[1] in winds and a[2] in winds)]
    dp['lcwa'] = compute_lcwa(dp['pruned_atoms_with_synsets'])
    dp['video_id'] = int(dp['video_id'][3:])
    dp['inds'] = list(set([tuple(i) for a in dp['pruned_atoms_with_synsets'] for i in a[1:]]))
    #dp['atoms_by_ind'] = [{ind[0] + " " + str(ind[1]):(atom.index(ind),[i for i in atom if i!=ind]) for atom in dp['pruned_atoms_with_synsets']  if ind in atom} for ind in dp['inds']]
    #dp['preds'] = list(set([tuple(a[0]) for a in dp['atoms_with_synsets'] if tuple(a[0]) in unique_predicates and not isnoun(a[0][1])]))
    dp['preds'] = list(set([(tuple(a[0]), len(a)-1) for a in dp['atoms_with_synsets'] if tuple(a[0]) in unique_predicates]))
d.sort(key=lambda x: x['video_id'])
final_d = {'inds':unique_individuals,'preds':unique_predicates,'dataset':d}
with open('msvd_pruned_filtered_linked_merged_parsed_captions.json','w') as f: json.dump(final_d,f)
short_d = final_d
short_d['dataset'] = [dp for dp in d if dp['video_id'] <= 10]
with open('10dp.json','w') as f: json.dump(short_d,f)
