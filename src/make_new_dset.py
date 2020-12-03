import random
from pdb import set_trace
import re
import json
import numpy as np
from nltk.corpus import wordnet
from collections import OrderedDict
from semantic_parser import tuplify

with open('msvd_linked_parsed_captions.json') as f: d=json.load(f)

#ignore_preds = ['take','do','be','have','try','go','stand','many']
ignore_preds = ['take','do','be','have']
ignore_inds = ['it','piece','thing','one','group','other']
all_atoms_as_lists  = [a for dp in d for a in dp['atoms_with_synsets']]
all_inds = [tuple(i) for dp in d for a in dp['atoms_with_synsets'] for i in a[1:]]
unique_individuals = set(all_inds)
all_classes = [tuple(a[0]) for dp in d for a in dp['atoms_with_synsets'] if len(a)==2]
all_relations = [tuple(a[0]) for dp in d for a in dp['atoms_with_synsets'] if len(a)==3]
#all_predicates = [i[0] for i in all_predicates]
unique_classes = set(all_classes)
unique_relations = set(all_relations)
class_counts = {p:all_classes.count(p) for p in unique_classes}
relation_counts = {p:all_relations.count(p) for p in unique_relations}
unique_classes = [p for p in unique_classes if class_counts[p]>50 and p[0][0] not in ignore_preds]
unique_relations = [p for p in unique_relations if relation_counts[p]>50 and p[0][0] not in ignore_preds]
counts = {p: all_inds.count(p) for p in unique_individuals}
unique_individuals = [i for i in unique_individuals if counts[i]>50 and i[0] not in ignore_inds]

unique_classes.sort(key=lambda x: class_counts[x],reverse=True)
unique_relations.sort(key=lambda x: relation_counts[x],reverse=True)
unique_individuals.sort(key=lambda x: counts[x],reverse=True)

#unique_predicates = [i[0] for i in unique_predicates]

winds = [i[0] for i in unique_individuals]
#wpreds = [p[0] for p in unique_predicates]

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
    other_options = unique_classes if len(atom)==2 else unique_relations
    return [tuple([new_pred] + list(atom[1:])) for new_pred in other_options if new_pred!=atom[0]]

def compute_lcwa(atoms):
    return [corruption for atom in atoms for corruption in corrupt_atom_pred(atom)]

lab_by_id = {k:np.zeros(len(unique_individuals),dtype=np.int32).tolist() if len(v) ==0 else sum([np.arange(len(unique_individuals)) == unique_individuals.index(ind) for ind in v]).tolist() for k,v in inds_by_id.items()}

for dp in d:
    pruned = [tuple(tuplify(a)) for a in dp['atoms_with_synsets'] if tuple(a[1]) in unique_individuals and ((len(a)==2 and tuple(a[0]) in unique_classes) or (len(a) == 3 and tuple(a[2]) in unique_individuals and tuple(a[0]) in unique_relations))]
    dp['pruned_atoms_with_synsets']=pruned
    dp['multiclass_inds'] = list(lab_by_id[dp['video_id']])
    #dp['pruned_atoms'] = [a for a in dp['atoms'] if a[0] in wpreds and (len(a) == 2 and a[1] in winds) or (len(a)==3 and a[1] in winds and a[2] in winds)]
    dp['lcwa'] = compute_lcwa(dp['pruned_atoms_with_synsets'])
    dp['video_id'] = int(dp['video_id'][3:])
    dp['inds'] = list(set([tuple(i) for a in dp['pruned_atoms_with_synsets'] for i in a[1:]]))
    #dp['preds'] = list(set([(tuple(a[0]), len(a)-1) for a in dp['atoms_with_synsets']]).intersection(set(unique_predicates)))
    dp['preds'] = list(set([tuple(a[0]) for a in pruned]))
d.sort(key=lambda x: x['video_id'])
inddict = OrderedDict({item[1]:np.arange(len(unique_individuals))==i for i,item in enumerate(unique_individuals)})
final_d = {'inds':unique_individuals,'classes':list(unique_classes),'relations':list(unique_relations),'dataset':d}
with open('msvd_pruned_filtered_linked_merged_parsed_captions.json','w') as f: json.dump(final_d,f)
with open('MSVD_final.json','w') as f: json.dump(final_d,f)
short_d = final_d
short_d['dataset'] = [dp for dp in d if dp['video_id'] <= 10]
with open('MSVD_10dp.json','w') as f: json.dump(short_d,f)
