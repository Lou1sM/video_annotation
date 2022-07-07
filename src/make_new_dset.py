import json
import numpy as np
from nltk.corpus import wordnet
from collections import OrderedDict
import argparse


def isnoun(offset):
    try: assert isinstance(offset,int)
    except: print(f'{offset} should be an int')
    try: wordnet.synset_from_pos_and_offset('n',offset); return True
    except: return False

class DatasetMaker():
    def __init__(self):
        self.ignore_preds = ['take','do','be','have']
        self.ignore_inds = ['it','piece','thing','one','group','other','type']

    def get_counts_dict_by_id(self,full_list):
        names_by_id = {x[1]:x[0] for x in full_list} # this also amounts to removing nonunique ids
        all_ids = np.array([x[1] for x in full_list])
        all_ids_counts = np.bincount(all_ids)
        counts_by_id = {k:all_ids_counts[k] for k in names_by_id.keys()}
        return names_by_id, counts_by_id

    def set_uniques(self,inds_list,classes_list,relations_list):
        unique_inds_by_id_, ind_counts = self.get_counts_dict_by_id(inds_list)
        unique_classes_by_id_, class_counts = self.get_counts_dict_by_id(classes_list)
        unique_relations_by_id_, relation_counts = self.get_counts_dict_by_id(relations_list)
        self.unique_inds_by_id = {k:v for k,v in unique_inds_by_id_.items()
                                    if v not in self.ignore_inds and ind_counts[k]>50}
        self.unique_classes_by_id = {k:v for k,v in unique_classes_by_id_.items()
                                    if v not in self.ignore_preds and class_counts[k]>50}
        self.unique_relations_by_id = {k:v for k,v in unique_relations_by_id_.items()
                                    if v not in self.ignore_preds and relation_counts[k]>50}

        self.n_unique_inds = len(self.unique_inds_by_id)
        self.n_unique_classes = len(self.unique_classes_by_id)
        self.n_unique_relations = len(self.unique_relations_by_id)

    def corrupt_atom_pred(self,atom):
        other_options_by_id = self.unique_classes_by_id if len(atom) == 2 else self.unique_relations_by_id
        # select 5 corruptions for each atom
        selected_ids = np.random.choice(list(other_options_by_id.keys()),5,replace=False)
        other_options_by_id = {int(k):other_options_by_id[k] for k in selected_ids}
        corruptions = []
        for k,v in other_options_by_id.items():
            if k == atom[0][1]: continue # Take all possibilities but the true one
            new_predicate = (v,k)
            corruptions.append(tuple([new_predicate] + list(atom[1:])))
        if len(corruptions) > 100:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return corruptions

    def compute_lcwa(self,atoms):
        return [corruption for atom in atoms for corruption in self.corrupt_atom_pred(atom)]

    def prune_atoms(self,atoms):
        pruned_atoms = []
        for a in atoms:
            if a[1][1] not in self.unique_inds_by_id:
                continue
            if len(a)==2:
                if a[0][1] not in self.unique_classes_by_id:
                    continue
            if len(a) == 3:
                if a[2][1] not in self.unique_inds_by_id:
                    continue
                if a[0][1] not in self.unique_relations_by_id:
                    continue
            pruned_atoms.append(a)
        return pruned_atoms

    def make_dset(self,dset_name):
        with open(f'{dset_name}_linked_parsed_captions.json') as f: d=json.load(f)

        # Ignore semantically weak words
        all_inds = [tuple(i) for dp in d for a in dp['atoms_with_synsets'] for i in a[1:]]
        all_classes = [tuple(a[0]) for dp in d for a in dp['atoms_with_synsets'] if len(a)==2]
        all_relations = [tuple(a[0]) for dp in d for a in dp['atoms_with_synsets'] if len(a)==3]
        self.set_uniques(all_inds,all_classes,all_relations)

        for dp in d:
            atoms = dp['atoms_with_synsets']
            all_inds_in_vid = [tuple(i) for a in atoms for i in a[1:]]
            inds_in_vid = set([i for i in all_inds_in_vid if i[1] in self.unique_inds_by_id])
            dp['inds'] = list(inds_in_vid)

            if len(inds_in_vid) == 0:
                multihot_inds = np.zeros(self.n_unique_inds).tolist()
            else:
                x = list(self.unique_inds_by_id.keys())
                ohes = [np.arange(self.n_unique_inds) == x.index(ind[1]) for ind in inds_in_vid]
                #multihot_inds = sum(ohes).tolist()
                multihot_inds = np.logical_or.reduce(ohes).astype(int).tolist()
            dp['multiclass_inds'] = list(multihot_inds)

            all_classes_in_vid = [tuple(a[0]) for a in atoms if len(a)==2]
            classes_in_vid = set([i for i in all_classes_in_vid
                                    if i[1] in self.unique_classes_by_id])
            dp['classes'] = list(classes_in_vid)

            all_relations_in_vid = [tuple(a[0]) for a in atoms if len(a)==3]
            relations_in_vid = set([i for i in all_relations_in_vid
                                    if i[1] in self.unique_relations_by_id])
            dp['relations'] = list(relations_in_vid)
            dp['pruned_atoms_with_synsets'] = self.prune_atoms(atoms)
            dp['lcwa'] = self.compute_lcwa(dp['pruned_atoms_with_synsets'])
            #print(len(inds_in_vid),len(classes_in_vid),len(relations_in_vid),len(dp['lcwa']))
            dp['video_id'] = int(dp['video_id'][5:])

        print('finishing iterating')
        d.sort(key=lambda x: x['video_id'])
        final_d = {'inds':[(v,k) for k,v in self.unique_inds_by_id.items()],
                   'classes':[(v,k) for k,v in self.unique_classes_by_id.items()],
                   'relations':[(v,k) for k,v in self.unique_relations_by_id.items()],
                   'dataset':d}

        with open(f'{dset_name}_final.json','w') as f: json.dump(final_d,f)
        short_d = final_d
        short_d['dataset'] = [dp for dp in d if dp['video_id'] <= 10]
        with open(f'{dset_name}_10dp.json','w') as f: json.dump(short_d,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset',type=str,required=True,choices=['MSVD','MSRVTT'])
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--verbose','-v',action='store_true')
    ARGS = parser.parse_args()

    dset_maker = DatasetMaker()
    dset_maker.make_dset(ARGS.dset)
