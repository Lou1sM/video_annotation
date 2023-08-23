"""Convert natural language captions to logical captions. The function atom_predict()
converts a single sentence and returns the logical caption. Running this script
will read a json file of captions for MSVD and MSRVTT at <dset_name>_captions.json
and write the parsed dataset, which also contains logical captions, to a new
json file at <dset_name>_parsed_captions.json.
"""

from time import time
import json
import stanza
from utils import tuplify
from nltk.stem import WordNetLemmatizer
import os
from copy import copy
from langdetect import detect_langs
import argparse

lemmatizer = WordNetLemmatizer()
#with open('5000_words.txt') as f: COMMON_ENG_WORDS=f.read().splitlines()


class SemanticParser():
    def __init__(self,syntactic_parser,single_words_only):
        self.syntactic_parser = syntactic_parser
        self.single_words_only = single_words_only
        self.init_times()

    def init_times(self):
        self.parse_time = 0
        self.detect_langs_time = 0
        self.atoms_time = 0

    def build_atoms(self,sent_words):
        """Extract logical atoms from a list of syntactic dependencies."""

        atoms = []
        root = [word for word in sent_words if word.deprel=='root' and word.head == 0][0]
        root_pos = int(root.id)

        if root.upos in ['VERB', 'AUX']:
            try:
                subj1 = [word for word in sent_words if word.deprel=='nsubj' and word.head == root_pos][0]
            except IndexError:
                return []
            try:
                obj1 = [word for word in sent_words if word.deprel=='obj' and word.head == root_pos][0]
                root_atom = [root.lemma, subj1.lemma, obj1.lemma]
            except:
                root_atom = [root.lemma, subj1.lemma]
        elif root.upos in ['ADJ']:
            try:
                subj1 = [word for word in sent_words if word.deprel=='nsubj' and word.head == root_pos][0]
                copula = [word for word in sent_words if word.deprel=='cop' and word.head == root_pos][0]
                root_atom = [root.lemma, subj1.lemma]
            except IndexError:
                return []
        elif root.upos in ["NOUN", 'PROPN']:
            try:
                copula = [word for word in sent_words if word.deprel=='cop' and word.head == root_pos][0]
                prep = [word for word in sent_words if word.deprel=='case' and word.upos == 'ADP' and word.head == root_pos][0]
                subj1 = [word for word in sent_words if word.deprel=='nsubj' and word.head == root_pos][0]
                root_atom = [prep.lemma, subj1.lemma, root.lemma]
            except IndexError:
                #TODO: catch some additional cases here, this exception normally means sentence is not a proper
                #sentece, eg just a np
                #import pdb; pdb.breakpoint()
                return []
        else:
            print(root.upos)
            return []

        atoms.append(root_atom)
        rdep = filter(lambda x: x not in [root, subj1, obj1], sent_words)
        for dep in sent_words:
            rel = dep.deprel
            if rel == 'amod':
                modifiee = sent_words[dep.head-1]
                atoms.append([dep.lemma, modifiee.lemma])
            if rel == 'compound':
                other_half = sent_words[dep.head-1]
                if self.single_words_only:
                    atoms.append([dep.lemma,other_half.lemma])
                    continue
                elif int(other_half.id) == int(dep.id)+1: #Pre-nominal modifier, most common in English
                    compound = dep.lemma + "_" + other_half.lemma
                elif int(other_half.id) == int(dep.id)-1: #Post-nominal modifier
                    compound = other_half.lemma + "_" + dep.lemma
                    print(compound)
                else:
                    continue
                for i, atom in enumerate(atoms): #Update this word to add the compound at all places it appears in all atoms
                    new_atom = [item if item != other_half.lemma else compound for item in atom]
                    atoms[i] = new_atom
            if rel == 'cc':
                thing_being_conjuncted_on = sent_words[dep.head-1]
                thing_its_being_conjuncted_to = sent_words[thing_being_conjuncted_on.head-1]
                root_pos2 = int(thing_being_conjuncted_on.id)
                if thing_its_being_conjuncted_to.upos == 'VERB':
                    # Eg 'the boy runs and jumps'
                    try:
                        subj2 = [word for word in sent_words if word.deprel=='nsubj' and word.head == root_pos2][0]
                    except IndexError:
                        subj2 = subj1
                    try:
                        obj2 = [word for word in sent_words if word.deprel=='obj' and word.head == root_pos2][0]
                        atom2 = [thing_being_conjuncted_on.lemma, subj2.lemma, obj2.lemma]
                        atoms.append(atom2)
                    except IndexError:
                        atom2 = [thing_being_conjuncted_on.lemma, subj2.lemma]
                        atoms.append(atom2)
                elif thing_its_being_conjuncted_to.upos == 'ADJ':
                    # Eg 'the black and white cat'
                    try:
                        subj2 = [word for word in sent_words if word.deprel=='nsubj' and word.head == root_pos2][0]
                    except IndexError:
                        subj2 = subj1
                    atom2 = [thing_being_conjuncted_on.lemma, subj2.lemma]
                    atoms.append(atom2)

                elif thing_its_being_conjuncted_to == subj1:
                    replace_idx = root_atom.index(subj1.lemma)
                    atom2 = copy(root_atom)
                    atom2[replace_idx] = thing_being_conjuncted_on.lemma
                    atoms.append(atom2)
                    #if len(root_atom) == 3: # Assume object in second clause should be the same as in first
                    #    try:
                    #        atom2 = [root.lemma, thing_being_conjuncted_on.lemma, obj1.lemma]
                    #    except Exception as e:
                    #        print(e)
                    #elif len(root_atom) == 2:
                    #    atom2 = [root.lemma, thing_being_conjuncted_on.lemma]
                elif 'obj1' in globals() or 'obj1' in locals():
                    if thing_its_being_conjuncted_to == obj1:
                        try:
                            subj2 = [word for word in sent_words if word.deprel=='subj' and word.head == root_pos2][0]
                        except IndexError:
                            subj2 = subj1
                        atom2 = [root.lemma, subj2.lemma, thing_being_conjuncted_on.lemma]
                        atoms.append(atom2)
        return atoms

    def atom_predict(self,sentence):
        """Turn NL sentence into set of atoms."""
        #if not is_english(sentence):
        if not isinstance(sentence,str):
            #math.isnan(sentence):
            return []
        lang_detect_start_time = time()
        try:
            lang_probs = detect_langs(sentence)
        except: breakpoint()
        self.detect_langs_time += time()-lang_detect_start_time
        en_prob = 0
        for lang_object in lang_probs:
            if lang_object.lang == 'en':
                en_prob = lang_object.prob
        if en_prob < 0.6:
            #print('REJECT', sentence)
            return []
        #else:
            #print('ACCEPT', sentence)
        parse_start_time = time()
        parsed = self.syntactic_parser(sentence)
        self.parse_time += time()-parse_start_time
        sent_words = [token.words[0] for token in parsed.sentences[0].tokens]
        atom_start_time = time()
        atoms = self.build_atoms(sent_words)
        self.atoms_time += time()-atom_start_time
        if ARGS.very_verbose:
            print(sentence)
            print(f'detect_langs time: {self.detect_langs_time:.3f}')
            print(f'parse time: {self.parse_time:.3f}')
            print(f'atom build time: {self.atoms_time:.3f}\n')
        return atoms

    def atoms_from_caption_list(self,caption_list,verbose):
        self.init_times()
        all_atoms_as_lists = []
        for caption in caption_list:
            new_atoms = self.atom_predict(caption)
            all_atoms_as_lists += new_atoms
        all_atoms_as_tuples = tuplify(all_atoms_as_lists)
        unique_atoms_as_tuples = list(set(all_atoms_as_tuples))
        if verbose:
            print(f'detect_langs time: {self.detect_langs_time:.3f}')
            print(f'parse time: {self.parse_time:.3f}')
            print(f'atom build time: {self.atoms_time:.3f}')
        if ARGS.very_verbose:
            print('\n=====================\n')
        return unique_atoms_as_tuples

def merge_graphs(data):
    """Takes, in the form of a json file, an entire dataset where different
    sentences for the same video are listed separately. Merges such sentences
    so that all sentences for the same video appear as on list of sentences
    under a single video_id.
    """
    merged_data={}
    for d in data:
        vid_id = d['video_id']
        if vid_id in merged_data.keys():
            try: merged_data[vid_id]['atoms'] = list(set(tuplify(merged_data[vid_id]['atoms']) + tuplify(d['atoms'])))
            except: breakpoint()
            #merged_data[vid_id]['atoms_with_synsets'] = list(set([tuple(tuplify(item)) for item in merged_data[vid_id]['atoms_with_synsets']] + [tuple(tuplify(item)) for item in d['atoms_with_synsets']]))
            merged_data[vid_id]['captions'] += [d['sentence']]
        else:
            merged_data[vid_id] = d
            merged_data[vid_id]['captions'] = [d['sentence']]
            merged_data[vid_id]['atoms'] = d['atoms']

            del merged_data[vid_id]['sentence']
    return merged_data

def convert_logical_caption(caption):
    inds = set([tuple(i) for atom in caption for i in atom[1:]])
    inds_dict = {ind[1]:(ind[0], f'ind{i}') for i,ind in enumerate(list(inds))}
    news = [[tuple(atom[0])] + [inds_dict[i[1]][1] for i in atom[1:]] for atom in caption] + [[(v[0],k),v[1]] for k,v in inds_dict.items()]
    return news

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--break_after',type=int,default=-1)
    parser.add_argument('--start_at',type=int,default=0)
    parser.add_argument('--dset',type=str,choices=['MSVD','MSRVTT'])
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--infpath',type=str)
    parser.add_argument('--outfpath',type=str)
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--verbose','-v',action='store_true')
    parser.add_argument('--very_verbose','-vv',action='store_true')
    ARGS = parser.parse_args()

    assert ARGS.dset is not None or (ARGS.infpath is not None and ARGS.outfpath is not None)

    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    stanza_parser = stanza.Pipeline()

    semantic_parser = SemanticParser(stanza_parser,single_words_only=False)
    infpath = ARGS.infpath if ARGS.infpath is not None else f'{ARGS.dset}_captions.json'
    with open(infpath) as f: data = json.load(f)
    logical_captions_by_vid_id = {}
    it = 0
    for vid_id,vid_caption_list in data.items():
        #atoms = list(set(tuplify([item for caption in caption_list for item in atom_predict(caption,False)])))
        if it < ARGS.start_at:
            it += 1
            continue
        print('processing', vid_id)
        assert type(vid_caption_list) in (list,str)
        if isinstance(vid_caption_list,str):
            vid_caption_list = [vid_caption_list]
        atoms = semantic_parser.atoms_from_caption_list(vid_caption_list,verbose=ARGS.verbose)
        #all_atoms_as_lists = [item for caption in caption_list for item in semantic_parser.atom_predict(caption,False)]
        #all_atoms_as_tuples = tuplify(all_atoms_as_lists)
        #unique_atoms_as_tuples = list(set(all_atoms_as_tuples))
        logical_captions_by_vid_id[vid_id] = {'atoms':atoms, 'captions':vid_caption_list, 'video_id':vid_id}
        print(atoms)
        if ARGS.test: break
        if it==ARGS.break_after: break
        it+=1
    if ARGS.outfpath is not None:
        outfpath = ARGS.outfpath
    else:
        outfpath = f'{ARGS.dset}_jim.json' if ARGS.test else f'{ARGS.dset}_parsed_captions.json'
    with open(outfpath, 'w') as f: json.dump(logical_captions_by_vid_id, f)
