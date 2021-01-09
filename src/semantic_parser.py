"""Convert natural language captions to logical captions. The function atom_predict()
converts a single sentence and returns the logical caption. Running this script
will read a json file of captions for MSVD and MSRVTT at <dset_name>_captions.json
and write the parsed dataset, which also contains logical captions, to a new
json file at <dset_name>_parsed_captions.json.
"""

from pdb import set_trace
import json
import stanfordnlp
from grammarbot import GrammarBotClient
from nltk.stem import WordNetLemmatizer
from copy import copy

lemmatizer = WordNetLemmatizer()
client = GrammarBotClient()


def build_atoms(sent_words, single_words_only=False):
    """Extract logical atoms from a list of syntactic dependencies."""

    atoms = []
    root = [word for word in sent_words if word.dependency_relation=='root' and word.governor == 0][0]
    root_pos = int(root.index)

    if root.upos in ['VERB', 'AUX']:
        try:
            subj1 = [word for word in sent_words if word.dependency_relation=='nsubj' and word.governor == root_pos][0]
        except IndexError:
            return []
        try:
            obj1 = [word for word in sent_words if word.dependency_relation=='obj' and word.governor == root_pos][0]
            root_atom = [root.lemma, subj1.lemma, obj1.lemma]
        except:
            root_atom = [root.lemma, subj1.lemma]
    elif root.upos in ['ADJ']:
        try:
            subj1 = [word for word in sent_words if word.dependency_relation=='nsubj' and word.governor == root_pos][0]
            copula = [word for word in sent_words if word.dependency_relation=='cop' and word.governor == root_pos][0]
            root_atom = [root.lemma, subj1.lemma]
        except IndexError:
            return []
    elif root.upos in ["NOUN", 'PROPN']:
        try:
            copula = [word for word in sent_words if word.dependency_relation=='cop' and word.governor == root_pos][0]
            prep = [word for word in sent_words if word.dependency_relation=='case' and word.upos == 'ADP' and word.governor == root_pos][0]
            subj1 = [word for word in sent_words if word.dependency_relation=='nsubj' and word.governor == root_pos][0]
            root_atom = [prep.lemma, subj1.lemma, root.lemma]
        except IndexError:
            #TODO: catch some additional cases here, this exception normally means sentence is not a proper
            #sentece, eg just a np
            #import pdb; pdb.set_trace()
            return []
    else:
        print(root.upos)
        return []

    atoms.append(root_atom)
    rdep = filter(lambda x: x not in [root, subj1, obj1], sent_words)
    for dep in sent_words:
        rel = dep.dependency_relation
        if rel == 'amod':
            modifiee = sent_words[dep.governor-1]
            atoms.append([dep.lemma, modifiee.lemma])
        if rel == 'compound':
            other_half = sent_words[dep.governor-1]
            if single_words_only:
                atoms.append([dep.lemma,other_half.lemma])
                continue
            elif int(other_half.index) == int(dep.index)+1: #Pre-nominal modifier, most common in English
                compound = dep.lemma + "_" + other_half.lemma
            elif int(other_half.index) == int(dep.index)-1: #Post-nominal modifier
                compound = other_half.lemma + "_" + dep.lemma
                print(compound)
            else:
                continue
            for i, atom in enumerate(atoms): #Update this word to add the compound at all places it appears in all atoms
                new_atom = [item if item != other_half.lemma else compound for item in atom]
                atoms[i] = new_atom
        if rel == 'cc':
            thing_being_conjuncted_on = sent_words[dep.governor-1]
            thing_its_being_conjuncted_to = sent_words[thing_being_conjuncted_on.governor-1]
            root_pos2 = int(thing_being_conjuncted_on.index)
            if thing_its_being_conjuncted_to.upos == 'VERB':
                try:
                    subj2 = [word for word in sent_words if word.dependency_relation=='nsubj' and word.governor == root_pos2][0]
                except IndexError:
                    subj2 = subj1
                try:
                    obj2 = [word for word in sent_words if word.dependency_relation=='obj' and word.governor == root_pos2][0]
                    atom2 = [thing_being_conjuncted_on.lemma, subj2.lemma, obj2.lemma]
                    atoms.append(atom2)
                except IndexError:
                    atom2 = [thing_being_conjuncted_on.lemma, subj2.lemma]
                    atoms.append(atom2)
            elif thing_its_being_conjuncted_to.upos == 'ADJ':
                try:
                    subj2 = [word for word in sent_words if word.dependency_relation=='nsubj' and word.governor == root_pos2][0]
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
                        subj2 = [word for word in sent_words if word.dependency_relation=='subj' and word.governor == root_pos2][0]
                    except IndexError:
                        subj2 = subj1
                    atom2 = [root.lemma, subj2.lemma, thing_being_conjuncted_on.lemma]
                    atoms.append(atom2)
    return atoms

def atom_predict(sentence,single_words_only):
    """Turn NL sentence into set of atoms."""
    parsed = nlp(sentence)
    sent_words = [token.words[0] for token in parsed.sentences[0].tokens]
    atoms = build_atoms(sent_words,single_words_only)
    return atoms

def tuplify(x): return [tuple(item) for item in x]
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
            except: pdb.set_trace()
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

    nlp = stanfordnlp.Pipeline()

    for dset_name in ['MSVD','MSRVTT']:
        with open(f'{dset_name}_captions.json') as f: data = json.load(f)
        logical_captions_by_vid_id = {}
        for vid_id,caption_list in data.items():
            try:
                atoms = list(set(tuplify([item for caption in caption_list for item in atom_predict(caption,False)])))
            except: set_trace()
            logical_captions_by_vid_id[vid_id] = {'atoms':atoms, 'captions':caption_list, 'video_id':vid_id}
            break
        with open(f'{dset_name}_parsed_captions.json', 'w') as f: json.dump(logical_captions_by_vid_id, f)
