from pdb import set_trace
import numpy as np
from time import time
import pdb
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import nltk
from scipy.special import softmax
from numpy import dot, array
from gensim import matutils
from nltk.stem import WordNetLemmatizer

def normalize_vec(x): return x/np.linalg.norm(x)

class WN_Linker():
    def __init__(self,w2v,stopwords=None,lemmatizer=None):
        if stopwords == None: stopwords = nltk.corpus.stopwords.words('english')
        self.w2v,self.stopwords,self.unfound_words,self.unfound_defs,self.no_synsets = w2v,stopwords,[],[],[]
        #Define map from Penn Treebank tags to WN tags
        from nltk.tag.perceptron import PerceptronTagger
        self.tagger = PerceptronTagger()
        pt_pss = self.tagger.classes
        self.pos_map = {k:'a' if k.startswith('J') else k[0].lower() if k[0] in ['N','V','R'] else None for k in pt_pss}
        self.pos_map[None] = None
        self.lemmatizer = lemmatizer if lemmatizer else WordNetLemmatizer()

    def convert_to_uni_tag(self,token): return '_'.join([token[0],nltk.map_tag('en-ptb','universal',token[1])])
    def tokenize(self, s): return [w for w in nltk.word_tokenize(s) if w not in self.stopwords]
    def tokenize_and_tag(self,sentence): return [self.convert_to_uni_tag(token) for token in nltk.pos_tag(nltk.word_tokenize(sentence)) if token[0] not in self.stopwords]
  
    def wv_similarity(self,w1,w2): return self.w2v.similarity(w1,w2)
    def wv_n_similarity(self,s1,s2): 
        if not(len(self.tokenize_and_tag(s1)) and len(self.tokenize_and_tag(s2))): return 0.
        try: return self.w2v.n_similarity(self.tokenize_and_tag(s1),self.tokenize_and_tag(s2)) 
        except KeyError:
            s1vecs = []
            s2vecs = []
            for w in self.tokenize_and_tag(s1):
                try: s1vecs.append(self.w2v[w])
                except KeyError: self.unfound_words.append(w)
            for w in self.tokenize_and_tag(s2):
                try: s2vecs.append(self.w2v[w])
                except KeyError: self.unfound_words.append(w)
            if not(len(s1vecs) and len(s2vecs)): self.unfound_defs.append(s2); return 0.
            return dot(matutils.unitvec(array(s1vecs).mean(axis=0)),matutils.unitvec(array(s2vecs).mean(axis=0)))

    def get_vecs_for_BOW(self,bag):
        try: return [self.w2v[word] for word in self.tokenize_and_tag(bag)]
        except KeyError:
            vecs = []
            for w in self.tokenize_and_tag(bag):
                try: vecs.append(self.w2v[w])
                except KeyError: self.unfound_words.append(w)
            if vecs == []: return [np.zeros(300)]
            return vecs

    def compute_similarity(self,word_context,synset): 
        return self.wv_n_similarity(word_context,synset.definition())

    def link_word_to_wn(self,word,context,pos=None,context_as_vec=False):
        orig_synsets = wordnet.synsets(word)
        synsets = [ss for ss in orig_synsets if ss.pos()==pos] if pos else orig_synsets
        if len(synsets) == 0: synsets = orig_synsets
        #sims = softmax([self.compute_similarity(context,ss) for ss in synsets])
        if len(synsets) == 0: 
            #set_trace()
            #print('no synsets for',word)
            self.no_synsets.append(word)
            return None
        if len(synsets) == 1: return synsets[0]
        synsets = synsets[:5] # Ignore rare senses in highly polysemous words
        if context_as_vec:
            synsets_vecs = [self.get_vecs_for_BOW(ss.definition()) for ss in synsets]
            synsets_vecs = [array(vecs).mean(0) for vecs in synsets_vecs if vecs]
            synsets_vecs = [normalize_vec(vec) for vec in synsets_vecs]
            sims = np.matmul(array(synsets_vecs),context)
        else:
            sims = [self.compute_similarity(context,ss) for ss in synsets]
        _, sense = max(zip(sims,synsets))
        return sense

    def lemmatize(self,w): return self.lemmatizer.lemmatize(w)

    def get_synsets_of_rule_parse(self,dp,use_offset=True,convert=False):
        context = ' '.join(dp['captions'])
        tokens = self.tokenize(context)
        vecs = self.get_vecs_for_BOW(context)
        context_vec = normalize_vec(array(vecs).mean(axis=0))
        context_vec = matutils.unitvec(context_vec)
        pos_tagged_context_dict = {self.lemmatize(k):v for k,v in self.tagger.tag(tokens)} # <token>: <pos>
        new_atoms = []
        unique_entities = set([x for atom in dp['atoms'] for x in atom])
        entity_id_dict = {}
        for entity in unique_entities:
            try:pt_pos = pos_tagged_context_dict[entity]
            except KeyError: pt_pos=self.tagger.tag([entity])[0][-1]
            pos=self.pos_map[pt_pos]
            synset = self.link_word_to_wn(entity,context_vec,context_as_vec=True,pos=pos)
            #synset = self.link_word_to_wn(entity,context,pos=pos)
            if synset is None: offset = None
            else: offset = synset.offset()
            if use_offset: entity_id_dict[entity] = offset
            else: entity_id_dict[entity] = synset
        for atom in dp['atoms']:
            new_atom = []
            for entity in atom: new_atom.append((entity,entity_id_dict[entity]))
            new_atoms.append(new_atom)

        return convert_logical_caption(new_atoms) if convert else new_atoms

def convert_logical_caption(caption):
    inds = set([tuple(i) for atom in caption for i in atom[1:]])
    inds_dict = {ind[1]:(ind[0], f'ind{i}') for i,ind in enumerate(list(inds))}
    news = [[tuple(atom[0])] + [inds_dict[i[1]][1] for i in atom[1:]] for atom in caption] + [[(v[0],k),v[1]] for k,v in inds_dict.items()]

    return news


if __name__ == "__main__":
      
    import json
    import sys
    dset = sys.argv[1]
    w = KeyedVectors.load_word2vec_format('/home/louis/model.bin',binary=True,limit=20000)
    print('model loaded')
    stopwords = nltk.corpus.stopwords.words('english')
    linker = WN_Linker(w,stopwords)
    json_fn = f'../nlp/{dset}_merged_parsed_captions.json'
    with open(json_fn) as f:
        d=json.load(f)
    new_dps = []
    for vidid,dp in d.items():
    #for dp in d:
        #atoms_with_synsets = linker.get_synsets_of_rule_parse(dp,convert=True)
        atoms_with_synsets = linker.get_synsets_of_rule_parse(dp,convert=False)
        new_dp = dict(dp, **{'atoms_with_synsets': [atom for atom in atoms_with_synsets if not any([i[1] == None for i in atom])]})
        new_dps.append(new_dp)

    with open(f'{dset}_linked_parsed_captions.json','w') as f:
        json.dump(new_dps,f)

    print(linker.unfound_defs)
    print(linker.no_synsets)
