import subprocess
import json
import sys
import os
import numpy as np

exp_name = sys.argv[1]
jsonfilename = '../experiments/{}/10d-det-val_{}_outputs.txt'.format(exp_name, exp_name)
gtfilename = '/data2/commons/rdf_video_captions/10d-det.json'
assert os.path.isfile(jsonfilename)

with open(jsonfilename, 'r') as jsonfile:
    jsondata = json.load(jsonfile)

def generate_random_noise(size, offset):
    a = np.random.normal(size=size)
    return a*offset/np.linalg.norm(a)

print(generate_random_noise(5, 0.4))

print(jsondata[0]['embeddings'][0])
for dp in jsondata:
    #dp['embeddings'] = [gt['embeddings'] for gt in gtdata if int(gt['videoId'])==int(dp['videoId'])][0]
    #print(dp['embeddings'])
    len_before = len(dp['embeddings'])
    norm_before = np.linalg.norm(np.array(dp['gt_embeddings'][0]))
    dp['embeddings'] = [list(emb+generate_random_noise(10, 0.46)) for emb in dp['gt_embeddings']]
    assert len(dp['embeddings']) == len_before
    assert np.linalg.norm(np.array(dp['gt_embeddings'][0])) <= norm_before + 0.4
    assert np.linalg.norm(np.array(dp['gt_embeddings'][0])) >= norm_before - 0.4
    #print(dp['embeddings'])
    #print([gt['embeddings'] for gt in gtdata if int(gt['videoId'])==int(dp['videoId'])][0])

with open('10d_noise0.4.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


with open(gtfilename, 'r') as gtfile:
    gtdata = json.load(gtfile)

print(jsondata[0]['embeddings'][0])

