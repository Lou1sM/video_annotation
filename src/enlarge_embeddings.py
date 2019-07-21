import numpy as np
import subprocess
import json
import sys
import os

exp_name = sys.argv[1]
jsonfilename = '../experiments/{}/10d-det-val_{}_outputs.txt'.format(exp_name, exp_name)
assert os.path.isfile(jsonfilename)
with open(jsonfilename, 'r') as jsonfile:
    jsondata = json.load(jsonfile)


#print('\nAccuracy for original:')
#a = subprocess.Popen(['vc-eval', jsonfilename], stdout=subprocess.PIPE)
#print(a.communicate()[0].decode())


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[.25*emb_component for emb_component in emb] for emb in dp['embeddings']]

with open('exp{}_0.25x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[(0.5/0.25)*emb_component for emb_component in emb] for emb in dp['embeddings']]

with open('exp{}_0.5x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[(1.5/0.5)*emb_component for emb_component in emb] for emb in dp['embeddings']]

with open('exp{}_1.5x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


#print('\nAccuracy for 1.5x:')
#a = subprocess.Popen(['vc-eval {}'.format(jsonfilename)], stdout=subprocess.PIPE)
#print(a.communicate()[0].decode())


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[(2/1.5)*emb_component for emb_component in emb] for emb in dp['embeddings']]


with open('exp{}_2x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[(5/2)*emb_component for emb_component in emb] for emb in dp['embeddings']]

with open('exp{}_5x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)


print(np.linalg.norm(jsondata[0]['embeddings'][0]))
for dp in jsondata:
    dp['embeddings'] = [[(10/5)*emb_component for emb_component in emb] for emb in dp['embeddings']]

with open('exp{}_10x.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)

print(np.linalg.norm(jsondata[0]['embeddings'][0]))
print(np.linalg.norm(jsondata[0]['embeddings'][0]/np.linalg.norm(jsondata[0]['embeddings'][0])))

for dp in jsondata:
    
    dp['embeddings'] = list([list(emb/np.linalg.norm(np.array(emb)))for emb in dp['embeddings']])

with open('exp{}_normalized.txt'.format(exp_name), 'w') as outfile:
    json.dump(jsondata, outfile)

print(np.linalg.norm(jsondata[0]['embeddings'][0]))


