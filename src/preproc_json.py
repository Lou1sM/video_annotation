import json
import sys
import os
from pdb import set_trace

d_in = sys.argv[1]
f_out = sys.argv[2]

json_paths = [os.path.join(d_in,f) for f in os.listdir(d_in)]

data = []
for jp in json_paths:
    with open(jp) as f: newdp=json.load(f)
    newdp['video_id'] = newdp['video_id'][3:] # Cut 'vid' from start of id
    newdp['gt_embeddings'] = newdp.pop('gt_embedding') # Add 's' to field name
    data.append(newdp)

with open(f_out,'w') as f: json.dump(data,f,indent=4)

