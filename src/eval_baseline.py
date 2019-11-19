import json
from pdb import set_trace

with open('/home/louis/msvd_baseline_linked_parsed_captions.json') as f: d = json.load(f)
with  open('/data1/louis/data/rdf_video_captions/MSVD-wordnet-25d.json') as f: gt = json.load(f)

gt = {i['video_id']: i for i in gt}
for dp in d:
    vidid = int(dp['video_id'][3:])
    gtdp = gt[vidid]
    set_trace()
    
