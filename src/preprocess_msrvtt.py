"""Process MSRVTT captions and save in a json, bundling together all captions
for a given video.
"""

import cv2
import numpy as np
import json
import math


with open('videodatainfo_2017.json') as f:msrvtt = json.load(f)

def reshape_video_tensor(ar):
    framed_ar = ar[range(ar.shape[0])[::math.ceil(ar.shape[0]/8)],:,:,:] # Use only 8 evenly spaced frames
    if framed_ar.shape[0] == 7:
        framed_ar = np.concatenate((framed_ar,ar[ar.shape[0]-1:ar.shape[0]]),axis=0)
    framed_ar = np.array([cv2.resize(frame,(256,256)) for frame in framed_ar])
    framed_ar = np.transpose(framed_ar,(0,3,1,2))
    assert framed_ar.shape[0] == 8
    return framed_ar

captions_by_video_id = {}

for caption_dict in msrvtt['sentences']:
    video_id = caption_dict['video_id']
    try:
        captions_by_video_id[video_id].append(caption_dict['caption'])
    except KeyError:
        captions_by_video_id[video_id] = [caption_dict['caption']]


with open('MSRVTT_captions.json', 'w') as f: json.dump(captions_by_video_id,f)
