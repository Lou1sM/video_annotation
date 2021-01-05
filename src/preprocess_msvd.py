"""Process MSVD .avi clips into np arrays and save in .npz files. Save all
captions for a video in a single list with corresponding int id, the position
that video appears in the unarchived youtube tarball.
"""

import cv2
import numpy as np
import os
import pandas as pd
import json
import math
from pdb import set_trace


captions_by_id = pd.read_csv('MSR Video Description Corpus.csv')
true_ids = [item[:11] for item in os.listdir('YouTubeClips')] # First 11 chars are the id

def reshape_video_tensor(ar):
    framed_ar = ar[range(ar.shape[0])[::math.ceil(ar.shape[0]/8)],:,:,:] # Use only 8 evenly spaced frames
    if framed_ar.shape[0] == 7:
        framed_ar = np.concatenate((framed_ar,ar[ar.shape[0]-1:ar.shape[0]]),axis=0)
    framed_ar = np.array([cv2.resize(frame,(256,256)) for frame in framed_ar])
    framed_ar = np.transpose(framed_ar,(0,3,1,2))
    assert framed_ar.shape[0] == 8
    return framed_ar

captions_by_int_id = {}
for int_id, filename in enumerate(os.listdir('YouTubeClips')):
    v = cv2.VideoCapture(os.path.join('YouTubeClips',filename))
    frame_list = []
    num_frames = 0
    while True:
        frame_exists, frame = v.read()
        if not frame_exists:
            break
        frame_list.append(frame)
        num_frames += 1
    video_tensor = np.stack(frame_list)
    true_id = filename[:11]
    print(true_id,int_id,num_frames)
    resized_video_tensor = reshape_video_tensor(video_tensor)
    np.save(os.path.join('../data/MSVD/frames',f'vid{int_id}_resized'),resized_video_tensor)
    captions_for_this_video = captions_by_id.loc[captions_by_id['VideoID']==true_id]['Description'].tolist()
    captions_by_int_id[int_id] = captions_for_this_video

with open('MSVD_captions.json', 'w') as f: json.dump(captions_by_int_id,f)
