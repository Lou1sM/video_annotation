"""Process MSVD .avi clips into np arrays and save in .npy files. Save all
captions for a video in a single list with corresponding int id.
"""

import cv2
import numpy as np
import os
import json
import math
from pdb import set_trace


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
"""
trainval_paths = [os.path.join('TrainValVideo', filename) for filename in os.listdir('TrainValVideo')]
test_paths = [os.path.join('TestVideo', filename) for filename in os.listdir('TestVideo')]
for video_path in trainval_paths + test_paths:
    int_id = video_path.split('/')[-1].split('.')[0][5:] # Names are 'video<some-number>.mp4'
    v = cv2.VideoCapture(video_path)
    frame_list = []
    num_frames = 0
    while True:
        frame_exists, frame = v.read()
        if not frame_exists:
            break
        frame_list.append(frame)
        num_frames += 1
    video_tensor = np.stack(frame_list)
    resized_video_tensor = reshape_video_tensor(video_tensor)
    np.save(os.path.join('../data/MSRVTT/frames',f'vid{int_id}_resized'),resized_video_tensor)
    print(int_id)
    #captions_for_this_video = captions_by_id.loc[captions_by_id['VideoID']==true_id]['Description'].tolist()
    #captions_by_int_id[int_id] = captions_for_this_video

"""
for caption_dict in msrvtt['sentences']:
    video_id = caption_dict['video_id']
    try:
        captions_by_video_id[video_id].append(caption_dict['caption'])
    except KeyError:
        captions_by_video_id[video_id] = [caption_dict['caption']]


with open('MSRVTT_captions.json', 'w') as f: json.dump(captions_by_video_id,f)
