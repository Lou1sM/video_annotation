"""Download and process MSRVTT clips using urls in csv file. Pair the captions
and video tensors by associating both with a video_id that is an int up to
10000.
"""

import cv2
import numpy as np
import os
import json
import youtube_dl
import subprocess
import skvideo.io
import math
from pdb import set_trace

ydl = youtube_dl.YoutubeDL()

def timeformat(time_in_seconds):
    minutes = time_in_seconds//60
    seconds = str(time_in_seconds - 60*minutes)
    minutes = str(int(minutes))
    if len(minutes) == 1:
        minutes = '0' + str(minutes)
    if len(seconds) == 3:
        seconds = '0' + str(seconds)
    return f"00:{minutes}:{seconds}"


def reshape_video_tensor(ar):
    framed_ar = ar[range(ar.shape[0])[::math.ceil(ar.shape[0]/8)],:,:,:] # Use only 8 evenly spaced frames
    framed_ar = np.array([cv2.resize(frame,(256,256)) for frame in framed_ar])
    return framed_ar

def download_video_segment(video_dict):
    url_info = ydl.extract_info(video_dict['url'],download=False)
    urls = [f['url'] for f in url_info['formats'] if f['vcodec'] != 'none' and f['acodec'] == 'none']
    video_filepath = os.path.join(video_dir,f'{video_dict["video_id"]}.mkv')
    args_to_run = ['ffmpeg','-loglevel','-8','-i',urls[0],'-y','-ss',timeformat(video_dict['start time']),'-to',timeformat(video_dict['end time']),video_filepath]
    for url in urls:
        args_to_run = ['ffmpeg','-i',url,'-y','-ss',timeformat(video_dict['start time']),'-to',timeformat(video_dict['end time']),video_filepath]
        subprocess.run(args_to_run)
        try:
            ar = skvideo.io.vread(video_filepath)
            return(reshape_video_tensor(ar))
        except: pass
    set_trace()
    print(f"Can't download video {video_dict['video_id'][5:]}")


video_dir = 'downloaded_videos'
if not os.path.isdir(video_dir): os.mkdir(video_dir)

with open('videodatainfo_2017.json') as f: d = json.load(f)

new_sentences_dict = {}
for sentence_dict in d['sentences']:
    vidid = sentence_dict['video_id'][5:]
    assert f'video{vidid}' == sentence_dict['video_id']
    try: new_sentences_dict[vidid].append(sentence_dict['caption'])
    except KeyError: new_sentences_dict[vidid] = [sentence_dict['caption']]

captions_and_vidids = []
for video_dict in d['videos']:
    vidid = video_dict['video_id'][5:]
    assert f'video{vidid}' == video_dict['video_id']
    framed_ar = download_video_segment(video_dict)
    captions = new_sentences_dict[vidid]
    new_dpoint = {'video_id':int(vidid),'captions':captions}
    captions_and_vidids.append(new_dpoint)

with open('MSVD_captions.json','w') as f: json.dump(captions_and_vidids,f)
