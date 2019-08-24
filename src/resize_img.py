import sys
import numpy as np
import matplotlib.pyplot as plt
import string
import argparse
import h5py
#from scipy.misc import imread, imresize
from skimage.transform import resize
import skvideo.io


def resize_video(infile, outfile, frame_rate):
    num_channel = 3 # thsi one should be a parameter
    resized_vid = np.zeros((frame_rate, num_channel, 224, 224))
    #input_vid = np.load(infile)['arr_0']
    input_vid = skvideo.io.vread(infile)
    num_frames = input_vid.shape[0]
    sampling_freq = int(num_frames/frame_rate)
    #print('freq', sampling_freq, 'num_frames', num_frames)

    for i in range(0, num_frames, sampling_freq):
    #for i in range(frame_rate):
        frame = input_vid[i]
        #plt.imshow(frame)
        #plt.show()
        #tmp = imresize(frame, (new_size, new_size))
        tmp = resize(frame, (256, 256))
        tmp = tmp[16:240, 16:240]
        tmp = np.rot90(tmp,3)
        #tmp = (tmp - [[[0.485]], [[0.456]], [[0.406]]]) / ([[[0.229]], [[0.224]], [[0.225]]])
        tmp = (tmp - [[[0.485, 0.456, 0.406]]]) / ([[[0.229, 0.224, 0.225]]])
        resized_idx = i/sampling_freq
        assert((resized_idx).is_integer()), '{} should be an integer'.format(i/sampling_freq)
        if resized_idx == 8: # This happens when frame_rate doesn't divide num_frames
            break
        resized_vid[int(i/sampling_freq)] = tmp.transpose(2,0,1)
        #plt.imshow(tmp)
        #plt.show()

    np.savez(outfile, resized_vid)
    return resized_vid



#for i in range(1, 1971):
#for i in range(1):
for i in range(0,10000):
    print(i)
    infile = "/data1/louis/frames-raw/msrvtt/train-video/video{}.mp4".format(i)
    outfile = "/data1/louis/frames-resized/MSRVTT/vid{}.npz".format(i)
    #outfile = "vid{}.npz".format(i)
    #outfile = "../data/frames_331_nasnet/vid{}.npz".format(i+1)
    #resize_video(infile, outfile, 331, 8)
    resize_video(infile, outfile, 8)
