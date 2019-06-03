import sys
import numpy as np
import matplotlib.pyplot as plt
import string
import argparse
import h5py
from scipy.misc import imread, imresize


def resize_video(infile, outfile, new_size, frame_rate):
    num_channel = 3 # thsi one should be a parameter
    resized_vid = np.zeros((frame_rate, num_channel, new_size, new_size))
    input_vid = np.load(infile)['arr_0']

    for i in range(frame_rate):
        frame = input_vid[i]
        #plt.imshow(frame)
        #plt.show()
        tmp = imresize(frame, (new_size, new_size))
        resized_vid[i] = tmp.transpose(2,0,1)
        #plt.imshow(tmp)
        #plt.show()

    np.savez(outfile, resized_vid)
    return resized_vid



#for i in range(1, 1971):
for i in range(0):
    print(i)
    infile = "../data/frames/vid{}.npz".format(i+1)
    outfile = "../data/frames_331_nasnet/vid{}.npz".format(i+1)
    resize_video(infile, outfile, 331, 8)
