"""
this function prepares the data to be consumed by the DataLayer
input paramters:
    --input_json: a json file.
        The json files must have a 'video_id' key or a 'gn' key. 
        It can have other parts but those are essential
    --num_val: Number of images to assign to validation data
    --output_json: output json file
    --output_h5: output h5 file
    --frame_dir: root location in which video frames are stored
    --num_test: number of test images (to withold until very very end)

example command: 
    p data_prep.py --frame_dir data/frames --input_json data/YT_40_structured_captions.json  --output_json
    some_file_name.json --output_h5 some_file_name.h5
"""

from __future__ import print_function
import os
import json
import numpy as np
import argparse
import h5py
from scipy.misc import imread, imresize
from utils import read_json_file
import random
from skimage import img_as_float


def split_data(vids, params):
    """Split the data into val/test/train."""
    count_val = 0
    count_test = 0
    count_train = 0
    for i, v in enumerate(vids):
        v['split'] = []
        if 'i_split' in v:
            v['split'] = v['i_split']
            if v['i_split'] == 'test':
                count_test = count_test + 1
            elif v['i_split'] == 'train':
                count_train = count_train + 1
            elif v['i_split'] == 'val':
                count_val = count_val + 1
            else:
                raise ValueError("Sth wrong with split", v['i_split'] )
        else:
            if params.only_test == 1:  # if process test data
                v['split'] = 'test'
                count_test = count_test + 1
            else:                      # process training data and create train/val/test splits
                if i < params.num_val:
                    v['split'] = 'val'
                    count_val = count_val + 1
                elif i < params.num_val + params.num_test:
                    v['split'] = 'test'
                    count_test = count_test + 1
                else:
                    v['split'] = 'train'
                    count_train = count_train + 1

    if params.only_test == 1:
        print ("This dataset will be used only for TEST puprpose. %d assigned to test"% len(vids) )
    else:
        print ("%d assigned to test, %d to val, and %d to train" % (count_test, count_val, count_train) )


def get_framerate(sample_video, params):
    """Return the framerate of a video."""
    return np.shape(np.load(params.frame_dir+'/'+sample_video['video_id']+params.v_f_name_ext)['arr_0'])[0]


def resize_video(params, frame_rate, input_vid):
    """Loop over video frames to resize them."""
    num_channel = 3 # thsi one should be a parameters
    resized_vid = np.zeros((frame_rate, num_channel, params.img_size, params.img_size))

    for i in range(frame_rate):
        tmp = imresize(input_vid[i], (params.img_size, params.img_size))
        resized_vid[i] = tmp.transpose(2,0,1)

    return resized_vid


def main(params):
    dtype = 'uint32'
    dtype_vid = 'float64' # this should be a parameter
    num_channel = 3     # this should be calculated from input data

    videos = read_json_file(params.input_json)
    random.seed(123)
    random.shuffle(videos)
    split_data(videos, params)
    dict_to_json = {}
    dict_to_json['video'] = []

    train_videos = [v for v in videos if v['split'] == 'train']
    val_videos = [v for v in videos if v['split'] == 'val']
    test_videos = [v for v in videos if v['split'] == 'test']
    data_dict = {'train': train_videos, 'val': val_videos, 'test': test_videos}
    print('\nData split is:')
    for data_split, videos in data_dict.items():
        print(data_split+': ',len(videos))
    # Rename the key to fit with Daniel's code
    for data_split, videos in data_dict.items():
        for v in videos:
            v['video_id'] = v['gn']
        print(videos[0].keys())
            
        if params.local:
            # Just for testing, when running locally
            videos = [v for v in videos if v['video_id'] in ["vid1", "vid2", "vid3", "vid4"]]
            if len(videos) == 0:
                continue
          ##Since we are dealing with videos, we need to get numebr of frame per video
        frame_rate = get_framerate(videos[np.random.randint(len(videos))], params)

        N = len(videos)
        print("\nProcessing the {} data. {} videos in this split".format(data_split, N))
        out_h5_file_name = data_split + '_' + params.output_h5
        out_json_file_name = data_split + '_' + params.output_json
        if params.local:
            out_h5_file_name = 'local_' + out_h5_file_name
            out_json_file_name = 'local_' + out_json_file_name
        h5_f = h5py.File(out_h5_file_name, 'w')
        vid_data = h5_f.create_dataset("videos", (N,frame_rate, num_channel, params.img_size, params.img_size), dtype = dtype_vid)
        
        dummy_seq_lens = np.random.randint(low=2, high=9, size=(N))
        dummy_embeddings = np.ones(shape=(N,10,300))
        print(dummy_seq_lens)
        print(dummy_embeddings.shape)
        embedding_data = h5_f.create_dataset("embeddings", data=dummy_embeddings)
        len_data = h5_f.create_dataset("seq_len", data=dummy_seq_lens)
        
        for idx, vid in enumerate(videos):
            i_file = params.frame_dir+'/'+vid['video_id']+params.v_f_name_ext
            assert os.path.exists(i_file) , "The %s file is not there, something is worng" % (i_file)

            try:
                input_vid = np.load(i_file)['arr_0']
            except Exception:
                print("Loading zip file failed; defaulted to zero vector.")
                input_vid = np.zeros((frame_rate, num_channel, params.img_size, params.img_size))
                pass
            int_img = resize_video(params, frame_rate, input_vid)
            d = np.array([[1,2],[3,4]])
            float_img = img_as_float(np.ndarray.astype(int_img, dtype=np.uint8))
            vid_data[idx] = float_img
            # show some progress
            if idx % 15 == 0:
                print("%d out of %d have been processed" % (idx + 1, len(videos)))

            # add data to a dict to be dumped into a json final
            dict_to_json['video'].append(vid)

    h5_f.close()
    print("Done with H5 file")
    json.dump(dict_to_json, open(out_json_file_name, 'w'))
    print("Done with everything")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--limit', default=0, type=int, help='max number of vids to prepare')
    parser.add_argument('--local', action='store_true', default=False, help='whether running locally, if so use just small number of data points')
    parser.add_argument('--output_json', default='dummy.json', help='output json file')
    parser.add_argument('--output_h5', default='dummy.h5', help='output h5 file')
    parser.add_argument('--old_json', default='', help='old json file to extend')
    parser.add_argument('--old_h5', default='', help='old h5 file to extend')
    parser.add_argument('--frame_dir', default='data/frames', help='root location in which video frames are stored')
    parser.add_argument('--only_test', type=int, default=0, help='This is used when use this code for process only test data 0|1(means test only)' )
    parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    parser.add_argument('--num_val',  default=0, type=int, help='number of images to assign to validation data')

    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--img_size', default=224, type =int, help = 'Each frame will be resize to this size')
    parser.add_argument('--v_f_name_ext', default='_f.npz', help = 'the format and extension of saved video name, e.g. video10_f.npz')

    args = parser.parse_args()
    if args.frame_dir == 'jade':
        args.frame_dir = '/jmain01/home/JAD015/ttl03/dxv38-ttl03/videocap/data/Youtube_frames_8'
    print('\nParsed input parameters:')
    print(json.dumps(vars(args), indent = 2))
    main(args)
   

