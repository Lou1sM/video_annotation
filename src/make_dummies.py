import h5py
import json
import numpy as np


#vid_ids = list(range(1201, 1301))
N = 10

def r():
    rand_seq_len = np.random.randint(high=10, low=2)
    #return np.random.uniform(size=(rand_seq_len, 300)), rand_seq_len
    return np.ones(size=(rand_seq_len, 300)), rand_seq_len

def get_dummy_data_point():
    rand_seq, rand_seq_len = r()
    print(rand_seq.shape)
    padding = np.zeros(shape=(N-rand_seq_len, 300))
    rand_seq = np.concatenate([rand_seq, padding], axis=0)
    print(rand_seq.shape)
    dummy_video_id = np.random.choice(vids)
    dummy_inds = np.random.choice(inds, rand_seq_len)
    dummy_preds = np.random.choice(preds, 3)
    new = {'video_id': dummy_video_id, 'embedding_seq': rand_seq, 'embedding_len': rand_seq_len, 'fred_inds': dummy_inds, 'fred_preds': dummy_preds}
    return new


def load_vid_from_id(vid_id):
    return np.load('../data/frames/{}_f.npz'.format(vid_id))['arr_0']


def get_dummy_json_dpoint():
    rand_seq_len = np.random.randint(high=10, low=2)
    #return [[np.random.uniform() for i in range(300)] for j in range(rand_seq_len)]
    return [[1]*300 for j in range(rand_seq_len)]
    

def make_json(size, file_path):
    print('making dummy json')
    data = []
    for _ in range(size):
        dp = {
            'embeddings': get_dummy_json_dpoint(),
            #'video_id': np.random.choice(vids)
            'videoId': np.random.randint(low=1201, high=1211)
        }
        data.append(dp)

    with open(file_path, 'w') as outfile:
        json.dump(data,outfile)


def padded_emb_seq_from_lists(list_of_lists, N=10):
    unpadded_array = np.array(list_of_lists)
    #print(unpadded_array.shape)
    padding = np.zeros(shape=(N-unpadded_array.shape[0], 300))
    padded_array = np.concatenate([unpadded_array, padding], axis=0)
    #print(padded_array.shape)
    #padded_list_of_lists = list_of_lists+[0. for _ in range(N-len(l))] for l in list_of_lists]
    return padded_array


dummy_vid_ids = []
dummy_emb_seqs = []
dummy_emb_lens = []
dummy_inds = []
dummy_preds = []

#vid_table_dict = {vid_id: load_vid_from_id(vid_id) for vid_id in vids}
make_json(100, '../data/mini/val_data.json')

def convert_json_to_h5(json_file_path, out_h5_file_path):
    print('reading json')
    with open(json_file_path, 'r') as infile:
        embeddings_and_ids = json.load(infile)
    N = len(embeddings_and_ids)
    max_individuals = 10
    h5_f = h5py.File(out_h5_file_path, 'w')
    id_dataset = h5_f.create_dataset('videoId', (N,), dtype='uint32')
    emb_seq_dataset = h5_f.create_dataset('embeddings', (N,max_individuals,300), dtype=np.float64)
    seq_len_dataset = h5_f.create_dataset('embedding_len', (N,), dtype='uint8')
    print('making h5')
    for idx, dp in enumerate(embeddings_and_ids):
        new_vid_id = dp['videoId']
        print(new_vid_id)
        id_dataset[idx] = new_vid_id
        print(id_dataset[idx])
        list_of_lists = dp['embeddings']
        seq_len_dataset[idx] = len(list_of_lists)
        emb_seq_dataset[idx] = padded_emb_seq_from_lists(list_of_lists)

    
    h5_f.close()

convert_json_to_h5('../data/mini/val_data.json', '../data/mini/val_data.h5')

"""
print(len(vid_table_dict))
for k,v in vid_table_dict.items():
    print(k, v.shape)
for i in range(20):
    new = get_dummy_data_point()

    dummy_vid_ids.append(new['video_id'])
    dummy_emb_seqs.append(new['embedding_seq'])
    dummy_emb_lens.append(new['embedding_len'])
    dummy_inds.append(new['fred_inds'])
    dummy_preds.append(new['fred_preds'])

h5_f = h5py.File('test.h5', 'w')
h5_f.create_dataset('embedding_len', dtype='uint8', data=dummy_emb_lens)
#h5_f.create_dataset('embedding_seq', dtype=np.float64, data=dummy_emb_seqs)
h5_f.create_dataset('embedding_seq', dtype=np.float64, data=dummy_emb_seqs)
#h5_f.create_dataset('video_ids', dtype=str, data=dummy_vid_ids)
#h5_f.create_dataset('fred_inds', dtype=str, data=dummy_inds)
#h5_f.create_dataset('fred_preds', dtype=str, data=dummy_preds)
h5_f.close()
"""

