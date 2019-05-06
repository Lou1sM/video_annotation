"""Functions for creating a h5 file from a json in the agree-upon format.
"""


import json
import numpy as np
import h5py


def padded_emb_seq_from_lists(list_of_lists, already_sorted=True, embedding_size=300, max_len=10):
    """Convert a list of lists to a sorted, padded np array.

    Json files can't store np arrays, so the agreed-upon json format stores
    embeddings as a list of lists of floats. If the input is not already sorted
    then this function sorts the outer list on the first element of the innner
    list, and zero pads up to the max number of individuals.

    Args:
      lists_of_lists: the embedddings
      already_sorted: bool indicating whether the input list of embeddings is
              already sorted by first element, if false then the current func-
              will perform sorting, default is True
      embedding_size: the dimension of each embeddiing
      max_len: the length to pad each sequence to
    """

    unsorted_unpadded_array = np.array(list_of_lists)
    if already_sorted:
        sorted_padded_array = unsorted_padded_array
    else:
        sorted_unpadded_array = unsorted_unpadded_array[np.argsort(unsorted_unpadded_array[:,0])]
    padding = np.zeros(shape=(max_len-sorted_unpadded_array.shape[0], embedding_size))
    sorted_padded_array = np.concatenate([sorted_unpadded_array, padding], axis=0)
    return sorted_padded_array


def make_eos_gt(num_embeddings, max_len=10):
    """Make the ground truth for the eos network.

    The eos predicts, at each output time step, whether the embedding sequence has
    ended (ie we are now in the padding) or not. A 0 means the sequence hasn't ended
    and a 1 means it has. 

    Args:
      num_embeddings: an int equal to the sequence length (before padding)
      max_len: an int equal to the maximum number of embeddings, ie what each seq-
              uence is padded up to
    """

    return np.concatenate([np.zeros(num_embeddings), np.ones(max_len-num_embeddings)])


def convert_json_to_h5s(json_file_path, out_h5_train_file_path, out_h5_val_file_path, out_h5_test_file_path, embedding_size=300, max_len=10):
    """Convert json in agreed-upon format to train, test and val h5 in format expected by the Dataset.

    The json contains a list of datapoint, each having keys 'videoId', 'embed-
    dings', 'individuals' and 'captions'. Only the first two are passed to the
    h5 file. The videoIds are ints in the json file and are passed as is to the
    h5 file. The embeddings are unpadded, unsorted lists of lists in the json
    file. They are converted to np arrays, padded and sorted (by first element)
    before being passed to the h5 file. Their unpadded length is also measured
    and passed to the h5 file as 'embedding_len'. 

    Each h5 file thus contains 3 datasets: videoId (ints), embeddings (np arrays)
    and embedding_lens (ints).

    Args:
      json_file_path: path of the json file to read from
      out_h5_train_file_path: path of the h5 file to write train data to
      out_h5_val_file_path: path of the h5 file to write val data to
      out_h5_test_file_path: path of the h5 file to write test data to
    """

    print('Reading json...')
    with open(json_file_path, 'r') as infile:
        embeddings_and_ids = json.load(infile)
    N = len(embeddings_and_ids)

    # First determine number of data points in each split
    num_test = 0
    num_val = 0
    num_train = 0
    for dp in embeddings_and_ids:
        if dp['videoId'] <= 1200:
            num_train += 1
        elif dp['videoId'] <= 1300:
            num_val += 1
        elif dp['videoId'] <= 1970:
            num_test += 1
        else:
            print("Unrecognised video id: {}".format(dp['videoId']))

    # Define the 3 data files and the datasets therein
    h5_f_train = h5py.File(out_h5_train_file_path, 'w')
    id_train_dataset = h5_f_train.create_dataset('videoId', (num_train,), dtype='uint32')
    emb_seq_train_dataset = h5_f_train.create_dataset('embeddings', (num_train,max_len,embedding_size), dtype=np.float64)
    seq_len_train_dataset = h5_f_train.create_dataset('embedding_len', (num_train,), dtype='uint8')
    eos_gt_train_dataset = h5_f_train.create_dataset('eos_gt', (num_train,max_len), dtype='uint8')
   
    h5_f_val = h5py.File(out_h5_val_file_path, 'w')
    id_val_dataset = h5_f_val .create_dataset('videoId', (num_val,), dtype='uint32')
    emb_seq_val_dataset = h5_f_val.create_dataset('embeddings', (num_val,max_len,embedding_size), dtype=np.float64)
    seq_len_val_dataset = h5_f_val.create_dataset('embedding_len', (num_val,), dtype='uint8')
    eos_gt_val_dataset = h5_f_val.create_dataset('eos_gt', (num_val,max_len), dtype='uint8')
    
    h5_f_test = h5py.File(out_h5_test_file_path, 'w')
    id_test_dataset = h5_f_test.create_dataset('videoId', (num_test,), dtype='uint32')
    emb_seq_test_dataset = h5_f_test.create_dataset('embeddings', (num_test,max_len,embedding_size), dtype=np.float64)
    seq_len_test_dataset = h5_f_test.create_dataset('embedding_len', (num_test,), dtype='uint8')
    eos_gt_test_dataset = h5_f_test.create_dataset('eos_gt', (num_test,max_len), dtype='uint8')


    
    # Now transfer the data from json to the h5s, with relevant intermediary processing
    idx_train = 0
    idx_val = 0
    idx_test = 0
    print('Making h5...')
    for idx, dp in enumerate(embeddings_and_ids):
        new_vid_id = dp['videoId']
        if new_vid_id <= 1200:
            id_train_dataset[idx_train] = new_vid_id
            list_of_lists = dp['embeddings']
            seq_len_train_dataset[idx_train] = len(list_of_lists)
            eos_gt_train_dataset[idx_train] = make_eos_gt(len(list_of_lists))
            emb_seq_train_dataset[idx_train] = padded_emb_seq_from_lists(list_of_lists, embedding_size, max_len)
            idx_train += 1
        
        elif new_vid_id <= 1300:
            print(new_vid_id)
            id_val_dataset[idx_val] = new_vid_id
            list_of_lists = dp['embeddings']
            seq_len_val_dataset[idx_val] = len(list_of_lists)
            eos_gt_val_dataset[idx_val] = make_eos_gt(len(list_of_lists))
            emb_seq_val_dataset[idx_val] = padded_emb_seq_from_lists(list_of_lists, embedding_size, max_len)
            idx_val += 1
        
        elif new_vid_id <= 1970:
            id_test_dataset[idx_test] = new_vid_id
            list_of_lists = dp['embeddings']
            seq_len_test_dataset[idx_test] = len(list_of_lists)
            eos_gt_test_dataset[idx_test] = make_eos_gt(len(list_of_lists))
            emb_seq_test_dataset[idx_test] = padded_emb_seq_from_lists(list_of_lists, embedding_size, max_len)
            idx_test += 1
    
    h5_f_train.close()
    h5_f_val.close()
    h5_f_test.close()


if __name__ == "__main__":
    
    convert_json_to_h5s('../data/dummy_data/data.json', '../data/dummy_data/train_data.h5', '../data/dummy_data/val_data.h5', '../data/dummy_data/test_data.h5')

    """
    from make_dummies import get_dummy_json_dpoint
    
    dummy = get_dummy_json_dpoint()
    dummy = [[7.,.6,2.,9.],[9.,6.,3.,2.,],[8.,5.,6.,3.]]
    print(dummy)
    #out = make_eos_gt(len(dummy), 10)
    out = padded_emb_seq_from_lists(dummy, embedding_size=4)
    print(out)
    print(make_eos_gt(len(dummy), 10))
    """





