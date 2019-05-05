"""Functions for creating a h5 file from a json in the agree-upon format.
"""


import json
import numpy as np
import h5py


def padded_emb_seq_from_lists(list_of_lists, embedding_size=300, max_len=10):
    """Convert a list of lists to a sorted, padded np array.

    Json files can't store np arrays, so the agreed-upon json format stores
    embeddings as a list of lists of floats. This function sorts the list on
    the first element, and zero pads up tot he max number of individuals.

    Args:
      lists_of_lists: the embedddings
      embedding_size: the dimension of each embeddiing
      max_len: the length to pad to sequence to
    """

    unsorted_unpadded_array = np.array(list_of_lists)
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


def convert_json_to_h5(json_file_path, out_h5_file_path, embedding_size=300, max_len=10):
    """Convert json in agreed-upon format to h5 in format expected by the Dataset.

    The json contains a list of datapoint, each having keys 'videoId', 'embed-
    dings', 'individuals' and 'captions'. Only the first two are passed to the
    h5 file. The videoIds are ints in the json file and are passed as is to the
    h5 file. The embeddings are unpadded, unsorted lists of lists in the json
    file. They are converted to np arrays, padded and sorted (by first element)
    before being passed to the h5 file. Their unpadded length is also measured
    and passed to the h5 file as 'embedding_len'. 

    The h5 file thus contains 3 datasets: videoId (ints), embeddings (np arrays)
    and embedding_lens (ints).

    Args:
      json_file_path: path of the json file to read from
      out_h5_file_path: path of the h5 file to write to
    """

    print('Reading json...')
    with open(json_file_path, 'r') as infile:
        embeddings_and_ids = json.load(infile)
    N = len(embeddings_and_ids)
    h5_f = h5py.File(out_h5_file_path, 'w')
    id_dataset = h5_f.create_dataset('videoId', (N,), dtype='uint32')
    emb_seq_dataset = h5_f.create_dataset('embeddings', (N,max_len,embedding_size), dtype=np.float64)
    seq_len_dataset = h5_f.create_dataset('embedding_len', (N,), dtype='uint8')
    eos_gt_dataset = h5_f.create_dataset('eos_gt', (N,max_len), dtype='uint8')
    print('Making h5...')
    for idx, dp in enumerate(embeddings_and_ids):
        new_vid_id = dp['videoId']
        id_dataset[idx] = new_vid_id
        list_of_lists = dp['embeddings']
        seq_len_dataset[idx] = len(list_of_lists)
        eos_gt_dataset[idx] = make_eos_gt(len(list_of_lists))
        emb_seq_dataset[idx] = padded_emb_seq_from_lists(list_of_lists, embedding_size, max_len)
    
    h5_f.close()


if __name__ == "__main__":
    
    convert_json_to_h5('../data/mini/val_data.json', '../data/mini/val_data.h5')

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





